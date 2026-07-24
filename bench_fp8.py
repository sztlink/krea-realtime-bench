"""fp8 companion bench (torchao W8A8 per-tensor), same instrumentation as bench_m0b.py.

Runs in a fresh process on purpose. Reloading the transformer inside the same
process OOMs, the old bf16 weights stay resident through the new load.
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_m0b"); OUT.mkdir(exist_ok=True)
RESULTS = {"meta": {}, "runs": []}
PROMPT = os.environ.get("BENCH_PROMPT", "A person dancing in an empty warehouse, dramatic lighting, camera static")
def flush(): (OUT / "results_fp8.json").write_text(json.dumps(RESULTS, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)
def mem():
    free, total = torch.cuda.mem_get_info()
    return {"torch_alloc_gb": gb(torch.cuda.memory_allocated()), "device_used_gb": gb(total - free)}

from release_server import load_merge_config, load_transformer, load_text_encoder, \
    load_vae, load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel
def _from_config(path, **kw):
    return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config("configs/self_forcing_server_14b.yaml")
config.enable_fp8 = True
t0 = time.time(); transformer = load_transformer(config); torch.cuda.synchronize()
RESULTS["meta"]["fp8_load"] = {"sec": round(time.time() - t0, 1), **mem()}
text_encoder = load_text_encoder()
vae_encoder, vae_decoder = load_vae()
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, text_encoder, vae_decoder)
models = Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)
flush(); print("FP8 LOAD OK", RESULTS["meta"])

cond = models.text_encoder(text_prompts=[PROMPT])
cond = {k: v.detach().clone() for k, v in cond.items()}
class StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in cond.items()}
real_t5 = models.text_encoder; models.text_encoder = StaticEnc()
real_t5.to("cpu"); torch.cuda.empty_cache()
RESULTS["meta"]["after_t5_offload"] = mem(); flush()

PHASE = {"tag": "denoise"}; EVENTS = []
orig_fwd = models.transformer.forward
def timed(*a, **k):
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record(); out = orig_fwd(*a, **k); e1.record()
    EVENTS.append((PHASE["tag"], e0, e1)); return out
models.transformer.forward = timed
_orig_recomp = GenerationSession.recompute_kv_cache
def timed_recomp(self, mdl):
    PHASE["tag"] = "prefill"
    try: return _orig_recomp(self, mdl)
    finally: PHASE["tag"] = "denoise"
GenerationSession.recompute_kv_cache = timed_recomp

def run_cfg(tag, kv_frames=3, steps=4, seed=42, num_blocks=9):
    print(f"===== RUN {tag} ====="); EVENTS.clear()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    params = GenerateParams(prompt=PROMPT, seed=seed, kv_cache_num_frames=kv_frames,
                            num_blocks=num_blocks, num_denoising_steps=steps)
    n = {"n": 0}
    def cb(pixels, frame_ids, event):
        event.synchronize(); n["n"] += pixels.shape[1]
    rec = {"tag": tag, "kv": kv_frames, "steps": steps, "seed": seed, "ok": False}
    try:
        session = GenerationSession(params, config, frame_callback=cb, models=models)
        block_times = []
        torch.cuda.synchronize(); t_all = time.time()
        for i in range(num_blocks):
            tb = time.time()
            try: session.generate_block(models)
            except asyncio.CancelledError: break
            torch.cuda.synchronize(); block_times.append(round(time.time() - tb, 3))
        wall = time.time() - t_all
        torch.cuda.synchronize()
        fw = [(t, round(e0.elapsed_time(e1), 2)) for t, e0, e1 in EVENTS]; EVENTS.clear()
        pre = [ms for t, ms in fw if t == "prefill"]; den = [ms for t, ms in fw if t == "denoise"]
        steady_wall = sum(block_times[1:]) if len(block_times) > 1 else wall
        rec.update({"ok": True, "wall_sec": round(wall, 2), "pixel_frames": n["n"],
            "fps_e2e": round(n["n"] / wall, 2),
            "fps_steady": round(max(0, n["n"] - 6) / steady_wall, 2) if steady_wall else None,
            "prefill_ms_mean": round(sum(pre)/len(pre), 1) if pre else None,
            "denoise_ms_mean": round(sum(den)/len(den), 1) if den else None,
            "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()), **mem()})
        torch.save(session.all_latents[:, :session.current_start_frame].cpu(), OUT / f"latents_{tag}.pt")
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc(); torch.cuda.empty_cache()
    RESULTS["runs"].append(rec); flush(); print(json.dumps(rec, default=str)[:800])

run_cfg("fp8_kv3_s4", 3, 4, 42)
run_cfg("fp8_kv3_s4_seed43", 3, 4, 43)
run_cfg("fp8_kv3_s5", 3, 5, 42)
print("===== FP8 DONE =====")
