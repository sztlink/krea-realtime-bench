"""Window-sweep extension of the Krea Realtime 14B baseline, for GPUs above 80GB.

Completes the recompute cost curve where the H100 hit its wall. Runs the
kv_cache_num_frames sweep 3 / 12 / 15 / 18 / 21 / 24 with the T5 encoder
swapped for a static embedding from the start. Same runtime instrumentation
as bench_m0b.py. Results in results_kv21/results.json.
"""
import os, sys, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_kv21"); OUT.mkdir(exist_ok=True)
RESULTS = {"meta": {}, "runs": []}
PROMPT = os.environ.get("BENCH_PROMPT", "A person dancing in an empty warehouse, dramatic lighting, camera static")
def flush(): (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)
def mem():
    free, total = torch.cuda.mem_get_info()
    return {"torch_alloc_gb": gb(torch.cuda.memory_allocated()), "device_used_gb": gb(total - free),
            "device_total_gb": gb(total)}

from release_server import load_merge_config, load_transformer, load_text_encoder, \
    load_vae, load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel
def _from_config(path, **kw):
    return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config("configs/self_forcing_server_14b.yaml")
RESULTS["meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "prompt": PROMPT}
t0 = time.time(); transformer = load_transformer(config); torch.cuda.synchronize()
RESULTS["meta"]["load_transformer"] = {"sec": round(time.time() - t0, 1), **mem()}
text_encoder = load_text_encoder()
vae_encoder, vae_decoder = load_vae()
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, text_encoder, vae_decoder)
models = Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)

cond = models.text_encoder(text_prompts=[PROMPT])
cond = {k: v.detach().clone() for k, v in cond.items()}
class StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in cond.items()}
real_t5 = models.text_encoder; models.text_encoder = StaticEnc()
real_t5.to("cpu"); torch.cuda.empty_cache()
RESULTS["meta"]["after_t5_offload"] = mem(); flush()
print("LOAD OK", RESULTS["meta"])

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

def kv_bytes(pipe):
    return sum(c["k"].numel() * c["k"].element_size() + c["v"].numel() * c["v"].element_size()
               for c in (pipe.kv_cache1 or []))

def run_cfg(tag, kv_frames, steps=4, seed=42, num_blocks=9, save_latents=False):
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
        rec["kv_cache_gb"] = gb(kv_bytes(models.pipeline))
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
            "block_sec": block_times,
            "fps_e2e": round(n["n"] / wall, 2),
            "fps_steady": round(max(0, n["n"] - 6) / steady_wall, 2) if steady_wall else None,
            "prefill_ms": {"mean": round(sum(pre)/len(pre), 1) if pre else None,
                           "per_block": [round(x, 1) for x in pre]},
            "denoise_ms_mean": round(sum(den)/len(den), 1) if den else None,
            "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()), **mem()})
        if save_latents:
            torch.save(session.all_latents[:, :session.current_start_frame].cpu(), OUT / f"latents_{tag}.pt")
        session.dispose()
    except torch.cuda.OutOfMemoryError as e:
        rec["error"] = f"OOM: {e}"; torch.cuda.empty_cache()
    except Exception:
        rec["error"] = traceback.format_exc(); torch.cuda.empty_cache()
    RESULTS["runs"].append(rec); flush()
    print(json.dumps({k: v for k, v in rec.items() if k != "block_sec"}, default=str)[:900])

run_cfg("kv3_s4_h200", 3)        # cross-GPU sanity point vs the H100 numbers
run_cfg("kv12_s4_h200", 12)      # curve overlap point
run_cfg("kv15_s4", 15)
run_cfg("kv18_s4", 18)
run_cfg("kv21_s4", 21, save_latents=True)   # the window that OOMed on 80GB
run_cfg("kv24_s4", 24)           # past-global, block 9 is the only full-window block
print("===== KV SWEEP DONE =====")
flush()
