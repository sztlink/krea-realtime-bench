"""L0 baseline bench for the Self-Forcing 1.3B on the local 4090.

Same instrumentation as bench_m0b.py (14B). Config via BENCH_CONFIG, defaults
to the 1.3B server config. Full-stack run first (T5 resident), then static
embedding swap, then the window sweep including the 21-frame global window,
which fits a 24GB card at 1.3B scale. Results in results_1p3b/results.json.
"""
import os, sys, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_1p3b"); OUT.mkdir(exist_ok=True)
RESULTS = {"meta": {}, "runs": []}
PROMPT = os.environ.get("BENCH_PROMPT", "A person dancing in an empty warehouse, dramatic lighting, camera static")
CONFIG_PATH = os.environ.get("BENCH_CONFIG", "configs/self_forcing_server.yaml")
def flush(): (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)
def mem():
    free, total = torch.cuda.mem_get_info()
    return {"torch_alloc_gb": gb(torch.cuda.memory_allocated()), "device_used_gb": gb(total - free),
            "device_total_gb": gb(total)}

from release_server import load_merge_config, load_transformer, \
    load_vae, load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel
def _from_config(path, **kw):
    return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config(CONFIG_PATH)
RESULTS["meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                   "config": CONFIG_PATH, "prompt": PROMPT}
t0 = time.time(); transformer = load_transformer(config); torch.cuda.synchronize()
RESULTS["meta"]["load_transformer"] = {"sec": round(time.time() - t0, 1), **mem()}
RESULTS["meta"]["param_count"] = sum(p.numel() for p in transformer.parameters())
# stock T5 loader allocates 22.7GB fp32 on GPU, impossible on a 24GB card.
# Embedding precomputed on CPU by make_embedding.py, loaded as a static encoder.
import re as _re, glob as _glob
_slug = _re.sub(r"[^a-z0-9]+", "-", PROMPT.lower())[:60].strip("-")
_emb = torch.load(f"embeddings/{_slug}.pt", map_location="cpu", weights_only=True)
_cond = {"prompt_embeds": _emb["prompt_embeds"].to("cuda")}
class _StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in _cond.items()}
text_encoder = _StaticEnc()
RESULTS["meta"]["t5_mode"] = "static-embedding-cpu-precomputed (stock loader needs 22.7GB fp32 on GPU)"
vae_encoder, vae_decoder = load_vae()
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, text_encoder, vae_decoder)
models = Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)
flush(); print("LOAD OK", json.dumps(RESULTS["meta"], default=str))

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

def run_cfg(tag, kv_frames=3, steps=4, seed=42, num_blocks=9, save_latents=False):
    print(f"===== RUN {tag} ====="); EVENTS.clear()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    params = GenerateParams(prompt=PROMPT, seed=seed, kv_cache_num_frames=kv_frames,
                            num_blocks=num_blocks, num_denoising_steps=steps)
    n = {"n": 0}; frames = []
    def cb(pixels, frame_ids, event):
        event.synchronize(); n["n"] += pixels.shape[1]
        frames.append(pixels[0, -1].float().cpu())
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
        if frames:
            from PIL import Image
            arr = ((frames[-1].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(OUT / f"frame_{tag}_last.png")
        session.dispose()
    except torch.cuda.OutOfMemoryError as e:
        rec["error"] = f"OOM: {e}"; torch.cuda.empty_cache()
    except Exception:
        rec["error"] = traceback.format_exc(); torch.cuda.empty_cache()
    RESULTS["runs"].append(rec); flush()
    print(json.dumps({k: v for k, v in rec.items() if k != "block_sec"}, default=str)[:900])

run_cfg("kv3_s4", 3, 4, save_latents=True)
run_cfg("kv3_s4_seed43", 3, 4, seed=43, save_latents=True)
run_cfg("kv3_s4_seed44", 3, 4, seed=44, save_latents=True)
run_cfg("kv3_s5", 3, 5)
run_cfg("kv6_s4", 6, 4)
run_cfg("kv12_s4", 12, 4)
run_cfg("kv21_s4", 21, 4, save_latents=True)   # global window fits at 1.3B scale
print("===== 1.3B BASELINE DONE =====")
flush()
