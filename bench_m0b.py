"""Baseline bench for Krea Realtime 14B. Run from inside the realtime-video repo.

    uv run python bench_m0b.py [--prompt "your prompt"]      (or BENCH_PROMPT env)

Measures per-stage load memory, peak VRAM, real KV-cache size, prefill (cache
rebuild) vs denoise cost per forward, end-to-end and steady-state fps, a sweep
over kv_cache_num_frames and step counts, and multi-seed runs as trajectory
references. It instruments the stock code path at runtime instead of forking
it, so what gets measured is what ships. Results stream to
results_m0b/results.json, latent trajectories are saved as .pt.
"""
import os, sys, json, time, asyncio, traceback
from pathlib import Path

os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_m0b"); OUT.mkdir(exist_ok=True)
RESULTS = {"meta": {}, "runs": []}
PROMPT = os.environ.get("BENCH_PROMPT", "A person dancing in an empty warehouse, dramatic lighting, camera static")
if "--prompt" in sys.argv:
    PROMPT = sys.argv[sys.argv.index("--prompt") + 1]

def flush():
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))

def gb(x): return round(x / 1e9, 3)

def mem_snapshot():
    free, total = torch.cuda.mem_get_info()
    return {"torch_alloc_gb": gb(torch.cuda.memory_allocated()),
            "torch_reserved_gb": gb(torch.cuda.memory_reserved()),
            "device_used_gb": gb(total - free), "device_total_gb": gb(total)}

# ---------------------------------------------------------------- load (staged)
from release_server import load_merge_config, load_transformer, load_text_encoder, \
    load_vae, load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel

# We only need the 14B config.json. Instantiate via from_config and let the Krea
# checkpoint state_dict overwrite everything (freqs is recomputed in __init__).
_orig_from_pretrained = CausalWanModel.from_pretrained.__func__ if hasattr(CausalWanModel.from_pretrained, "__func__") else CausalWanModel.from_pretrained
def _from_config(path, **kw):
    conf = CausalWanModel.load_config(str(path))
    model = CausalWanModel.from_config(conf, **kw)
    return model
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config("configs/self_forcing_server_14b.yaml")

RESULTS["meta"] = {
    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    "config": "self_forcing_server_14b.yaml", "prompt": PROMPT,
    "do_compile": os.environ.get("DO_COMPILE"),
}
stages = {}
t0 = time.time(); transformer = load_transformer(config); torch.cuda.synchronize()
stages["transformer"] = {"sec": round(time.time() - t0, 1), **mem_snapshot()}
model_cfg = transformer.model.config
RESULTS["meta"]["model_config"] = {k: getattr(model_cfg, k, None) for k in
    ["dim", "ffn_dim", "num_heads", "num_layers", "patch_size", "in_dim", "out_dim"]}
RESULTS["meta"]["param_count_transformer"] = sum(p.numel() for p in transformer.parameters())

t0 = time.time(); text_encoder = load_text_encoder(); torch.cuda.synchronize()
stages["text_encoder"] = {"sec": round(time.time() - t0, 1), **mem_snapshot()}
RESULTS["meta"]["param_count_t5"] = sum(p.numel() for p in text_encoder.parameters())

t0 = time.time(); vae_encoder, vae_decoder = load_vae(); torch.cuda.synchronize()
stages["vae"] = {"sec": round(time.time() - t0, 1), **mem_snapshot()}

pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, text_encoder, vae_decoder)
models = Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)
RESULTS["meta"]["load_stages"] = stages
flush()
print("LOAD OK", json.dumps(stages, indent=1))

# ------------------------------------------------------- instrumentation
PHASE = {"tag": "denoise"}
EVENTS = []            # (tag, ev0, ev1)
def wrap_transformer(m):
    orig = m.forward
    def timed(*a, **k):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); out = orig(*a, **k); e1.record()
        EVENTS.append((PHASE["tag"], e0, e1))
        return out
    m.forward = timed
wrap_transformer(models.transformer)

_orig_recomp = GenerationSession.recompute_kv_cache
def timed_recomp(self, mdl):
    PHASE["tag"] = "prefill"
    try: return _orig_recomp(self, mdl)
    finally: PHASE["tag"] = "denoise"
GenerationSession.recompute_kv_cache = timed_recomp

def drain_events():
    torch.cuda.synchronize()
    out = [(tag, round(e0.elapsed_time(e1), 2)) for tag, e0, e1 in EVENTS]
    EVENTS.clear()
    return out

def kv_bytes(pipe):
    b = 0
    for c in (pipe.kv_cache1 or []):
        b += c["k"].numel() * c["k"].element_size() + c["v"].numel() * c["v"].element_size()
    x = 0
    for c in (getattr(pipe, "crossattn_cache", None) or []):
        x += c["k"].numel() * c["k"].element_size() + c["v"].numel() * c["v"].element_size()
    return b, x

# ------------------------------------------------------------------ one run
def run_cfg(tag, kv_frames=3, steps=5, seed=42, num_blocks=9, save_latents=True):
    print(f"\n===== RUN {tag} (kv={kv_frames} steps={steps} seed={seed}) =====")
    EVENTS.clear()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    params = GenerateParams(prompt=PROMPT, seed=seed, kv_cache_num_frames=kv_frames,
                            num_blocks=num_blocks, num_denoising_steps=steps)
    n_frames = {"n": 0}; last_frames = []
    def cb(pixels, frame_ids, event):
        event.synchronize()
        n_frames["n"] += pixels.shape[1]
        last_frames.append(pixels[0, -1].float().cpu())
    rec = {"tag": tag, "kv_cache_num_frames": kv_frames, "steps": steps, "seed": seed,
           "num_blocks": num_blocks, "ok": False}
    try:
        session = GenerationSession(params, config, frame_callback=cb, models=models)
        kvb, xb = kv_bytes(models.pipeline)
        rec["kv_cache_gb"] = gb(kvb); rec["crossattn_cache_gb"] = gb(xb)
        rec["kv_cache_tokens"] = list(models.pipeline.kv_cache1[0]["k"].shape)
        block_times = []
        torch.cuda.synchronize(); t_all = time.time()
        for i in range(num_blocks):
            tb = time.time()
            try: session.generate_block(models)
            except asyncio.CancelledError: break
            torch.cuda.synchronize()
            block_times.append(round(time.time() - tb, 3))
        wall = time.time() - t_all
        fw = drain_events()
        pre = [ms for t, ms in fw if t == "prefill"]
        den = [ms for t, ms in fw if t == "denoise"]
        steady_wall = sum(block_times[1:]) if len(block_times) > 1 else wall
        frames_b0 = 9 - 3  # block 0 delivers 6 frames (first 3 skipped)
        steady_frames = max(0, n_frames["n"] - frames_b0)
        rec.update({
            "ok": True, "wall_sec": round(wall, 2), "block_sec": block_times,
            "pixel_frames": n_frames["n"],
            "fps_e2e": round(n_frames["n"] / wall, 2),
            "fps_steady": round(steady_frames / steady_wall, 2) if steady_wall else None,
            "prefill_ms": {"n": len(pre), "mean": round(sum(pre)/len(pre), 1) if pre else None,
                           "per_block": [round(x, 1) for x in pre]},
            "denoise_ms": {"n": len(den), "mean": round(sum(den)/len(den), 1) if den else None,
                           "steady_mean": round(sum(den[steps:])/max(1, len(den)-steps), 1) if len(den) > steps else None},
            "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
            "peak_reserved_gb": gb(torch.cuda.max_memory_reserved()),
            **{f"mem_{k}": v for k, v in mem_snapshot().items()},
        })
        if save_latents:
            torch.save(session.all_latents[:, :session.current_start_frame].cpu(), OUT / f"latents_{tag}.pt")
        if last_frames:
            from PIL import Image
            import numpy as np
            for name, fr in [("first", last_frames[0]), ("last", last_frames[-1])]:
                arr = ((fr.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
                Image.fromarray(arr).save(OUT / f"frame_{tag}_{name}.png")
        session.dispose()
    except torch.cuda.OutOfMemoryError as e:
        rec["error"] = f"OOM: {e}"; torch.cuda.empty_cache()
    except Exception as e:
        rec["error"] = traceback.format_exc(); torch.cuda.empty_cache()
    RESULTS["runs"].append(rec); flush()
    print(json.dumps({k: v for k, v in rec.items() if k != "block_sec"}, indent=1, default=str)[:1200])
    return rec

# 1. full run with the T5 encoder resident, the memory profile of the whole stack
run_cfg("full_kv3_s5", kv_frames=3, steps=5, seed=42)

# 2. swap T5 for a static embedding and move it to CPU (frees ~11.4GB)
cond = models.text_encoder(text_prompts=[PROMPT])
cond = {k: v.detach().clone() for k, v in cond.items()}
class StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in cond.items()}
real_t5 = models.text_encoder
models.text_encoder = StaticEnc()
real_t5.to("cpu"); torch.cuda.empty_cache()
RESULTS["meta"]["after_t5_offload"] = mem_snapshot(); flush()

# 3. main sweep
run_cfg("kv3_s4", 3, 4)
run_cfg("kv6_s4", 6, 4)
run_cfg("kv12_s4", 12, 4)
run_cfg("kv21_s4", 21, 4)          # near-global window, OOM-guarded
# 4. multi-seed runs at the perf-claim config, trajectory references
run_cfg("kv3_s4_seed43", 3, 4, seed=43)
run_cfg("kv3_s4_seed44", 3, 4, seed=44)

# 5. fp8 (torchao W8A8 per-tensor, a code path the repo already ships)
try:
    print("\n===== reload fp8 =====")
    del models.transformer, models.pipeline, transformer, pipeline
    torch.cuda.empty_cache()
    import copy as _copy
    cfg8 = _copy.deepcopy(config); cfg8.enable_fp8 = True
    t0 = time.time(); tr8 = load_transformer(cfg8); torch.cuda.synchronize()
    RESULTS["meta"]["fp8_load"] = {"sec": round(time.time() - t0, 1), **mem_snapshot()}
    pipe8 = load_pipeline(cfg8, torch.cuda.current_device(), tr8, models.text_encoder, models.vae_decoder)
    models.transformer = tr8; models.pipeline = pipe8
    wrap_transformer(models.transformer)
    flush()
    run_cfg("fp8_kv3_s4", 3, 4)
    run_cfg("fp8_kv3_s4_seed43", 3, 4, seed=43)
except Exception:
    RESULTS["meta"]["fp8_error"] = traceback.format_exc(); flush()

print("\n===== M0.B DONE =====")
flush()
