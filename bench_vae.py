"""Is there a torch.compile win waiting behind the memory wall? (VAE decode, isolated)

In the full loop the compiled VAE decoder dies with a 586MB OOM: at the M1
operating point the card holds 22.47GB and has ~250MB free, while inductor wants
one more (1,96,4,480,832) fp32 buffer. That says the compile question is blocked
by memory, not answered by it.

So ask it where memory is free: load ONLY the VAE decoder and replay the real
latent trajectory saved at the M1 gate, block by block, with the same rolling
cache protocol as the server. Eager vs compiled, same tensors, same order.

If compiled wins here, the win is real and KV-cache 4-bit is what unlocks it in
the loop. If it does not, torch.compile is simply not the lever for this frame.

Run:
  DISABLE_SAGEATTENTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python bench_vae.py
"""
import os, json, time, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_fps"); OUT.mkdir(exist_ok=True)
RES = {"meta": {}, "phases": []}
def flush(): (OUT / "vae.json").write_text(json.dumps(RES, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)

from release_server import load_vae

vae_encoder, vae_decoder = load_vae()
lat = torch.load("results_v/latents_kv3_s4.pt", map_location="cpu", weights_only=True)
RES["meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "latents_shape": list(lat.shape), "latents_dtype": str(lat.dtype)}
print("latents", lat.shape, lat.dtype, flush=True)

# the server decodes one block of num_frame_per_block=3 latent frames at a time,
# threading a 55-slot rolling cache through every call
NFPB = 3
n_blocks = lat.shape[1] // NFPB
blocks = [lat[:, i * NFPB:(i + 1) * NFPB].to("cuda").half() for i in range(n_blocks)]
print("blocks", n_blocks, blocks[0].shape, flush=True)

def decode_all(dec, reps=1):
    """One full pass over the trajectory, fresh cache each pass (as a session does)."""
    per_block = []
    for _ in range(reps):
        cache = [None] * 55
        for b in blocks:
            torch.cuda.synchronize(); t = time.perf_counter()
            pixels, cache = dec(b, *cache)
            torch.cuda.synchronize()
            per_block.append(time.perf_counter() - t)
    return per_block, pixels

def phase(tag, dec, reps=2, note=""):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rec = {"tag": tag, "note": note, "ok": False}
    try:
        t0 = time.time()
        per_block, pixels = decode_all(dec, reps=reps)
        # first block of the first pass pays cache warmup; steady = the rest
        steady = per_block[1:]
        rec.update({"ok": True, "wall_sec": round(time.time() - t0, 2),
                    "blocks_timed": len(per_block),
                    "sec_per_block_mean": round(sum(steady) / len(steady), 4),
                    "sec_per_block_min": round(min(steady), 4),
                    "first_block_sec": round(per_block[0], 4),
                    "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
                    "pixels_shape": list(pixels.shape),
                    "pixels_finite": bool(torch.isfinite(pixels).all())})
    except Exception:
        rec["error"] = traceback.format_exc()[-1000:]
        torch.cuda.empty_cache()
    RES["phases"].append(rec); flush()
    print(json.dumps(rec, default=str)[:700], flush=True)
    return rec

# --- eager control -----------------------------------------------------------
eager = phase("eager", vae_decoder, reps=2, note="control")

# keep an eager reference for the fidelity check
ref_cache = [None] * 55
with torch.no_grad():
    ref_pixels, _ = vae_decoder(blocks[0], *ref_cache)
ref_pixels = ref_pixels.float().cpu()

# --- compiled ----------------------------------------------------------------
try:
    t = time.time()
    dec_c = torch.compile(vae_decoder, fullgraph=True)
    RES["meta"]["compile_call_sec"] = round(time.time() - t, 2)
    t = time.time()
    warm = phase("compiled_warmup", dec_c, reps=1, note="compilation happens here")
    RES["meta"]["first_pass_wall_sec"] = round(time.time() - t, 2)
    comp = phase("compiled", dec_c, reps=2, note="steady")
    if eager.get("ok") and comp.get("ok"):
        RES["meta"]["speedup_per_block"] = round(
            eager["sec_per_block_mean"] / comp["sec_per_block_mean"], 3)
        RES["meta"]["extra_peak_gb"] = round(
            comp["peak_alloc_gb"] - eager["peak_alloc_gb"], 3)
    # fidelity: compiled output must match eager
    c_cache = [None] * 55
    c_pixels, _ = dec_c(blocks[0], *c_cache)
    c_pixels = c_pixels.float().cpu()
    RES["meta"]["fidelity_vs_eager"] = {
        "max_abs_diff": float((c_pixels - ref_pixels).abs().max()),
        "mean_abs_diff": float((c_pixels - ref_pixels).abs().mean()),
        "finite": bool(torch.isfinite(c_pixels).all())}
except Exception:
    RES["meta"]["compile_error"] = traceback.format_exc()[-1500:]

flush()
print("DONE", flush=True)
print(json.dumps(RES["meta"], indent=1, default=str), flush=True)
