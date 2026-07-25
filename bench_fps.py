"""M1+ / fps: where the frame time goes, and what torch.compile actually buys.

The M1 gate left the 14B W4A4 at 2.81 fps steady (kv3, 4 steps) on the 4090. The
distance to real time is the next question, and torch.compile is the lever that
costs nothing and needs no infra decision (inductor codegens Triton; no CUDA
toolkit involved).

The catch, found by reading the runtime: the Nunchaku W4A4 Linears enter through
a pybind11 extension (`nunchaku._C.ops`), NOT the torch dispatcher. Dynamo cannot
trace an opaque pybind call, so compiling the transformer must graph-break at
every quantized slot (10 per block x 40 blocks). This bench does not assume that
verdict, it measures it, and it measures the attribution first so we know which
piece is even worth compiling.

Phases, all in one process so the control is exact (same load, same weights):
  A. eager, uninstrumented   -> the honest control fps
  B. eager, instrumented     -> split: recompute / denoise / vae decode / rest
  C. + compiled VAE decoder  -> the clean target (pure torch ops, fullgraph upstream)
  D. + compiled transformer  -> the graph-break question, answered with a number

What it found (2026-07-24, kv3 s4, 9 blocks):
  A/B  control reproduces the M1 gate exactly: 2.81 fps steady, 22.777GB peak.
       Frame splits into denoise ~68%, recompute 14.8%, vae decode 15.7%.
       (recompute calls the transformer too, so the "denoise" timer at 83.3%
       covers both; the pure-denoise share is the difference.)
  C/D  every compiled run OOMs by 586MB. Not a compile failure: at this operating
       point the card has ~250MB free and inductor wants one more
       (1,96,4,480,832) fp32 buffer. The compile question is memory-blocked, so
       it gets asked where memory is free instead -> bench_vae.py, which finds
       1.31x on the VAE decode for +6.32GB of peak. On a 15.7% slice that is
       ~+3.7% end to end (2.81 -> ~2.92 fps) for 6.3GB we do not have.

Verdict: torch.compile is not the lever for this frame. The 83% that matters is
behind opaque kernels, and the 16% that is compilable costs more memory than the
whole win is worth. What binds this card is memory, which is KV-cache 4-bit.

Run (on the 4090):
  DISABLE_SAGEATTENTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python bench_fps.py
"""
import os, json, time, asyncio, traceback
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open

OUT = Path("results_fps"); OUT.mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = os.environ.get("V_QDIR", "checkpoints-14b-w4a4-ckv")
NUM_BLOCKS = int(os.environ.get("FPS_BLOCKS", "9"))
RES = {"meta": {}, "runs": []}
def flush(): (OUT / "results.json").write_text(json.dumps(RES, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)
def mem():
    free, total = torch.cuda.mem_get_info()
    return {"torch_alloc_gb": gb(torch.cuda.memory_allocated()), "device_used_gb": gb(total - free)}

from release_server import load_merge_config, load_vae, load_pipeline, \
    GenerateParams, GenerationSession, Models
from utils.wan_wrapper import WanDiffusionWrapper
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import rope_params
from nunchaku_causal_wan import load_w4a4_blocks

def _from_config(path, **kw):
    # meta only for the transformer skeleton; the wrapper's scheduler must be real
    with torch.device("meta"):
        return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

QUANT_LOCALS = ("self_attn.q.", "self_attn.k.", "self_attn.v.", "self_attn.o.",
                "cross_attn.q.", "cross_attn.k.", "cross_attn.v.", "cross_attn.o.",
                "ffn.0.", "ffn.2.")

config = load_merge_config(CONFIG_PATH)
t0 = time.time()

transformer = WanDiffusionWrapper(model_name="Wan2.1-T2V-14B",
                                  timestep_shift=getattr(config, "timestep_shift", 5.0),
                                  is_causal=True)
model = transformer.model

needed = {}
with safe_open(KREA_CKPT, framework="pt") as f:
    for k in f.keys():
        if not k.startswith("model.blocks."):
            needed[k] = f.get_tensor(k)
with safe_open(str(Path(QDIR) / "unquantized_layers.safetensors"), framework="pt") as f:
    for k in f.keys():
        needed[k] = f.get_tensor(k)
needed = {k: v.to(torch.bfloat16) for k, v in needed.items()}
missing, unexpected = transformer.load_state_dict(needed, strict=False, assign=True)
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
leftover = [k for k in missing if not any(q in k for q in QUANT_LOCALS)]
assert not leftover, f"non-quantized params left meta: {leftover[:8]}"

d = model.dim // model.num_heads
model.freqs = torch.cat([
    rope_params(1024, d - 4 * (d // 6)),
    rope_params(1024, 2 * (d // 6)),
    rope_params(1024, 2 * (d // 6)),
], dim=1)

for blk in model.blocks:
    blk.self_attn.fuse_projections()
load_w4a4_blocks(model, QDIR, device="cuda")

still_meta = [n for n, p in transformer.named_parameters() if p.device.type == "meta"]
assert not still_meta, f"meta params remain: {still_meta[:8]}"
transformer = transformer.to("cuda").to(torch.bfloat16)
transformer.eval(); transformer.requires_grad_(False)
torch.cuda.synchronize(); torch.cuda.empty_cache()
RES["meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "triton": __import__("triton").__version__,
               "blocks": NUM_BLOCKS,
               "load": {"sec": round(time.time() - t0, 1), **mem()}}
print("LOAD OK", json.dumps(RES["meta"], default=str), flush=True)

import re as _re
_slug = _re.sub(r"[^a-z0-9]+", "-", PROMPT.lower())[:60].strip("-")
_emb = torch.load(f"embeddings/{_slug}.pt", map_location="cpu", weights_only=True)
_cond = {"prompt_embeds": _emb["prompt_embeds"].to("cuda", torch.bfloat16)}
class _StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in _cond.items()}
text_encoder = _StaticEnc()
vae_encoder, vae_decoder = load_vae()
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, text_encoder, vae_decoder)
models = Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)
flush()

# ---------------------------------------------------------------- attribution
# Timers sync around each piece: that serialization costs a little wall clock,
# so attribution runs are never the runs we quote as fps. The uninstrumented
# control is the number.
TIMERS = defaultdict(float); COUNTS = defaultdict(int); INSTRUMENT = {"on": False}

def _timed(name, fn):
    def wrapper(*a, **kw):
        if not INSTRUMENT["on"]:
            return fn(*a, **kw)
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn(*a, **kw)
        torch.cuda.synchronize()
        TIMERS[name] += time.perf_counter() - t; COUNTS[name] += 1
        return out
    return wrapper

GenerationSession.recompute_kv_cache = _timed("recompute", GenerationSession.recompute_kv_cache)
WanDiffusionWrapper.forward = _timed("denoise", WanDiffusionWrapper.forward)
_vae_fwd = vae_decoder.forward
vae_decoder.forward = _timed("vae_decode", _vae_fwd)

def _dynamo_counters():
    try:
        from torch._dynamo.utils import counters
        gb_ = counters.get("graph_break", {})
        return {"graph_break_kinds": len(gb_), "graph_breaks_total": int(sum(gb_.values())),
                "top": sorted(gb_.items(), key=lambda kv: -kv[1])[:4]}
    except Exception as e:
        return {"error": str(e)}

def run_cfg(tag, kv_frames=3, steps=4, seed=42, num_blocks=NUM_BLOCKS, instrument=False, note=""):
    print(f"===== RUN {tag} =====", flush=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    TIMERS.clear(); COUNTS.clear(); INSTRUMENT["on"] = instrument
    params = GenerateParams(prompt=PROMPT, seed=seed, kv_cache_num_frames=kv_frames,
                            num_blocks=num_blocks, num_denoising_steps=steps)
    n = {"n": 0}; frames = []
    def cb(pixels, frame_ids, event):
        event.synchronize(); n["n"] += pixels.shape[1]
        frames.append(pixels[0, -1].float().cpu())
    rec = {"tag": tag, "kv": kv_frames, "steps": steps, "seed": seed,
           "instrumented": instrument, "note": note, "ok": False}
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
        steady_wall = sum(block_times[1:]) if len(block_times) > 1 else wall
        latents = session.all_latents[:, :session.current_start_frame].cpu()
        rec.update({"ok": True, "wall_sec": round(wall, 2), "pixel_frames": n["n"],
            "block_sec": block_times,
            "fps_e2e": round(n["n"] / wall, 2),
            "fps_steady": round(max(0, n["n"] - 6) / steady_wall, 2) if steady_wall else None,
            "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
            "latents_finite": bool(torch.isfinite(latents).all()), **mem()})
        if instrument:
            acc = {k: round(v, 3) for k, v in TIMERS.items()}
            acc_sum = sum(TIMERS.values())
            rec["attribution_sec"] = acc
            rec["attribution_calls"] = dict(COUNTS)
            rec["attribution_pct"] = {k: round(100 * v / wall, 1) for k, v in TIMERS.items()}
            rec["attribution_pct"]["unattributed"] = round(100 * (wall - acc_sum) / wall, 1)
        if frames:
            from PIL import Image
            arr = ((frames[-1].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(OUT / f"frame_{tag}_last.png")
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc()[-1200:]; torch.cuda.empty_cache()
    finally:
        INSTRUMENT["on"] = False
    RES["runs"].append(rec); flush()
    print(json.dumps({k: v for k, v in rec.items() if k != "block_sec"}, default=str)[:900], flush=True)
    return rec

# A. eager control -------------------------------------------------------------
run_cfg("A_eager_warmup", note="first block pays cache init")
run_cfg("A_eager", note="control fps, uninstrumented")
# B. attribution ---------------------------------------------------------------
run_cfg("B_eager_attrib", instrument=True, note="syncs added; not an fps number")

# C. compiled VAE decoder ------------------------------------------------------
try:
    t = time.time()
    # compile the ORIGINAL forward and re-wrap the timer around the compiled
    # callable: wrapping the other way puts time.perf_counter() (a pybind
    # builtin) inside the traced region and dynamo refuses it.
    vae_decoder.forward = _timed("vae_decode", torch.compile(_vae_fwd, fullgraph=True))
    RES["meta"]["vae_compile_call_sec"] = round(time.time() - t, 2)
    run_cfg("C_vae_compiled_warmup", note="compilation happens here")
    run_cfg("C_vae_compiled", note="uninstrumented")
    run_cfg("C_vae_compiled_attrib", instrument=True, note="where the time moved")
    RES["meta"]["dynamo_after_C"] = _dynamo_counters()
except Exception:
    RES["meta"]["vae_compile_error"] = traceback.format_exc()[-1200:]
flush()

# D. + compiled transformer ----------------------------------------------------
try:
    t = time.time()
    models.transformer = torch.compile(models.transformer)
    RES["meta"]["tf_compile_call_sec"] = round(time.time() - t, 2)
    run_cfg("D_tf_compiled_warmup", note="compilation + graph breaks happen here")
    run_cfg("D_tf_compiled", note="uninstrumented")
    RES["meta"]["dynamo_after_D"] = _dynamo_counters()
except Exception:
    RES["meta"]["tf_compile_error"] = traceback.format_exc()[-1200:]

flush()
print("DONE", flush=True)
print(json.dumps({r["tag"]: {"fps_steady": r.get("fps_steady"), "peak": r.get("peak_alloc_gb"),
                             "err": bool(r.get("error"))} for r in RES["runs"]}, indent=1), flush=True)
