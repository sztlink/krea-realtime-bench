"""The QuantKVCache in the real loop, and the window that starts fitting.

The M1 gate left the 14B W4A4 at 2.81 fps with a 22.77 GB peak and 0.92 GB free. The 6 and
12 frame windows died inside _initialize_kv_cache, short by 138 and 230 megabytes. This
measures what quantizing the cache buys.

The bf16 control runs first, in the same process, then the schemes. Each window runs as far
as it fits.
  kv3   bf16 cache 7.67 GB   ->  int4 about 2.10 GB
  kv6   bf16 cache 11.5 GB (does not fit)  ->  int4 about 3.15 GB
  kv12  bf16 cache 19.2 GB (impossible)    ->  int4 about 5.24 GB

Note on method. Any run that follows an out of memory failure in the same process is not
trustworthy. Allocator state after OOM degrades and empty_cache does not undo it, which
cost a 2x measurement error here before it was isolated in a clean process.

Run:
  DISABLE_SAGEATTENTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python bench_qkv.py
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open

import quant_kv

OUT = Path("results_qkv"); OUT.mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = os.environ.get("V_QDIR", "checkpoints-14b-w4a4-ckv")
BLOCKS = int(os.environ.get("QKV_BLOCKS", "9"))
RES = {"meta": {}, "runs": []}
def flush(): (OUT / "bench.json").write_text(json.dumps(RES, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)

from release_server import load_merge_config, load_vae, load_pipeline, \
    GenerateParams, GenerationSession, Models
from utils.wan_wrapper import WanDiffusionWrapper
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import rope_params
from nunchaku_causal_wan import load_w4a4_blocks

def _from_config(path, **kw):
    with torch.device("meta"):
        return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config(CONFIG_PATH)
transformer = WanDiffusionWrapper(model_name="Wan2.1-T2V-14B",
                                  timestep_shift=getattr(config, "timestep_shift", 5.0), is_causal=True)
model = transformer.model
needed = {}
with safe_open(KREA_CKPT, framework="pt") as f:
    for k in f.keys():
        if not k.startswith("model.blocks."): needed[k] = f.get_tensor(k)
with safe_open(str(Path(QDIR) / "unquantized_layers.safetensors"), framework="pt") as f:
    for k in f.keys(): needed[k] = f.get_tensor(k)
needed = {k: v.to(torch.bfloat16) for k, v in needed.items()}
transformer.load_state_dict(needed, strict=False, assign=True)
d = model.dim // model.num_heads
model.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)),
                         rope_params(1024, 2 * (d // 6))], dim=1)
for blk in model.blocks: blk.self_attn.fuse_projections()
load_w4a4_blocks(model, QDIR, device="cuda")
transformer = transformer.to("cuda").to(torch.bfloat16)
transformer.eval(); transformer.requires_grad_(False)
torch.cuda.empty_cache()

import re as _re
_slug = _re.sub(r"[^a-z0-9]+", "-", PROMPT.lower())[:60].strip("-")
_emb = torch.load(f"embeddings/{_slug}.pt", map_location="cpu", weights_only=True)
_cond = {"prompt_embeds": _emb["prompt_embeds"].to("cuda", torch.bfloat16)}
class _StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in _cond.items()}
vae_encoder, vae_decoder = load_vae()
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, _StaticEnc(), vae_decoder)
models = Models(_StaticEnc(), transformer, pipeline, vae_encoder, vae_decoder)
ORIG_INIT = pipeline._initialize_kv_cache
RES["meta"] = {"gpu": torch.cuda.get_device_name(0), "blocks": BLOCKS}
print("LOAD OK", flush=True)

def cache_bytes():
    c = getattr(pipeline, "kv_cache1", None)
    if not c: return 0
    e = c[0]["k"]
    if isinstance(e, quant_kv.QuantKVTensor):
        return sum(x["k"].storage_bytes() + x["v"].storage_bytes() for x in c)
    return sum(x["k"].numel() * x["k"].element_size() * 2 for x in c)

def run(tag, kv_frames, scheme=None, save_clip=False):
    """scheme None = bf16 upstream; senao (k_bits, v_bits, group_mode, sink_frames)."""
    print(f"===== {tag} =====", flush=True)
    pipeline.kv_cache1 = []                     # forca realocacao
    if scheme is None:
        pipeline._initialize_kv_cache = ORIG_INIT
    else:
        kb, vb, mode, sink = scheme
        quant_kv.install(pipeline, k_bits=kb, v_bits=vb, group_mode=mode,
                         sink_frames=sink, verbose=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rec = {"tag": tag, "kv_frames": kv_frames,
           "scheme": "bf16" if scheme is None else f"k{scheme[0]}v{scheme[1]}_{scheme[2]}_sink{scheme[3]}",
           "ok": False}
    frames = []
    def cb(pixels, frame_ids, event):
        event.synchronize()
        if save_clip:
            frames.append(pixels[0].float().cpu())
    try:
        params = GenerateParams(prompt=PROMPT, seed=42, kv_cache_num_frames=kv_frames,
                                num_blocks=BLOCKS, num_denoising_steps=4)
        session = GenerationSession(params, config, frame_callback=cb, models=models)
        rec["cache_gb"] = gb(cache_bytes())
        torch.cuda.synchronize(); t0 = time.time()
        block_times = []
        n = 0
        for _ in range(BLOCKS):
            tb = time.time()
            try: out = session.generate_block(models)
            except asyncio.CancelledError: break
            torch.cuda.synchronize(); block_times.append(round(time.time() - tb, 3))
            n += out.shape[1]
        wall = time.time() - t0
        steady = sum(block_times[1:]) if len(block_times) > 1 else wall
        latents = session.all_latents[:, :session.current_start_frame].cpu()
        free, total = torch.cuda.mem_get_info()
        rec.update({"ok": True, "wall_sec": round(wall, 2), "frames": n,
                    "fps_e2e": round(n / wall, 2),
                    "fps_steady": round(max(0, n - 6) / steady, 2) if steady else None,
                    "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
                    "device_used_gb": gb(total - free), "device_free_gb": gb(free),
                    "latents_finite": bool(torch.isfinite(latents).all()),
                    "latent_absmax": round(float(latents.abs().max()), 3)})
        if save_clip and frames:
            import subprocess
            from PIL import Image
            cd = OUT / tag; cd.mkdir(parents=True, exist_ok=True)
            i = 0
            for blk in frames:
                for j in range(blk.shape[0]):
                    arr = ((blk[j].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
                    Image.fromarray(arr).save(cd / f"f{i:04d}.png"); i += 1
            subprocess.run(["ffmpeg", "-y", "-framerate", "16", "-i", str(cd / "f%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                            str(OUT / f"{tag}.mp4")], check=True, capture_output=True)
            rec["mp4"] = f"{tag}.mp4"
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc()[-800:]
        torch.cuda.empty_cache()
    RES["runs"].append(rec); flush()
    print(json.dumps(rec, default=str)[:600], flush=True)
    torch.cuda.empty_cache()
    return rec

BANDS4 = (4, 4, "bands", 0)
K8V4 = (8, 4, "bands", 0)

run("bf16_kv3",  3, None,   save_clip=True)     # controle: 2.81 fps, 22.77 GB
run("q44_kv3",   3, BANDS4, save_clip=True)
run("q84_kv3",   3, K8V4,   save_clip=True)
run("bf16_kv6",  6, None)                        # esperado: OOM no init
run("q44_kv6",   6, BANDS4, save_clip=True)      # the window that starts fitting
run("q84_kv6",   6, K8V4)
run("q44_kv12", 12, BANDS4, save_clip=True)      # o premio grande

flush()
print("DONE", flush=True)
print(json.dumps([{k: r.get(k) for k in ("tag", "scheme", "kv_frames", "cache_gb",
                                         "fps_steady", "peak_alloc_gb", "latents_finite")}
                  for r in RES["runs"]], indent=1), flush=True)
