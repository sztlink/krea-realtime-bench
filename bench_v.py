"""Stage V: the 14B W4A4 in the causal loop on the RTX 4090 — the M1 gate.

The bf16 14B cannot exist on a 24GB card, so the loader never materializes it.
The model is built on the meta device, the non-quantized tensors are assigned in
from the original Krea checkpoint (non-block modules) and from the converted
unquantized_layers.safetensors (per-block norms, modulation, cross k/v), the
quantized slots are swapped for SVDQW4A4Linear straight onto the GPU, RoPE freqs
are recomputed (plain fp-complex attribute, deliberately not a buffer upstream),
and a scan asserts no meta parameter survives before the first forward.

Measures fps/memory across kv windows, saves latent trajectories (3 seeds) and
last frames. Fidelity reading is distributional (the L5 lesson): stability,
finite latents, internal diversity, coherent frames.

Run:
  DISABLE_SAGEATTENTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python bench_v.py
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open

OUT = Path("results_v"); OUT.mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = os.environ.get("V_QDIR", "checkpoints-14b-w4a4-ckv")
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

# 1. wrapper real, transformer skeleton on meta (via the from_pretrained patch)
transformer = WanDiffusionWrapper(model_name="Wan2.1-T2V-14B",
                                  timestep_shift=getattr(config, "timestep_shift", 5.0),
                                  is_causal=True)
model = transformer.model

# 2. assign real tensors for everything the quantizer did not consume
needed = {}
with safe_open(KREA_CKPT, framework="pt") as f:
    for k in f.keys():
        if not k.startswith("model.blocks."):
            needed[k] = f.get_tensor(k)          # patch/text/time embeddings, head
with safe_open(str(Path(QDIR) / "unquantized_layers.safetensors"), framework="pt") as f:
    for k in f.keys():
        needed[k] = f.get_tensor(k)              # norms, modulation, cross k/v
needed = {k: v.to(torch.bfloat16) for k, v in needed.items()}
missing, unexpected = transformer.load_state_dict(needed, strict=False, assign=True)
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
leftover = [k for k in missing if not any(q in k for q in QUANT_LOCALS)]
assert not leftover, f"non-quantized params left meta: {leftover[:8]}"

# 3. RoPE freqs: plain complex attribute, recomputed exactly as upstream __init__
d = model.dim // model.num_heads
model.freqs = torch.cat([
    rope_params(1024, d - 4 * (d // 6)),
    rope_params(1024, 2 * (d // 6)),
    rope_params(1024, 2 * (d // 6)),
], dim=1)

# 4. fuse slots (meta garbage, L1 dedup drops q/k/v) then swap in the real kernels
for blk in model.blocks:
    blk.self_attn.fuse_projections()
load_w4a4_blocks(model, QDIR, device="cuda")

# 5. nothing meta may survive
still_meta = [n for n, p in transformer.named_parameters() if p.device.type == "meta"]
assert not still_meta, f"meta params remain: {still_meta[:8]}"
transformer = transformer.to("cuda").to(torch.bfloat16)
transformer.eval(); transformer.requires_grad_(False)
torch.cuda.synchronize(); torch.cuda.empty_cache()
RES["meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
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

def run_cfg(tag, kv_frames=3, steps=4, seed=42, num_blocks=9, save_latents=False):
    print(f"===== RUN {tag} =====", flush=True)
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
        if save_latents:
            torch.save(latents, OUT / f"latents_{tag}.pt")
        if frames:
            from PIL import Image
            arr = ((frames[-1].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(OUT / f"frame_{tag}_last.png")
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc()[-1000:]; torch.cuda.empty_cache()
    RES["runs"].append(rec); flush()
    print(json.dumps({k: v for k, v in rec.items() if k != "block_sec"}, default=str)[:600], flush=True)

run_cfg("kv3_s4_warmup", 3, 4)
run_cfg("kv3_s4", 3, 4, save_latents=True)
run_cfg("kv3_s4_seed43", 3, 4, seed=43, save_latents=True)
run_cfg("kv3_s4_seed44", 3, 4, seed=44, save_latents=True)
run_cfg("kv6_s4", 6, 4)
print("DONE", flush=True)
