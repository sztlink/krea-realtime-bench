"""Memory profile of the 14B W4A4 loop on the 4090 — the open M0.a checklist item.

M0.a projected the KV cache at 7.67GB bf16 for the server window (6 physical
frames, 9360 tokens, 40 layers, k+v) and ~2.2GB at 4 bits. The fps probe then
found the card sitting with ~250MB free at the M1 operating point. If that 7.67GB
is real, KV-cache 4-bit is not "the other half of the work", it is the only place
5+GB can come from on this card.

So count the bytes instead of projecting them: walk the live cache tensors after
the session initializes them, and stage the allocator around load / cache init /
steady state.

Run:
  DISABLE_SAGEATTENTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python bench_mem.py
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open

OUT = Path("results_fps"); OUT.mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = os.environ.get("V_QDIR", "checkpoints-14b-w4a4-ckv")
RES = {"stages": {}, "caches": {}, "windows": []}
def flush(): (OUT / "mem.json").write_text(json.dumps(RES, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)
def stage(name):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    RES["stages"][name] = {"alloc_gb": gb(torch.cuda.memory_allocated()),
                           "reserved_gb": gb(torch.cuda.memory_reserved()),
                           "device_used_gb": gb(total - free),
                           "device_free_gb": gb(free)}
    print(name, json.dumps(RES["stages"][name]), flush=True)

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

QUANT_LOCALS = ("self_attn.q.", "self_attn.k.", "self_attn.v.", "self_attn.o.",
                "cross_attn.q.", "cross_attn.k.", "cross_attn.v.", "cross_attn.o.",
                "ffn.0.", "ffn.2.")

config = load_merge_config(CONFIG_PATH)
stage("00_empty")

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
transformer.load_state_dict(needed, strict=False, assign=True)
d = model.dim // model.num_heads
model.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)),
                         rope_params(1024, 2 * (d // 6)),
                         rope_params(1024, 2 * (d // 6))], dim=1)
for blk in model.blocks:
    blk.self_attn.fuse_projections()
load_w4a4_blocks(model, QDIR, device="cuda")
transformer = transformer.to("cuda").to(torch.bfloat16)
transformer.eval(); transformer.requires_grad_(False)
torch.cuda.synchronize(); torch.cuda.empty_cache()
stage("10_weights_loaded")

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
stage("20_vae_and_pipeline")

def walk(obj, depth=0):
    """Sum tensor bytes in the nested list/dict cache structures, by key."""
    tot = 0; per_key = {}
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size(), {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            b, _ = walk(v, depth + 1)
            tot += b; per_key[k] = per_key.get(k, 0) + b
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            b, sub = walk(v, depth + 1)
            tot += b
            for k2, b2 in sub.items():
                per_key[k2] = per_key.get(k2, 0) + b2
    return tot, per_key

def profile_window(kv_frames, steps=4, blocks=4):
    print(f"===== WINDOW kv={kv_frames} =====", flush=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rec = {"kv_frames": kv_frames, "ok": False}
    try:
        params = GenerateParams(prompt=PROMPT, seed=42, kv_cache_num_frames=kv_frames,
                                num_blocks=blocks, num_denoising_steps=steps)
        before = torch.cuda.memory_allocated()
        session = GenerationSession(params, config, frame_callback=lambda *a: None, models=models)
        torch.cuda.synchronize()
        after_init = torch.cuda.memory_allocated()
        kv_bytes, kv_keys = walk(models.pipeline.kv_cache1)
        cx_bytes, cx_keys = walk(models.pipeline.crossattn_cache)
        t0 = time.time()
        for _ in range(blocks):
            try: session.generate_block(models)
            except asyncio.CancelledError: break
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        rec.update({"ok": True,
            "session_init_delta_gb": gb(after_init - before),
            "kv_cache_gb": gb(kv_bytes), "kv_per_key_gb": {k: gb(v) for k, v in kv_keys.items()},
            "crossattn_cache_gb": gb(cx_bytes), "crossattn_per_key_gb": {k: gb(v) for k, v in cx_keys.items()},
            "kv_layers": len(models.pipeline.kv_cache1),
            "kv_k_shape": list(models.pipeline.kv_cache1[0]["k"].shape),
            "kv_k_dtype": str(models.pipeline.kv_cache1[0]["k"].dtype),
            "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
            "device_used_gb": gb(total - free), "device_free_gb": gb(free),
            "wall_sec": round(time.time() - t0, 2)})
        # what 4-bit would free: int4 + per-token/per-channel scales, bf16 sinks kept
        q = kv_bytes / 4 + kv_bytes / 2 / 64 * 2   # 4-bit payload + bf16 scales @ group 64
        rec["kv_4bit_projected_gb"] = gb(q)
        rec["kv_4bit_frees_gb"] = gb(kv_bytes - q)
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc()[-1000:]
        torch.cuda.empty_cache()
    RES["windows"].append(rec); flush()
    print(json.dumps({k: v for k, v in rec.items() if not k.startswith("kv_per")}, default=str)[:900], flush=True)
    torch.cuda.empty_cache()

for kvf in (3, 6, 12):
    profile_window(kvf)

flush()
print("DONE", flush=True)
print(json.dumps(RES["stages"], indent=1), flush=True)
