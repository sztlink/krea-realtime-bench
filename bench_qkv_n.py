"""Quantized cache against reset period, with conversion counters.

The quantized cache unlocked the 6 and 12 frame windows, which are impossible in bf16, and
this separates two costs that add up in the long window case.

  H1  dequantization on the read path is expensive and grows with the window
  H2  the cache rebuild rewrites the WHOLE cache every block, and rewriting now means
      REQUANTIZING a larger window

If H2 dominates, the fix already exists, since the resident regime rewrites nothing and only
appends new tokens.

Measures, per window, rebuild every block against never, with counters for how many tokens
were quantized and dequantized and how long each side took.

Run:
  DISABLE_SAGEATTENTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python bench_qkv_n.py
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open

import quant_kv
from quant_kv import QuantKVTensor

OUT = Path("results_qkv"); OUT.mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = os.environ.get("V_QDIR", "checkpoints-14b-w4a4-ckv")
BLOCKS = int(os.environ.get("QKV_BLOCKS", "9"))
MODE = os.environ.get("QKV_MODE", "bands")
RES = {"runs": []}
def flush(): (OUT / "bench_n.json").write_text(json.dumps(RES, indent=1, default=str))
def gb(x): return round(x / 1e9, 3)

# ------------------------------------------------- contadores dentro do proxy
CNT = {"q_tokens": 0, "q_sec": 0.0, "d_tokens": 0, "d_sec": 0.0}
_q, _d = QuantKVTensor._quantize, QuantKVTensor._dequantize

def q_counted(self, x):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = _q(self, x)
    torch.cuda.synchronize()
    CNT["q_sec"] += time.perf_counter() - t; CNT["q_tokens"] += x.shape[0]
    return out

def d_counted(self, packed, scales):
    torch.cuda.synchronize(); t = time.perf_counter()
    out = _d(self, packed, scales)
    torch.cuda.synchronize()
    CNT["d_sec"] += time.perf_counter() - t; CNT["d_tokens"] += packed.shape[0]
    return out

QuantKVTensor._quantize = q_counted
QuantKVTensor._dequantize = d_counted

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
print("LOAD OK", flush=True)

# ------------------------------------------------------------ the reset period
_orig_recompute = GenerationSession.recompute_kv_cache
PERIOD = {"n": 1}

def patched_recompute(self, models):
    if self.block_idx == 0 or (PERIOD["n"] is not None and self.block_idx % PERIOD["n"] == 0):
        return _orig_recompute(self, models)
    for b in models.pipeline.generator.model.blocks:
        b.self_attn.num_frame_per_block = models.pipeline.num_frame_per_block
    gei = models.pipeline.kv_cache1[0]["global_end_index"]
    return int(gei) // models.pipeline.frame_seq_length

GenerationSession.recompute_kv_cache = patched_recompute

def unlock(models, attn_size, sink_frames=1):
    for b in models.pipeline.generator.model.blocks:
        b.self_attn.local_attn_size = attn_size
        b.self_attn.sink_size = sink_frames
        b.self_attn.max_attention_size = attn_size * 1560

def run(tag, kv_frames, period, sink_frames=1):
    print(f"===== {tag} =====", flush=True)
    pipeline.kv_cache1 = []
    quant_kv.install(pipeline, k_bits=4, v_bits=4, group_mode=MODE,
                     sink_frames=sink_frames, verbose=False)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for k in CNT: CNT[k] = 0 if isinstance(CNT[k], int) else 0.0
    PERIOD["n"] = period
    rec = {"tag": tag, "kv_frames": kv_frames, "N": period or "inf", "mode": MODE,
           "kernels": quant_kv.USE_KERNELS, "ok": False}
    try:
        params = GenerateParams(prompt=PROMPT, seed=42, kv_cache_num_frames=kv_frames,
                                num_blocks=BLOCKS, num_denoising_steps=4)
        session = GenerationSession(params, config, frame_callback=lambda *a: None, models=models)
        unlock(models, kv_frames + 3, sink_frames)
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        for _ in range(BLOCKS):
            try: out = session.generate_block(models)
            except asyncio.CancelledError: break
            n += out.shape[1]
        torch.cuda.synchronize(); wall = time.time() - t0
        latents = session.all_latents[:, :session.current_start_frame].cpu()
        rec.update({"ok": True, "wall_sec": round(wall, 2), "frames": n,
                    "fps": round(n / wall, 2),
                    "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
                    "quant_sec": round(CNT["q_sec"], 2), "quant_Mtokens": round(CNT["q_tokens"] / 1e6, 2),
                    "dequant_sec": round(CNT["d_sec"], 2), "dequant_Mtokens": round(CNT["d_tokens"] / 1e6, 2),
                    "quant_pct": round(100 * CNT["q_sec"] / wall, 1),
                    "dequant_pct": round(100 * CNT["d_sec"] / wall, 1),
                    "latents_finite": bool(torch.isfinite(latents).all())})
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc()[-600:]
        torch.cuda.empty_cache()
    RES["runs"].append(rec); flush()
    print(json.dumps(rec, default=str)[:600], flush=True)
    torch.cuda.empty_cache()

for kvf in (3, 6, 12):
    run(f"kv{kvf}_N1", kvf, 1)
    run(f"kv{kvf}_Ninf", kvf, None)

flush()
print("DONE", flush=True)
print(json.dumps([{k: r.get(k) for k in ("tag", "fps", "quant_pct", "dequant_pct",
                                         "quant_Mtokens", "dequant_Mtokens", "peak_alloc_gb")}
                  for r in RES["runs"]], indent=1), flush=True)
