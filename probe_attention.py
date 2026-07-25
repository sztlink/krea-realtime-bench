"""The clay hypothesis. The grey is attention spreading instead of selecting.

Reading offered while watching the clips. Mixing modelling clay of every colour gives grey,
because an average of many different things tends toward the middle. Attention does this
literally. When the distribution spreads instead of selecting, the output is a mean over
many values, and a mean of diverse states has low variance, which is what low contrast is.

The hypothesis explains every control already measured.
  longer window, more grey        -> more colours in the mixture
  resident worse than rebuild     -> the rebuild derives every key from one clean context
                                     at timestep zero, homogeneous, while the resident cache
                                     accumulates keys from many blocks and steps
  no anchor is catastrophic       -> the sink is where attention dumps the mass it cannot
                                     place. Without it the excess spreads over the whole
                                     window and becomes an average
  quantization modulates only     -> the error flattens the logits a little

Measures, per block and per layer, over a sample of query positions, the entropy of the
attention distribution normalized by log of the window size so windows of different sizes
compare, the maximum weight, and the fraction of mass landing on the anchor. Then correlate
with the frames where contrast collapses.

Run:
  PROBE_TAG=res PROBE_PERIOD=inf python probe_attention.py
"""
import os, json, math, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open
import quant_kv

OUT = Path("results_probe"); OUT.mkdir(exist_ok=True)
PROMPT = os.environ.get("PROBE_PROMPT",
    "Camera tracking alongside a woman in a bikini rollerskating down a Miami boardwalk at "
    "golden hour, palm trees and neon signs streaking past, lens flare, smooth steadicam "
    "motion, saturated colors, cinematic, 35mm")
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = "checkpoints-14b-w4a4-ckv"
BLOCKS = int(os.environ.get("PROBE_BLOCKS", "18"))
KVF = int(os.environ.get("PROBE_KV", "6"))
SEED = int(os.environ.get("PROBE_SEED", "42"))
TAG = os.environ.get("PROBE_TAG", "res")
_p = os.environ.get("PROBE_PERIOD", "inf")
PERIOD = None if _p == "inf" else int(_p)
SINK_FRAMES = int(os.environ.get("PROBE_SINK", "1"))
PROBE_LAYERS = {0, 13, 27, 39}
NQ = 32

from release_server import load_merge_config, load_vae, load_pipeline, \
    GenerateParams, GenerationSession, Models
from utils.wan_wrapper import WanDiffusionWrapper
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import rope_params
from nunchaku_causal_wan import load_w4a4_blocks
import wan.modules.causal_model as cm

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

# ------------------------------------------------------------------ a sonda
_orig_attention = cm.attention
STATE = {"block": -1, "call": 0}
PROBE = []
SINK_TOKENS = SINK_FRAMES * 1560

def probed_attention(q, k, v, *a, **kw):
    i = STATE["call"]; STATE["call"] += 1
    layer = i % 40
    if layer in PROBE_LAYERS and STATE["block"] >= 0:
        with torch.no_grad():
            Lq, Lk, D = q.shape[1], k.shape[1], q.shape[-1]
            step = max(1, Lq // NQ)
            qs = q[0, ::step][:NQ].float()                       # [nq, H, D]
            ks = k[0].float()                                    # [Lk, H, D]
            logits = torch.einsum("qhd,khd->hqk", qs, ks) / math.sqrt(D)
            p = logits.softmax(-1)
            ent = -(p * (p + 1e-9).log()).sum(-1)                # [H, nq]
            PROBE.append({
                "block": STATE["block"], "layer": layer, "Lk": int(Lk),
                "entropia": round(float(ent.mean()), 4),
                # normalizada pelo maximo possivel, para comparar janelas de tamanhos diferentes
                "entropia_norm": round(float(ent.mean() / math.log(Lk)), 4),
                "peso_max": round(float(p.max(-1).values.mean()), 5),
                "massa_ancora": round(float(p[..., :SINK_TOKENS].sum(-1).mean()), 5),
            })
    return _orig_attention(q, k, v, *a, **kw)

cm.attention = probed_attention

# ------------------------------------------------------------------ periodo do reset
_orig_recompute = GenerationSession.recompute_kv_cache
def patched(self, models):
    if self.block_idx == 0 or (PERIOD is not None and self.block_idx % PERIOD == 0):
        return _orig_recompute(self, models)
    for b in models.pipeline.generator.model.blocks:
        b.self_attn.num_frame_per_block = models.pipeline.num_frame_per_block
    gei = models.pipeline.kv_cache1[0]["global_end_index"]
    return int(gei) // models.pipeline.frame_seq_length
GenerationSession.recompute_kv_cache = patched

pipeline.kv_cache1 = []
quant_kv.install(pipeline, k_bits=4, v_bits=4, group_mode="bands4",
                 sink_frames=SINK_FRAMES, verbose=True)
params = GenerateParams(prompt=PROMPT, seed=SEED, kv_cache_num_frames=KVF,
                        num_blocks=BLOCKS, num_denoising_steps=4)
frames = []
def cb(pixels, frame_ids, event):
    event.synchronize(); frames.append(pixels[0].float().cpu())
session = GenerationSession(params, config, frame_callback=cb, models=models)
for b in models.pipeline.generator.model.blocks:
    b.self_attn.local_attn_size = KVF + 3
    b.self_attn.sink_size = SINK_FRAMES
    b.self_attn.max_attention_size = (KVF + 3) * 1560

for i in range(BLOCKS):
    STATE["block"] = i
    try: session.generate_block(models)
    except asyncio.CancelledError: break
torch.cuda.synchronize()

# contraste por frame, para correlacionar
import numpy as np
con = []
for blk in frames:
    for j in range(blk.shape[0]):
        a = ((blk[j].clamp(-1, 1) + 1) / 2).numpy()
        y = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]
        con.append(float(y.std()))
con = np.array(con)
med = float(np.median(con))

# agrega a sonda por bloco
by_block = {}
for r in PROBE:
    by_block.setdefault(r["block"], []).append(r)
blocos = []
for b in sorted(by_block):
    rs = by_block[b]
    blocos.append({
        "bloco": b,
        "entropia_norm": round(sum(r["entropia_norm"] for r in rs) / len(rs), 4),
        "peso_max": round(sum(r["peso_max"] for r in rs) / len(rs), 5),
        "massa_ancora": round(sum(r["massa_ancora"] for r in rs) / len(rs), 5),
        "Lk": rs[0]["Lk"],
    })

# mapeia frame -> bloco (bloco 0 entrega menos frames)
nb = len(frames)
per_block = [f.shape[0] for f in frames]
edges, acc = [], 0
for n in per_block:
    edges.append((acc, acc + n)); acc += n
for i, (lo, hi) in enumerate(edges):
    if i < len(blocos):
        seg = con[lo:hi]
        blocos[i]["contraste"] = round(float(seg.mean()), 4)
        blocos[i]["contraste_min"] = round(float(seg.min()), 4)
        blocos[i]["cinza"] = bool(seg.min() < 0.85 * med)

res = {"tag": TAG, "kv": KVF, "period": PERIOD or "inf", "sink": SINK_FRAMES,
       "seed": SEED, "contraste_mediana": round(med, 4), "blocos": blocos}
(OUT / f"probe_{TAG}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1)[:2000], flush=True)
print("DONE", flush=True)
