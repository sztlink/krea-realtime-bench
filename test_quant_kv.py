"""QuantKVCache fixtures. Interface semantics, and error on REAL keys and values.

Two parts.

1. Semantics. The proxy has to behave like a tensor across the three operations the runtime
   performs, including the bf16 anchor case. Synthetic data, exact checks.

2. Error. Runs two blocks with the normal bf16 cache, copies the real K and V out of a few
   layers (K is post rotary and post RMSNorm, V is raw) and measures round trip error per
   scheme. This is where the band hypothesis becomes a number, band aligned groups against
   blind groups, and keys against values.

Run:
  DISABLE_SAGEATTENTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python test_quant_kv.py
"""
import os, json, asyncio
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open

from quant_kv import QuantKVTensor, ROPE_BANDS

OUT = Path("results_qkv"); OUT.mkdir(exist_ok=True)
RES = {"semantica": {}, "erro": {}}

# ---------------------------------------------------------------- 1. semantics
def check_semantics():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S, H, D = 4680, 8, 128
    t = QuantKVTensor([1, S, H, D], dev, bits=4, group_mode="bands", sink_tokens=1560)
    x = torch.randn(1, 1560, H, D, dtype=torch.bfloat16, device=dev)
    t[:, 0:1560] = x
    back = t[:, 0:1560]
    sink_exact = torch.equal(back, x)                      # sink guardado em bf16
    y = torch.randn(1, 3120, H, D, dtype=torch.bfloat16, device=dev)
    t[:, 1560:4680] = y
    yb = t[:, 1560:4680]
    rel = ((yb.float() - y.float()).norm() / y.float().norm()).item()
    shape_ok = tuple(t.shape) == (1, S, H, D) and tuple(yb.shape) == (1, 3120, H, D)
    partial = t[:, 1000:2000]                              # slice cruzando a borda do sink
    partial_ok = tuple(partial.shape) == (1, 1000, H, D) and \
        torch.equal(partial[:, :560], x[:, 1000:1560])
    t.zero_()
    zeroed = float(t[:, 0:100].abs().max())
    RES["semantica"] = {
        "sink_bf16_exato": bool(sink_exact),
        "erro_rel_corpo_int4": round(rel, 5),
        "shapes_ok": bool(shape_ok),
        "slice_cruzando_sink_ok": bool(partial_ok),
        "zero_ok": zeroed == 0.0,
        "bytes_vs_bf16": round(t.storage_bytes() / (S * H * D * 2), 4),
    }
    print(json.dumps(RES["semantica"], indent=1), flush=True)
    assert sink_exact and shape_ok and partial_ok and zeroed == 0.0

check_semantics()

# ---------------------------------------------------------------- 2. erro real
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = os.environ.get("V_QDIR", "checkpoints-14b-w4a4-ckv")

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

params = GenerateParams(prompt=PROMPT, seed=42, kv_cache_num_frames=3,
                        num_blocks=2, num_denoising_steps=4)
session = GenerationSession(params, config, frame_callback=lambda *a: None, models=models)
for _ in range(2):
    try: session.generate_block(models)
    except asyncio.CancelledError: break
torch.cuda.synchronize()

LAYERS = [0, 13, 27, 39]
snap = {}
for li in LAYERS:
    e = models.pipeline.kv_cache1[li]
    n = e["local_end_index"]
    snap[li] = {"k": e["k"][:, :n].clone(), "v": e["v"][:, :n].clone()}
print("snapshot", {li: tuple(s["k"].shape) for li, s in snap.items()}, flush=True)
session.dispose(); del session
torch.cuda.empty_cache()

def roundtrip_err(x, bits, mode):
    """x: [1, L, H, D] bf16 -> erro relativo global e por banda."""
    _, L, H, D = x.shape
    t = QuantKVTensor([1, L, H, D], x.device, bits=bits, group_mode=mode)
    t[:, 0:L] = x
    y = t[:, 0:L]
    xf, yf = x.float(), y.float()
    rel = ((yf - xf).norm() / xf.norm()).item()
    per_band = {}
    for name, (lo, hi) in zip(("temporal", "altura", "largura"), ROPE_BANDS):
        a, b = xf[..., lo:hi], yf[..., lo:hi]
        per_band[name] = round(((b - a).norm() / a.norm()).item(), 5)
    return round(rel, 5), per_band, round(t.storage_bytes() / (L * H * D * 2), 4)

SCHEMES = [(4, "bands"), (4, "bands2"), (4, "bands4"),
           (4, "blind64"), (4, "blind32"), (8, "bands"), (8, "bands2")]
for li, s in snap.items():
    RES["erro"][f"layer{li}"] = {}
    for which in ("k", "v"):
        for bits, mode in SCHEMES:
            rel, per_band, frac = roundtrip_err(s[which], bits, mode)
            RES["erro"][f"layer{li}"][f"{which}_int{bits}_{mode}"] = {
                "rel": rel, "por_banda": per_band, "bytes_vs_bf16": frac}
            print(f"  L{li:2d} {which} int{bits} {mode:8s} rel={rel:.5f} "
                  f"banda={per_band} bytes={frac}", flush=True)

(OUT / "fixtures.json").write_text(json.dumps(RES, indent=1))

# summary: mean across layers per scheme
resumo = {}
for li, per in RES["erro"].items():
    for scheme, v in per.items():
        resumo.setdefault(scheme, []).append(v["rel"])
RES["resumo"] = {k: round(sum(v) / len(v), 5) for k, v in sorted(resumo.items())}
(OUT / "fixtures.json").write_text(json.dumps(RES, indent=1))
print("=== RESUMO (erro relativo medio entre camadas) ===", flush=True)
print(json.dumps(RES["resumo"], indent=1), flush=True)
print("DONE", flush=True)
