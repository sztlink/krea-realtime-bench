"""Where the 1.64s block actually goes at one denoise step.

The step sweep gave block = 4d + F with d = 0.80s per denoise step and F = 0.83s fixed,
so at one step half the block is no longer denoise. Before optimizing the fixed part,
measure what it is made of. Timers wrap each piece with a sync, which serializes and adds
wall clock, so the attribution run is never the run quoted as fps.
"""
import os, json, time, asyncio, traceback
from collections import defaultdict
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open
import quant_kv

OUT = Path("results_qkv"); OUT.mkdir(exist_ok=True)
PROMPT = ("Camera tracking alongside a woman in a bikini rollerskating down a Miami boardwalk "
          "at golden hour, palm trees and neon signs streaking past, lens flare, smooth "
          "steadicam motion, saturated colors, cinematic, 35mm")
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = "checkpoints-14b-w4a4-ckv"
BLOCKS = int(os.environ.get("ATTR_BLOCKS", "12"))
RES = []

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

T = defaultdict(float); C = defaultdict(int); ON = {"v": False}
def timed(name, fn):
    def w(*a, **kw):
        if not ON["v"]: return fn(*a, **kw)
        torch.cuda.synchronize(); t = time.perf_counter()
        o = fn(*a, **kw)
        torch.cuda.synchronize()
        T[name] += time.perf_counter() - t; C[name] += 1
        return o
    return w

GenerationSession.recompute_kv_cache = timed("recompute", GenerationSession.recompute_kv_cache)
WanDiffusionWrapper.forward = timed("denoise_fwd", WanDiffusionWrapper.forward)
_vf = vae_decoder.forward
# Compilar o forward ORIGINAL e re-envolver o timer POR FORA. O contrario poe
# time.perf_counter() dentro da regiao tracada e o dynamo recusa.
if os.environ.get("COMPILE_VAE", "0") == "1":
    _vf = torch.compile(_vf, fullgraph=True)
    print("VAE compilado", flush=True)
vae_decoder.forward = timed("vae_decode", _vf)
from quant_kv import QuantKVTensor
QuantKVTensor._quantize = timed("quantize", QuantKVTensor._quantize)
QuantKVTensor._dequantize = timed("dequantize", QuantKVTensor._dequantize)

_orig_rc = GenerationSession.recompute_kv_cache
PERIOD = 4
def patched(self, models):
    if self.block_idx == 0 or self.block_idx % PERIOD == 0:
        return _orig_rc(self, models)
    for b in models.pipeline.generator.model.blocks:
        b.self_attn.num_frame_per_block = models.pipeline.num_frame_per_block
    gei = models.pipeline.kv_cache1[0]["global_end_index"]
    return int(gei) // models.pipeline.frame_seq_length
GenerationSession.recompute_kv_cache = patched

for steps in (1,):
    pipeline.kv_cache1 = []
    quant_kv.install(pipeline, k_bits=4, v_bits=4, group_mode="bands4", sink_frames=1, verbose=False)
    T.clear(); C.clear(); ON["v"] = True
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    params = GenerateParams(prompt=PROMPT, seed=42, kv_cache_num_frames=3,
                            num_blocks=BLOCKS, num_denoising_steps=steps)
    n = {"n": 0}
    session = GenerationSession(params, config, frame_callback=lambda p, f, e: (e.synchronize(), n.__setitem__("n", n["n"] + p.shape[1])), models=models)
    for b in models.pipeline.generator.model.blocks:
        b.self_attn.local_attn_size = 6; b.self_attn.sink_size = 1
        b.self_attn.max_attention_size = 6 * 1560
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(BLOCKS):
        try: session.generate_block(models)
        except asyncio.CancelledError: break
    torch.cuda.synchronize(); wall = time.time() - t0
    ON["v"] = False
    acc = dict(T)
    rec = {"steps": steps, "wall": round(wall, 2), "bloco_s": round(wall / BLOCKS, 3),
           "frames": n["n"],
           "sec": {k: round(v, 2) for k, v in acc.items()},
           "pct": {k: round(100 * v / wall, 1) for k, v in acc.items()},
           "calls": dict(C),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    RES.append(rec)
    (OUT / f"attrib_vae_{os.environ.get('COMPILE_VAE','0')}.json").write_text(json.dumps(RES, indent=1))
    print(json.dumps(rec), flush=True)
    session.dispose(); torch.cuda.empty_cache()
print("DONE", flush=True)
