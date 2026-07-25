"""Reset conditioned on attention spread, against a fixed period.

The grey investigation showed the contrast collapse is attention spreading instead of
selecting, and that the mean maximum attention weight predicts it at r = 0.79 in the
resident regime. The cache rebuild works because it derives every key from one clean context
at timestep zero, leaving them homogeneous and comparable.

Hence the idea. Stop choosing between resident and rebuild, and sense instead. The reset
fires when the mixture starts turning grey.

The rule is RELATIVE, because an absolute threshold depends on the model and the window. The
sensor learns a baseline from the first blocks with the window already full and fires when
the reading falls below RATIO of it.

The sensor reads the PREVIOUS block, which is causal and cheap. It does not prevent the
first collapse, it prevents the degraded regime that follows.

Result, kept here because the negative is the point. The gate fires once in eighteen blocks
and lands where the resident regime already was. Homogeneity behaves as a continuous
property rather than a state that gets repaired, so by the time a sensor reports, the damage
already sits in the cache. What replaces the sensor is the fixed period curve this script
also measures.

Timing note. Calling float() on a device tensor inside a CUDA loop measures queue drain
rather than work, and inflated the sensor cost tenfold here before it was caught. Accumulate
on device, read once at the end.

Run:
  AD_KV=12 AD_MODES=1,2,4,inf AD_SEEDS=42,43,44,45 python adaptive_reset.py
"""
import os, json, math, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
import numpy as np
from safetensors import safe_open
import quant_kv

OUT = Path("results_adaptive"); OUT.mkdir(exist_ok=True)
PROMPT = ("Camera tracking alongside a woman in a bikini rollerskating down a Miami boardwalk "
          "at golden hour, palm trees and neon signs streaking past, lens flare, smooth "
          "steadicam motion, saturated colors, cinematic, 35mm")
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = "checkpoints-14b-w4a4-ckv"
BLOCKS = int(os.environ.get("AD_BLOCKS", "18"))
KVF = int(os.environ.get("AD_KV", "6"))
SEEDS = [int(x) for x in os.environ.get("AD_SEEDS", "42,43,44").split(",")]
SENSOR_LAYER = 27
NQ = 8
RATIO = float(os.environ.get("AD_RATIO", "0.92"))
# The baseline has to be learned with the window ALREADY FULL. In the first blocks it is
# still filling (4680, 9360, 14040 tokens) and the maximum weight lives in another regime,
# which raises the baseline and makes the gate almost never fire.
BASELINE_FROM = int(os.environ.get("AD_BASE_FROM", "4"))
BASELINE_TO = int(os.environ.get("AD_BASE_TO", "8"))
RES = {"runs": []}
def flush(): (OUT / "results.json").write_text(json.dumps(RES, indent=1, default=str))

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

# ---------------------------------------------------------------- sensor barato
_orig_attention = cm.attention
S = {"call": 0, "on": False, "acc": 0.0, "n": 0, "sec": 0.0}

def sensing_attention(q, k, v, *a, **kw):
    i = S["call"]; S["call"] += 1
    if S["on"] and (i % 40) == SENSOR_LAYER:
        t0 = time.perf_counter()
        with torch.no_grad():
            Lq, D = q.shape[1], q.shape[-1]
            step = max(1, Lq // NQ)
            # Do NOT cast ks to fp32. k[0].float() materialises the whole window (288 MB
            # at kv6) on every call. The einsum runs in the native dtype and only the
            # result, which is small ([H, nq, Lk] = 4.5M), becomes fp32 for the softmax.
            qs = q[0, ::step][:NQ]
            ks = k[0]
            logits = torch.einsum("qhd,khd->hqk", qs, ks).float() / math.sqrt(D)
            S["acc"] += float(logits.softmax(-1).max(-1).values.mean()); S["n"] += 1
        S["sec"] += time.perf_counter() - t0
    return _orig_attention(q, k, v, *a, **kw)

cm.attention = sensing_attention

# ---------------------------------------------------------------- the three regimes
_orig_recompute = GenerationSession.recompute_kv_cache
G = {"mode": "inf", "baseline": None, "last": None, "hist": [], "resets": 0, "fires": []}

def gated_recompute(self, models):
    b = self.block_idx
    if b == 0:
        G["resets"] += 1
        return _orig_recompute(self, models)
    do = False
    if G["mode"].isdigit():
        do = (b % int(G["mode"])) == 0
    elif G["mode"] == "adaptive":
        # decide from the previous block's reading. Causal and free.
        if G["baseline"] is not None and G["last"] is not None:
            do = G["last"] < RATIO * G["baseline"]
    if do:
        G["resets"] += 1; G["fires"].append(b)
        return _orig_recompute(self, models)
    for blk in models.pipeline.generator.model.blocks:
        blk.self_attn.num_frame_per_block = models.pipeline.num_frame_per_block
    gei = models.pipeline.kv_cache1[0]["global_end_index"]
    return int(gei) // models.pipeline.frame_seq_length

GenerationSession.recompute_kv_cache = gated_recompute

def run(mode, seed):
    tag = f"{mode}_s{seed}"
    print(f"===== {tag} =====", flush=True)
    pipeline.kv_cache1 = []
    quant_kv.install(pipeline, k_bits=4, v_bits=4, group_mode="bands4",
                     sink_frames=1, verbose=False)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    G.update({"mode": mode, "baseline": None, "last": None, "hist": [], "resets": 0, "fires": []})
    S.update({"call": 0, "on": mode == "adaptive", "acc": 0.0, "n": 0, "sec": 0.0})
    rec = {"tag": tag, "mode": mode, "seed": seed, "kv": KVF, "ok": False}
    frames = []
    def cb(pixels, frame_ids, event):
        event.synchronize(); frames.append(pixels[0].float().cpu())
    try:
        params = GenerateParams(prompt=PROMPT, seed=seed, kv_cache_num_frames=KVF,
                                num_blocks=BLOCKS, num_denoising_steps=4)
        session = GenerationSession(params, config, frame_callback=cb, models=models)
        for blk in models.pipeline.generator.model.blocks:
            blk.self_attn.local_attn_size = KVF + 3
            blk.self_attn.sink_size = 1
            blk.self_attn.max_attention_size = (KVF + 3) * 1560
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        for i in range(BLOCKS):
            S["acc"] = 0.0; S["n"] = 0
            try: out = session.generate_block(models)
            except asyncio.CancelledError: break
            n += out.shape[1]
            if S["n"]:
                pm = S["acc"] / S["n"]
                G["hist"].append(round(pm, 5)); G["last"] = pm
                # baseline: mean of the first blocks already in the full regime
                if len(G["hist"]) == BASELINE_TO:
                    win = G["hist"][BASELINE_FROM:BASELINE_TO]
                    G["baseline"] = sum(win) / len(win)
        torch.cuda.synchronize(); wall = time.time() - t0
        con = []
        for blkf in frames:
            for j in range(blkf.shape[0]):
                a = ((blkf[j].clamp(-1, 1) + 1) / 2).numpy()
                y = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]
                con.append(float(y.std()))
        con = np.array(con); med = float(np.median(con))
        rec.update({"ok": True, "fps": round(n / wall, 2), "frames": n,
                    "resets": G["resets"], "disparos": G["fires"],
                    "sensor_baseline": round(G["baseline"], 5) if G["baseline"] else None,
                    "sensor_hist": G["hist"],
                    "sensor_overhead_pct": round(100 * S["sec"] / wall, 2),
                    "contraste_mediana": round(med, 4),
                    "queda_max_pct": round(100 * (med - float(con.min())) / med, 1),
                    "frames_cinza": int((con < 0.85 * med).sum()),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)})
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc()[-600:]
        torch.cuda.empty_cache()
    RES["runs"].append(rec); flush()
    print(json.dumps({k: v for k, v in rec.items() if k != "sensor_hist"}, default=str)[:500], flush=True)
    torch.cuda.empty_cache()

MODES = os.environ.get("AD_MODES", "inf,adaptive,1").split(",")
for seed in SEEDS:
    for mode in MODES:
        run(mode, seed)

flush()
print("DONE", flush=True)
for m in MODES:
    rs = [r for r in RES["runs"] if r.get("mode") == m and r.get("ok")]
    if rs:
        print(f'{m:9s} fps={sum(r["fps"] for r in rs)/len(rs):.2f} '
              f'cinza={[r["frames_cinza"] for r in rs]} '
              f'resets={[r["resets"] for r in rs]}', flush=True)
