"""L2 collect harness: per-Linear activation capture on the REAL causal loop (1.3B, 4090).

DeepCompressor's native diffusion collect caches whole-model forward inputs and
replays them — impossible here: the causal loop carries mutable GB-scale kv/crossattn
caches inside the call args. So the adapter inverts the flow: hooks capture each
quantization target's input during the real generation loop (denoise steps +
kv-cache recompute), and L3 feeds DeepCompressor's smooth/low-rank/range calibration
from these caches directly.

Capture streams per block (7 streams cover all 10 Linears — shared inputs):
  self_qkv  -> self_attn.{q,k,v} input (modulated norm1 x)
  self_o    -> self_attn.o input
  cross_q   -> cross_attn.q input (norm3 x)
  cross_kv  -> cross_attn.{k,v} input (text context; runs ONCE per session, crossattn_cache)
  cross_o   -> cross_attn.o input
  ffn_up    -> ffn.0 input (norm2 x)
  ffn_down  -> ffn.2 input (GELU output, ffn_dim wide)

Per stream per block: channelwise absmax/absmean (fp32) + token reservoir
(K tokens per call, fp16, tagged with call idx -> timestep recoverable).
Per call: phase (denoise/recompute), chunk, timesteps -> timestep histogram.
Fixture: full inputs of blocks 0 and 15 for two mid-generation calls (L4 unit tests).

Outputs in collect_out/: block{b}/{stream}.pt, calls.json, timestep_histogram.json,
fixtures/, summary.json.
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path(os.environ.get("COLLECT_OUT", "collect_out")); OUT.mkdir(exist_ok=True)
(OUT / "fixtures").mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server.yaml"
SEEDS = [42, 43, 44]
KV_FRAMES, STEPS, NUM_BLOCKS = 3, 4, 9
RESERVOIR_CAP = int(os.environ.get("RESERVOIR_CAP", "2048"))   # tokens per stream per block
FIXTURE_CALLS = {20, 21}      # global call idx within seed-42 session (mid-generation)
FIXTURE_BLOCKS = {0, 15}

from release_server import load_merge_config, load_transformer, \
    load_vae, load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel, CausalWanSelfAttention
def _from_config(path, **kw):
    return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)
# keep q/k/v as separate Linears: quantization targets must match checkpoint names
CausalWanSelfAttention.fuse_projections = lambda self: None

config = load_merge_config(CONFIG_PATH)
transformer = load_transformer(config)
model = transformer.model
n_blocks_model = len(model.blocks)

import re as _re
_slug = _re.sub(r"[^a-z0-9]+", "-", PROMPT.lower())[:60].strip("-")
_emb = torch.load(f"embeddings/{_slug}.pt", map_location="cpu", weights_only=True)
_cond = {"prompt_embeds": _emb["prompt_embeds"].to("cuda")}
class _StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in _cond.items()}
text_encoder = _StaticEnc()
vae_encoder, vae_decoder = load_vae()
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, text_encoder, vae_decoder)
models = Models(text_encoder, transformer, pipeline, vae_encoder, vae_decoder)
print("LOAD OK (unfused), blocks:", n_blocks_model)

# ---- call context, set by the wrapper hook / run loop ----
CTX = {"call": -1, "phase": "denoise", "chunk": -1, "session": None, "fixture": False}
CALLS = []  # one dict per model forward

class StreamStats:
    def __init__(self, dim):
        self.absmax = torch.zeros(dim, dtype=torch.float32, device="cuda")
        self.abssum = torch.zeros(dim, dtype=torch.float32, device="cuda")
        self.tokens = 0
        self.calls = 0
        self.res_chunks = []   # [K, dim] fp16 cpu tensors
        self.res_call_ids = []  # per-chunk call idx (row tag)
        self.res_tokens = 0

    def update(self, x, k_per_call):
        # x: [..., dim] -> [tokens, dim]
        flat = x.detach().reshape(-1, x.shape[-1])
        a = flat.abs()
        self.absmax = torch.maximum(self.absmax, a.amax(dim=0).float())
        self.abssum += a.sum(dim=0, dtype=torch.float32)
        self.tokens += flat.shape[0]
        self.calls += 1
        if self.res_tokens < RESERVOIR_CAP:
            k = min(k_per_call, RESERVOIR_CAP - self.res_tokens, flat.shape[0])
            idx = torch.randperm(flat.shape[0], device=flat.device)[:k]
            self.res_chunks.append(flat[idx].to(torch.float16).cpu())
            self.res_call_ids.append(torch.full((k,), CTX["call"], dtype=torch.int32))
            self.res_tokens += k

STREAMS = {}   # (block, stream) -> StreamStats
FIXTURE = {}   # (call, block, stream) -> tensor

def make_hook(b, stream, dim, k_per_call):
    def hook(module, args):
        x = args[0]
        key = (b, stream)
        st = STREAMS.get(key)
        if st is None:
            st = STREAMS[key] = StreamStats(dim)
        st.update(x, k_per_call)
        if CTX["fixture"] and b in FIXTURE_BLOCKS:
            FIXTURE[(CTX["call"], b, stream)] = x.detach().to(torch.float16).cpu()
        return None
    return hook

# calls per stream over the whole collect (~44/session x 3) sets K for even coverage
EXPECTED_CALLS = len(SEEDS) * (NUM_BLOCKS * STEPS + NUM_BLOCKS)   # denoise + recompute upper bound
K_PER_CALL = max(8, RESERVOIR_CAP // EXPECTED_CALLS)

hooks = []
dim = model.blocks[0].self_attn.q.in_features
ffn_dim = model.blocks[0].ffn[2].in_features
for b, blk in enumerate(model.blocks):
    for stream, mod, d, k in [
        ("self_qkv", blk.self_attn.q, dim, K_PER_CALL),
        ("self_o", blk.self_attn.o, dim, K_PER_CALL),
        ("cross_q", blk.cross_attn.q, dim, K_PER_CALL),
        ("cross_kv", blk.cross_attn.k, dim, RESERVOIR_CAP),  # ~1 call/session: take all
        ("cross_o", blk.cross_attn.o, dim, K_PER_CALL),
        ("ffn_up", blk.ffn[0], dim, K_PER_CALL),
        ("ffn_down", blk.ffn[2], ffn_dim, K_PER_CALL),
    ]:
        hooks.append(mod.register_forward_pre_hook(make_hook(b, stream, d, k)))
print(f"hooks: {len(hooks)} (K_PER_CALL={K_PER_CALL}, cap={RESERVOIR_CAP})")

orig_fwd = models.transformer.forward
def wrapped(*a, **k):
    CTX["call"] += 1
    t = k.get("timestep", a[2] if len(a) > 2 else None)
    ts = sorted(set(round(float(v), 2) for v in t.flatten())) if t is not None else []
    CTX["fixture"] = (CTX["session"] == 42 and CTX["call"] in FIXTURE_CALLS)
    CALLS.append({"call": CTX["call"], "session": CTX["session"], "chunk": CTX["chunk"],
                  "phase": CTX["phase"], "timesteps": ts})
    return orig_fwd(*a, **k)
models.transformer.forward = wrapped

_orig_recomp = GenerationSession.recompute_kv_cache
def tagged_recomp(self, mdl):
    CTX["phase"] = "recompute"
    try: return _orig_recomp(self, mdl)
    finally: CTX["phase"] = "denoise"
GenerationSession.recompute_kv_cache = tagged_recomp

t0 = time.time()
for seed in SEEDS:
    CTX["session"] = seed; CTX["call"] = -1
    params = GenerateParams(prompt=PROMPT, seed=seed, kv_cache_num_frames=KV_FRAMES,
                            num_blocks=NUM_BLOCKS, num_denoising_steps=STEPS)
    def cb(pixels, frame_ids, event): event.synchronize()
    session = GenerationSession(params, config, frame_callback=cb, models=models)
    for c in range(NUM_BLOCKS):
        CTX["chunk"] = c
        try: session.generate_block(models)
        except asyncio.CancelledError: break
    torch.cuda.synchronize()
    session.dispose()
    print(f"seed {seed} done, calls this session: {CTX['call'] + 1}, elapsed {time.time()-t0:.0f}s")

for h in hooks: h.remove()
models.transformer.forward = orig_fwd
GenerationSession.recompute_kv_cache = _orig_recomp

# ---- persist ----
summary = {"prompt": PROMPT, "seeds": SEEDS, "kv_frames": KV_FRAMES, "steps": STEPS,
           "num_chunks": NUM_BLOCKS, "reservoir_cap": RESERVOIR_CAP, "k_per_call": K_PER_CALL,
           "n_blocks": n_blocks_model, "dim": dim, "ffn_dim": ffn_dim,
           "total_calls": len(CALLS), "streams": {}}
total_bytes = 0
for b in range(n_blocks_model):
    bdir = OUT / f"block{b:02d}"; bdir.mkdir(exist_ok=True)
    for stream in ["self_qkv", "self_o", "cross_q", "cross_kv", "cross_o", "ffn_up", "ffn_down"]:
        st = STREAMS.get((b, stream))
        if st is None: continue
        res = torch.cat(st.res_chunks) if st.res_chunks else torch.empty(0)
        call_ids = torch.cat(st.res_call_ids) if st.res_call_ids else torch.empty(0, dtype=torch.int32)
        payload = {"reservoir": res, "call_ids": call_ids,
                   "absmax": st.absmax.cpu(), "absmean": (st.abssum / max(st.tokens, 1)).cpu(),
                   "tokens_seen": st.tokens, "calls_seen": st.calls}
        p = bdir / f"{stream}.pt"; torch.save(payload, p)
        total_bytes += p.stat().st_size
        if b == 0:
            summary["streams"][stream] = {"calls_seen": st.calls, "tokens_seen": st.tokens,
                "reservoir_tokens": int(res.shape[0]), "dim": int(st.absmax.shape[0]),
                "absmax_max": float(st.absmax.max()), "absmax_p50": float(st.absmax.median())}

for (call, b, stream), x in FIXTURE.items():
    torch.save(x, OUT / "fixtures" / f"call{call:03d}_block{b:02d}_{stream}.pt")

(OUT / "calls.json").write_text(json.dumps(CALLS, indent=0))
hist = {}
for c in CALLS:
    for t in c["timesteps"]:
        key = f"{c['phase']}@{t}"
        hist[key] = hist.get(key, 0) + 1
summary["timestep_histogram"] = dict(sorted(hist.items()))
summary["disk_bytes"] = total_bytes
(OUT / "timestep_histogram.json").write_text(json.dumps(summary["timestep_histogram"], indent=1))
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print(f"DONE in {time.time()-t0:.0f}s, {total_bytes/1e9:.2f}GB in {OUT}/")
