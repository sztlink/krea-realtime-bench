"""The reset period as a control. N in {1, 2, 4, infinity}, same memory, 3 seeds.

Krea holds long context stable by rebuilding the whole KV cache every block. It zeroes
the cache, assembles a clean context of the anchor frame plus the most recent denoised
latents, and refills with one forward at timestep zero, so the cache never holds keys and
values derived from noisy intermediate states. That is a generative rule rather than an
optimization, and replacing it with a resident cache is a change of medium presented as a
change of cost. This sweeps the period instead of assuming it away, and it runs before
quantizing anything.

  N=1        upstream, rebuild every block. Control, must reproduce the M1 gate.
  N=2, N=4   periodic reset
  N=inf      pure resident, incremental append plus rolling eviction with sinks

Three things the released code leaves unreachable, read in the source rather than assumed.

  - `init_models` sets `block.self_attn.local_attn_size = -1` twice, redundantly, while
    only the pipeline receives the real window. With -1 the rolling eviction never fires
    and `max_attention_size` stays 32760, so skipping the recompute overflows the cache.
  - `sink_size` is 0 everywhere. Without sinks the eviction discards frame 0, and the
    resident regime would lose the anchor to a confound rather than to the rule.
  - `do_kv_recomp` sits in both configs and in `test_request.py` and is never read by
    `release_server.py`. The recompute runs unconditionally.

All three attributes are applied identically in every regime, including N=1, so that the
only difference is N. With N=1 the eviction never fires anyway and the read slice is
unchanged.

Honest caveat in the design. Physical memory is identical across regimes, but the
effective context is not. The recompute attends over a short clean context while the
resident regime fills the whole physical window. That is not a confound to hide, it is
part of what a resident cache buys.

Position bookkeeping gotcha. When skipping the recompute, the start frame must come from
what the cache itself believes (`global_end_index // frame_seq_length`), not from the
global frame index. The recompute rebases positions onto a short clean context, and mixing
the two coordinate systems inflates `local_end_index` until the write slice comes out
empty.

Run:
  DISABLE_SAGEATTENTION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    N_BLOCKS=18 N_PROMPT="..." python bench_n.py
"""
import os, json, time, asyncio, subprocess, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open

OUT = Path(os.environ.get("N_OUT", "results_n")); OUT.mkdir(exist_ok=True)
# The first round used a dancer and the SUBJECT was rejected, not the regimes. All four
# failed the same way (the person changing sex across the clip, the body turning without
# the head, the face always blurred). With the model's floor above the difference being
# measured, the test discriminates nothing. A full body at 832x480 leaves a few dozen
# pixels for a face. The skater carries identity in silhouette and motion instead.
PROMPT = os.environ.get("N_PROMPT",
    "A person dancing in an empty warehouse, dramatic lighting, camera static")
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = os.environ.get("V_QDIR", "checkpoints-14b-w4a4-ckv")
NUM_BLOCKS = int(os.environ.get("N_BLOCKS", "18"))
SEEDS = [int(s) for s in os.environ.get("N_SEEDS", "42,43,44").split(",")]
RES = {"meta": {}, "runs": []}
def flush(): (OUT / "results.json").write_text(json.dumps(RES, indent=1, default=str))
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
torch.cuda.empty_cache()

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
RES["meta"] = {"gpu": torch.cuda.get_device_name(0), "blocks": NUM_BLOCKS,
               "seeds": SEEDS, "prompt": PROMPT}
print("LOAD OK", flush=True)

# ------------------------------------------------------------ the period as a parameter
_orig_recompute = GenerationSession.recompute_kv_cache
PERIOD = {"n": 1}          # None = infinito (residente puro)
CALLS = {"recompute": 0}

def patched_recompute(self, models):
    if self.block_idx == 0:
        CALLS["recompute"] += 1
        return _orig_recompute(self, models)
    p = PERIOD["n"]
    if p is not None and self.block_idx % p == 0:
        CALLS["recompute"] += 1
        return _orig_recompute(self, models)
    # resident. Keep the cache and append incrementally. The position must come from
    # what the cache ITSELF believes, not from the global frame index. The recompute
    # rebases positions onto a short clean context, so mixing the two coordinate
    # systems inflates local_end_index until the write slice comes out empty.
    for block in models.pipeline.generator.model.blocks:
        block.self_attn.num_frame_per_block = models.pipeline.num_frame_per_block
    gei = models.pipeline.kv_cache1[0]["global_end_index"]
    return int(gei) // models.pipeline.frame_seq_length

GenerationSession.recompute_kv_cache = patched_recompute

def unlock_resident_path(models, attn_size=6, sink_frames=1):
    """Undoes the init_models mutation that makes sink eviction unreachable."""
    for block in models.pipeline.generator.model.blocks:
        block.self_attn.local_attn_size = attn_size
        block.self_attn.sink_size = sink_frames
        block.self_attn.max_attention_size = attn_size * 1560

def run(tag, period, seed, blocks=NUM_BLOCKS):
    print(f"===== {tag} (N={period or 'inf'}, seed {seed}) =====", flush=True)
    clip_dir = OUT / tag; clip_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    PERIOD["n"] = period; CALLS["recompute"] = 0
    rec = {"tag": tag, "N": period or "inf", "seed": seed, "blocks": blocks, "ok": False}
    params = GenerateParams(prompt=PROMPT, seed=seed, kv_cache_num_frames=3,
                            num_blocks=blocks, num_denoising_steps=4)
    count = {"n": 0}
    from PIL import Image
    def cb(pixels, frame_ids, event):
        event.synchronize()
        for j in range(pixels.shape[1]):
            arr = ((pixels[0, j].float().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(clip_dir / f"f{count['n']:04d}.png")
            count["n"] += 1
    try:
        session = GenerationSession(params, config, frame_callback=cb, models=models)
        unlock_resident_path(models)      # after init_models, which is what sets -1
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(blocks):
            try: session.generate_block(models)
            except asyncio.CancelledError: break
        torch.cuda.synchronize(); wall = time.time() - t0
        latents = session.all_latents[:, :session.current_start_frame].cpu()
        rec.update({"ok": True, "wall_sec": round(wall, 2), "frames": count["n"],
                    "fps": round(count["n"] / wall, 2),
                    "recompute_calls": CALLS["recompute"],
                    "peak_alloc_gb": gb(torch.cuda.max_memory_allocated()),
                    "latents_finite": bool(torch.isfinite(latents).all()),
                    "latent_absmax": round(float(latents.abs().max()), 3)})
        torch.save(latents, clip_dir / "latents.pt")
        session.dispose()
        subprocess.run(["ffmpeg", "-y", "-framerate", "16", "-i", str(clip_dir / "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        str(OUT / f"{tag}.mp4")], check=True, capture_output=True)
        rec["mp4"] = f"{tag}.mp4"
    except Exception:
        rec["error"] = traceback.format_exc()[-1200:]
        torch.cuda.empty_cache()
    RES["runs"].append(rec); flush()
    print(json.dumps(rec, default=str)[:600], flush=True)
    return rec

# fixed blind labels, so the clips are judged without knowing which regime is which
REGIMES = [(1, "C"), (2, "A"), (4, "D"), (None, "B")]
key = {}
for period, letter in REGIMES:
    for seed in SEEDS:
        tag = f"{letter}-s{seed}"
        key[tag] = {"N": period or "inf", "seed": seed}
        run(tag, period, seed)
(OUT / "blind-key.json").write_text(json.dumps(key, indent=1))

flush()
print("DONE", flush=True)
print(json.dumps([{k: r.get(k) for k in ("tag", "N", "seed", "fps", "recompute_calls",
                                         "peak_alloc_gb", "latents_finite", "latent_absmax")}
                  for r in RES["runs"]], indent=1), flush=True)
