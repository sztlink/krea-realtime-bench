"""V2V: the regime FLAMA actually needs, and the one this lane never measured.

The interactive part of FLAMA films a person and shows them catching fire, the way Lucy
demonstrated. That is video to video. Everything measured in this lineage so far was text
to video, generating from pure noise, and the verdict that one and two denoise steps look
bad came from that regime.

V2V is different in a way that matters for speed. The schedule is
`linspace(strength * 1000, 0, steps)`, so at strength 0.7 the first timestep is 700 rather
than 1000. The model starts from a partially noised camera frame instead of from nothing,
so each step covers less noise, and few steps may hold where they did not before.

Measures fps and produces clips across strength and step count, on a real input video.

Run:
  DISABLE_SAGEATTENTION=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    V2V_INPUT=<video.mp4> .venv/bin/python bench_v2v.py
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch
from safetensors import safe_open
import quant_kv

OUT = Path("results_v2v"); OUT.mkdir(exist_ok=True)
PROMPT = os.environ.get("V2V_PROMPT",
    "A person engulfed in flames, fire wrapping around the body, glowing embers rising, "
    "dark background, cinematic")
INPUT = os.environ.get("V2V_INPUT", "results_qkv/clip_kv3.mp4")
CONFIG_PATH = "configs/self_forcing_server_14b.yaml"
KREA_CKPT = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = "checkpoints-14b-w4a4-ckv"
BLOCKS = int(os.environ.get("V2V_BLOCKS", "12"))
STRENGTHS = [float(x) for x in os.environ.get("V2V_STRENGTHS", "0.7").split(",")]
STEPS = [int(x) for x in os.environ.get("V2V_STEPS", "4,2,1").split(",")]
USE_TAEHV = os.environ.get("V2V_TAEHV", "0") == "1"
RES = {"runs": []}
def flush(): (OUT / "results.json").write_text(json.dumps(RES, indent=1, default=str))

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
if USE_TAEHV:
    from taehv_stream import TAEHVDecoderWrapper
    vae_decoder = TAEHVDecoderWrapper()
    print("decoder: TAEHV", flush=True)
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, _StaticEnc(), vae_decoder)
models = Models(_StaticEnc(), transformer, pipeline, vae_encoder, vae_decoder)
print("LOAD OK, entrada:", INPUT, flush=True)

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

def run(strength, steps):
    tag = f"v2v_s{strength}_st{steps}"
    print(f"===== {tag} =====", flush=True)
    pipeline.kv_cache1 = []
    quant_kv.install(pipeline, k_bits=4, v_bits=4, group_mode="bands4", sink_frames=1, verbose=False)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rec = {"tag": tag, "strength": strength, "steps": steps, "entrada": INPUT, "ok": False}
    frames = []
    def cb(pixels, frame_ids, event):
        event.synchronize(); frames.append(pixels[0].float().cpu())
    try:
        params = GenerateParams(prompt=PROMPT, seed=42, kv_cache_num_frames=3,
                                num_blocks=BLOCKS, num_denoising_steps=steps,
                                input_video=INPUT, strength=strength)
        t_enc = time.time()
        session = GenerationSession(params, config, frame_callback=cb, models=models)
        rec["encode_sec"] = round(time.time() - t_enc, 2)
        for b in models.pipeline.generator.model.blocks:
            b.self_attn.local_attn_size = 6; b.self_attn.sink_size = 1
            b.self_attn.max_attention_size = 6 * 1560
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        for _ in range(BLOCKS):
            try: out = session.generate_block(models)
            except asyncio.CancelledError: break
            n += out.shape[1]
        torch.cuda.synchronize(); wall = time.time() - t0
        lat = session.all_latents[:, :session.current_start_frame].cpu()
        rec.update({"ok": True, "wall_sec": round(wall, 2), "frames": n,
                    "fps": round(n / wall, 2), "bloco_s": round(wall / BLOCKS, 3),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "latents_finite": bool(torch.isfinite(lat).all())})
        if frames:
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
        rec["error"] = traceback.format_exc()[-700:]
        torch.cuda.empty_cache()
    RES["runs"].append(rec); flush()
    print(json.dumps(rec, default=str)[:420], flush=True)
    torch.cuda.empty_cache()

for s in STRENGTHS:
    for st in STEPS:
        run(s, st)
flush()
print("DONE", flush=True)
