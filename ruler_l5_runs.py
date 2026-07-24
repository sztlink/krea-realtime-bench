"""L5 runs: instrumented generations feeding the fidelity ruler.

One process, two model states: the bf16 model runs first, then the same instance is
swapped in place to W4A4 (load_w4a4_blocks) and runs again. Captures pixel frames at
fixed indices for the visual strip, plus a 2x-length W4A4 run (18 chunks) for drift.
Latents for the standard runs already exist (results_1p3b, results_w4a4); this adds
the long-run latents and the frame strips. Everything else is ruler_l5_metrics.py.
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_l5"); OUT.mkdir(exist_ok=True)
PROMPT = "A person dancing in an empty warehouse, dramatic lighting, camera static"
CONFIG_PATH = "configs/self_forcing_server.yaml"
STRIP_IDX = [0, 33, 66, 101]          # pixel frame indices for the 9-chunk strip
STRIP_IDX_LONG = [0, 50, 101, 152, 203]  # for the 18-chunk run

from release_server import load_merge_config, load_transformer, \
    load_vae, load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel
from nunchaku_causal_wan import load_w4a4_blocks
def _from_config(path, **kw):
    return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config(CONFIG_PATH)
transformer = load_transformer(config)

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
print("LOAD OK (bf16)")

RES = {"runs": []}
def run(tag, seed=42, num_blocks=9, strip=STRIP_IDX, save_latents=None):
    print(f"===== {tag} =====")
    torch.cuda.empty_cache()
    params = GenerateParams(prompt=PROMPT, seed=seed, kv_cache_num_frames=3,
                            num_blocks=num_blocks, num_denoising_steps=4)
    frames, count = {}, {"n": 0}
    def cb(pixels, frame_ids, event):
        event.synchronize()
        for j in range(pixels.shape[1]):
            if count["n"] + j in strip:
                frames[count["n"] + j] = pixels[0, j].float().cpu()
        count["n"] += pixels.shape[1]
    rec = {"tag": tag, "seed": seed, "chunks": num_blocks, "ok": False}
    try:
        session = GenerationSession(params, config, frame_callback=cb, models=models)
        t0 = time.time()
        for _ in range(num_blocks):
            try: session.generate_block(models)
            except asyncio.CancelledError: break
        torch.cuda.synchronize()
        latents = session.all_latents[:, :session.current_start_frame].cpu()
        rec.update({"ok": True, "sec": round(time.time() - t0, 1),
                    "pixel_frames": count["n"],
                    "latents_finite": bool(torch.isfinite(latents).all()),
                    "latent_frames": int(latents.shape[1])})
        if save_latents:
            torch.save(latents, OUT / save_latents)
        from PIL import Image
        for idx, fr in frames.items():
            arr = ((fr.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(OUT / f"strip_{tag}_f{idx:03d}.png")
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc()
    RES["runs"].append(rec)
    (OUT / "runs.json").write_text(json.dumps(RES, indent=1, default=str))
    print(json.dumps(rec, default=str)[:400])

# bf16 strip (latents for these already exist in results_1p3b)
run("bf16_kv3_s42", seed=42)

# swap the SAME instance to W4A4 and rerun
load_w4a4_blocks(models.transformer.model, "ptq_out/self-forcing-1p3b-w4a4")
run("w4a4_kv3_s42", seed=42)
run("w4a4_long18_s42", seed=42, num_blocks=18, strip=STRIP_IDX_LONG,
    save_latents="latents_w4a4_long18_s42.pt")
print("DONE")
