"""V2V on the Self-Forcing 1.3B in bf16, for the question abstraction raises.

Leitura do Felipe: se o alvo virou abstrato (elemental de fogo, tinta, plasma), a
vantagem do 14B — fidelidade semantica, anatomia, realismo — deixa de estar em jogo,
e o modelo menor talvez volte a fazer sentido.

O 1.3B roda a 7.5 fps em bf16 no regime T2V contra 2.88 do 14B W4A4. Em V2V com
poucos passos entra em outro patamar. Este bench responde primeiro a pergunta
aesthetic question first, in bf16, because quantising is only worth it if the look holds.

Caveat. This 1.3B is the Self-Forcing sibling of Wan 2.1, NOT a smaller Krea Realtime.
Different model, different look, never judged.
"""
import os, json, time, asyncio, traceback
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_v2v_1p3b"); OUT.mkdir(exist_ok=True)
PROMPT = os.environ["V2V_PROMPT"]
INPUT = os.environ.get("V2V_INPUT", "results_n/C-s42.mp4")
CONFIG_PATH = "configs/self_forcing_server.yaml"
BLOCKS = int(os.environ.get("V2V_BLOCKS", "8"))
STRENGTHS = [float(x) for x in os.environ.get("V2V_STRENGTHS", "0.85").split(",")]
STEPS = [int(x) for x in os.environ.get("V2V_STEPS", "4").split(",")]
TAG = os.environ.get("V2V_TAG", "p13")
RES = {"runs": []}
def flush(): (OUT / "results.json").write_text(json.dumps(RES, indent=1, default=str))

from release_server import load_merge_config, load_transformer, load_vae, \
    load_pipeline, GenerateParams, GenerationSession, Models
from wan.modules.causal_model import CausalWanModel
def _from_config(path, **kw):
    return CausalWanModel.from_config(CausalWanModel.load_config(str(path)), **kw)
CausalWanModel.from_pretrained = staticmethod(_from_config)

config = load_merge_config(CONFIG_PATH)
transformer = load_transformer(config)
torch.cuda.synchronize()
print("1.3B carregado", round(torch.cuda.memory_allocated()/1e9, 2), "GB", flush=True)

import re as _re
_slug = _re.sub(r"[^a-z0-9]+", "-", PROMPT.lower())[:60].strip("-")
_emb = torch.load(f"embeddings/{_slug}.pt", map_location="cpu", weights_only=True)
_cond = {"prompt_embeds": _emb["prompt_embeds"].to("cuda", torch.bfloat16)}
class _StaticEnc(torch.nn.Module):
    def forward(self, text_prompts): return {k: v.clone() for k, v in _cond.items()}
vae_encoder, vae_decoder = load_vae()
pipeline = load_pipeline(config, torch.cuda.current_device(), transformer, _StaticEnc(), vae_decoder)
models = Models(_StaticEnc(), transformer, pipeline, vae_encoder, vae_decoder)
print("LOAD OK, entrada:", INPUT, flush=True)

def run(strength, steps):
    tag = f"{TAG}_s{strength}_st{steps}"
    print(f"===== {tag} =====", flush=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rec = {"tag": tag, "strength": strength, "steps": steps, "modelo": "1.3B bf16",
           "entrada": INPUT, "ok": False}
    frames = []
    def cb(pixels, frame_ids, event):
        event.synchronize(); frames.append(pixels[0].float().cpu())
    try:
        params = GenerateParams(prompt=PROMPT, seed=42, kv_cache_num_frames=3,
                                num_blocks=BLOCKS, num_denoising_steps=steps,
                                input_video=INPUT, strength=strength)
        session = GenerationSession(params, config, frame_callback=cb, models=models)
        torch.cuda.synchronize(); t0 = time.time(); n = 0
        for _ in range(BLOCKS):
            try: out = session.generate_block(models)
            except asyncio.CancelledError: break
            n += out.shape[1]
        torch.cuda.synchronize(); wall = time.time() - t0
        lat = session.all_latents[:, :session.current_start_frame].cpu()
        rec.update({"ok": True, "wall_sec": round(wall, 2), "frames": n,
                    "fps": round(n / wall, 2), "bloco_s": round(wall / BLOCKS, 3),
                    "peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
                    "latents_finite": bool(torch.isfinite(lat).all())})
        if frames:
            import subprocess
            from PIL import Image
            cd = OUT / tag; cd.mkdir(parents=True, exist_ok=True)
            i = 0
            for blk in frames:
                for j in range(blk.shape[0]):
                    arr = ((blk[j].clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
                    Image.fromarray(arr).save(cd / f"f{i:04d}.png"); i += 1
            subprocess.run(["ffmpeg","-y","-framerate","16","-i",str(cd/"f%04d.png"),
                            "-c:v","libx264","-pix_fmt","yuv420p","-crf","18",
                            str(OUT / f"{tag}.mp4")], check=True, capture_output=True)
            rec["mp4"] = f"{tag}.mp4"
        session.dispose()
    except Exception:
        rec["error"] = traceback.format_exc()[-700:]
        torch.cuda.empty_cache()
    RES["runs"].append(rec); flush()
    print(json.dumps(rec, default=str)[:400], flush=True)
    torch.cuda.empty_cache()

for s in STRENGTHS:
    for st in STEPS:
        run(s, st)
flush(); print("DONE", flush=True)
