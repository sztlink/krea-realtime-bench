"""TAEHV against the full Wan VAE decoder. Quality first, then speed.

The block attribution at one denoise step puts the VAE decode at 42.6 percent of the
block, the largest single slice left. The server hardcodes the heavy decoder and never
reads the `use_taehv` flag that sits in both configs, so the tiny decoder that ships in
demo_utils/taehv.py has never run in this lineage.

Weights come from a third party mirror (the original madebyollin repo is gone), so they
get verified rather than trusted. Decode the same real latents through both decoders,
compare the pixels, and time them.

Run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python test_taehv.py
"""
import os, json, time
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch

OUT = Path("results_taehv"); OUT.mkdir(exist_ok=True)
RES = {}

lat = torch.load("results_v/latents_kv3_s4.pt", map_location="cpu", weights_only=True)
print("latentes", tuple(lat.shape), lat.dtype, flush=True)
NFPB = 3
n_blocks = lat.shape[1] // NFPB
blocks = [lat[:, i * NFPB:(i + 1) * NFPB].cuda().half() for i in range(n_blocks)]

# ---------------------------------------------------------------- decoder pesado
from release_server import load_vae
vae_encoder, vae_decoder = load_vae()

@torch.no_grad()
def run_heavy(reps=2):
    per = []
    for _ in range(reps):
        cache = [None] * 55
        for b in blocks:
            torch.cuda.synchronize(); t = time.perf_counter()
            px, cache = vae_decoder(b, *cache)
            torch.cuda.synchronize(); per.append(time.perf_counter() - t)
    return per, px

per_h, px_h = run_heavy()
RES["pesado"] = {"seg_por_bloco": round(sum(per_h[1:]) / len(per_h[1:]), 4),
                 "saida": list(px_h.shape), "min": round(float(px_h.min()), 3),
                 "max": round(float(px_h.max()), 3)}
print("pesado", json.dumps(RES["pesado"]), flush=True)
ref = px_h.float().cpu().clone()

# ---------------------------------------------------------------- decoder minusculo
from demo_utils.taehv import TAEHV
tae = TAEHV(checkpoint_path="taew2_1.pth").cuda().half().eval()
n_par = sum(p.numel() for p in tae.parameters())
print("TAEHV params", round(n_par / 1e6, 2), "M", flush=True)

@torch.no_grad()
def run_tiny(reps=2, parallel=False):
    per = []
    for _ in range(reps):
        for b in blocks:
            torch.cuda.synchronize(); t = time.perf_counter()
            px = tae.decode_video(b, parallel=parallel, show_progress_bar=False)
            torch.cuda.synchronize(); per.append(time.perf_counter() - t)
    return per, px

for parallel in (False, True):
    try:
        per_t, px_t = run_tiny(parallel=parallel)
        key = "minusculo_parallel" if parallel else "minusculo_sequencial"
        RES[key] = {"seg_por_bloco": round(sum(per_t[1:]) / len(per_t[1:]), 4),
                    "saida": list(px_t.shape), "min": round(float(px_t.min()), 3),
                    "max": round(float(px_t.max()), 3),
                    "aceleracao": round(RES["pesado"]["seg_por_bloco"] / (sum(per_t[1:]) / len(per_t[1:])), 2)}
        print(key, json.dumps(RES[key]), flush=True)
    except Exception as e:
        RES[f"erro_parallel_{parallel}"] = f"{type(e).__name__}: {e}"
        print("ERRO", parallel, type(e).__name__, e, flush=True)

# ---------------------------------------------------------------- fidelidade
# The heavy decoder returns about [-1,1] and TAEHV about [0,1]. Normalise both first.
try:
    a = ((ref.clamp(-1, 1) + 1) / 2)
    b = px_t.float().cpu().clamp(0, 1)
    if a.shape == b.shape:
        d = (a - b).abs()
        RES["fidelidade"] = {"mae": round(float(d.mean()), 4), "max": round(float(d.max()), 4),
                             "shapes_batem": True}
    else:
        RES["fidelidade"] = {"shapes_batem": False, "pesado": list(a.shape), "minusculo": list(b.shape)}
    print("fidelidade", json.dumps(RES["fidelidade"]), flush=True)
except Exception as e:
    RES["fidelidade"] = f"{type(e).__name__}: {e}"
    print("fidelidade ERRO", e, flush=True)

# side by side images, for the eye
try:
    from PIL import Image
    import numpy as np
    def to_img(t, lo, hi):
        x = ((t[0, -1].float().cpu().clamp(lo, hi) - lo) / (hi - lo) * 255).byte()
        return Image.fromarray(x.permute(1, 2, 0).numpy())
    ih, it = to_img(px_h, -1, 1), to_img(px_t, 0, 1)
    w, h = ih.size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(ih, (0, 0)); canvas.paste(it, (w, 0))
    canvas.save(OUT / "pesado_vs_minusculo.jpg", quality=93)
    print("imagem salva", canvas.size, flush=True)
except Exception as e:
    print("imagem ERRO", e, flush=True)

(OUT / "resultado.json").write_text(json.dumps(RES, indent=1))
print("DONE", flush=True)
