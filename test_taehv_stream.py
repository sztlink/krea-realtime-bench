"""Streaming TAEHV against block-independent TAEHV, looking for the seam.

If temporal memory does not survive between calls, a seam appears every twelve pixel
frames. This decodes the same real latent trajectory both ways and measures frame to
frame difference, which spikes at a seam.
"""
import os, json
from pathlib import Path
os.environ.setdefault("DO_COMPILE", "false")
import torch, numpy as np
from taehv_stream import TAEHVDecoderWrapper, decode_stream
from demo_utils.taehv import TAEHV

OUT = Path("results_taehv"); OUT.mkdir(exist_ok=True)
lat = torch.load("results_v/latents_kv3_s4.pt", map_location="cpu", weights_only=True)
NFPB = 3
blocks = [lat[:, i*NFPB:(i+1)*NFPB].cuda().half() for i in range(lat.shape[1] // NFPB)]
print("blocos", len(blocks), flush=True)

def frame_diffs(px_all):
    x = px_all.float()
    d = (x[:, 1:] - x[:, :-1]).abs().mean(dim=(0, 2, 3, 4))
    return d.cpu().numpy()

RES = {}
# 1. no state between calls, which is what upstream would do
tae = TAEHV(checkpoint_path="taew2_1.pth").cuda().half().eval()
outs = []
with torch.no_grad():
    for b in blocks:
        outs.append(tae.decode_video(b, parallel=False, show_progress_bar=False))
px_nostate = torch.cat(outs, 1)

# 2. com estado carregado
w = TAEHVDecoderWrapper()
outs = []
cache = [None] * 55
for b in blocks:
    px, cache = w(b, *cache)
    outs.append((px + 1) / 2)
px_state = torch.cat(outs, 1)

# 3. control. The heavy decoder, which carries real state in its 55 slots
from release_server import load_vae
_, heavy = load_vae()
outs = []; cache = [None]*55
import torch as _t
with _t.no_grad():
    for b in blocks:
        px, cache = heavy(b, *cache)
        outs.append((px.float().clamp(-1,1)+1)/2)
px_heavy = _t.cat(outs, 1)

for name, px in (("pesado_controle", px_heavy), ("sem_estado", px_nostate), ("com_estado", px_state)):
    d = frame_diffs(px)
    nf = px.shape[1] // len(blocks)
    seams = [i for i in range(len(d)) if (i + 1) % nf == 0]
    outros = [i for i in range(len(d)) if (i + 1) % nf != 0]
    RES[name] = {
        "frames": int(px.shape[1]),
        "frames_por_bloco": int(nf),
        "diff_media_geral": round(float(d.mean()), 5),
        "diff_media_nas_emendas": round(float(d[seams].mean()), 5),
        "diff_media_fora": round(float(d[outros].mean()), 5),
        "razao_emenda": round(float(d[seams].mean() / d[outros].mean()), 3),
    }
    print(name, json.dumps(RES[name]), flush=True)

(OUT / "stream.json").write_text(json.dumps(RES, indent=1))
print("DONE", flush=True)
