"""Identity ruler. The CLIP(t) against CLIP(0) retention curve, per regime.

First attempt at a proxy for identity drift, which is the failure mode the eye catches
first in autoregressive video. It is published as a negative result, because it does not
measure what it looks like it measures.

Calibration before verdict. The dancer clips have a STATIC camera, so the fall in
similarity against frame 0 is dominated by the subject changing, and that is where the
proxy has to register what the eye already saw. It does, around 20 to 23 percent loss by
the last third, in every regime. But the spread between seeds is larger than the spread
between regimes, so it separates nothing at n=3.

The skater clips have a MOVING camera, and there the similarity falls because new scene
keeps entering the frame, independently of any drift. Retention then rises monotonically
as resets get rarer, which would rank the upstream regime last. What the metric is
actually tracking is how much the camera moved. A whole-frame embedding is the wrong
probe. The next ruler crops the subject.

The comparison is never trajectory-glued. Four regimes produce four different valid
videos, because the sampler is chaotic (see the chaos floor measurement in the L5 stage),
so what gets read is a distribution, the mean per regime with its spread across seeds.

Run:
  RULER_STRIDE=4 python ruler_identity.py results_n results_n_skate
"""
import sys, json, re
from pathlib import Path
import torch
from PIL import Image

STRIDE = int(__import__("os").environ.get("RULER_STRIDE", "4"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE)
model.eval()
print("CLIP carregado", flush=True)

@torch.no_grad()
def embed_clip_frames(clip_dir, stride=STRIDE, batch=64):
    frames = sorted(clip_dir.glob("f*.png"))
    if not frames:
        return None, 0
    picked = frames[::stride]
    embs = []
    for i in range(0, len(picked), batch):
        ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in picked[i:i + batch]]).to(DEVICE)
        e = model.encode_image(ims)
        embs.append(torch.nn.functional.normalize(e, dim=-1).float().cpu())
    return torch.cat(embs), len(frames)

def curve(embs):
    """Retenção contra o frame 0 e coerência local (frame a frame)."""
    ref = embs[:1]
    to_ref = (embs @ ref.T).squeeze(1)                  # cos(t, 0)
    local = (embs[1:] * embs[:-1]).sum(-1)              # cos(t, t-1)
    n = len(to_ref)
    third = max(1, n // 3)
    return {
        "n_samples": n,
        "to_ref_final": round(float(to_ref[-1]), 4),
        "to_ref_last_third_mean": round(float(to_ref[-third:].mean()), 4),
        "to_ref_min": round(float(to_ref.min()), 4),
        "to_ref_slope_per_sample": round(float((to_ref[-1] - to_ref[0]) / max(1, n - 1)), 6),
        "local_coherence_mean": round(float(local.mean()), 4),
        "local_coherence_min": round(float(local.min()), 4),
        "to_ref_curve": [round(float(x), 4) for x in to_ref],
    }

def run_set(root):
    root = Path(root)
    key_path = root / "blind-key.json"
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    res = json.loads((root / "results.json").read_text())
    tag_to_n = {r["tag"]: r["N"] for r in res.get("runs", [])}
    out = {}
    for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tag = clip_dir.name
        embs, n_frames = embed_clip_frames(clip_dir)
        if embs is None:
            continue
        c = curve(embs)
        c["frames_total"] = n_frames
        c["N"] = tag_to_n.get(tag, key.get(tag, {}).get("N"))
        c["seed"] = int(re.search(r"s(\d+)$", tag).group(1)) if re.search(r"s(\d+)$", tag) else None
        out[tag] = c
        print(f"  {tag:10s} N={str(c['N']):4s} seed={c['seed']} "
              f"final={c['to_ref_final']:.3f} last3rd={c['to_ref_last_third_mean']:.3f} "
              f"localmin={c['local_coherence_min']:.3f}", flush=True)
    return out

def by_regime(out):
    agg = {}
    for tag, c in out.items():
        agg.setdefault(str(c["N"]), []).append(c)
    summary = {}
    for n, items in sorted(agg.items(), key=lambda kv: (kv[0] == "inf", kv[0])):
        f = [i["to_ref_last_third_mean"] for i in items]
        lo = [i["local_coherence_min"] for i in items]
        summary[n] = {
            "seeds": len(items),
            "retencao_ultimo_terco_media": round(sum(f) / len(f), 4),
            "retencao_ultimo_terco_min": round(min(f), 4),
            "retencao_ultimo_terco_max": round(max(f), 4),
            "coerencia_local_min_media": round(sum(lo) / len(lo), 4),
        }
    return summary

for root in sys.argv[1:]:
    print(f"===== {root} =====", flush=True)
    out = run_set(root)
    summ = by_regime(out)
    Path(root, "ruler_identity.json").write_text(json.dumps(
        {"per_clip": out, "por_regime": summ}, indent=1))
    print(json.dumps(summ, indent=1), flush=True)
print("DONE", flush=True)
