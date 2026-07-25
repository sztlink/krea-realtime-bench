"""Locates the transient grey collapses, and which block they fall in.

Report from watching the clips. The 12 frame context clip has two short moments where
everything goes slightly grey and recovers, the 6 frame clip has one, and the 3 frame clip
was not mentioned. The frequency rising with the window is the interesting part.

Measures mean saturation and contrast per frame, finds the dips, and converts frame index
into block index.

Method note that cost a wrong conclusion here. Counting events against a threshold RELATIVE
to each clip's own distribution does not compare clips, because a stable clip flags a
shallow dip and an unstable one hides a deep one. The comparable measure is absolute, how
far the worst frame falls below the median and what fraction of the clip sits below 85
percent of it.

Run:
  python find_grey.py results_n/clip_a results_n/clip_b
"""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image

def analyze(clip_dir):
    frames = sorted(Path(clip_dir).glob("f*.png"))
    sat, con, lum = [], [], []
    for p in frames:
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        mx, mn = a.max(axis=2), a.min(axis=2)
        s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        sat.append(float(s.mean())); con.append(float(y.std())); lum.append(float(y.mean()))
    return np.array(sat), np.array(con), np.array(lum), len(frames)

def dips(v, k=2.5):
    """indices onde v cai k desvios abaixo da mediana. Criterio RELATIVO ao proprio
    clipe, entao serve para localizar o evento e NAO para comparar clipes entre si."""
    med, sd = np.median(v), v.std()
    return [i for i, x in enumerate(v) if x < med - k * sd]

def depth(v, frac=0.85):
    """Medida ABSOLUTA, comparavel entre clipes: quanto o pior frame cai em relacao a
    mediana, e que fracao do clipe fica abaixo de `frac` da mediana. Contar eventos com
    limiar relativo engana, porque um clipe estavel marca queda rasa e um instavel
    esconde queda funda."""
    med = float(np.median(v))
    return {"queda_max_pct": round(100 * (med - float(v.min())) / med, 1),
            "frames_abaixo_de_85pct": int((v < frac * med).sum()),
            "fracao_abaixo_de_85pct": round(float((v < frac * med).mean()), 4)}

def runs(idx):
    """agrupa indices consecutivos em intervalos"""
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out

RES = {}
for d in sys.argv[1:]:
    sat, con, lum, n = analyze(d)
    name = Path(d).name
    ds, dc = dips(sat), dips(con)
    both = sorted(set(ds) & set(dc))
    RES[name] = {
        "frames": n,
        "sat_mediana": round(float(np.median(sat)), 4),
        "sat_min": round(float(sat.min()), 4),
        "con_mediana": round(float(np.median(con)), 4),
        "con_min": round(float(con.min()), 4),
        "quedas_saturacao": runs(ds),
        "quedas_contraste": runs(dc),
        "quedas_nas_duas": runs(both),
        "profundidade_contraste": depth(con),
        "profundidade_saturacao": depth(sat),
    }
    # bloco: o bloco 0 entrega 9 frames (descarta 3), os demais 12
    def blk(f):
        return 0 if f < 9 else 1 + (f - 9) // 12
    RES[name]["blocos_das_quedas"] = sorted({blk(f) for f in both})
    print(name, json.dumps(RES[name], indent=1), flush=True)

Path("results_qkv/grey.json").write_text(json.dumps(RES, indent=1))
print("DONE", flush=True)
