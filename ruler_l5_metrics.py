"""L5 metrics: the fidelity ruler over saved trajectories.

Reads the bf16 references (results_1p3b), the W4A4 trajectories (results_w4a4) and
the 18-chunk W4A4 long run (results_l5). Everything is per latent frame index with
controls, never a single-frame verdict:

  divergence   W4A4 vs bf16, same seed, per frame (rel err + cosine)
  control      bf16 vs bf16 across seeds = what two DIFFERENT valid videos measure
  diversity    W4A4 across seeds vs bf16 across seeds (collapse check, ratio ~1)
  health       per-frame latent std, plus slope over the 54-frame long run (drift)
  early        first chunk (same init noise) = near-teacher-forced per-step error

Writes results_l5/ruler.json and prints the verdict table.
"""
import json
import torch
from pathlib import Path

L0, L4, L5 = Path("results_1p3b"), Path("results_w4a4"), Path("results_l5")
SEEDS = {42: "latents_kv3_s4.pt", 43: "latents_kv3_s4_seed43.pt", 44: "latents_kv3_s4_seed44.pt"}

def load(p): return torch.load(p, map_location="cpu", weights_only=True).float()

bf16 = {s: load(L0 / f) for s, f in SEEDS.items()}
w4a4 = {s: load(L4 / f) for s, f in SEEDS.items()}
bf16_kv21, w4a4_kv21 = load(L0 / "latents_kv21_s4.pt"), load(L4 / "latents_kv21_s4.pt")
long18 = load(L5 / "latents_w4a4_long18_s42.pt")

def per_frame_rel(a, b):
    d = (a - b).flatten(2)
    return (d.pow(2).mean(-1) / b.flatten(2).pow(2).mean(-1).clamp_min(1e-12)).sqrt()[0]

def per_frame_cos(a, b):
    af, bf = a.flatten(2)[0], b.flatten(2)[0]
    return torch.nn.functional.cosine_similarity(af, bf, dim=-1)

R = {}

# 1. same-seed divergence per frame, 3 seeds
div_rel = torch.stack([per_frame_rel(w4a4[s], bf16[s]) for s in SEEDS])
div_cos = torch.stack([per_frame_cos(w4a4[s], bf16[s]) for s in SEEDS])
R["divergence_rel_mean"] = div_rel.mean(0).tolist()
R["divergence_rel_std"] = div_rel.std(0).tolist()
R["divergence_cos_mean"] = div_cos.mean(0).tolist()

# 2. cross-seed controls (all 3 pairs), per frame
pairs = [(42, 43), (42, 44), (43, 44)]
ctrl_bf = torch.stack([per_frame_rel(bf16[a], bf16[b]) for a, b in pairs])
ctrl_w4 = torch.stack([per_frame_rel(w4a4[a], w4a4[b]) for a, b in pairs])
R["control_bf16_rel_mean"] = ctrl_bf.mean(0).tolist()
R["control_w4a4_rel_mean"] = ctrl_w4.mean(0).tolist()
R["diversity_ratio_mean"] = (ctrl_w4.mean() / ctrl_bf.mean()).item()

# 3. kv21 (global window) same-seed divergence
R["kv21_divergence_rel"] = per_frame_rel(w4a4_kv21, bf16_kv21).tolist()

# 4. latent health: per-frame std; long-run slope
std_bf = torch.stack([bf16[s].flatten(2).std(-1)[0] for s in SEEDS]).mean(0)
std_w4 = torch.stack([w4a4[s].flatten(2).std(-1)[0] for s in SEEDS]).mean(0)
R["latent_std_bf16"] = std_bf.tolist()
R["latent_std_w4a4"] = std_w4.tolist()
long_std = long18.flatten(2).std(-1)[0]
t = torch.arange(long_std.shape[0], dtype=torch.float32)
slope = ((t - t.mean()) * (long_std - long_std.mean())).sum() / (t - t.mean()).pow(2).sum()
R["long18_std"] = long_std.tolist()
R["long18_std_slope_per_frame"] = slope.item()
R["long18_finite"] = bool(torch.isfinite(long18).all())

# 5. aggregates by chunk position
def agg(x, lo, hi): return {"mean": float(x[:, lo:hi].mean()), "std": float(x[:, lo:hi].std())}
R["summary"] = {
    "chunk1_frames0-2": agg(div_rel, 0, 3),
    "chunk5_frames12-14": agg(div_rel, 12, 15),
    "chunk9_frames24-26": agg(div_rel, 24, 27),
    "control_bf16_chunk9": agg(ctrl_bf, 24, 27),
    "ratio_to_control_chunk9": float(div_rel[:, 24:27].mean() / ctrl_bf[:, 24:27].mean()),
    "ratio_to_control_chunk1": float(div_rel[:, 0:3].mean() / ctrl_bf[:, 0:3].mean()),
}

# chaos floor: 1% single-forward perturbation on pure bf16 (ruler_l5_chaos.py)
chaos = json.loads((L5 / "chaos_floor.json").read_text())
R["chaos_floor_rel"] = chaos["rel"]
R["chaos_amplification_27f"] = chaos["rel"][-1] / 0.01

# settled-region slopes (the first frames ramp as the video gains motion; that ramp
# is the process, so drift is judged on frames 10+ against bf16's own settled slope)
def settled_slope(x, lo):
    seg = x[lo:]
    tt = torch.arange(seg.shape[0], dtype=torch.float32)
    return (((tt - tt.mean()) * (seg - seg.mean())).sum() / (tt - tt.mean()).pow(2).sum()).item()
R["long18_settled_slope"] = settled_slope(long_std, 10)
R["bf16_settled_slope"] = settled_slope(std_bf, 10)
R["w4a4_kv3_settled_slope"] = settled_slope(std_w4, 10)

# verdict: chaos makes trajectory closeness unattainable for ANY sustained per-step
# perturbation (bf16's own rounding included), so the gate asks for divergence no
# worse than two valid videos, preserved diversity, healthy stats, and no drift.
s = R["summary"]
checks = {
    "below_two_valid_videos_early": s["ratio_to_control_chunk1"] < 0.66,
    "at_or_below_two_valid_videos_late": s["ratio_to_control_chunk9"] < 1.05,
    "diversity_not_collapsed": 0.8 < R["diversity_ratio_mean"] < 1.25,
    "latent_std_tracks_bf16": bool(abs(std_w4.mean() - std_bf.mean()) / std_bf.mean() < 0.10),
    "long_run_no_drift": bool(abs(R["long18_settled_slope"])
                              < max(2 * abs(R["bf16_settled_slope"]), 0.003))
                         and R["long18_finite"],
}
R["checks"] = checks
R["gate"] = all(checks.values())

(L5 / "ruler.json").write_text(json.dumps(R, indent=1))
print("frame:      ", "  ".join(f"{i:5d}" for i in [0, 1, 2, 13, 26]))
print("div rel:    ", "  ".join(f"{R['divergence_rel_mean'][i]:.3f}" for i in [0, 1, 2, 13, 26]))
print("chaos 1%:   ", "  ".join(f"{R['chaos_floor_rel'][i]:.3f}" for i in [0, 1, 2, 13, 26]))
print("ctrl bf16:  ", "  ".join(f"{R['control_bf16_rel_mean'][i]:.3f}" for i in [0, 1, 2, 13, 26]))
print("div cos:    ", "  ".join(f"{R['divergence_cos_mean'][i]:.3f}" for i in [0, 1, 2, 13, 26]))
print(json.dumps({**s, "diversity_ratio": R["diversity_ratio_mean"],
                  "chaos_amp_27f": round(R["chaos_amplification_27f"], 1),
                  "long18_settled_slope": R["long18_settled_slope"],
                  "bf16_settled_slope": R["bf16_settled_slope"]}, indent=1))
print("checks:", json.dumps(checks, indent=1))
print("GATE:", "PASS" if R["gate"] else "FAIL")
