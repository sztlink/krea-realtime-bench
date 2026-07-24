"""Quick outlier analysis over collect_out: per stream, aggregated across blocks,
the channelwise outlier ratio (absmax / absmean) that motivates smoothing, and
worst blocks. Informs the L3 skip/keep list."""
import json, torch
from pathlib import Path

OUT = Path("collect_out")
n_blocks = 30
streams = ["self_qkv", "self_o", "cross_q", "cross_kv", "cross_o", "ffn_up", "ffn_down"]
report = {}
for s in streams:
    ratios, worst = [], []
    for b in range(n_blocks):
        d = torch.load(OUT / f"block{b:02d}" / f"{s}.pt", map_location="cpu", weights_only=True)
        amax, amean = d["absmax"], d["absmean"].clamp_min(1e-6)
        r = (amax / amean)
        ratios.append(r.median().item())
        worst.append((r.max().item(), b))
    worst.sort(reverse=True)
    report[s] = {
        "outlier_ratio_median_of_blocks": round(sum(ratios) / len(ratios), 1),
        "worst_channel_ratio": round(worst[0][0], 1),
        "worst_block": worst[0][1],
        "top3_blocks": [b for _, b in worst[:3]],
    }
print(json.dumps(report, indent=1))
(OUT / "outlier_report.json").write_text(json.dumps(report, indent=1))
