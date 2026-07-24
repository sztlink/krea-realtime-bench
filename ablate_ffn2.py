"""Error decomposition on the worst PTQ module (ffn.2 block 13) + a mid one (block 0).
Which side carries the error: W4 (weights), A4 (activations), and what the rank-32
branch buys. Informs whether the unsigned-shift trick is worth building in L4."""
import os, sys, json, types
stub = types.ModuleType("deepcompressor.csrc.load"); stub._C = None
sys.modules["deepcompressor.csrc.load"] = stub
import torch
from pathlib import Path
from safetensors import safe_open
sys.path.insert(0, ".")
from ptq_wan import quant_sint4_sim, quant_token_sint4_sim, svdquant_branch, svd_rank
from deepcompressor.calib.smooth import get_smooth_scale

DEV = "cuda"
report = {}
with safe_open("checkpoints/self_forcing_dmd.sft", framework="pt") as f:
    for blk in [13, 0]:
        w = f.get_tensor(f"model.blocks.{blk}.ffn.2.weight").to(DEV, torch.float32)
        res = torch.load(f"collect_out/block{blk:02d}/ffn_down.pt",
                         map_location="cpu", weights_only=True)
        x = res["reservoir"].to(DEV, torch.float32)
        ref = x @ w.T
        den = ref.pow(2).mean().clamp_min(1e-12)

        def rel(out): return ((out - ref).pow(2).mean() / den).sqrt().item()

        # best smooth from the ptq report was ~alpha .3-.45 beta 0; recompute a quick pick
        x_span, w_span = x.abs().amax(0), w.abs().amax(0)
        best = (None, float("inf"))
        for alpha in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            s = get_smooth_scale(alpha=alpha, beta=0.0, alpha_base=x_span,
                                 beta_base=w_span).clamp_min(1e-5)
            b, a, Rq, _ = svdquant_branch(w * s.unsqueeze(0), num_iters=4)
            x_s = x / s.unsqueeze(0)
            e = rel(quant_token_sint4_sim(x_s) @ Rq.T + (x_s @ a.T) @ b.T)
            if e < best[1]: best = (s, e, alpha)
        s = best[0]; x_s = x / s.unsqueeze(0)
        w_s = w * s.unsqueeze(0)
        b, a, Rq, _ = svdquant_branch(w_s, num_iters=64)

        cases = {}
        # full W4A4 (the shipped artifact)
        cases["w4a4_branch"] = rel(quant_token_sint4_sim(x_s) @ Rq.T + (x_s @ a.T) @ b.T)
        # W4 only: activations fp
        cases["w4_fp_acts"] = rel(x_s @ Rq.T + (x_s @ a.T) @ b.T)
        # A4 only: weights fp
        cases["a4_fp_wgts"] = rel(quant_token_sint4_sim(x_s) @ w_s.T)
        # no branch: plain smooth + RTN
        Rq_nb, _ = quant_sint4_sim(w_s)
        cases["w4a4_no_branch"] = rel(quant_token_sint4_sim(x_s) @ Rq_nb.T)
        # no smooth at all
        b2, a2, Rq2, _ = svdquant_branch(w, num_iters=16)
        cases["w4a4_no_smooth"] = rel(quant_token_sint4_sim(x) @ Rq2.T + (x @ a2.T) @ b2.T)
        report[f"block{blk}"] = {"alpha": best[2], **{k: round(v, 4) for k, v in cases.items()}}
        print(f"block {blk}:", json.dumps(report[f"block{blk}"]))

Path("ablate_ffn2.json").write_text(json.dumps(report, indent=1))
