"""L3 driver: SVDQuant W4A4 INT4 PTQ of Self-Forcing 1.3B, fed by the L2 collect.

DeepCompressor as a library, not a framework. Its diffusion app collects activations
by replaying whole-model forwards, which the causal loop cannot do (mutable KV state
in the args). Here the activations come from collect_out/ (real-loop reservoirs) and
calibration runs per linear group with upstream conventions:
  smooth      -> spans via ChannelMetric.abs_max + get_smooth_scale (exact upstream
                 math), grid over (alpha, beta) pairs scored by W4A4-simulated output
                 error on the reservoir (OutputsError objective at the module level)
  branch      -> SVDQuant rank-32, alternating refinement (randomized SVD), split
                 a = Vh [rank, ic], b = U*S [oc, rank] matching LowRankBranch
  weights     -> sint4 symmetric RTN, group 64, scale = groupwise absmax / 7
Outputs match the upstream ptq.py contract consumed by the nunchaku convert:
  model.pt  = fake-quant Q(W_smooth - branch) weights bf16 + all unquantized tensors
  scale.pt  = {name}.weight.scale.0 [oc,1,ng,1] bf16
  smooth.pt = {block}.{group key} -> s [ic] fp32 (y = (x/s) @ (W*s).T convention)
  branch.pt = {block}.{group key} -> {"a.weight" [rank,ic], "b.weight" [oc,rank]}
              in SMOOTHED space (convert un-smooths lora_down itself)

Quantized targets per block (skip list from the L2 outlier report: cross k/v and
everything non-linear stays bf16):
  self_attn.{q,k,v} as one group (shared input/smooth/branch, keyed self_attn.q)
  self_attn.o | cross_attn.q | cross_attn.o | ffn.0 | ffn.2

Run on the 4090:
  cd ~/realtime-diffusion/realtime-video
  PYTHONPATH=~/realtime-diffusion/src/deepcompressor ../venv-ptq/bin/python ptq_wan.py
"""
import os, sys, json, time, types
from pathlib import Path

# codebook kernels JIT-compile CUDA at import and need nvcc; int4 never touches them
stub = types.ModuleType("deepcompressor.csrc.load"); stub._C = None
sys.modules["deepcompressor.csrc.load"] = stub

import torch
from safetensors import safe_open
from deepcompressor.calib.smooth import get_smooth_scale

CKPT = os.environ.get("PTQ_CKPT", "checkpoints/self_forcing_dmd.sft")
COLLECT = Path(os.environ.get("PTQ_COLLECT", "collect_out"))
OUT = Path(os.environ.get("PTQ_OUT", "ptq_out")); OUT.mkdir(exist_ok=True)
N_BLOCKS = int(os.environ.get("PTQ_BLOCKS", "30"))
RANK = 32
GROUP = 64
DEV = "cuda"
# candidate (alpha, beta) pairs: coupled SmoothQuant family + act-only anchors + identity,
# mirroring the upstream beta=-2 grid at num_grids=20
ALPHAS = [i / 20 for i in range(1, 20)]
PAIRS = [(a, 1 - a) for a in ALPHAS] + [(a, 0.0) for a in ALPHAS] + [(0.0, 0.0)]

# collect stream -> (output key local name, [checkpoint linears in the group])
TARGETS = [
    ("self_qkv", "self_attn.q", ["self_attn.q", "self_attn.k", "self_attn.v"]),
    ("self_o", "self_attn.o", ["self_attn.o"]),
    ("cross_q", "cross_attn.q", ["cross_attn.q"]),
    ("cross_o", "cross_attn.o", ["cross_attn.o"]),
    ("ffn_up", "ffn.0", ["ffn.0"]),
    ("ffn_down", "ffn.2", ["ffn.2"]),
]

def quant_sint4_sim(w, group=GROUP):
    """Symmetric groupwise sint4 fake-quant. Returns (dequantized w, scale [oc,1,ng,1])."""
    oc, ic = w.shape
    ng = ic // group
    wg = w.view(oc, ng, group)
    scale = wg.abs().amax(dim=-1, keepdim=True).div_(7.0).clamp_min_(1e-8)
    q = wg.div(scale).round_().clamp_(-8, 7)
    return (q * scale).view(oc, ic), scale.view(oc, 1, ng, 1)

def quant_token_sint4_sim(x, group=GROUP):
    """Dynamic per-token groupwise sint4 on activations (what the W4A4 kernel does)."""
    n, ic = x.shape
    ng = ic // group
    xg = x.view(n, ng, group)
    scale = xg.abs().amax(dim=-1, keepdim=True).div_(7.0).clamp_min_(1e-8)
    q = xg.div(scale).round_().clamp_(-8, 7)
    return (q * scale).view(n, ic)

def svd_rank(w, rank=RANK):
    """Top-`rank` randomized SVD, upstream LowRankBranch split: a=Vh, b=U*S."""
    U, S, V = torch.svd_lowrank(w, q=rank + 16, niter=4)
    return U[:, :rank] * S[:rank].unsqueeze(0), V[:, :rank].T.contiguous()

def svdquant_branch(w_s, num_iters, tol=0.999):
    """Alternating refinement: branch absorbs what int4 cannot carry.
    w_s ~= b@a + Q(R). Returns (b, a, Rq fake-quant, scale)."""
    b, a = svd_rank(w_s)
    prev_err = None
    for _ in range(num_iters):
        Rq, scale = quant_sint4_sim(w_s - (b @ a))
        b_new, a_new = svd_rank(w_s - Rq)
        err = (w_s - (b_new @ a_new) - Rq).pow(2).sum().item()
        if prev_err is not None and err >= prev_err * tol:
            break
        b, a, prev_err = b_new, a_new, err
    Rq, scale = quant_sint4_sim(w_s - (b @ a))
    return b, a, Rq, scale

def w4a4_sim_out(x, s, b, a, Rq):
    """Simulated W4A4 kernel output: quantized main path + fp lora path."""
    x_s = x / s.unsqueeze(0)
    x_q = quant_token_sint4_sim(x_s)
    return x_q @ Rq.T + (x_s @ a.T) @ b.T

def main():
    t0 = time.time()
    ckpt = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.blocks."):
                ckpt[k] = f.get_tensor(k)
    print(f"checkpoint: {len(ckpt)} block tensors")

    model_sd, scale_sd, smooth_sd, branch_sd = {}, {}, {}, {}
    report = {"config": {"rank": RANK, "group": GROUP, "pairs": len(PAIRS),
                         "eval_iters": 4, "final_iters": 64}, "blocks": {}}

    for blk in range(N_BLOCKS):
        bname = f"model.blocks.{blk}"
        brep = {}
        for stream, gkey, locals_ in TARGETS:
            res = torch.load(COLLECT / f"block{blk:02d}" / f"{stream}.pt",
                             map_location="cpu", weights_only=True)
            x = res["reservoir"].to(DEV, torch.float32)
            ws = [ckpt[f"{bname}.{ln}.weight"].to(DEV, torch.float32) for ln in locals_]
            w_cat = torch.cat(ws, dim=0)
            ref_out = x @ w_cat.T

            # AbsMax spans per in-channel (same reduction ChannelMetric.abs_max performs)
            x_span = x.abs().amax(dim=0)
            w_span = w_cat.abs().amax(dim=0)

            best = (None, float("inf"), None)
            for alpha, beta in PAIRS:
                if alpha == 0.0 and beta == 0.0:
                    s = torch.ones_like(x_span)
                else:
                    s = get_smooth_scale(alpha=alpha, beta=beta,
                                         alpha_base=x_span, beta_base=w_span).clamp_min(1e-5)
                b, a, Rq, _ = svdquant_branch(w_cat * s.unsqueeze(0), num_iters=4)
                err = (w4a4_sim_out(x, s, b, a, Rq) - ref_out).pow(2).mean().item()
                if err < best[1]:
                    best = ((alpha, beta), err, s)
            (alpha, beta), _, s = best

            # final calibration at the chosen s with the full refinement budget
            b, a, Rq, scale = svdquant_branch(w_cat * s.unsqueeze(0), num_iters=64)
            out = w4a4_sim_out(x, s, b, a, Rq)
            rel = ((out - ref_out).pow(2).mean()
                   / ref_out.pow(2).mean().clamp_min(1e-12)).sqrt().item()

            # persist in the upstream ptq.py layout, split back per source linear
            off = 0
            for ln, w_i in zip(locals_, ws):
                oc = w_i.shape[0]
                name = f"{bname}.{ln}"
                model_sd[f"{name}.weight"] = Rq[off:off + oc].to(torch.bfloat16).cpu()
                model_sd[f"{name}.bias"] = ckpt[f"{name}.bias"].to(torch.bfloat16)
                scale_sd[f"{name}.weight.scale.0"] = scale[off:off + oc].to(torch.bfloat16).cpu()
                off += oc
            smooth_sd[f"{bname}.{gkey}"] = s.float().cpu()
            branch_sd[f"{bname}.{gkey}"] = {"a.weight": a.to(torch.bfloat16).cpu(),
                                            "b.weight": b.to(torch.bfloat16).cpu()}
            brep[gkey] = {"alpha": alpha, "beta": beta, "rel_out_err": round(rel, 5)}
            del x, ws, w_cat, ref_out, b, a, Rq
        report["blocks"][blk] = brep
        torch.cuda.empty_cache()
        errs = [v["rel_out_err"] for v in brep.values()]
        print(f"block {blk:02d} rel_err {min(errs):.4f}..{max(errs):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # everything not quantized passes through unchanged for the converter's bf16 file
    for k, v in ckpt.items():
        if k not in model_sd:
            model_sd[k] = v.to(torch.bfloat16)

    torch.save(model_sd, OUT / "model.pt")
    torch.save(scale_sd, OUT / "scale.pt")
    torch.save(smooth_sd, OUT / "smooth.pt")
    torch.save(branch_sd, OUT / "branch.pt")
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    all_errs = [v["rel_out_err"] for br in report["blocks"].values() for v in br.values()]
    print(f"DONE in {time.time()-t0:.0f}s -> {OUT}/ | rel_err median "
          f"{sorted(all_errs)[len(all_errs)//2]:.4f} max {max(all_errs):.4f}")

if __name__ == "__main__":
    main()
