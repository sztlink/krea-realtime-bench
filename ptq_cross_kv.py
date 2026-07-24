"""Stage V contingency: quantize cross-attention k/v of the 14B to W4A4 and merge
into the converted checkpoint. Frees ~3.6GB of bf16, the difference between OOM at
the VAE decode and a comfortable fit on the 24GB card.

k and v share the context input, so one smooth vector serves both (grid-scored on
their concatenated output error). Each keeps its own rank-32 branch and scales.
Packing goes through the stock nunchaku converter, tensors keyed exactly like the
existing slots. Output: checkpoints-14b-w4a4-ckv/ with cross k/v moved from the
bf16 file into the quantized one.

Run on the 4090 (venv-ptq):
  PYTHONPATH=~/realtime-diffusion/src/deepcompressor ../venv-ptq/bin/python ptq_cross_kv.py
"""
import sys, types, json, time
from pathlib import Path

stub = types.ModuleType("deepcompressor.csrc.load"); stub._C = None
sys.modules["deepcompressor.csrc.load"] = stub

import torch
from safetensors import safe_open
import safetensors.torch
from deepcompressor.calib.smooth import get_smooth_scale
from deepcompressor.backend.nunchaku.convert import convert_to_nunchaku_w4x4y16_linear_state_dict

sys.path.insert(0, ".")
from ptq_wan import quant_sint4_sim, quant_token_sint4_sim, svdquant_branch, w4a4_sim_out, PAIRS

KREA = "checkpoints/krea-realtime-video-14b.safetensors"
QDIR = Path("checkpoints-14b-w4a4")
OUT = Path("checkpoints-14b-w4a4-ckv"); OUT.mkdir(exist_ok=True)
COLLECT = Path("collect14b")
N_BLOCKS = 40
DEV = "cuda"

t0 = time.time()
new_tensors = {}
report = {}
with safe_open(KREA, framework="pt") as f:
    for blk in range(N_BLOCKS):
        res = torch.load(COLLECT / f"block{blk:02d}" / "cross_kv.pt",
                         map_location="cpu", weights_only=True)
        x = res["reservoir"].to(DEV, torch.float32)
        ws = {ln: f.get_tensor(f"model.blocks.{blk}.cross_attn.{ln}.weight").to(DEV, torch.float32)
              for ln in ("k", "v")}
        bs = {ln: f.get_tensor(f"model.blocks.{blk}.cross_attn.{ln}.bias").to(DEV, torch.float32)
              for ln in ("k", "v")}
        w_cat = torch.cat([ws["k"], ws["v"]], dim=0)
        ref = x @ w_cat.T
        x_span, w_span = x.abs().amax(0), w_cat.abs().amax(0)

        best = (None, float("inf"), None)
        for alpha, beta in PAIRS:
            s = (torch.ones_like(x_span) if alpha == beta == 0.0 else
                 get_smooth_scale(alpha=alpha, beta=beta, alpha_base=x_span,
                                  beta_base=w_span).clamp_min(1e-5))
            b, a, Rq, _ = svdquant_branch(w_cat * s.unsqueeze(0), num_iters=4)
            err = (w4a4_sim_out(x, s, b, a, Rq) - ref).pow(2).mean().item()
            if err < best[1]:
                best = ((alpha, beta), err, s)
        (alpha, beta), _, s = best

        rel_tot = 0.0
        for ln in ("k", "v"):
            w_s = ws[ln] * s.unsqueeze(0)
            b, a, Rq, scale = svdquant_branch(w_s, num_iters=64)
            out = w4a4_sim_out(x, s, b, a, Rq)
            ref_l = x @ ws[ln].T
            rel = ((out - ref_l).pow(2).mean() / ref_l.pow(2).mean().clamp_min(1e-12)
                   ).sqrt().item()
            rel_tot += rel
            sd = convert_to_nunchaku_w4x4y16_linear_state_dict(
                weight=Rq.to(torch.bfloat16), scale=scale.to(torch.bfloat16),
                bias=bs[ln].to(torch.bfloat16), smooth=s.to(torch.bfloat16),
                lora=(a.to(torch.bfloat16), b.to(torch.bfloat16)))
            for k2, v2 in sd.items():
                new_tensors[f"model.blocks.{blk}.cross_attn.{ln}.{k2}"] = v2.cpu()
        report[blk] = {"alpha": alpha, "beta": beta, "rel_mean": round(rel_tot / 2, 5)}
        if blk % 8 == 0:
            print(f"block {blk} rel {report[blk]['rel_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)

# merge: quantized file gains cross k/v, bf16 file loses them
with safe_open(str(QDIR / "transformer_blocks.safetensors"), framework="pt") as f:
    meta = f.metadata() or {}
    merged = {k: f.get_tensor(k) for k in f.keys()}
merged.update(new_tensors)
safetensors.torch.save_file(merged, str(OUT / "transformer_blocks.safetensors"), metadata=meta)

with safe_open(str(QDIR / "unquantized_layers.safetensors"), framework="pt") as f:
    kept = {k: f.get_tensor(k) for k in f.keys()
            if ".cross_attn.k." not in k and ".cross_attn.v." not in k}
safetensors.torch.save_file(kept, str(OUT / "unquantized_layers.safetensors"))

rels = [r["rel_mean"] for r in report.values()]
(OUT / "ckv_report.json").write_text(json.dumps(report, indent=1))
import statistics
print(f"DONE {time.time()-t0:.0f}s | rel median {statistics.median(rels):.4f} max {max(rels):.4f}")
qb = sum(v.numel() * v.element_size() for v in merged.values())
kb = sum(v.numel() * v.element_size() for v in kept.values())
print(f"quantized {qb/1e9:.2f}GB, bf16 {kb/1e9:.2f}GB")
