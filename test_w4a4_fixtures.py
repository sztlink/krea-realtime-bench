"""L4 unit test: the real nunchaku kernel vs bf16 reference on the L2 fixtures.

For blocks 0 and 15, calls 20/21 (mid-generation denoise inputs captured from the
real loop): run each quantized slot on the fixture input and compare against the
fp32 reference output of the original checkpoint weights. Prints per-slot relative
error next to the PTQ simulation's prediction from ptq_out/report.json."""
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open

from nunchaku_causal_wan import SVDQW4A4Linear, _rename

DEV = "cuda"
CKPT = "checkpoints/self_forcing_dmd.sft"
QDIR = Path("ptq_out/self-forcing-1p3b-w4a4")
FIX = Path("collect_out/fixtures")
report = json.load(open("ptq_out/report.json"))

# fixture stream -> (quantized slot local name, [reference linears])
CASES = [
    ("self_qkv", "self_attn.to_qkv", ["self_attn.q", "self_attn.k", "self_attn.v"]),
    ("self_o", "self_attn.o", ["self_attn.o"]),
    ("cross_q", "cross_attn.q", ["cross_attn.q"]),
    ("cross_o", "cross_attn.o", ["cross_attn.o"]),
    ("ffn_up", "ffn.0", ["ffn.0"]),
    ("ffn_down", "ffn.2", ["ffn.2"]),
]
REPORT_KEY = {"self_attn.to_qkv": "self_attn.q", "self_attn.o": "self_attn.o",
              "cross_attn.q": "cross_attn.q", "cross_attn.o": "cross_attn.o",
              "ffn.0": "ffn.0", "ffn.2": "ffn.2"}

sd = {}
with safe_open(str(QDIR / "transformer_blocks.safetensors"), framework="pt") as f:
    for k in f.keys():
        sd[_rename(k)] = f.get_tensor(k)

ok = True
with safe_open(CKPT, framework="pt") as ck:
    for blk in [0, 15]:
        print(f"--- block {blk} ---")
        for stream, slot, refs in CASES:
            prefix = f"model.blocks.{blk}.{slot}."
            sub = {k[len(prefix):]: v.to(DEV) for k, v in sd.items() if k.startswith(prefix)}
            in_f = sub["smooth_factor"].shape[0]
            out_f = sub["qweight"].shape[0]
            q = SVDQW4A4Linear(in_f, out_f, rank=32, bias=True, precision="int4",
                               torch_dtype=torch.bfloat16, device=DEV)
            q.load_state_dict(sub)

            w = torch.cat([ck.get_tensor(f"model.blocks.{blk}.{r}.weight") for r in refs]
                          ).to(DEV, torch.float32)
            b = torch.cat([ck.get_tensor(f"model.blocks.{blk}.{r}.bias") for r in refs]
                          ).to(DEV, torch.float32)

            errs = []
            for call in [20, 21]:
                p = FIX / f"call{call:03d}_block{blk:02d}_{stream}.pt"
                if not p.exists():
                    continue
                x = torch.load(p, map_location=DEV, weights_only=True)
                x3 = x.to(torch.bfloat16)
                if x3.dim() == 2:
                    x3 = x3.unsqueeze(0)
                ref = x.reshape(-1, in_f).to(torch.float32) @ w.T + b
                out = q(x3).reshape(-1, out_f).to(torch.float32)
                rel = ((out - ref).pow(2).mean() / ref.pow(2).mean().clamp_min(1e-12)
                       ).sqrt().item()
                errs.append(rel)
                if not torch.isfinite(out).all():
                    ok = False
                    print(f"  {slot}: NON-FINITE OUTPUT")
            sim = report["blocks"][str(blk)][REPORT_KEY[slot]]["rel_out_err"]
            kernel = sum(errs) / len(errs)
            flag = "" if kernel < max(2.5 * sim, sim + 0.05) else "  <-- DIVERGES FROM SIM"
            if flag:
                ok = False
            print(f"  {slot:18s} kernel {kernel:.4f} | sim {sim:.4f}{flag}")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
