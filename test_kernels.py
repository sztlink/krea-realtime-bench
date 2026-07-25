"""Fused kernels against the reference. Correctness first, then time.

The reference, a Python loop over bands, is the oracle, because it produced the fixtures
and the on card results. The kernel is only worth having if it gives the same answer.

Three checks.
  1. dequantizing the SAME packed and scales through both paths must be bit identical
  2. quantizing through both may differ only on rounding ties (the reference uses torch
     round-half-to-even, libdevice rounds away from zero), so the tolerance is one nibble
     and the round trip error has to match
  3. time, per scheme, including the finest grouping that the Python loop made expensive

Run:
  python test_kernels.py
"""
import json, time
from pathlib import Path
import torch

import quant_kv
from quant_kv import QuantKVTensor, _band_edges

assert quant_kv.HAVE_KERNELS, "triton not available"
OUT = Path("results_qkv"); OUT.mkdir(exist_ok=True)
RES = {"correcao": {}, "tempo": {}}
dev = "cuda"
L, H, D = 9360, 40, 128

torch.manual_seed(0)
x = (torch.randn(L, H, D, device=dev, dtype=torch.bfloat16) * 3.0)
# alguns canais outliers, como o K real tem
x[:, :, 7] *= 40
x[:, :, 60] *= 15

def make(mode, kernels):
    t = QuantKVTensor([1, L, H, D], dev, bits=4, group_mode=mode)
    t.use_kernels = kernels and quant_kv.USE_KERNELS
    if t.use_kernels and t.chan_band is None:
        import quant_kv_kernels as k
        t.chan_band = k.build_chan_band(t.bands, D, dev)
    return t

for mode in ("bands", "bands2", "bands4"):
    ref, ker = make(mode, False), make(mode, True)
    p_ref, s_ref = ref._quantize(x)
    p_ker, s_ker = ker._quantize(x)

    # 1. dequantising the SAME content must match exactly
    d_ref = ref._dequantize(p_ref, s_ref)
    d_ker = ker._dequantize(p_ref, s_ref)
    same = torch.equal(d_ref, d_ker)

    # 2. quantizacao pode diferir em empates
    q_ref = (p_ref & 0xF).to(torch.int16)
    q_ker = (p_ker & 0xF).to(torch.int16)
    dq_max = int((q_ref - q_ker).abs().max())
    scale_close = torch.allclose(s_ref.float(), s_ker.float(), rtol=1e-2, atol=1e-3)
    err_ref = ((ref._dequantize(p_ref, s_ref).float() - x.float()).norm() / x.float().norm()).item()
    err_ker = ((ker._dequantize(p_ker, s_ker).float() - x.float()).norm() / x.float().norm()).item()

    RES["correcao"][mode] = {
        "dequant_bit_identico": bool(same),
        "quant_delta_max_nibble": dq_max,
        "escalas_batem": bool(scale_close),
        "erro_referencia": round(err_ref, 5),
        "erro_kernel": round(err_ker, 5),
        "delta_erro": round(abs(err_ref - err_ker), 6),
    }
    print(mode, json.dumps(RES["correcao"][mode]), flush=True)
    assert same, f"{mode}: dequantisation diverged"
    assert dq_max <= 1, f"{mode}: quantisation diverged beyond a rounding tie ({dq_max})"
    assert abs(err_ref - err_ker) < 1e-3, f"{mode}: round trip error diverged"

def bench(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000

for mode in ("bands", "bands2", "bands4"):
    row = {}
    for label, kernels in (("referencia", False), ("kernel", True)):
        t = make(mode, kernels)
        p, s = t._quantize(x)
        row[f"quant_ms_{label}"] = round(bench(lambda: t._quantize(x)), 3)
        row[f"dequant_ms_{label}"] = round(bench(lambda: t._dequantize(p, s)), 3)
    row["speedup_quant"] = round(row["quant_ms_referencia"] / row["quant_ms_kernel"], 2)
    row["speedup_dequant"] = round(row["dequant_ms_referencia"] / row["dequant_ms_kernel"], 2)
    RES["tempo"][mode] = row
    print(mode, json.dumps(row), flush=True)

(OUT / "kernels.json").write_text(json.dumps(RES, indent=1))
print("DONE", flush=True)
