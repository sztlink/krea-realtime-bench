"""Fused Triton kernels for KV cache quantization and dequantization.

Why. The reference implementation in quant_kv.py is written to be correct rather than
fast, looping over bands in Python with an int16 intermediate and window sized temporaries.
Measured inside the real loop that costs 7 to 8 percent of the frame quantizing and 11 to
23 percent dequantizing, and the dequantization share grows with the window. That is 20 to
31 percent of the frame spent on format conversion.

Beyond the direct cost, the Python loop is what prevents choosing the scheme by error.
Subdividing inside the rotary bands cuts key error from 0.119 to 0.080, but every extra
band is another round of kernel launches. Fusing takes the group count out of the equation.

Layout, identical to the reference so the two paths are interchangeable.
  packed     uint8  [S, H, D/2], byte i holds channel 2i in the low nibble and 2i+1 high
  scales     bf16   [S, H, G]
  chan_band  int32  [D], the group index of each channel, precomputed once

The (2i, 2i+1) pair is the rotary pair, so every byte carries a complete pair, and group
boundaries always fall on even indices.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                    # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _dequant_kernel(packed_ptr, scales_ptr, band_ptr, out_ptr,
                        n_rows, D: tl.constexpr, G: tl.constexpr, HALF: tl.constexpr):
        """One program per (token, head) row. Reads HALF bytes, writes D values."""
        pid = tl.program_id(0)
        if pid >= n_rows:
            return
        i = tl.arange(0, HALF)
        b = tl.load(packed_ptr + pid * HALF + i).to(tl.int32)
        q_even = (b & 0xF) - 8
        q_odd = ((b >> 4) & 0xF) - 8

        band_e = tl.load(band_ptr + 2 * i)
        band_o = tl.load(band_ptr + 2 * i + 1)
        s_e = tl.zeros([HALF], dtype=tl.float32)
        s_o = tl.zeros([HALF], dtype=tl.float32)
        for g in tl.static_range(G):
            sg = tl.load(scales_ptr + pid * G + g).to(tl.float32)
            s_e = tl.where(band_e == g, sg, s_e)
            s_o = tl.where(band_o == g, sg, s_o)

        tl.store(out_ptr + pid * D + 2 * i, (q_even.to(tl.float32) * s_e).to(tl.bfloat16))
        tl.store(out_ptr + pid * D + 2 * i + 1, (q_odd.to(tl.float32) * s_o).to(tl.bfloat16))

    @triton.jit
    def _quant_kernel(x_ptr, packed_ptr, scales_ptr, band_ptr,
                      n_rows, qmax, D: tl.constexpr, G: tl.constexpr, HALF: tl.constexpr):
        """Um programa por linha. Reduz por banda, quantiza e empacota numa passada."""
        pid = tl.program_id(0)
        if pid >= n_rows:
            return
        i = tl.arange(0, HALF)
        x_e = tl.load(x_ptr + pid * D + 2 * i).to(tl.float32)
        x_o = tl.load(x_ptr + pid * D + 2 * i + 1).to(tl.float32)
        band_e = tl.load(band_ptr + 2 * i)
        band_o = tl.load(band_ptr + 2 * i + 1)

        s_e = tl.zeros([HALF], dtype=tl.float32)
        s_o = tl.zeros([HALF], dtype=tl.float32)
        for g in tl.static_range(G):
            me = tl.max(tl.where(band_e == g, tl.abs(x_e), 0.0), axis=0)
            mo = tl.max(tl.where(band_o == g, tl.abs(x_o), 0.0), axis=0)
            m = tl.maximum(me, mo)
            sg = tl.where(m > 0, m / qmax, 1.0)
            tl.store(scales_ptr + pid * G + g, sg.to(tl.bfloat16))
            s_e = tl.where(band_e == g, sg, s_e)
            s_o = tl.where(band_o == g, sg, s_o)

        q_e = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.round(x_e / s_e), -qmax), qmax).to(tl.int32) + 8
        q_o = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.round(x_o / s_o), -qmax), qmax).to(tl.int32) + 8
        packed = (q_e & 0xF) | ((q_o & 0xF) << 4)
        tl.store(packed_ptr + pid * HALF + i, packed.to(tl.uint8))


def _rows(t):
    return t.shape[0] * t.shape[1]


def dequantize(packed, scales, chan_band, D, dtype=torch.bfloat16):
    """packed [L,H,D/2] uint8, scales [L,H,G] bf16 -> [L,H,D] bf16."""
    L, H, half = packed.shape
    G = scales.shape[-1]
    out = torch.empty((L, H, D), dtype=dtype, device=packed.device)
    n = L * H
    _dequant_kernel[(n,)](packed, scales, chan_band, out, n,
                          D=D, G=G, HALF=half, num_warps=2)
    return out


def quantize(x, chan_band, G, qmax, dtype=torch.bfloat16):
    """x [L,H,D] bf16 -> (packed [L,H,D/2] uint8, scales [L,H,G] bf16)."""
    L, H, D = x.shape
    half = D // 2
    packed = torch.empty((L, H, half), dtype=torch.uint8, device=x.device)
    scales = torch.empty((L, H, G), dtype=dtype, device=x.device)
    n = L * H
    _quant_kernel[(n,)](x.contiguous(), packed, scales, chan_band, n, float(qmax),
                        D=D, G=G, HALF=half, num_warps=2)
    return packed, scales


def build_chan_band(bands, D, device):
    """An int32 [D] vector with each channel's group index, precomputed once."""
    cb = torch.zeros(D, dtype=torch.int32, device=device)
    for gi, (lo, hi) in enumerate(bands):
        cb[lo:hi] = gi
    return cb
