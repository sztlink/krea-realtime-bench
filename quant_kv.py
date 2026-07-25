"""A four bit KV cache for the causal Wan loop, behind the same interface.

Why. Measured on the 4090, the W4A4 weights take 8.23 GB and the KV cache beside them
takes 7.67 GB, with a 22.77 GB peak and 0.92 GB free. The model was quantized to four
bits and the cache is now almost the size of the model. Quantizing the weights moved the
bottleneck instead of removing it.

What. The cache is a list of 40 dicts {k, v, global_end_index, local_end_index} with k and
v of shape [1, S, 40, 128] in bf16. The runtime touches those tensors in exactly three
operations, read in wan/modules/causal_model.py rather than assumed.

  1. slice write     kv_cache["k"][:, a:b] = roped_key
  2. rolling evict   kv_cache["k"][:, a:b] = kv_cache["k"][:, c:d].clone()
  3. slice read      attention(q, kv_cache["k"][:, lo:hi], kv_cache["v"][:, lo:hi])

Plus `.shape`, `.zero_()` and `.dtype`. So nothing upstream needs forking. An object that
behaves like a tensor across those operations is enough, and that is what QuantKVTensor is.

Reads dequantize the requested window. The transient is per layer, about 192 MB of live
bf16 window at a time against 7.67 GB resident, so the capacity win survives. This buys
fitting rather than speed. Read bandwidth stays the same or worse, and the fps comes later,
from the window that now fits.

## Scale granularity, and the rotary bands

Rotary embedding pairs adjacent channels (2i, 2i+1), verified in causal_rope_apply, so a
contiguous group never splits a pair. But the frequencies are split into three bands,
freqs.split([c - 2*(c//3), c//3, c//3]) with c = 64, which is [22, 21, 21] complex
components, that is real channels [0:44) temporal, [44:86) height, [86:128) width. A blind
group of 64 straddles the temporal and spatial bands and puts one scale across channels
that rotate in different regimes. Error that tracks frame position is the worst kind to
have in video, so the default is band aligned. `bandsN` subdivides each band into N groups
without ever crossing a boundary. test_quant_kv.py measures all of them on real tensors.

## K and V asymmetry

A key is an address and a value is content. Measured, keys quantize 11 percent harder than
values, so `k_bits` can exceed `v_bits`.

## The anchor

The first `sink_frames * 1560` tokens can stay in bf16. Default 0, which matches upstream,
because with the cache rebuild running the anchor lives in the clean context rather than in
sink tokens. Set it to 1 for resident regimes, where the rolling eviction actually fires.
Measured there, dropping the anchor is as damaging as quadrupling the window.
"""
import os
import torch

try:
    import quant_kv_kernels as _kern
    HAVE_KERNELS = _kern.HAVE_TRITON
except Exception:
    HAVE_KERNELS = False
USE_KERNELS = HAVE_KERNELS and os.environ.get("QKV_KERNELS", "1") != "0"

FRAME_SEQ_LEN = 1560
# bordas reais das bandas do RoPE para head_dim 128 (ver docstring)
ROPE_BANDS = ((0, 44), (44, 86), (86, 128))


def _band_edges(head_dim, mode):
    """bands = the three rotary bands. bandsN = each band subdivided into N groups without
    ever crossing a boundary. blindN = blind groups of N channels, which do cross."""
    if mode.startswith("bands"):
        assert head_dim == 128, f"bandas do RoPE mapeadas para head_dim 128, veio {head_dim}"
        n = int(mode[5:]) if len(mode) > 5 else 1
        if n == 1:
            return ROPE_BANDS
        out = []
        for lo, hi in ROPE_BANDS:
            width = hi - lo
            # subdivide into N pieces. The rotary pair is (2i, 2i+1), so the inner
            # boundaries land on even indices and never split a pair
            step = max(2, ((width + n - 1) // n + 1) // 2 * 2)
            for s in range(lo, hi, step):
                out.append((s, min(s + step, hi)))
        return tuple(out)
    if mode.startswith("blind"):
        g = int(mode[5:])
        return tuple((i, min(i + g, head_dim)) for i in range(0, head_dim, g))
    raise ValueError(f"modo de grupo desconhecido: {mode}")


class QuantKVTensor:
    """Behaves like a [1, S, H, D] bf16 tensor across the operations the runtime uses.

    Armazena int4 empacotado (dois valores por byte) mais uma escala bf16 por
    per (token, head, band). The first `sink_tokens` stay in bf16 when asked for.
    """

    def __init__(self, shape, device, dtype=torch.bfloat16, bits=4,
                 group_mode="bands", sink_tokens=0):
        b, s, h, d = shape
        assert b == 1, "o loop causal do server roda com batch 1"
        assert bits in (4, 8), "int4 ou int8"
        self.logical_shape = torch.Size(shape)
        self.dtype = dtype
        self.device = device
        self.bits = bits
        self.n_tokens, self.n_heads, self.head_dim = s, h, d
        self.bands = _band_edges(d, group_mode)
        self.group_mode = group_mode
        self.qmax = (1 << (bits - 1)) - 1          # 7 para int4, 127 para int8
        self.sink_tokens = min(sink_tokens, s)

        packed_d = d // 2 if bits == 4 else d
        self.packed = torch.zeros((s, h, packed_d), dtype=torch.uint8, device=device)
        self.scales = torch.zeros((s, h, len(self.bands)), dtype=dtype, device=device)
        self.sink = (torch.zeros((self.sink_tokens, h, d), dtype=dtype, device=device)
                     if self.sink_tokens else None)
        # fast path: one fused kernel per (token, head) row, which takes the group
        # count out of the equation and lets the scheme be chosen by error
        self.use_kernels = USE_KERNELS and bits == 4 and str(device) != "cpu"
        self.chan_band = (_kern.build_chan_band(self.bands, d, device)
                          if self.use_kernels else None)

    # ------------------------------------------------------------------ tensor-like
    @property
    def shape(self):
        return self.logical_shape

    def size(self, dim=None):
        return self.logical_shape if dim is None else self.logical_shape[dim]

    def numel(self):
        return int(self.logical_shape.numel())

    def zero_(self):
        self.packed.zero_(); self.scales.zero_()
        if self.sink is not None:
            self.sink.zero_()
        return self

    def storage_bytes(self):
        n = self.packed.numel() + self.scales.numel() * self.scales.element_size()
        if self.sink is not None:
            n += self.sink.numel() * self.sink.element_size()
        return n

    # ------------------------------------------------------------------ quant / dequant
    def _quantize(self, x):
        """x: [L, H, D] bf16 -> (packed [L, H, D/2] uint8, scales [L, H, B])"""
        if self.use_kernels:
            return _kern.quantize(x, self.chan_band, len(self.bands), self.qmax, self.dtype)
        L, H, D = x.shape
        q = torch.empty((L, H, D), dtype=torch.int16, device=x.device)
        scales = torch.empty((L, H, len(self.bands)), dtype=self.dtype, device=x.device)
        for bi, (lo, hi) in enumerate(self.bands):
            seg = x[..., lo:hi].float()
            s = seg.abs().amax(dim=-1, keepdim=True) / self.qmax
            s = torch.where(s > 0, s, torch.ones_like(s))       # banda toda zero
            q[..., lo:hi] = torch.round(seg / s).clamp(-self.qmax, self.qmax).to(torch.int16)
            scales[..., bi] = s.squeeze(-1).to(self.dtype)
        if self.bits == 8:
            return (q + 128).clamp(0, 255).to(torch.uint8), scales
        # int4: shift to [0, 15] and pack the adjacent pair, which is the rotary pair
        u = (q + 8).clamp(0, 15).to(torch.uint8)
        return (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous(), scales

    def _dequantize(self, packed, scales):
        """(packed [L, H, *], scales [L, H, B]) -> [L, H, D] bf16"""
        if self.use_kernels:
            return _kern.dequantize(packed, scales, self.chan_band, self.head_dim, self.dtype)
        if self.bits == 8:
            q = packed.to(torch.int16) - 128
        else:
            L, H, P = packed.shape
            q = torch.empty((L, H, P * 2), dtype=torch.int16, device=packed.device)
            q[..., 0::2] = (packed & 0x0F).to(torch.int16) - 8
            q[..., 1::2] = ((packed >> 4) & 0x0F).to(torch.int16) - 8
        out = torch.empty(q.shape, dtype=self.dtype, device=q.device)
        for bi, (lo, hi) in enumerate(self.bands):
            out[..., lo:hi] = (q[..., lo:hi].to(torch.float32)
                               * scales[..., bi:bi + 1].float()).to(self.dtype)
        return out

    # ------------------------------------------------------------------ slicing
    @staticmethod
    def _token_slice(idx):
        """Accepts [:, a:b] and [:, a:b, ...], which is all the runtime uses."""
        if not isinstance(idx, tuple):
            raise TypeError("QuantKVTensor expects [:, a:b] indexing")
        assert idx[0] == slice(None), "the batch dimension is always taken whole"
        sl = idx[1]
        assert isinstance(sl, slice) and sl.step in (None, 1), "contiguous slices only"
        return sl

    def __getitem__(self, idx):
        sl = self._token_slice(idx)
        a, b, _ = sl.indices(self.n_tokens)
        if b <= a:
            return torch.empty((1, 0, self.n_heads, self.head_dim),
                               dtype=self.dtype, device=self.device)
        out = self._dequantize(self.packed[a:b], self.scales[a:b])
        if self.sink is not None and a < self.sink_tokens:
            end = min(b, self.sink_tokens)
            out[: end - a] = self.sink[a:end]
        return out.unsqueeze(0)

    def __setitem__(self, idx, value):
        sl = self._token_slice(idx)
        a, b, _ = sl.indices(self.n_tokens)
        if b <= a:
            return
        v = value
        if v.dim() == 4:
            assert v.shape[0] == 1
            v = v[0]
        assert v.shape[0] == b - a, f"escrita de {v.shape[0]} tokens em slice de {b - a}"
        v = v.to(self.dtype)
        self.packed[a:b], self.scales[a:b] = self._quantize(v)
        if self.sink is not None and a < self.sink_tokens:
            end = min(b, self.sink_tokens)
            self.sink[a:end] = v[: end - a]

    # the eviction calls kv_cache["k"][:, c:d].clone(), and __getitem__ already returns
    # a real tensor, so provenance is detected in __setitem__ instead
    def clone(self):
        raise RuntimeError("clone() no cache inteiro nao e usado pelo runtime")

    def __repr__(self):
        gb = self.storage_bytes() / 1e9
        return (f"QuantKVTensor({tuple(self.logical_shape)}, int{self.bits}, "
                f"{self.group_mode}, sink={self.sink_tokens}, {gb:.3f} GB)")


def install(pipeline, k_bits=4, v_bits=4, group_mode="bands", sink_frames=0, verbose=True):
    """Swaps the kv_cache1 allocation for QuantKVTensor, preserving the interface.

    Patches the instance's `_initialize_kv_cache`. The cross attention cache is NOT
    touched. It carries the prompt, weighs 0.42 GB, and is the semantic anchor.
    """
    sink_tokens = sink_frames * FRAME_SEQ_LEN
    orig = pipeline._initialize_kv_cache

    def _initialize_kv_cache(batch_size, dtype, device):
        if pipeline.local_attn_size != -1:
            size = pipeline.local_attn_size * pipeline.frame_seq_length
        else:
            size = 32760
        cfg = pipeline.generator.model.config
        shape = [batch_size, size, cfg.num_heads, cfg.dim // cfg.num_heads]
        existing = getattr(pipeline, "kv_cache1", None)
        if existing and isinstance(existing[0]["k"], QuantKVTensor) \
                and list(existing[0]["k"].shape) == shape:
            for e in existing:
                e["k"].zero_(); e["v"].zero_()
                e["global_end_index"] = 0; e["local_end_index"] = 0
            return
        cache = []
        for _ in range(pipeline.num_transformer_blocks):
            cache.append({
                "k": QuantKVTensor(shape, device, dtype, k_bits, group_mode, sink_tokens),
                "v": QuantKVTensor(shape, device, dtype, v_bits, group_mode, sink_tokens),
                "global_end_index": 0,
                "local_end_index": 0,
            })
        pipeline.k_shape = shape; pipeline.v_shape = shape
        pipeline.kv_cache1 = cache
        if verbose:
            tot = sum(e["k"].storage_bytes() + e["v"].storage_bytes() for e in cache)
            bf16 = 2 * len(cache) * shape[0] * shape[1] * shape[2] * shape[3] * 2
            print(f"QuantKVCache: {cache[0]['k']}  |  {len(cache)} camadas  "
                  f"{tot/1e9:.3f} GB vs {bf16/1e9:.3f} GB bf16  "
                  f"(libera {(bf16-tot)/1e9:.3f} GB)", flush=True)

    pipeline._initialize_kv_cache = _initialize_kv_cache
    return orig
