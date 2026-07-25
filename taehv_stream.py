"""Streaming TAEHV decoder that honours the server's decoder contract.

The block attribution at one denoise step puts the VAE decode at 42.6 percent of the
block, the largest slice left after the transformer. The tiny decoder in
demo_utils/taehv.py decodes the same three latent frames 25.6 times faster, but the
server cannot use it as shipped for two reasons.

First, `load_vae()` hardcodes the heavy decoder and never reads the `use_taehv` flag that
sits in both configs, which makes it the fourth dead flag found in this codebase.

Second, and this is the real work, `apply_model_with_memblocks` allocates its memory list
fresh on every call. The server decodes one block of three latent frames at a time, so
every call would start with empty temporal memory and leave a seam every twelve pixel
frames. This module hoists that memory out of the traversal and carries it across calls.

The server calls the decoder as `pixels, cache = vae_decoder(latents, *cache)` with a
55 slot cache. The wrapper accepts that signature, keeps its own memory internally, and
passes the server's cache back untouched. A fresh session arrives with every slot None,
which is the reset signal.

Output range. The heavy decoder returns about [-1, 1] and TAEHV returns about [0, 1], so
the wrapper converts, otherwise every downstream clamp and byte conversion is wrong.
"""
import torch
import torch.nn as nn

from demo_utils.taehv import TAEHV, MemBlock, TPool, TGrow, TWorkItem


def decode_stream(model, x, mem=None):
    """`apply_model_with_memblocks` in sequential mode, with memory carried in and out.

    Same graph traversal as upstream. The only change is that `mem` arrives as an
    argument and leaves as a return value, so temporal state survives between calls.
    """
    assert x.ndim == 5, f"TAEHV opera em NTCHW, veio {x.ndim} dims"
    N, T, C, H, W = x.shape
    if mem is None:
        mem = [None] * len(model)
    out = []
    work_queue = [TWorkItem(xt, 0) for xt in x.reshape(N, T * C, H, W).chunk(T, dim=1)]
    while work_queue:
        xt, i = work_queue.pop(0)
        if i == len(model):
            out.append(xt)
            continue
        b = model[i]
        if isinstance(b, MemBlock):
            if mem[i] is None:
                xt_new = b(xt, xt * 0)
                mem[i] = xt.clone()
            else:
                xt_new = b(xt, mem[i])
                mem[i] = xt.clone()
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(b, TPool):
            if mem[i] is None:
                mem[i] = []
            mem[i].append(xt)
            if len(mem[i]) > b.stride:
                raise ValueError("fila do TPool passou do stride")
            if len(mem[i]) == b.stride:
                n, c, h, w = xt.shape
                xt = b(torch.cat(mem[i], 1).view(n * b.stride, c, h, w))
                mem[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            _nt, c, h, w = xt.shape
            for xt_next in reversed(xt.view(N, b.stride * c, h, w).chunk(b.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i + 1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    return (torch.stack(out, 1) if out else None), mem


class TAEHVDecoderWrapper(nn.Module):
    """Drop-in for the server's `vae_decoder`.

    Signature and returns match, so nothing upstream changes. The 55 slot cache the
    server threads through is passed back untouched, because the temporal state that
    matters lives in `self.mem`.
    """

    def __init__(self, checkpoint_path="taew2_1.pth", dtype=torch.float16, device="cuda"):
        super().__init__()
        self.tae = TAEHV(checkpoint_path=checkpoint_path).to(device=device, dtype=dtype).eval()
        self.tae.requires_grad_(False)
        self.mem = None
        self.blocks_seen = 0
        # o primeiro modulo do decoder e um Clamp, sem pesos; pegar o dtype de um parametro
        self._dtype = next(self.tae.decoder.parameters()).dtype
        self._marker = torch.zeros(1, device=device, dtype=dtype)

    def reset(self):
        self.mem = None
        self.blocks_seen = 0

    @torch.no_grad()
    def forward(self, latents, *cache):
        # A fresh session arrives with every slot None. But if the cache is handed back
        # untouched, the next call looks fresh again and the state resets every block,
        # which was the first version's bug. So slot 0 comes back with a marker.
        if cache and all(c is None for c in cache):
            self.reset()
        px, self.mem = decode_stream(self.tae.decoder, latents.to(self._dtype), self.mem)
        self.blocks_seen += 1
        # TAEHV entrega ~[0,1], o servidor espera ~[-1,1]
        out_cache = list(cache)
        if out_cache:
            out_cache[0] = self._marker
        return px * 2.0 - 1.0, out_cache
