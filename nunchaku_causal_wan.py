"""L4: Nunchaku W4A4 runtime for the causal Wan 1.3B (Self-Forcing).

Analog of the Krea2 port (nunchaku PR #947 lineage): only the block Linears that the
L3 checkpoint quantized are swapped for SVDQW4A4Linear; attention math, RoPE, norms,
KV cache and the whole server loop stay stock. The fused to_qkv slot created by
fuse_projections (with the L1 dedup) is the natural insertion point for the fused
qkv GEMM. Cross-attention k/v stay bf16, matching the calibration skip list.

nunchaku's package __init__ imports flux models that need a newer diffusers than the
pinned server env, so nunchaku.models.linear is loaded surgically without executing
the package __init__ (only the compiled ops are pulled in).

Checkpoint: the convert_wan.py two-file format. Keys arrive as
  model.blocks.N.{self_attn.to_qkv|self_attn.o|cross_attn.q|cross_attn.o|ffn.0|ffn.2}.
  {qweight|wscales|bias|smooth|smooth_orig|lora_down|lora_up}
and are renamed to the module's parameter names on load.
"""
import sys, types
from pathlib import Path

import torch


def _import_svdq_linear():
    import importlib.util
    site = Path(torch.__file__).parent.parent
    np = site / "nunchaku"
    for name, sub in [("nunchaku", ""), ("nunchaku.models", "models"), ("nunchaku.ops", "ops")]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(np / sub)]
            sys.modules[name] = m
    import nunchaku.models.linear as nl
    return nl.SVDQW4A4Linear


SVDQW4A4Linear = _import_svdq_linear()

RENAMES = [(".lora_down", ".proj_down"), (".lora_up", ".proj_up"),
           (".smooth_orig", ".smooth_factor_orig"), (".smooth", ".smooth_factor")]
# module local name -> (in_features, out_features)
QUANT_SLOTS = [
    ("self_attn.to_qkv", None), ("self_attn.o", None),
    ("cross_attn.q", None), ("cross_attn.o", None),
    ("ffn.0", None), ("ffn.2", None),
]


def _rename(key):
    for old, new in RENAMES:
        if key.endswith(old):
            return key[: -len(old)] + new
    return key


def load_w4a4_blocks(model, ckpt_dir, device="cuda", torch_dtype=torch.bfloat16):
    """Swap the quantized Linears of every block for SVDQW4A4Linear loaded from the
    converted checkpoint. `model` is the CausalWanModel (already fused, on device)."""
    from safetensors import safe_open
    sd = {}
    with safe_open(str(Path(ckpt_dir) / "transformer_blocks.safetensors"), framework="pt") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            sd[_rename(k)] = f.get_tensor(k)
    rank = 32
    if "quantization_config" in meta:
        import json
        rank = json.loads(meta["quantization_config"]).get("rank", 32)

    n_swapped = 0
    for i, blk in enumerate(model.blocks):
        prefix = f"model.blocks.{i}."
        for local, _ in QUANT_SLOTS:
            parent = blk
            *path, leaf = local.split(".")
            for p in path:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            sub = {k[len(prefix) + len(local) + 1:]: v for k, v in sd.items()
                   if k.startswith(prefix + local + ".")}
            assert sub, f"no tensors for {prefix}{local}"
            in_f = sub["smooth_factor"].shape[0]
            out_f = sub["qweight"].shape[0]
            q = SVDQW4A4Linear(in_f, out_f, rank=rank, bias="bias" in sub,
                               precision="int4", torch_dtype=torch_dtype, device=device)
            q.load_state_dict({k: v.to(device) for k, v in sub.items()})
            if leaf.isdigit():
                parent[int(leaf)] = q
            else:
                setattr(parent, leaf, q)
            n_swapped += 1
        blk.self_attn.fused_projections = True
    torch.cuda.empty_cache()
    print(f"W4A4: swapped {n_swapped} linears across {len(model.blocks)} blocks (rank {rank})")
    return model
