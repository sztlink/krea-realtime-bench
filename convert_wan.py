"""Convert the ptq_wan.py SVDQuant W4A4 checkpoint of Self-Forcing 1.3B into the
two-file Nunchaku format (transformer_blocks.safetensors + unquantized_layers.safetensors).

Mirrors deepcompressor's convert_krea2.py driver. Quantized per block: the fused
self-attention qkv group (one smooth and one branch cached under self_attn.q, the
shared-input convention), self_attn.o, cross_attn.q, cross_attn.o, ffn.0, ffn.2.
Cross-attention k/v, norms, and modulation pass through in bf16. Non-block modules
(patch/text/time embeddings, head) are NOT here; the L4 loader keeps reading them
from the original self_forcing_dmd.sft.

Usage (venv-ptq):
  PYTHONPATH=~/realtime-diffusion/src/deepcompressor ../venv-ptq/bin/python \
      convert_wan.py --quant-path ptq_out --output-root ptq_out --rank 32
"""
import argparse, json, os

import safetensors.torch
import torch

from deepcompressor.backend.nunchaku.convert import (
    convert_to_nunchaku_transformer_block_state_dict, update_state_dict)

WAN_LOCAL_NAME_MAP = {
    "self_attn.to_qkv": ["self_attn.q", "self_attn.k", "self_attn.v"],
    "self_attn.o": "self_attn.o",
    "cross_attn.q": "cross_attn.q",
    "cross_attn.o": "cross_attn.o",
    "ffn.0": "ffn.0",
    "ffn.2": "ffn.2",
}
WAN_SMOOTH_NAME_MAP = {
    "self_attn.to_qkv": "self_attn.q",
    "self_attn.o": "self_attn.o",
    "cross_attn.q": "cross_attn.q",
    "cross_attn.o": "cross_attn.o",
    "ffn.0": "ffn.0",
    "ffn.2": "ffn.2",
}
WAN_BRANCH_NAME_MAP = dict(WAN_SMOOTH_NAME_MAP)
WAN_CONVERT_MAP = {k: "linear" for k in WAN_LOCAL_NAME_MAP}

_QUANTIZED_LOCALS = ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                     "cross_attn.q", "cross_attn.o", "ffn.0", "ffn.2"]


def convert_wan_state_dicts(state_dict, scale_dict, smooth_dict, branch_dict,
                            float_point=False):
    block_names = sorted(
        {".".join(k.split(".")[:3]) for k in state_dict if k.startswith("model.blocks.")},
        key=lambda x: int(x.split(".")[-1]))
    print(f"Converting {len(block_names)} CausalWan blocks...")

    consumed = set()
    for bn in block_names:
        for local in _QUANTIZED_LOCALS:
            consumed.add(f"{bn}.{local}.weight")
            consumed.add(f"{bn}.{local}.bias")

    converted = {}
    for bn in block_names:
        update_state_dict(
            converted,
            convert_to_nunchaku_transformer_block_state_dict(
                state_dict=state_dict, scale_dict=scale_dict,
                smooth_dict=smooth_dict, branch_dict=branch_dict,
                block_name=bn,
                local_name_map=WAN_LOCAL_NAME_MAP,
                smooth_name_map=WAN_SMOOTH_NAME_MAP,
                branch_name_map=WAN_BRANCH_NAME_MAP,
                convert_map=WAN_CONVERT_MAP,
                float_point=float_point,
            ),
            prefix=bn,
        )
    other = {k: v for k, v in state_dict.items() if k not in consumed}
    return converted, other


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quant-path", required=True)
    p.add_argument("--output-root", default="")
    p.add_argument("--model-name", default="self-forcing-1p3b-w4a4")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--float-point", action="store_true")
    args = p.parse_args()
    out_root = args.output_root or args.quant_path

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(os.path.join(args.quant_path, "model.pt"), map_location=dev,
                            weights_only=False)
    scale_dict = torch.load(os.path.join(args.quant_path, "scale.pt"), map_location="cpu",
                            weights_only=False)
    smooth_dict = torch.load(os.path.join(args.quant_path, "smooth.pt"), map_location=dev,
                             weights_only=False)
    branch_dict = torch.load(os.path.join(args.quant_path, "branch.pt"), map_location=dev,
                             weights_only=False)
    # packer expects fp16/bf16 weights and fp32 smooth vectors cast at pack time
    smooth_dict = {k: v.to(torch.bfloat16) if torch.is_tensor(v) else v
                   for k, v in smooth_dict.items()}

    converted, other = convert_wan_state_dicts(state_dict, scale_dict, smooth_dict,
                                               branch_dict, float_point=args.float_point)
    outdir = os.path.join(out_root, args.model_name)
    os.makedirs(outdir, exist_ok=True)
    meta = {"quantization_config": json.dumps(
        {"rank": args.rank, "precision": "nvfp4" if args.float_point else "int4"})}
    safetensors.torch.save_file(converted, os.path.join(outdir, "transformer_blocks.safetensors"),
                                metadata=meta)
    safetensors.torch.save_file(other, os.path.join(outdir, "unquantized_layers.safetensors"))
    qb = sum(v.numel() * v.element_size() for v in converted.values())
    ob = sum(v.numel() * v.element_size() for v in other.values())
    print(f"saved to {outdir}: quantized {len(converted)} tensors {qb/1e9:.2f}GB, "
          f"unquantized {len(other)} tensors {ob/1e9:.2f}GB")
