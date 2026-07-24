"""Convert the UMT5-XXL encoder from .pth to the .safetensors that realtime-video loads.

The krea-ai/realtime-video code opens wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.safetensors.
That file exists in no public repository. The official Wan-AI/Wan2.1-T2V-1.3B repo ships a .pth.
This script bridges the gap. Run it once from the realtime-video repo root after downloading
the .pth (see README).
"""
import torch
from safetensors.torch import save_file

SRC = "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
DST = "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.safetensors"

sd = torch.load(SRC, map_location="cpu", weights_only=True)
sd = {k: v.clone().contiguous() for k, v in sd.items()}
save_file(sd, DST)
print(f"wrote {DST} ({len(sd)} tensors)")
