#!/usr/bin/env bash
# M0.b setup — roda no pod (H100 80GB, imagem runpod/pytorch torch280-ubuntu2404).
# Downloads enxutos: T5+VAE+tokenizer do 1.3B, SÓ config.json do 14B original
# (o bench instancia via from_config e carrega o checkpoint Krea por cima), checkpoint Krea 28GB.
set -euo pipefail
cd /workspace
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/hf
export HF_HUB_DISABLE_XET=1   # evita cache duplicado de chunks no volume

apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1 &
APT_PID=$!

[ -d realtime-video ] || git clone -q https://github.com/krea-ai/realtime-video
cd realtime-video
echo "MODEL_FOLDER=wan_models" > .env

# downloads em paralelo com o build do env (hf CLI no python de sistema)
pip install -q --break-system-packages "huggingface_hub[cli]" 2>/dev/null
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --include "models_t5*" "Wan2.1_VAE.pth" "google/*" "config.json" \
  --local-dir wan_models/Wan2.1-T2V-1.3B >/tmp/dl_13b.log 2>&1 &
DL1=$!
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --include "config.json" \
  --local-dir wan_models/Wan2.1-T2V-14B >/tmp/dl_14b.log 2>&1 &
DL2=$!
huggingface-cli download krea/krea-realtime-video krea-realtime-video-14b.safetensors \
  --local-dir checkpoints >/tmp/dl_krea.log 2>&1 &
DL3=$!

pip install -q --break-system-packages uv 2>/dev/null
uv sync >/tmp/uv_sync.log 2>&1
uv pip install -q libs/sageattention-2.2.1-cp311-cp311-linux_x86_64.whl
echo "=== env pronto ==="
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

wait $DL1 $DL2 $DL3 $APT_PID
echo "=== downloads prontos ==="
ls -lh checkpoints/ wan_models/Wan2.1-T2V-14B/ 2>/dev/null
du -sh wan_models checkpoints /workspace/hf 2>/dev/null
df -h /workspace | tail -1
echo "=== SETUP OK ==="
