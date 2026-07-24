#!/usr/bin/env bash
# Estágio C — setup no pod (A100 80GB, runpod/pytorch torch280-ubuntu2404).
# Mais enxuto que o M0.b: SEM T5 (embedding estático sobe por scp e serve pro 14B),
# SEM wheel do Sage (sm80; SDPA é o caminho da linhagem). Downloads: VAE + config.json
# do 14B + checkpoint Krea 28GB (completado pelo driver com retry/stall-detect).
set -euo pipefail
cd /workspace
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1

[ -d realtime-video ] || git clone -q https://github.com/krea-ai/realtime-video
cd realtime-video
echo "MODEL_FOLDER=wan_models" > .env

pip install -q --break-system-packages "huggingface_hub[cli]" hf_transfer 2>/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Wan-AI/Wan2.1-T2V-1.3B --include "Wan2.1_VAE.pth" \
  --local-dir wan_models/Wan2.1-T2V-1.3B >/tmp/dl_vae.log 2>&1 &
DL1=$!
hf download Wan-AI/Wan2.1-T2V-14B --include "config.json" \
  --local-dir wan_models/Wan2.1-T2V-14B >/tmp/dl_cfg.log 2>&1 &
DL2=$!
nohup hf download krea/krea-realtime-video krea-realtime-video-14b.safetensors \
  --local-dir checkpoints >/workspace/dl_krea.log 2>&1 &

# fix do uv.lock: clip quebra com setuptools>=81 em build isolation (M0B §4)
python3 - <<'PY'
from pathlib import Path
p = Path("pyproject.toml"); s = p.read_text()
if "setuptools<81" not in s:
    # a secao extra-build-dependencies JA EXISTE no upstream; a linha entra DENTRO dela
    s = s.replace('nvidia-pyindex = ["pip"]', 'nvidia-pyindex = ["pip"]\nclip = ["setuptools<81"]')
    p.write_text(s); print("clip fix aplicado")
PY
pip install -q --break-system-packages uv 2>/dev/null
uv sync >/tmp/uv_sync.log 2>&1 || { tail -5 /tmp/uv_sync.log; exit 1; }
uv run python -c "import torch; print('torch', torch.__version__, 'cuda ok', torch.cuda.is_available())"

wait $DL1 $DL2
mkdir -p embeddings results_c
echo "=== SETUP OK (checkpoint segue baixando em background) ==="
df -h /workspace | tail -1
