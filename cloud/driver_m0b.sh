#!/usr/bin/env bash
# Driver autonomo do M0.b no pod: completa downloads (retry+stall-detect+resume),
# depois dispara o bench. Sobrevive a quedas de SSH; status em /workspace/status.txt.
cd /workspace/realtime-video
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
STATUS=/workspace/status.txt
echo "driver start $(date -u +%H:%M:%S)" > "$STATUS"

need_t5() { ! ls wan_models/Wan2.1-T2V-1.3B/models_t5*.safetensors >/dev/null 2>&1; }
need_ck() { ! ls checkpoints/krea-realtime-video-14b.safetensors >/dev/null 2>&1; }

for round in $(seq 1 12); do
  if ! need_t5 && ! need_ck; then break; fi
  pkill -f "hf download" 2>/dev/null; sleep 3
  find checkpoints wan_models -name "*.lock" -delete 2>/dev/null
  if need_t5; then nohup hf download Wan-AI/Wan2.1-T2V-1.3B --include "models_t5*" \
    --local-dir wan_models/Wan2.1-T2V-1.3B >> /workspace/dl_t5.log 2>&1 & fi
  if need_ck; then nohup hf download krea/krea-realtime-video krea-realtime-video-14b.safetensors \
    --local-dir checkpoints >> /workspace/dl_krea.log 2>&1 & fi
  LAST=0; STALL=0
  while { need_t5 || need_ck; } && [ "$STALL" -lt 4 ]; do
    sleep 30
    CUR=$(du -sb checkpoints wan_models 2>/dev/null | awk '{s+=$1} END {print s}')
    if [ "$CUR" = "$LAST" ]; then STALL=$((STALL+1)); else STALL=0; fi
    LAST=$CUR
    echo "round=$round gb=$((CUR/1024/1024/1024)) stall=$STALL t5=$(need_t5 && echo falta || echo ok) ck=$(need_ck && echo falta || echo ok) $(date -u +%H:%M:%S)" >> "$STATUS"
  done
done

if need_t5 || need_ck; then echo "DOWNLOAD-FALHOU" >> "$STATUS"; exit 1; fi
echo "DOWNLOADS-OK $(date -u +%H:%M:%S)" >> "$STATUS"
cp /workspace/bench_m0b.py .
uv run python bench_m0b.py > /workspace/bench.log 2>&1
echo "BENCH-EXIT=$? $(date -u +%H:%M:%S)" >> "$STATUS"
