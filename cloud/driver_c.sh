#!/usr/bin/env bash
# Estágio C — driver autônomo no pod (A100 80GB): a cadeia provada no 1.3B, escalada.
# Etapas idempotentes com marcador em disco; sobrevive a queda de SSH; status em
# /workspace/status.txt. Sem T5 (embedding estático sobe por scp, serve pro 14B).
# Sem Sage (A100 é sm80, wheel do repo é sm90; SDPA = consistente com a linhagem).
cd /workspace/realtime-video
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
export DISABLE_SAGEATTENTION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STATUS=/workspace/status.txt
log() { echo "$1 $(date -u +%H:%M:%S)" >> "$STATUS"; }
log "driver_c start"

# ---- downloads (só o checkpoint Krea; VAE+configs vieram do setup) ----
need_ck() { ! ls checkpoints/krea-realtime-video-14b.safetensors >/dev/null 2>&1; }
for round in $(seq 1 12); do
  need_ck || break
  pkill -f "hf download" 2>/dev/null; sleep 3
  find checkpoints -name "*.lock" -delete 2>/dev/null
  nohup hf download krea/krea-realtime-video krea-realtime-video-14b.safetensors \
    --local-dir checkpoints >> /workspace/dl_krea.log 2>&1 &
  LAST=0; STALL=0
  while need_ck && [ "$STALL" -lt 4 ]; do
    sleep 30
    CUR=$(du -sb checkpoints 2>/dev/null | awk '{print $1}')
    if [ "$CUR" = "$LAST" ]; then STALL=$((STALL+1)); else STALL=0; fi
    LAST=$CUR
    log "dl round=$round gb=$((CUR/1024/1024/1024)) stall=$STALL"
  done
done
need_ck && { log "DOWNLOAD-FALHOU"; exit 1; }
log "DOWNLOADS-OK"

# ---- collect (14B, loop real, 3 seeds) ----
if [ ! -f collect_out/summary.json ]; then
  log "COLLECT-START"
  COLLECT_CONFIG=configs/self_forcing_server_14b.yaml \
  COLLECT_FIXTURE_BLOCKS=0,20,39 \
  uv run python collect_wan.py > /workspace/collect.log 2>&1
  RC=$?
  log "COLLECT-EXIT=$RC"
  [ $RC -ne 0 ] && exit 1
else
  log "COLLECT-SKIP (summary existe)"
fi
[ -f collect_out/outlier_report.json ] || ANALYZE_BLOCKS=40 uv run python analyze_collect.py >> /workspace/collect.log 2>&1

# ---- ptq (SVDQuant W4A4, DeepCompressor como biblioteca) ----
if [ ! -f ptq_out/report.json ]; then
  log "PTQ-START"
  [ -d /workspace/deepcompressor ] || git clone -q https://github.com/nunchaku-tech/deepcompressor /workspace/deepcompressor \
    || git clone -q https://github.com/mit-han-lab/deepcompressor /workspace/deepcompressor
  uv pip install -q omniconfig scipy 2>>/workspace/ptq.log
  PTQ_BLOCKS=40 PTQ_CKPT=checkpoints/krea-realtime-video-14b.safetensors \
  PYTHONPATH=/workspace/deepcompressor \
  uv run python ptq_wan.py > /workspace/ptq.log 2>&1
  RC=$?
  log "PTQ-EXIT=$RC"
  [ $RC -ne 0 ] && exit 1
else
  log "PTQ-SKIP (report existe)"
fi

# ---- convert (2-arquivos Nunchaku) ----
QDIR=ptq_out/self-forcing-14b-w4a4
if [ ! -f "$QDIR/transformer_blocks.safetensors" ]; then
  log "CONVERT-START"
  PYTHONPATH=/workspace/deepcompressor \
  uv run python convert_wan.py --quant-path ptq_out --model-name self-forcing-14b-w4a4 \
    > /workspace/convert.log 2>&1
  RC=$?
  log "CONVERT-EXIT=$RC"
  [ $RC -ne 0 ] && exit 1
else
  log "CONVERT-SKIP (checkpoint existe)"
fi

# ---- eval smoke (opcional, não-fatal: kernel nunchaku em sm80 é incerto) ----
if [ ! -f results_c/results.json ]; then
  log "EVAL-START"
  uv pip install -q --no-deps \
    "https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1%2Bcu12.8torch2.8-cp311-cp311-linux_x86_64.whl" \
    2>>/workspace/eval.log
  EVAL_QDIR=$QDIR uv run python eval_c.py > /workspace/eval.log 2>&1
  log "EVAL-EXIT=$? (não-fatal)"
else
  log "EVAL-SKIP"
fi

# ---- recibos compactos (fixtures e reservoirs ficam no volume p/ pull opcional) ----
tar czf /workspace/receipts_c.tar.gz \
  collect_out/summary.json collect_out/timestep_histogram.json collect_out/outlier_report.json \
  collect_out/calls.json ptq_out/report.json results_c 2>>/workspace/tar.log || true
log "DRIVER-DONE"
