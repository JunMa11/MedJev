#!/usr/bin/env bash
# Wait for the MedJev baseline inference jobs to release the GPUs, then start the
# fine-tune on whichever GPU has the most free memory.
#
#   scripts/launch_training.sh [OUT_DIR]
#
# Polls every 60s for any running `medjev.base_probe` process (the zero-shot
# baselines) and for a GPU with at least $NEED_MB free. Training peaks at ~6.1 GB
# at --max_state 1024, so 12 GB of headroom is conservative.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-runs/medjev-0.8b}"
NEED_MB="${NEED_MB:-12000}"
PY=.venv/bin/python

baselines_running() {
  pgrep -u "$(id -u)" -f "medjev\.base_probe" >/dev/null
}

free_gpu() {
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
    awk -F', *' -v need="$NEED_MB" '{free=$2-$3; if (free>best) {best=free; idx=$1}} END {if (best>=need) print idx}'
}

echo "$(date -Is) waiting for baseline inference to finish..."
while baselines_running; do
  sleep 60
done
echo "$(date -Is) baselines done."

while :; do
  GPU="$(free_gpu)"
  [ -n "$GPU" ] && break
  echo "$(date -Is) no GPU with ${NEED_MB}MB free yet; retrying"
  sleep 60
done

# A baseline may have been launched while we were picking a GPU.
if baselines_running; then exec "$0" "$OUT"; fi

mkdir -p "$(dirname "$OUT")"
LOG="${OUT%/}.log"
echo "$(date -Is) starting training on GPU $GPU -> $OUT (log: $LOG)"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" -m medjev.train \
  --out "$OUT" \
  --base Qwen3.5-0.8B-Base \
  --data data/medjev-v1/train.jsonl \
  --epochs 2 \
  --questions_per_record 4 \
  --max_state 1024 \
  --lr 1e-4 \
  --batch 1 --accum 8 \
  --dtype bf16 --checkpointing 1 \
  --ord_w 0.3 \
  --save_every 500 \
  --log_every 25 \
  >"$LOG" 2>&1 &
echo "$(date -Is) training pid $! on GPU $GPU"
wait $!
echo "$(date -Is) training exited with $?"
