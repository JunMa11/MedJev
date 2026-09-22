#!/usr/bin/env bash
# Wait for the MedJev fine-tune to exit, then score it on development and print
# the comparison against the baselines.
#
#   scripts/after_training.sh [RUN_DIR]
#
# Development only: the test split stays locked until a checkpoint has been
# chosen on development, which is the whole point of the --allow-test gate.
set -uo pipefail
cd "$(dirname "$0")/.."

RUN="${1:-runs/medjev-0.8b}"
NEED_MB="${NEED_MB:-10000}"
PY=.venv/bin/python

echo "$(date -Is) waiting for training in $RUN to finish..."
while pgrep -u "$(id -u)" -f "medjev\.train .*--out $RUN( |$)" >/dev/null; do
  sleep 120
done

if [ ! -f "$RUN/head.pt" ]; then
  echo "$(date -Is) no final checkpoint at $RUN/head.pt — training did not complete; leaving evaluation to a human"
  exit 1
fi

while :; do
  GPU="$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
    awk -F', *' -v need="$NEED_MB" '{free=$2-$3; if (free>best) {best=free; idx=$1}} END {if (best>=need) print idx}')"
  [ -n "$GPU" ] && break
  echo "$(date -Is) waiting for a GPU with ${NEED_MB}MB free"
  sleep 60
done

echo "$(date -Is) scoring $RUN on development (GPU $GPU)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m medjev.evaluate --run "$RUN" --split development 2>&1 |
  tee "${RUN%/}-eval-development.log"
"$PY" -m medjev.compare --split development | tee "${RUN%/}-compare-development.md"
echo "$(date -Is) done"
