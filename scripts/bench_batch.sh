#!/usr/bin/env bash
# Measure training throughput against --batch and --checkpointing on one GPU, so the
# next run's settings are measured rather than inherited from kev's defaults.
#
#   scripts/bench_batch.sh [GPU] [RECORDS]
#
# Reports steady-state seconds per record and peak allocated memory. Read the LAST
# log line of each config: the cumulative average still carries Triton autotune
# warm-up for the first ~20 records.
#
# What to watch for: rows within one record share a state length, so --batch 1 pads
# almost nothing. Larger batches pad every row up to the longest note in the batch
# (note lengths run p50 656 / p95 1352 tokens), so the extra throughput is not free.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
RECORDS="${2:-180}"
OUT="$(mktemp -d)"
PY=.venv/bin/python

printf '%-8s %-6s %-14s %-10s\n' batch ckpt s/record peak_GB
for ckpt in 1 0; do
  for batch in 1 2 4; do
    dir="$OUT/b${batch}-c${ckpt}"
    log="$OUT/b${batch}-c${ckpt}.log"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m medjev.train \
      --out "$dir" --max_records "$RECORDS" --questions_per_record 0 \
      --max_state 2048 --max_branch 2560 --epochs 1 \
      --batch "$batch" --accum 1 --dtype bf16 --checkpointing "$ckpt" \
      --val_every 0 --log_every 1000000 --wandb 0 >"$log" 2>&1
    if [ $? -ne 0 ]; then
      printf '%-8s %-6s %-14s %-10s\n' "$batch" "$ckpt" "OOM/failed" "-"
      grep -oE "CUDA out of memory|Error[^\"]*" "$log" | head -1
      continue
    fi
    "$PY" - "$dir" <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/training_metrics.json"))
c = json.load(open(f"{sys.argv[1]}/training_config.json"))["args"]
print("%-8s %-6s %-14.3f %-10.1f" % (c["batch"], c["checkpointing"],
      m["wall_seconds"] / m["records_seen"], m["peak_device_bytes"] / 2**30))
PY
  done
done
echo "(scratch: $OUT)"
