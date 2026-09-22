#!/usr/bin/env bash
# MedJev fine-tune: 3 GPUs, 2048-token states, all ~9 questions per record.
#
#   scripts/train_3gpu.sh [OUT_DIR]
#
# --accum 3 with 3 ranks keeps the effective batch at 9 records, matching the
# single-GPU recipe (batch 1 x accum 8), so --lr 1e-4 carries over unchanged.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-runs/medjev-0.8b}"
exec .venv/bin/torchrun --nproc_per_node=3 --master_port="${MASTER_PORT:-29580}" \
  -m medjev.train \
  --out "$OUT" \
  --base Qwen3.5-0.8B-Base \
  --data data/medjev-v1/train.jsonl \
  --val_data data/medjev-v1/development.jsonl \
  --epochs 2 \
  --questions_per_record 0 \
  --max_state 2048 --max_branch 2560 \
  --lr 1e-4 \
  --batch 4 --accum 3 \
  --dtype bf16 --checkpointing 1 \
  --ord_w 0.3 \
  --save_every 100 --log_every 25 \
  --val_every 200 --val_records 400 \
  --wandb 1 --wandb_project medjev --wandb_name medjev-0.8b-s2048
