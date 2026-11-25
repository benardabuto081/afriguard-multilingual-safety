#!/usr/bin/env bash
# Backbone pilot: 150 train examples per language, 1 epoch, then full-dev evaluation.
# usage: [DTYPE=float16|bfloat16] run_pilot.sh <model_id> <gpu_index> <tag>
set -u
MODEL="$1"; GPU="$2"; TAG="$3"; DTYPE="${DTYPE:-float16}"
OUT=experiments/multitask/pilot_$TAG
CUDA_VISIBLE_DEVICES=$GPU python scripts/train.py --model "$MODEL" --per-language 150 --epochs 1 \
  --log-every 10 --dtype "$DTYPE" --out-dir "$OUT" || { echo "TRAIN FAILED: $TAG"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_ft.py --model "$MODEL" --adapter "$OUT/adapter" \
  --split dev --batch-size 8 --dtype "$DTYPE" --out-dir "$OUT/eval_dev" || echo "EVAL FAILED: $TAG"
echo "PILOT DONE: $TAG"
