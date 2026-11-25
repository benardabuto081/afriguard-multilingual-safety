#!/usr/bin/env bash
# usage: [DTYPE=..] [TASK_MODE=joint|A|B] [TRAIN_ARGS=".."] [EVAL_ARGS=".."] run_ft.sh <model> <gpu> <out_dir>
# Resumes automatically if <out_dir>/state.pt exists.
set -u
MODEL="$1"; GPU="$2"; OUT="$3"; DTYPE="${DTYPE:-float16}"; TM="${TASK_MODE:-joint}"
RESUME=""; [ -f "$OUT/state.pt" ] && RESUME="--resume"
CUDA_VISIBLE_DEVICES=$GPU python scripts/train.py --model "$MODEL" --task-mode "$TM" --dtype "$DTYPE" \
  --out-dir "$OUT" $RESUME ${TRAIN_ARGS:-} || { echo "TRAIN FAILED: $OUT"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_ft.py --model "$MODEL" --adapter "$OUT/adapter" --task-mode "$TM" \
  --split dev --batch-size 8 --dtype "$DTYPE" --out-dir "$OUT/eval_dev" ${EVAL_ARGS:-} || echo "EVAL FAILED: $OUT"
echo "FT DONE: $OUT"
