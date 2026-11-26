#!/usr/bin/env bash
# usage: [TRAIN_ARGS=..] [EVAL_ARGS=..] run_lbl.sh <model> <gpu> <out_dir>   (resumes if <out_dir>/state.pt exists)
set -u
MODEL="$1"; GPU="$2"; OUT="$3"; LOCK="$OUT.lock"
mkdir -p "$(dirname "$OUT")"
if ! mkdir "$LOCK" 2>/dev/null; then echo "LOCKED: $OUT (already running? remove $LOCK after a crash)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT
RESUME=""; [ -f "$OUT/state.pt" ] && RESUME="--resume"
CUDA_VISIBLE_DEVICES=$GPU python scripts/train_lbl.py --model "$MODEL" --out-dir "$OUT" $RESUME ${TRAIN_ARGS:-} || { echo "TRAIN FAILED: $OUT"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_lbl.py --model "$MODEL" --adapter "$OUT/adapter" --out-dir "$OUT/eval_lbl" ${EVAL_ARGS:-} || echo "EVAL FAILED: $OUT"
echo "LBL DONE: $OUT"
