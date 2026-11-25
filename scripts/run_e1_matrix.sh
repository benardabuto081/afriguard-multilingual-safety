#!/usr/bin/env bash
# E1 matrix for one model on one GPU: {0,3}-shot x {generate,choice}, prompt+response, full dev.
# usage: run_e1_matrix.sh <model_id> <gpu_index> [extra run_baselines.py args]
set -u
MODEL="$1"; GPU="$2"; shift 2
for SHOTS in 0 3; do
  for MODE in generate choice; do
    echo ">>> $MODEL shots=$SHOTS mode=$MODE"
    CUDA_VISIBLE_DEVICES=$GPU python scripts/run_baselines.py --model "$MODEL" \
      --split dev --mode "$MODE" --shots "$SHOTS" "$@" || echo "FAILED: $MODEL shots=$SHOTS mode=$MODE"
  done
done
echo "MATRIX DONE: $MODEL"
