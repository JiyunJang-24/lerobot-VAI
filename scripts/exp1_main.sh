#!/usr/bin/env bash
# Experiment 1 proper: every method on the same deterministic split, one GPU each.
#   bash scripts/exp1_main.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-outputs/exp1}
STEPS=${STEPS:-4000}
SEED=${SEED:-0}

run () {                       # run <method> <gpu>
  local M=$1 G=$2
  echo "[$M] gpu $G training"
  CUDA_VISIBLE_DEVICES=$G python -m lerobot.scripts.exp1_train \
    --method "$M" --out "$OUT/$M" --steps "$STEPS" --seed "$SEED" \
    > "$OUT/$M.train.log" 2>&1 || { echo "[$M] TRAIN FAILED"; return 1; }
  echo "[$M] gpu $G evaluating"
  CUDA_VISIBLE_DEVICES=$G python -m lerobot.scripts.exp1_eval \
    --checkpoint "$OUT/$M" --states-per-scene 6 --max-embodiments 8 --seed "$SEED" \
    > "$OUT/$M.eval.log" 2>&1 || { echo "[$M] EVAL FAILED"; return 1; }
  echo "[$M] done"
}

mkdir -p "$OUT"
[ -f "$OUT/split.json" ] || python tools/exp1_split.py --out "$OUT/split.json"

run A 0 &
run B 1 &
run C 2 &
run D 3 &
run E 4 &
wait
echo "all methods finished"
python tools/exp1_report.py --root "$OUT"
