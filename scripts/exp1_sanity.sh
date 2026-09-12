#!/usr/bin/env bash
# Tiny end-to-end check: 3 scenes, 40 steps, 2 states/scene at eval. ~4 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-/tmp/exp1_sanity}
for M in B C D E; do
  CUDA_VISIBLE_DEVICES=${GPU:-0} python -m lerobot.scripts.exp1_train \
    --method "$M" --out "$OUT/$M" --steps 40 --log-every 10 --max-train-scenes 3
  CUDA_VISIBLE_DEVICES=${GPU:-0} python -m lerobot.scripts.exp1_eval \
    --checkpoint "$OUT/$M" --states-per-scene 2 --max-embodiments 4
done
