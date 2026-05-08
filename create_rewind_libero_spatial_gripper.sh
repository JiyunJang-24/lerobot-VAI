#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

INPUT_ROOT="${1:-${SCRIPT_DIR}/dataset_git/libero_spatial_reproduce}"
OUTPUT_ROOT="${2:-${SCRIPT_DIR}/dataset_git/libero_spatial_reproduce_rewind_gripper}"

conda run --no-capture-output -n lerobot \
  python tools/create_rewind_dataset.py \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --conditional-gripper
