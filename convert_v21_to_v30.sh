#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ID="v-1.000-1.000_num36"
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/RMA_ex02_scaling/RMA_vla_scaling_50_01_0.0_0.0"

if [[ ! -d "${DATASET_ROOT}/${REPO_ID}" ]]; then
    echo "Local dataset not found: ${DATASET_ROOT}/${REPO_ID}"
    exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id "${REPO_ID}" \
    --root "${DATASET_ROOT}" \
    --push-to-hub false
