#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

DATASET_ROOT="${1:-${SCRIPT_DIR}/dataset_git/libero_spatial_reproduce}"
REPO_ID="${2:-v-1.000-1.000_num1}"
EPISODE_INDEX="${3:-0}"

conda run --no-capture-output -n lerobot \
  python tools/visualize_episode_actions.py \
    --dataset-root "${DATASET_ROOT}" \
    --repo-id "${REPO_ID}" \
    --episode-index "${EPISODE_INDEX}"
