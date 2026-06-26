#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_BASE="${DATASET_BASE:-${SCRIPT_DIR}/dataset_git/RMA_ex02_scaling}"
MAX_JOBS="${MAX_JOBS:-5}"
CONDA_ENV="${CONDA_ENV:-smolvla}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/convert_v21_to_v30_multiple}"

cd "${SCRIPT_DIR}" || exit 1
mkdir -p "${LOG_DIR}"

if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found. Activate conda first or update this script's conda setup."
    exit 1
fi

conda activate "${CONDA_ENV}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

if ! [[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer. Current value: ${MAX_JOBS}"
    exit 1
fi

convert_one() {
    local dataset_path="$1"
    local repo_id
    local root
    local root_name
    local log_file

    repo_id="$(basename "${dataset_path}")"
    root="$(dirname "${dataset_path}")"
    root_name="$(basename "${root}")"
    log_file="${LOG_DIR}/${root_name}__${repo_id}.log"

    echo "[START] ${root_name}/${repo_id}"
    python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
        --repo-id "${repo_id}" \
        --root "${root}" \
        --push-to-hub "${PUSH_TO_HUB}" \
        >"${log_file}" 2>&1
    local status=$?

    if [[ ${status} -eq 0 ]]; then
        echo "[DONE]  ${root_name}/${repo_id}"
    else
        echo "[FAIL]  ${root_name}/${repo_id} (see ${log_file})"
    fi

    return "${status}"
}

wait_for_batch() {
    local failed=0
    local pid

    for pid in "$@"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done

    return "${failed}"
}

mapfile -d '' DATASETS < <(
    find "${DATASET_BASE}" -mindepth 2 -maxdepth 2 -type d -name 'v-*' -print0 | sort -z
)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No datasets found under ${DATASET_BASE}"
    exit 1
fi

echo "Found ${#DATASETS[@]} datasets under ${DATASET_BASE}"
echo "Converting up to ${MAX_JOBS} datasets in parallel"
echo "Logs: ${LOG_DIR}"

pids=()
failed=0

for dataset_path in "${DATASETS[@]}"; do
    convert_one "${dataset_path}" &
    pids+=("$!")

    if [[ ${#pids[@]} -ge ${MAX_JOBS} ]]; then
        if ! wait_for_batch "${pids[@]}"; then
            failed=1
        fi
        pids=()
    fi
done

if [[ ${#pids[@]} -gt 0 ]]; then
    if ! wait_for_batch "${pids[@]}"; then
        failed=1
    fi
fi

if [[ ${failed} -ne 0 ]]; then
    echo "One or more conversions failed. Check logs in ${LOG_DIR}"
    exit 1
fi

echo "All conversions completed successfully."
