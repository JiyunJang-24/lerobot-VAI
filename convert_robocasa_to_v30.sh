#!/usr/bin/env bash

# Converts every LeRobot-formatted dataset found under DATASET_BASE to codebase_version "v3.0", in
# place (mirrors convert_v21_to_v30_multiple.sh's convention, just discovering datasets by the
# presence of meta/info.json anywhere under DATASET_BASE instead of a fixed "v-*" naming pattern,
# since RoboCasa's datasets live at varying depths/names).
#
# For each dataset found:
#   - "v2.0" (no per-episode stats -- some RoboCasa MimicGen exports): first run
#     tools/convert_v20_to_v21_local.py (also fixes any feature wrongly declared dtype "object" --
#     see that script's docstring), then the step below.
#   - "v2.1": run the project's existing src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py
#     (--push-to-hub false), which keeps a "<name>_old" backup of the pre-conversion directory next
#     to it.
#   - already "v3.0": format conversion is skipped.
#   - In all three cases, finally runs (both idempotent, no-op if nothing to fix):
#     - tools/ensure_frame_index.py: some RoboCasa "mg" (MimicGen) exports never had a
#       `frame_index` column, which makes this project's `lerobot.datasets.dataset_tools`
#       (split_dataset/remove_feature, used by tools/prepare_robocasa_dataset.py to build a
#       training-ready subset afterwards) raise a "Keys mismatch" error the moment they touch
#       such a dataset.
#     - tools/fix_episode_file_index.py: convert_dataset_v21_to_v30.py mis-declares which output
#       data file the first episode of each new size-based file chunk actually landed in (off by
#       one file at every such boundary), which makes dataset_tools' split_dataset crash with
#       "cannot convert float NaN to integer" on episode selections that cross one of those
#       boundaries.
#
# This is a pure format conversion: it does not filter cameras, cap episode counts, or otherwise
# reshape any dataset -- every camera and every episode present in the source stays. Everything else
# (camera selection, episode-count caps for training) is handled separately at training-prep time by
# tools/prepare_robocasa_dataset.py (invoked automatically by train_smolVLA_robocasa.sh).
#
# Usage:
#   ./convert_robocasa_to_v30.sh [DATASET_BASE]
#   MAX_JOBS=4 ./convert_robocasa_to_v30.sh /root/Desktop/workspace/jiyun/robocasa/datasets

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_BASE="${1:-${DATASET_BASE:-/root/Desktop/workspace/jiyun/robocasa/datasets}}"
MAX_JOBS="${MAX_JOBS:-3}"
V20_SHIM_NUM_WORKERS="${V20_SHIM_NUM_WORKERS:-16}"
V20_SHIM_VIDEO_SAMPLE_FRAMES="${V20_SHIM_VIDEO_SAMPLE_FRAMES:-8}"
CONDA_ENV="${CONDA_ENV:-smolvla}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/convert_robocasa_to_v30}"

cd "${SCRIPT_DIR}" || exit 1
mkdir -p "${LOG_DIR}"

if [[ ! -d "${DATASET_BASE}" ]]; then
  echo "DATASET_BASE not found: ${DATASET_BASE}" >&2
  exit 1
fi

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
  echo "conda command not found. Activate conda first or update this script's conda setup." >&2
  exit 1
fi

conda activate "${CONDA_ENV}" || exit 1
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

if ! [[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_JOBS must be a positive integer. Current value: ${MAX_JOBS}" >&2
  exit 1
fi

convert_one() {
  local dataset_path="$1"
  local repo_id root log_file version

  repo_id="$(basename "${dataset_path}")"
  root="$(dirname "${dataset_path}")"
  log_file="${LOG_DIR}/$(echo "${dataset_path}" | tr '/' '_').log"

  version="$(python -c "import json; print(json.load(open('${dataset_path}/meta/info.json')).get('codebase_version', 'unknown'))")"

  echo "[START] ${dataset_path} (${version})"
  {
    if [[ "${version}" == "v3.0" ]]; then
      echo "--- already v3.0, skipping format conversion ---"
    else
      if [[ "${version}" == "v2.0" ]]; then
        echo "--- v2.0 -> v2.1 shim ---"
        python "${SCRIPT_DIR}/tools/convert_v20_to_v21_local.py" \
          --root "${dataset_path}" \
          --num-workers "${V20_SHIM_NUM_WORKERS}" \
          --video-sample-frames "${V20_SHIM_VIDEO_SAMPLE_FRAMES}" \
          --video-backend pyav
        shim_status=$?
        if [[ ${shim_status} -ne 0 ]]; then
          exit "${shim_status}"
        fi
        echo "--- v2.1 -> v3.0 ---"
      fi
      python "${SCRIPT_DIR}/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py" \
        --repo-id "${repo_id}" \
        --root "${root}" \
        --push-to-hub false
      convert_status=$?
      if [[ ${convert_status} -ne 0 ]]; then
        exit "${convert_status}"
      fi
    fi
    echo "--- ensure frame_index ---"
    python "${SCRIPT_DIR}/tools/ensure_frame_index.py" --root "${dataset_path}"
    frame_index_status=$?
    if [[ ${frame_index_status} -ne 0 ]]; then
      exit "${frame_index_status}"
    fi
    echo "--- fix episode data-file-index metadata ---"
    python "${SCRIPT_DIR}/tools/fix_episode_file_index.py" --root "${dataset_path}"
  } >"${log_file}" 2>&1
  local status=$?

  if [[ ${status} -eq 0 ]]; then
    echo "[DONE]  ${dataset_path}"
  else
    echo "[FAIL]  ${dataset_path} (see ${log_file})"
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

mapfile -t INFO_FILES < <(find "${DATASET_BASE}" -type f -name info.json -path "*/meta/*" | sort)
DATASETS=()
for info_file in "${INFO_FILES[@]}"; do
  dataset_dir="$(dirname "$(dirname "${info_file}")")"
  # skip convert_dataset_v21_to_v30.py's own "<name>_old" pre-conversion backups (and any nested
  # "_old_old" from repeated runs) -- otherwise each rerun would re-discover and re-convert its own
  # backups, stacking more backups forever.
  if [[ "$(basename "${dataset_dir}")" == *_old ]]; then
    continue
  fi
  DATASETS+=("${dataset_dir}")
done

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No datasets (meta/info.json) found under ${DATASET_BASE}"
  exit 1
fi

echo "Found ${#DATASETS[@]} dataset(s) under ${DATASET_BASE}"
echo "Converting up to ${MAX_JOBS} in parallel"
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
