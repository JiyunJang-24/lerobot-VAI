#!/usr/bin/env bash

# End-to-end pipeline for the PnP cross-embodiment mix pulled from
# ChiefJang/robocasa_x_atomic_ur5e_iiwa:
#
#   panda  = PnPCounterToStove  (108 human + 892 mg = 1000)
#   iiwa   = PnPCounterToSink   (1000)
#   ur5e   = PnPSinkToCounter   (1000)
#
# The download (tools/download_robocasa_x_atomic.py) is assumed to be already running -- this
# script does NOT start it. It watches the four subset directories and kicks off each one's v3.0
# conversion the moment that subset's files have all landed, rather than waiting for the whole
# 9.5 GB fetch to finish. That matters because ~72% of the files (and ~73% of the bytes) belong to
# the single panda_mg subset, so the three small subsets are downloadable-and-convertible long
# before mg is done; overlapping their conversion with mg's download takes them off the critical
# path entirely.
#
# Once all four are v3.0 it runs train_smolVLA_robocasa_x.sh, which does its own episode-cap +
# camera-drop prep (dropping robot0_agentview_left, keeping agentview_right + eye_in_hand) and then
# the 8-GPU accelerate launch.
#
# Usage:
#   ./run_robocasa_x_atomic_pipeline.sh
# Skip straight to training if everything is already converted:
#   SKIP_CONVERT=true ./run_robocasa_x_atomic_pipeline.sh

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BASE="${BASE:-${SCRIPT_DIR}/dataset_git/robocasa_x_atmoic}"
LOG_DIR="${SCRIPT_DIR}/outputs/logs"
DOWNLOAD_LOG="${LOG_DIR}/hf_download_atmoic.log"
SKIP_CONVERT="${SKIP_CONVERT:-false}"
POLL_S="${POLL_S:-30}"
mkdir -p "${LOG_DIR}"

# label | download subtree (also the DATASET_BASE handed to the converter) | expected file count |
# dataset root relative to BASE (where meta/info.json lives)
# The expected counts come from list_repo_tree on the HF repo; a subset's download is complete when
# its file count reaches this, since snapshot_download only moves a file into place once fully
# written. That count is only a *download* signal though -- conversion to v3.0 packs thousands of
# per-episode mp4/parquet files into a handful, so an already-converted subset sits far below its
# expected count and is instead recognised by its dataset root already reporting v3.0.
SUBSETS=(
  "panda_human|PandaOmron/pretrain/PnPCounterToStove|764|PandaOmron/pretrain/PnPCounterToStove"
  "iiwa|IIWAOmron/pretrain/PnPCounterToSink|7008|IIWAOmron/pretrain/PnPCounterToSink/lerobot"
  "ur5e|UR5eOmron/pretrain/PnPSinkToCounter|7008|UR5eOmron/pretrain/PnPSinkToCounter/lerobot"
  "panda_mg|mg/PandaOmron/pretrain/PnPCounterToStove|38560|mg/PandaOmron/pretrain/PnPCounterToStove"
)

# Dataset roots (where meta/info.json actually lives) -- note IIWA/UR5e nest theirs under lerobot/.
ROOT_PANDA_HUMAN="${BASE}/PandaOmron/pretrain/PnPCounterToStove"
ROOT_PANDA_MG="${BASE}/mg/PandaOmron/pretrain/PnPCounterToStove"
ROOT_IIWA="${BASE}/IIWAOmron/pretrain/PnPCounterToSink/lerobot"
ROOT_UR5E="${BASE}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

download_alive() { pgrep -f "download_robocasa_x_atomic.py" >/dev/null 2>&1; }

count_files() { find "$1" -type f 2>/dev/null | wc -l; }

# True once $1 (a dataset root) reports codebase_version v3.0, i.e. it has already been converted.
already_v30() {
  local info="$1/meta/info.json"
  [[ -f "${info}" ]] || return 1
  python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('codebase_version')=='v3.0' else 1)" \
    "${info}" 2>/dev/null
}

# Blocks until $2 has at least $3 files, or until $4 is already v3.0 (a rerun after conversion --
# conversion collapses thousands of files into a handful, so the raw count would never match again).
# Fails only if the downloader has exited AND the count is still short.
wait_for_subset() {
  local label="$1" dir="$2" expected="$3" root="$4" n
  while true; do
    if already_v30 "${root}"; then
      log "${label}: already converted to v3.0"
      return 0
    fi
    n="$(count_files "${dir}")"
    if [[ "${n}" -ge "${expected}" ]]; then
      log "${label}: download complete (${n}/${expected} files)"
      return 0
    fi
    if ! download_alive; then
      sleep 5  # let any in-flight rename settle before the final verdict
      n="$(count_files "${dir}")"
      if [[ "${n}" -ge "${expected}" ]] || already_v30 "${root}"; then
        log "${label}: download complete (${n}/${expected} files)"
        return 0
      fi
      log "${label}: FAILED -- downloader exited with only ${n}/${expected} files"
      return 1
    fi
    sleep "${POLL_S}"
  done
}

# Waits for a subset, then converts it to v3.0 in place. Run one of these per subset in parallel.
stage_subset() {
  local label="$1" subtree="$2" expected="$3" root_rel="$4"
  local dir="${BASE}/${subtree}"
  local root="${BASE}/${root_rel}"
  local log_file="${LOG_DIR}/convert_${label}.log"

  wait_for_subset "${label}" "${dir}" "${expected}" "${root}" || return 1

  if [[ "${SKIP_CONVERT}" == "true" ]]; then
    log "${label}: SKIP_CONVERT=true, not converting"
    return 0
  fi

  # convert_robocasa_to_v30.sh handles this repo's two export quirks itself (normalize_video_dirs
  # before the format conversion, flatten_singleton_columns after), so there is nothing to fix up
  # here -- see that script's header.
  log "${label}: starting v3.0 conversion -> ${log_file}"
  local started
  started=$(date +%s)
  MAX_JOBS=1 "${SCRIPT_DIR}/convert_robocasa_to_v30.sh" "${dir}" >"${log_file}" 2>&1
  local status=$?
  local elapsed=$(( $(date +%s) - started ))

  if [[ ${status} -eq 0 ]]; then
    log "${label}: conversion OK ($((elapsed / 60))m$((elapsed % 60))s)"
  else
    log "${label}: conversion FAILED exit ${status} after $((elapsed / 60))m -- see ${log_file}"
  fi
  return "${status}"
}

echo "=================================================================================="
echo "robocasa_x atomic PnP pipeline  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "  base:     ${BASE}"
echo "  download: $(download_alive && echo 'running' || echo 'not running')"
echo "  stages:   per-subset wait -> v3.0 convert (parallel) -> prep+train (8 GPU)"
echo "=================================================================================="

pids=()
labels=()
for entry in "${SUBSETS[@]}"; do
  IFS='|' read -r label subtree expected root_rel <<<"${entry}"
  stage_subset "${label}" "${subtree}" "${expected}" "${root_rel}" &
  pids+=("$!")
  labels+=("${label}")
done

failed=()
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    failed+=("${labels[$i]}")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  log "ABORTING before training -- these subsets failed: ${failed[*]}"
  exit 1
fi

log "all four subsets are v3.0; verifying before training"
for pair in "panda_human:${ROOT_PANDA_HUMAN}" "panda_mg:${ROOT_PANDA_MG}" \
            "iiwa:${ROOT_IIWA}" "ur5e:${ROOT_UR5E}"; do
  name="${pair%%:*}"
  path="${pair#*:}"
  if [[ ! -f "${path}/meta/info.json" ]]; then
    log "${name}: missing ${path}/meta/info.json -- aborting"
    exit 1
  fi
  python - "$name" "$path" <<'PY'
import json, sys
name, path = sys.argv[1], sys.argv[2]
i = json.load(open(f"{path}/meta/info.json"))
print(f"  {name:12s} version={i.get('codebase_version')} episodes={i.get('total_episodes')} "
      f"frames={i.get('total_frames')} robot={i.get('robot_type')}")
if i.get("codebase_version") != "v3.0":
    raise SystemExit(f"{name} is still {i.get('codebase_version')}, not v3.0")
PY
  if [[ $? -ne 0 ]]; then
    log "${name}: version check failed -- aborting"
    exit 1
  fi
done

TRAIN_LOG="${LOG_DIR}/robocasa_x_pnp_$(date +%Y%m%d_%H%M%S).log"
log "starting training -> ${TRAIN_LOG}"

PANDA_TOTAL_EPISODES=1000 \
IIWA_EPISODES=1000 \
UR5E_EPISODES=1000 \
SOURCE_PANDA_HUMAN="${ROOT_PANDA_HUMAN}" \
SOURCE_PANDA_MG="${ROOT_PANDA_MG}" \
SOURCE_IIWA="${ROOT_IIWA}" \
SOURCE_UR5E="${ROOT_UR5E}" \
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/robocasa_x_pnp_p1000_i1000_u1000" \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${TRAIN_LOG}"
train_status=${PIPESTATUS[0]}

echo "=================================================================================="
if [[ ${train_status} -eq 0 ]]; then
  log "pipeline finished OK -- training log: ${TRAIN_LOG}"
else
  log "training FAILED exit ${train_status} -- see ${TRAIN_LOG}"
fi
echo "  free disk: $(df -h --output=avail "${SCRIPT_DIR}" | tail -1 | tr -d ' ')"
echo "=================================================================================="
exit "${train_status}"
