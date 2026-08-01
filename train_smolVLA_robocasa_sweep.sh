#!/usr/bin/env bash

# Runs train_smolVLA_robocasa.sh back-to-back over a series of dataset sizes, largest first:
#   3000 -> 2000 -> 1000 -> 500 episodes
# Each run is a full STEPS-step training (the step count does NOT scale with dataset size), so this
# is roughly 4 x ~9.5h ~= 38h of wall clock on 8x H100.
#
# Two things this script owns that the single-run script deliberately doesn't:
#
#   1. Per-size datasets that persist. Every subset is built once under
#      dataset_git/robocasa_sweep/ep<N>/ and kept, so you can inspect/reuse all four afterwards
#      instead of them overwriting each other at a single shared path. All subsets are built up
#      front (phase 1) rather than just-in-time, so a prep failure surfaces in minutes instead of
#      a day into the sweep.
#
#   2. Checkpoint pruning. A 403M-trainable-param checkpoint is several GB and save_freq=5000 means
#      10 of them per run; four runs would blow past the free space on this box. Only the last two
#      checkpoints (45000 and 50000) are kept per run. Pruning happens *during* the run, not just at
#      the end, so peak disk stays near two checkpoints instead of ten -- while always leaving the
#      newest one on disk so a crashed run is still resumable.
#
# Usage:
#   ./train_smolVLA_robocasa_sweep.sh
#   EPISODE_COUNTS="1000 500" ./train_smolVLA_robocasa_sweep.sh   # override the series

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
cd "${SCRIPT_DIR}" || exit 1

read -r -a EPISODE_COUNTS <<<"${EPISODE_COUNTS:-3000 2000 1000 500}"

STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
export STEPS SAVE_FREQ

# Sources are the same for every size -- only the episode cap changes.
ROBOCASA_TASK_ROOT="${SCRIPT_DIR}/dataset_git/pretrain/atomic/TurnOnSinkFaucet/20250819"
SOURCE_HUMAN="${SOURCE_HUMAN:-${ROBOCASA_TASK_ROOT}/lerobot}"
SOURCE_MG="${SOURCE_MG:-${ROBOCASA_TASK_ROOT}/mg/demo/2025-08-21-12-24-03/lerobot}"
CAMERAS="${CAMERAS:-observation.images.robot0_agentview_right observation.images.robot0_eye_in_hand}"

DATASET_SWEEP_ROOT="${DATASET_SWEEP_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_sweep}"
SWEEP_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/train/sweep_${SWEEP_STAMP}}"

# Keep only the final two checkpoints: with STEPS=50000 / SAVE_FREQ=5000 that's 045000 and 050000.
# lerobot pads step dirs to max(6, len(str(STEPS))) digits (see get_step_identifier).
CKPT_DIGITS=$(( ${#STEPS} > 6 ? ${#STEPS} : 6 ))
KEEP_CHECKPOINTS=(
  "$(printf "%0${CKPT_DIGITS}d" "$((STEPS - SAVE_FREQ))")"
  "$(printf "%0${CKPT_DIGITS}d" "${STEPS}")"
)

is_keeper() {
  local name="$1" k
  for k in "${KEEP_CHECKPOINTS[@]}"; do
    [[ "${name}" == "${k}" ]] && return 0
  done
  return 1
}

# Deletes non-keeper checkpoints. With keep_newest=true the most recent checkpoint is spared even if
# it isn't a keeper, so an interrupted run still has something to resume from; the final sweep pass
# calls it with keep_newest=false, but only once the run has actually produced the last checkpoint.
prune_checkpoints() {
  local ckpt_dir="$1" keep_newest="$2"
  [[ -d "${ckpt_dir}" ]] || return 0

  local dirs=() name newest=""
  # -type d skips the `last` symlink, which we always leave alone.
  while IFS= read -r name; do
    dirs+=("${name}")
  done < <(find "${ckpt_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort)

  (( ${#dirs[@]} == 0 )) && return 0
  [[ "${keep_newest}" == "true" ]] && newest="${dirs[-1]}"

  for name in "${dirs[@]}"; do
    [[ "${name}" == "${newest}" ]] && continue
    is_keeper "${name}" && continue
    echo "  [prune] removing checkpoint ${name}"
    rm -rf "${ckpt_dir:?}/${name}"
  done
}

PRUNER_PID=""
stop_pruner() {
  if [[ -n "${PRUNER_PID}" ]] && kill -0 "${PRUNER_PID}" 2>/dev/null; then
    kill "${PRUNER_PID}" 2>/dev/null
    wait "${PRUNER_PID}" 2>/dev/null
  fi
  PRUNER_PID=""
}
trap 'stop_pruner' EXIT INT TERM

echo "=================================================================================="
echo "smolVLA RoboCasa episode-count sweep"
echo "  episode counts : ${EPISODE_COUNTS[*]}"
echo "  steps per run  : ${STEPS} (save_freq ${SAVE_FREQ})"
echo "  keep ckpts     : ${KEEP_CHECKPOINTS[*]}"
echo "  datasets       : ${DATASET_SWEEP_ROOT}/ep<N>   (kept, one per size)"
echo "  outputs        : ${SWEEP_OUTPUT_ROOT}/ep<N>"
echo "  free disk      : $(df -h "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')"
echo "=================================================================================="

# --- Phase 1: build every subset up front ---------------------------------------------------------
echo
echo "### Phase 1/2: preparing datasets"
for n in "${EPISODE_COUNTS[@]}"; do
  ds_root="${DATASET_SWEEP_ROOT}/ep${n}"
  echo
  echo "--- dataset for ${n} episodes -> ${ds_root}"
  # shellcheck disable=SC2086
  PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${PYTHONPATH:-}" \
    python "${SCRIPT_DIR}/tools/prepare_robocasa_dataset.py" \
      --source-human "${SOURCE_HUMAN}" \
      --source-mg "${SOURCE_MG}" \
      --output-root "${ds_root}" \
      --total-episodes "${n}" \
      --cameras ${CAMERAS}
  if [[ $? -ne 0 ]]; then
    echo "Dataset prep failed for ${n} episodes; aborting the whole sweep." >&2
    exit 1
  fi
done

echo
echo "### Phase 1 complete -- all datasets ready:"
du -sh "${DATASET_SWEEP_ROOT}"/ep* 2>/dev/null

# --- Phase 2: train each size in order -------------------------------------------------------------
echo
echo "### Phase 2/2: training"
declare -a RESULTS=()

for n in "${EPISODE_COUNTS[@]}"; do
  run_out="${SWEEP_OUTPUT_ROOT}/ep${n}"
  ckpt_dir="${run_out}/checkpoints"

  echo
  echo "=================================================================================="
  echo "### Training ${n} episodes   ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "###   output: ${run_out}"
  echo "=================================================================================="

  # Prune intermediate checkpoints while the run is in flight so disk stays flat.
  ( while true; do
      sleep 180
      prune_checkpoints "${ckpt_dir}" true
    done ) &
  PRUNER_PID=$!

  start_ts=$(date +%s)
  TOTAL_EPISODES="${n}" \
  DATASET_ROOT="${DATASET_SWEEP_ROOT}/ep${n}" \
  OUTPUT_DIR="${run_out}" \
  FORCE=false \
    "${SCRIPT_DIR}/train_smolVLA_robocasa.sh"
  status=$?
  elapsed=$(( $(date +%s) - start_ts ))

  stop_pruner

  # Only collapse down to the keepers once the final checkpoint actually exists -- otherwise a run
  # that died early would have its one usable checkpoint deleted.
  final_ckpt="${ckpt_dir}/${KEEP_CHECKPOINTS[-1]}"
  if [[ -d "${final_ckpt}" ]]; then
    echo "### Pruning checkpoints for ep${n} down to: ${KEEP_CHECKPOINTS[*]}"
    prune_checkpoints "${ckpt_dir}" false
  else
    echo "### ${final_ckpt} missing -- leaving checkpoints untouched for ep${n}" >&2
  fi

  if [[ ${status} -eq 0 ]]; then
    RESULTS+=("ep${n}: OK    ($((elapsed / 3600))h$(( (elapsed % 3600) / 60 ))m)  ${run_out}")
  else
    RESULTS+=("ep${n}: FAILED exit ${status} ($((elapsed / 3600))h$(( (elapsed % 3600) / 60 ))m)  ${run_out}")
    echo "### ep${n} exited ${status} -- continuing with the remaining sizes." >&2
  fi

  echo "### Remaining checkpoints for ep${n}:"
  ls -1 "${ckpt_dir}" 2>/dev/null | sed 's/^/    /'
done

echo
echo "=================================================================================="
echo "Sweep finished  ($(date '+%Y-%m-%d %H:%M:%S'))"
printf '  %s\n' "${RESULTS[@]}"
echo "  datasets kept under: ${DATASET_SWEEP_ROOT}"
echo "  free disk: $(df -h "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')"
echo "=================================================================================="
