#!/usr/bin/env bash

# Runs train_smolVLA_robocasa_x_visual_robust.sh, and if it dies of CUDA OOM, retries with a
# progressively cheaper configuration instead of just failing.
#
# Ladder order is chosen to protect result comparability, not just to free memory:
#
#   1. encoder_chunk_size  32 -> 16 -> 8
#        Pure plumbing: it only controls how many auxiliary images are pushed through the vision
#        encoder per forward call. Same images, same loss, same gradients -- so this is free to
#        shrink and is tried first.
#   2. visual_robust_batch_size  8 -> 6 -> 4
#        Real cost: the supervised-contrastive term draws its negatives from within the auxiliary
#        batch, so a smaller batch means fewer negatives and a weaker signal. The action loss is
#        untouched.
#   3. main batch_size  64 -> 56 -> 48
#        Last resort. Every previous robocasa_x run used 64/GPU (512 effective), so changing it
#        makes this run's loss curve no longer directly comparable to them. Only used if the two
#        cheaper knobs are exhausted.
#
# OOM in this setup shows up within the first minutes -- the auxiliary loss runs from step 1 and the
# per-step image count is fixed -- so a config that survives startup is very likely to survive the
# whole run.
#
# A non-OOM failure stops the ladder immediately: retrying with less memory would not fix it.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
SUMMARY_LOG="${SCRIPT_DIR}/outputs/logs/visual_robust_oom_backoff.log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

# Both batch sizes come from the caller and are only ever stepped *down*. Earlier this file
# hard-coded them on every rung and passed them explicitly to the child, silently overriding whatever
# was asked for -- a run meant to match a batch-48 baseline trained at 64, and a request for
# VISUAL_ROBUST_BATCH_SIZE=32 ran at 8. Deriving the rungs from the requested values keeps the
# ladder a fallback rather than a config of its own.
BASE_BATCH="${BATCH_SIZE:-64}"
BASE_VR_BATCH="${VISUAL_ROBUST_BATCH_SIZE:-8}"

# chunk_size | vr_batch | main_batch
# Order protects comparability: encoder_chunk first (pure plumbing, identical gradients), then the
# auxiliary batch (weakens the contrastive signal), and only last the main batch (which would make
# the run non-comparable to its baseline).
LADDER=(
  "32|${BASE_VR_BATCH}|${BASE_BATCH}"
  "16|${BASE_VR_BATCH}|${BASE_BATCH}"
  "8|$(( BASE_VR_BATCH * 3 / 4 ))|${BASE_BATCH}"
  "8|$(( BASE_VR_BATCH / 2 ))|${BASE_BATCH}"
  "8|$(( BASE_VR_BATCH / 2 ))|$(( BASE_BATCH * 7 / 8 ))"
  "8|$(( BASE_VR_BATCH / 2 ))|$(( BASE_BATCH * 3 / 4 ))"
)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${SUMMARY_LOG}"; }

wait_for_free_gpus() {
  # After an OOM the dead ranks can hold memory for a few seconds; starting again too early would
  # OOM for the wrong reason and burn a rung of the ladder.
  for _ in $(seq 1 60); do
    pgrep -f "lerobot_train.*\.py" >/dev/null 2>&1 || break
    sleep 5
  done
  pkill -9 -f "lerobot_train_with_visual_robust.py" 2>/dev/null || true
  sleep 10
  for _ in $(seq 1 60); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)"
    [[ "${used:-9999}" -lt 2000 ]] && return 0
    sleep 5
  done
  log "WARNING: GPUs still show memory in use; continuing anyway"
}

log "=================================================================================="
log "visual robust training with OOM backoff (${#LADDER[@]} rungs)"
log "=================================================================================="

attempt=0
for rung in "${LADDER[@]}"; do
  attempt=$((attempt + 1))
  IFS='|' read -r chunk vrbatch mainbatch <<<"${rung}"

  log "attempt ${attempt}/${#LADDER[@]}: encoder_chunk=${chunk} vr_batch=${vrbatch} batch=${mainbatch}"
  wait_for_free_gpus

  run_log="${SCRIPT_DIR}/outputs/logs/vr_attempt${attempt}_c${chunk}_v${vrbatch}_b${mainbatch}_$(date +%Y%m%d_%H%M%S).log"

  VISUAL_ROBUST_ENCODER_CHUNK_SIZE="${chunk}" \
  VISUAL_ROBUST_BATCH_SIZE="${vrbatch}" \
  BATCH_SIZE="${mainbatch}" \
    "${SCRIPT_DIR}/train_smolVLA_robocasa_x_visual_robust.sh" >"${run_log}" 2>&1
  status=$?

  if [[ ${status} -eq 0 ]]; then
    log "attempt ${attempt} SUCCEEDED (encoder_chunk=${chunk} vr_batch=${vrbatch} batch=${mainbatch})"
    log "log: ${run_log}"
    exit 0
  fi

  if grep -qiE "CUDA out of memory|torch\.cuda\.OutOfMemoryError|OutOfMemoryError|CUDA error: out of memory" "${run_log}"; then
    log "attempt ${attempt} hit CUDA OOM; stepping down"
    if [[ ${attempt} -eq ${#LADDER[@]} ]]; then
      log "ladder exhausted -- even the cheapest configuration OOMs. Not retrying."
      log "log: ${run_log}"
      exit 1
    fi
    continue
  fi

  log "attempt ${attempt} failed (exit ${status}) for a non-OOM reason; stopping."
  log "last lines of ${run_log}:"
  tail -25 "${run_log}" | tee -a "${SUMMARY_LOG}"
  exit "${status}"
done
