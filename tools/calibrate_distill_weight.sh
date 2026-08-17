#!/usr/bin/env bash

# Short probes to pick VISION_DISTILL_WEIGHT, the same way tools/calibrate_l2sp_weight.sh sizes the
# weight-space penalty.
#
# At weight 1.0 the term contributed 0.005 of a 0.445 total loss (~1%), so the useful range is well
# above 1. Read `vision_distill_relative_error` = ||f_student - f_teacher|| / ||f_teacher||: the
# control run says how far the features drift on their own, and each weight says how much of that
# drift the penalty removes.
#
# weight=1e-6 is the control: numerically inert, but it still builds the teacher and logs the
# relative error, giving the unregularised feature drift at the same step for free.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)/.."
SCRIPT_DIR="$(cd -- "${SCRIPT_DIR}" >/dev/null 2>&1 && pwd -P)"
STEPS="${STEPS:-210}"
WEIGHTS="${WEIGHTS:-1e-6 10 100}"
GPU_IDS="${GPU_IDS:-}"
OUT="${SCRIPT_DIR}/outputs/logs/distill_calibration_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${OUT}"; }

log "distill calibration: ${STEPS} steps per weight, weights: ${WEIGHTS}"

for w in ${WEIGHTS}; do
  run_log="${SCRIPT_DIR}/outputs/logs/distill_probe_w${w}_$(date +%H%M%S).log"
  log "### weight=${w} -> ${run_log}"
  VISION_DISTILL_WEIGHT="${w}" \
  STEPS="${STEPS}" \
  SAVE_FREQ=100000 \
  WANDB_MODE=offline \
  NUM_WORKERS="${NUM_WORKERS:-6}" \
  GPU_IDS="${GPU_IDS}" \
  MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29560}" \
  JOB_TAG="distill_probe_w${w}" \
  LOG="${run_log}" \
    "${SCRIPT_DIR}/run_barx_frontonly_distill.sh" >/dev/null 2>&1
  status=$?

  err="$(tr '\r' '\n' < "${run_log}" | grep -oE "'vision_distill_relative_error': [0-9.e-]+" | tail -1 | awk '{print $2}')"
  dl="$(tr '\r' '\n' < "${run_log}" | grep -oE "'vision_distill_loss': [0-9.e-]+" | tail -1 | awk '{print $2}')"
  loss="$(tr '\r' '\n' < "${run_log}" | grep -oE 'loss:[0-9.]+' | tail -1)"
  rate="$(tr '\r' '\n' < "${run_log}" | grep -oE "[0-9]+/${STEPS} \[[^]]*\]" | tail -1)"
  log "### weight=${w} exit=${status}  ${loss}  rel_err=${err:-?}  distill=${dl:-?}"
  log "###   rate: ${rate:-?}"
done

log "distill calibration done -> ${OUT}"
