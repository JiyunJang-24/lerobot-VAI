#!/usr/bin/env bash

# Short probes to pick VISION_L2SP_WEIGHT before spending 7 hours on the real run.
#
# The penalty is a sum over 86.4M parameters, so its loss value says nothing useful about how hard it
# is actually pulling -- 1e-3 reads as "2.5 next to an action loss of 0.069" yet its per-parameter
# gradient is 2*1e-3*(w-w0), which is tiny. The only honest way to size it is to run a few hundred
# steps and read the two numbers that matter:
#
#   vision_l2sp_relative_drift   how far the tower has moved  (unregularised 50k end-state: 0.1029)
#   loss                         whether the action task is being strangled
#
# weight=1e-12 is the control: numerically inert, but it still captures the reference and logs the
# drift, giving the unregularised drift at the same step for free.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)/.."
SCRIPT_DIR="$(cd -- "${SCRIPT_DIR}" >/dev/null 2>&1 && pwd -P)"
STEPS="${STEPS:-400}"
WEIGHTS="${WEIGHTS:-1e-12 1e-4 1e-3 1e-2}"
OUT="${SCRIPT_DIR}/outputs/logs/l2sp_calibration_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${OUT}"; }

log "L2-SP calibration: ${STEPS} steps per weight, weights: ${WEIGHTS}"

for w in ${WEIGHTS}; do
  run_log="${SCRIPT_DIR}/outputs/logs/l2sp_probe_w${w}_$(date +%H%M%S).log"
  log "### weight=${w} -> ${run_log}"
  VISION_L2SP_WEIGHT="${w}" \
  STEPS="${STEPS}" \
  SAVE_FREQ="${STEPS}" \
  WANDB_MODE=offline \
  JOB_TAG="l2sp_probe_w${w}" \
  LOG="${run_log}" \
    "${SCRIPT_DIR}/run_barx_frontonly_l2sp.sh" >/dev/null 2>&1
  status=$?

  drift="$(tr '\r' '\n' < "${run_log}" | grep -oE "'vision_l2sp_relative_drift': [0-9.e-]+" | tail -1 | awk '{print $2}')"
  l2sp="$(tr '\r' '\n' < "${run_log}" | grep -oE "'vision_l2sp_loss': [0-9.e-]+" | tail -1 | awk '{print $2}')"
  loss="$(tr '\r' '\n' < "${run_log}" | grep -oE 'loss:[0-9.]+' | tail -1)"
  log "### weight=${w} exit=${status}  ${loss}  drift=${drift:-?}  l2sp=${l2sp:-?}"
done

log "calibration done -> ${OUT}"
