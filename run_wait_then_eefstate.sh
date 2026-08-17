#!/usr/bin/env bash

# Waits for the L2-SP and feature-distillation runs to finish, then trains the EEF-state variant.
#
# Order of business once the GPUs are free:
#   1. a 210-step smoke run, because this is the first time the EEF-state head runs end to end on
#      GPU under DDP -- a standalone head called outside the policy's forward is exactly the shape
#      that produced "marked as ready twice" before, and a 14-hour run is the wrong place to find out
#   2. the full 50k run, only if the smoke passed
#
# Waiting is on the PROCESSES, not on a log marker: a crashed run leaves no terminal line and a
# marker wait would block forever. The patterns are built at runtime from pieces so this script's own
# command line cannot match them -- a self-matching pgrep has already cost this project one wait loop
# that blocked on itself.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
WEIGHT="${VR_STATE_WEIGHT:-0.1}"
SUMMARY="${SCRIPT_DIR}/outputs/logs/wait_then_eefstate.log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SUMMARY}"; }

L2SP_PAT="job_name=smolvla_robocasa_x_barx_frontonly_l2sp""vision"
DISTILL_PAT="job_name=smolvla_robocasa_x_barx_frontonly_distill""_w"

log "waiting for L2-SP and distillation to finish ..."
while pgrep -f "${L2SP_PAT}" >/dev/null 2>&1 || pgrep -f "${DISTILL_PAT}" >/dev/null 2>&1; do
  sleep 120
done
log "both finished; GPUs free"

# Both pooling modes, because the open design question is whether a fingertip POSITION can be
# regressed at all from pooled tokens. 600 steps rather than 210: at 210 the head is still mostly
# untrained, and "did the error come down" is the whole point of the probe.
for pool in ${POOLS:-mean attn}; do
  log "=== probe: pool=${pool}, ${PROBE_STEPS:-600} steps, weight ${WEIGHT} ==="
  probe_log="${SCRIPT_DIR}/outputs/logs/eefstate_probe_${pool}_$(date +%Y%m%d_%H%M%S).log"
  VR_STATE_WEIGHT="${WEIGHT}" VR_STATE_POOL="${pool}" \
  STEPS="${PROBE_STEPS:-600}" SAVE_FREQ=100000 WANDB_MODE=offline JOB_TAG="eefstate_probe_${pool}" \
  LOG="${probe_log}" \
    "${SCRIPT_DIR}/run_barx_frontonly_eefstate.sh" >/dev/null 2>&1
  status=$?

  pick() { tr '\r' '\n' < "${probe_log}" | grep -oE "'$1': [0-9.e-]+" | tail -1 | awk '{print $2}'; }
  first() { tr '\r' '\n' < "${probe_log}" | grep -oE "'$1': [0-9.e-]+" | head -1 | awk '{print $2}'; }
  log "### pool=${pool} exit=${status}"
  log "###   state_loss  $(first visual_robust_state_loss) -> $(pick visual_robust_state_loss)"
  log "###   pos_err_m   $(first visual_robust_state_pos_err_m) -> $(pick visual_robust_state_pos_err_m)"
  log "###   rot_err_deg $(first visual_robust_state_rot_err_deg) -> $(pick visual_robust_state_rot_err_deg)"
  log "###   view_spread $(first visual_robust_state_view_spread_m) -> $(pick visual_robust_state_view_spread_m)"
  log "###   policy loss $(tr '\r' '\n' < "${probe_log}" | grep -oE 'loss:[0-9.]+' | tail -1)"
done

rm -rf "${SCRIPT_DIR}"/outputs/train/*/*eefstate_probe* 2>/dev/null || true
log "PROBES DONE -- stopping here for review before the 50k run."
log "Start it with:  VR_STATE_WEIGHT=<w> VR_STATE_POOL=<mean|attn> ./run_barx_frontonly_eefstate.sh"
exit 0
