#!/usr/bin/env bash

# Waits for pnpsink_fusion_vrloss_maintower_ki_lap (no dropout) to finish, then runs the same
# config with --policy.fusion_aux_dropout=0.5.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
WAIT_TAG="${WAIT_TAG:-pnpsink_fusion_vrloss_maintower_ki_lap}"

running () {
  for p in $(pgrep -f lerobot_train_with_visual_robust 2>/dev/null); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "JOB_TAG=${WAIT_TAG}" && return 0
  done
  return 1
}

echo "waiting for ${WAIT_TAG} to finish ..."
while running; do sleep 120; done

LOG="$(ls -t "${SCRIPT_DIR}"/outputs/logs/${WAIT_TAG}_*.log 2>/dev/null | head -1)"
if [[ -z "${LOG}" ]] || ! grep -aq "End of training" "${LOG}"; then
  echo "${WAIT_TAG} did not reach 'End of training' (${LOG:-log not found}); not starting the dropout run."
  exit 1
fi
echo "${WAIT_TAG} finished cleanly (${LOG}); starting the aux-dropout=0.5 run."

GPU_IDS=0,1,2,3,4,5,6,7 BATCH_SIZE="${BATCH_SIZE:-48}" VR_FRAMES="${VR_FRAMES:-16}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29599}" AUX_DROPOUT="${AUX_DROPOUT:-0.5}" \
JOB_TAG="${JOB_TAG:-pnpsink_fusion_vrloss_maintower_dropout_ki_lap}" \
  "${SCRIPT_DIR}/run_pnpsink_fusion_vrloss_onetower.sh"
exit $?
