#!/usr/bin/env bash

# Waits for the GPU 0-3 fusion run (main=stock, aux=contrastive, aux frozen) to finish, then starts
# the "both towers trainable" fusion condition on the same cards.
#
# Batch is lower than the frozen-aux runs (96): a trainable aux tower keeps a full backward graph
# instead of running under no_grad, which costs roughly one more tower's worth of memory (measured
# single-process: 23.3 GiB frozen vs 37.1 GiB trainable at batch 48). The earlier both-trainable run
# OOMed at 96/rank and was stable at 64/rank (measured 64.2 GiB), so this uses 64.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
WAIT_TAG="${WAIT_TAG:-pnpsink_fusion_aux_contrastive_frozen_ki_lap}"

running () {
  for p in $(pgrep -f lerobot_train_with_visual_robust 2>/dev/null); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "JOB_TAG=${WAIT_TAG}" && return 0
  done
  return 1
}

echo "waiting for ${WAIT_TAG} (GPU 0-3) to finish ..."
while running; do sleep 120; done

# Known at launch time: the GPU 0-3 run's own log (main=stock, aux=contrastive, frozen).
WAIT_LOG="${WAIT_LOG:-${SCRIPT_DIR}/outputs/logs/pnpsink_dualenc_ki_lap_20260824_205615.log}"
if [[ -z "${WAIT_LOG}" ]] || ! grep -aq "End of training" "${WAIT_LOG}"; then
  echo "${WAIT_TAG} did not reach 'End of training' (log: ${WAIT_LOG:-not found}); not starting."
  exit 1
fi
echo "${WAIT_TAG} finished cleanly (${WAIT_LOG}); starting both-trainable fusion on GPU 0-3."

WAIT=false GPU_IDS=0,1,2,3 BATCH_SIZE="${BATCH_SIZE:-64}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29595}" \
FREEZE_AUX=false \
AUX_TOWER="${AUX_TOWER:-${SCRIPT_DIR}/outputs/siglip_pretrain/both_cmean_eefattn_6views_b16/vision_tower.safetensors}" \
JOB_TAG="${JOB_TAG:-pnpsink_fusion_both_trainable_ki_lap}" \
  "${SCRIPT_DIR}/run_pnpsink_dualencoder_ki_lap.sh"
exit $?
