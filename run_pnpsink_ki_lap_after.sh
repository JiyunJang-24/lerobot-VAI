#!/usr/bin/env bash

# PnPSinkToCounter (all three robots) + knowledge insulation on the LAP objective. No VQA, no
# contrastive, stock SigLIP tower -- the middle cell of the comparison:
#
#   pnpsink_baseline          0.0947   plain, no KI
#   pnpsink_ki_lap            this run
#   pnpsink_vqa_ki_lap        KI-LAP + the VQA term
#
# Chained behind the VQA run rather than sharing the GPUs: two jobs on the same eight cards halve
# each other, so the total wall clock is the same while the first result arrives much later.
# Starts only if the VQA run reached "End of training".

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
WAIT_TAG="${WAIT_TAG:-pnpsink_vqa_ki_lap}"

running () {
  for p in $(pgrep -f lerobot_train_with_visual_robust 2>/dev/null); do
    tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "JOB_TAG=${WAIT_TAG}" && return 0
  done
  return 1
}

echo "waiting for ${WAIT_TAG} to finish ..."
while running; do sleep 180; done

WAIT_LOG="$(ls -t "${SCRIPT_DIR}"/outputs/logs/${WAIT_TAG}_*.log 2>/dev/null | head -1)"
if ! grep -aq "End of training" "${WAIT_LOG}"; then
  echo "${WAIT_TAG} stopped without reaching 'End of training' (${WAIT_LOG}); not starting."
  exit 1
fi
echo "${WAIT_TAG} finished cleanly; starting PnPSinkToCounter + KI(LAP)."

LOG="${SCRIPT_DIR}/outputs/logs/pnpsink_ki_lap_$(date +%Y%m%d_%H%M%S).log"
EXTRA_TRAIN_ARGS_STR=$'--policy.knowledge_insulation=true\n--policy.ki_objective=lap\n--policy.ki_token_loss_weight=1.0' \
PANDA_TOTAL_EPISODES=900 IIWA_EPISODES=1000 UR5E_EPISODES=1000 \
USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
SOURCE_PANDA_MG="${BARX}/PandaOmron/pretrain/PnPSinkToCounter/lerobot" \
SOURCE_IIWA="${BARX}/IIWAOmron/pretrain/PnPSinkToCounter/lerobot" \
SOURCE_UR5E="${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/barx_pnpsink_p900_i1000_u1000" \
CAMERAS=observation.images.robot0_agentview_right USE_WRIST_CAM=false \
FREEZE_VISION_ENCODER=false \
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" BATCH_SIZE="${BATCH_SIZE:-48}" \
NUM_WORKERS="${NUM_WORKERS:-10}" \
STEPS="${STEPS:-50000}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT=29575 \
JOB_TAG=pnpsink_ki_lap \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
