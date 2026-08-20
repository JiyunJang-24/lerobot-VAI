#!/usr/bin/env bash

# Wait for the KI-LAP run on GPUs 4-7 to finish, then start the frozen-tower run on the same cards.
#
# Chained rather than run side by side: sharing four GPUs halved both jobs (3.15 -> 1.04 it/s for
# frozen, 1.76 -> 1.06 for KI-LAP), so the total wall clock was the same while the frozen result
# arrived eight hours later than it needed to.
#
# It waits on the actual training processes rather than on a PID recorded at launch, so it survives
# a restart of the job it is waiting for, and it refuses to start if KI-LAP did not reach
# "End of training".

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
TOWER="${TOWER:-${SCRIPT_DIR}/outputs/siglip_pretrain/both_cmean_eefattn_b16/vision_tower.safetensors}"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
WAIT_TAG="${WAIT_TAG:-vrenc_cmean_eef_ki_lap}"

running () {
  for p in $(pgrep -f lerobot_train_with_visual_robust 2>/dev/null); do
    if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "JOB_TAG=${WAIT_TAG}"; then return 0; fi
  done
  return 1
}

echo "waiting for ${WAIT_TAG} to finish ..."
while running; do sleep 120; done

KI_LOG="$(ls -t "${SCRIPT_DIR}"/outputs/logs/vrenc_cmean_eef_ki_lap_*.log 2>/dev/null | head -1)"
if ! grep -aq "End of training" "${KI_LOG}"; then
  echo "${WAIT_TAG} stopped without reaching 'End of training' (${KI_LOG}); not starting the frozen run."
  exit 1
fi
echo "${WAIT_TAG} finished cleanly; starting the frozen run."

LOG="${SCRIPT_DIR}/outputs/logs/vrenc_cmean_eef_frozen_$(date +%Y%m%d_%H%M%S).log"
EXTRA_TRAIN_ARGS_STR="--policy.vision_encoder_path=${TOWER}" \
PANDA_TOTAL_EPISODES=900 IIWA_EPISODES=1000 UR5E_EPISODES=1000 \
USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
SOURCE_PANDA_MG="${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot" \
SOURCE_IIWA="${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot" \
SOURCE_UR5E="${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000" \
CAMERAS=observation.images.robot0_agentview_right USE_WRIST_CAM=false \
FREEZE_VISION_ENCODER=true \
GPU_IDS="${GPUS:-4,5,6,7}" BATCH_SIZE="${BATCH:-96}" NUM_WORKERS="${NUM_WORKERS:-10}" \
STEPS="${STEPS:-50000}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT=29541 \
JOB_TAG=vrenc_cmean_eef_frozen \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
