#!/usr/bin/env bash

# GPUs 4-7: the frozen-tower run, then the knowledge-insulation (LAP) run on the same cards.
#
# Both start from the contrastive+EEF pre-trained SigLIP tower
# (outputs/siglip_pretrain/both_cmean_eefattn_b16). The frozen run finishes first because a frozen
# tower has no backward graph, which is why the second run is chained behind it rather than
# competing for the same GPUs.
#
# The two blocks below are deliberately spelled out rather than shared through a helper: env
# assignments passed through a function's "$@" are parsed as a command name, not as assignments,
# which is how the first version of this script died with "FREEZE_VISION_ENCODER=true: command not
# found" after printing that it had started.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
TOWER="${TOWER:-${SCRIPT_DIR}/outputs/siglip_pretrain/both_cmean_eefattn_b16/vision_tower.safetensors}"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
GPUS="${GPUS:-4,5,6,7}"
BATCH="${BATCH:-96}"          # 96 x 4 GPUs = 384 effective, matching the knowledge-insulation runs
STEPS="${STEPS:-50000}"
NUM_WORKERS="${NUM_WORKERS:-10}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

if [[ ! -f "${TOWER}" ]]; then
  echo "pre-trained tower not found: ${TOWER}"
  exit 1
fi

echo "=== [1/2] pre-trained tower, FROZEN ==="
LOG1="${SCRIPT_DIR}/outputs/logs/vrenc_cmean_eef_frozen_$(date +%Y%m%d_%H%M%S).log"
EXTRA_TRAIN_ARGS_STR="--policy.vision_encoder_path=${TOWER}" \
PANDA_TOTAL_EPISODES=900 IIWA_EPISODES=1000 UR5E_EPISODES=1000 \
USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
SOURCE_PANDA_MG="${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot" \
SOURCE_IIWA="${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot" \
SOURCE_UR5E="${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000" \
CAMERAS=observation.images.robot0_agentview_right USE_WRIST_CAM=false \
FREEZE_VISION_ENCODER=true \
GPU_IDS="${GPUS}" BATCH_SIZE="${BATCH}" NUM_WORKERS="${NUM_WORKERS}" \
STEPS="${STEPS}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT=29541 \
JOB_TAG=vrenc_cmean_eef_frozen \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG1}"
frozen_status=${PIPESTATUS[0]}

if [[ ${frozen_status} -ne 0 ]]; then
  echo "=== frozen run FAILED (exit ${frozen_status}); not starting the knowledge-insulation run ==="
  exit "${frozen_status}"
fi

echo "=== [2/2] pre-trained tower, trainable, + knowledge insulation (LAP) ==="
LOG2="${SCRIPT_DIR}/outputs/logs/vrenc_cmean_eef_ki_lap_$(date +%Y%m%d_%H%M%S).log"
EXTRA_TRAIN_ARGS_STR=$'--policy.vision_encoder_path='"${TOWER}"$'\n--policy.knowledge_insulation=true\n--policy.ki_objective=lap\n--policy.ki_token_loss_weight=1.0' \
PANDA_TOTAL_EPISODES=900 IIWA_EPISODES=1000 UR5E_EPISODES=1000 \
USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
SOURCE_PANDA_MG="${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot" \
SOURCE_IIWA="${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot" \
SOURCE_UR5E="${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000" \
CAMERAS=observation.images.robot0_agentview_right USE_WRIST_CAM=false \
FREEZE_VISION_ENCODER=false \
GPU_IDS="${GPUS}" BATCH_SIZE="${BATCH}" NUM_WORKERS="${NUM_WORKERS}" \
STEPS="${STEPS}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT=29542 \
JOB_TAG=vrenc_cmean_eef_ki_lap \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG2}"
ki_status=${PIPESTATUS[0]}

echo "=== done: frozen exit ${frozen_status}, ki-lap exit ${ki_status} ==="
exit "${ki_status}"
