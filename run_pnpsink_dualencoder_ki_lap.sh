#!/usr/bin/env bash

# PnPSinkToCounter + knowledge insulation (LAP) with TWO vision towers.
#
#   trainable   the stock SigLIP, free to adapt to the task
#   frozen      the contrastive+EEF pre-trained SigLIP, held fixed
#
# Both see the same image and produce [N, 1024, 768]; the two are concatenated on the feature axis
# and projected back to 768, so the token count and the connector are unchanged. The projection is
# initialised as an exact identity on the trainable tower (first half identity, second half zero),
# so step 0 computes bit for bit what a single-tower run computes and the frozen tower's
# contribution has to be learned rather than injected as noise.
#
# The point: earlier runs asked ONE tower to be embodiment-invariant and task-adapted at once, and
# policy training pulled it back toward reading the robot (§4: gap -0.05 -> -0.138). Freezing the
# invariant tower means task adaptation can no longer overwrite it.
#
# Cost: the frozen tower runs under no_grad, so it adds a forward pass but no activations.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
AUX_TOWER="${AUX_TOWER:-${SCRIPT_DIR}/outputs/siglip_pretrain/both_cmean_eefattn_6views_b16/vision_tower.safetensors}"
WAIT_FILE="${WAIT_FILE:-${AUX_TOWER}}"

if [[ "${WAIT:-true}" == "true" ]]; then
  echo "waiting for ${WAIT_FILE} ..."
  while [[ ! -f "${WAIT_FILE}" ]]; do sleep 120; done
  # The file appears at the end of pre-training, but give the writer a moment to finish flushing.
  sleep 30
fi
if [[ ! -f "${AUX_TOWER}" ]]; then echo "auxiliary tower not found: ${AUX_TOWER}"; exit 1; fi
echo "auxiliary (frozen) tower: ${AUX_TOWER}"

LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/pnpsink_dualenc_ki_lap_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

EXTRA_TRAIN_ARGS_STR=$'--policy.knowledge_insulation=true\n--policy.ki_objective=lap\n--policy.ki_token_loss_weight=1.0\n--policy.aux_vision_encoder_path='"${AUX_TOWER}" \
PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-900}" \
IIWA_EPISODES="${IIWA_EPISODES:-1000}" \
UR5E_EPISODES="${UR5E_EPISODES:-1000}" \
USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
SOURCE_PANDA_MG="${BARX}/PandaOmron/pretrain/PnPSinkToCounter/lerobot" \
SOURCE_IIWA="${BARX}/IIWAOmron/pretrain/PnPSinkToCounter/lerobot" \
SOURCE_UR5E="${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/barx_pnpsink_p900_i1000_u1000" \
CAMERAS=observation.images.robot0_agentview_right USE_WRIST_CAM=false \
FREEZE_VISION_ENCODER=false \
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" BATCH_SIZE="${BATCH_SIZE:-48}" \
NUM_WORKERS="${NUM_WORKERS:-10}" \
STEPS="${STEPS:-50000}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29585}" \
JOB_TAG=pnpsink_dualenc_ki_lap \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
