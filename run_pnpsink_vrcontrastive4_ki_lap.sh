#!/usr/bin/env bash

# PnPSinkToCounter + knowledge insulation (LAP) + visual-robust contrastive over FOUR embodiments:
#
#     UR5eOmron               UR5e            + Robotiq85
#     PandaOmronPandaGripper  Panda           + Panda gripper
#     JacoOmron               Jaco            + Robotiq85
#     JacoOmronPandaGripper   Jaco            + Panda gripper
#
# IIWAOmron and PandaOmron are left out. What remains is a deliberate cross: two arms sharing a
# gripper (Panda+PandaGripper vs Jaco+PandaGripper) and one arm across two grippers (Jaco+Robotiq85
# vs Jaco+PandaGripper), so the positive group varies arm and gripper independently rather than
# only bundling them together.
#
# --dataset.visual_robust_include_views names the subset explicitly. Without it the only controls
# were "first N alphabetically" (which biases toward one embodiment) or "random N", neither of
# which can express this selection. A name matching nothing raises rather than silently shrinking
# the positive group.
#
# Contrastive keeps same-episode batching: its negatives are other timesteps of the same episode,
# which is the hard-negative design the loss was built around. (The VQA run turned that off because
# it has no negatives at all.)

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_new_barx_ur5e/new_barx}"
VR_REPO_IDS="${VR_REPO_IDS:-UR5eOmron_PnPSinkToCounter/lerobot}"
VIEWS="${VIEWS:-UR5eOmron,PandaOmronPandaGripper,JacoOmron,JacoOmronPandaGripper}"
VR_WEIGHT="${VR_WEIGHT:-0.5}"
VR_BATCH="${VR_BATCH:-24}"      # 24 frames x 4 views = 96 images, matching the earlier contrastive run
KI_WEIGHT="${KI_WEIGHT:-1.0}"
WAIT_TAG="${WAIT_TAG:-pnpsink_ki_lap}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/pnpsink_vrcontrastive4_ki_lap_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

if [[ "${WAIT:-true}" == "true" ]]; then
  running () {
    for p in $(pgrep -f lerobot_train_with_visual_robust 2>/dev/null); do
      tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "JOB_TAG=${WAIT_TAG}" && return 0
    done
    return 1
  }
  echo "waiting for ${WAIT_TAG} to finish ..."
  while running; do sleep 120; done
  echo "${WAIT_TAG} is done."
fi

echo "PnPSink + KI(LAP) + contrastive over 4 embodiments -> ${LOG}"

EXTRA_TRAIN_ARGS=(
  --policy.knowledge_insulation=true
  --policy.ki_objective=lap
  --policy.ki_token_loss_weight="${KI_WEIGHT}"
  --dataset.visual_robust_repo_id="[${VR_REPO_IDS}]"
  --dataset.visual_robust_root="${VR_ROOT}"
  --dataset.visual_robust_front_prefixes=observation.images.
  --dataset.visual_robust_include_views="${VIEWS}"
  --dataset.visual_robust_contrastive_weight="${VR_WEIGHT}"
  --dataset.visual_robust_front_objective=contrastive
  --dataset.visual_robust_head_mode=none
  --dataset.visual_robust_batch_size="${VR_BATCH}"
  --dataset.visual_robust_max_views=4
  --dataset.visual_robust_random_views=false
  --dataset.visual_robust_same_episode_negatives=true
  --dataset.visual_robust_encoder_chunk_size=48
  --dataset.visual_robust_num_workers="${VR_WORKERS:-6}"
  --dataset.visual_robust_cache_in_memory=false
  --dataset.visual_robust_vqa_weight=0.0
)
EXTRA_TRAIN_ARGS_STR="$(printf '%s\n' "${EXTRA_TRAIN_ARGS[@]}")"
export EXTRA_TRAIN_ARGS_STR

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
NUM_WORKERS="${NUM_WORKERS:-8}" \
STEPS="${STEPS:-50000}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29580}" \
JOB_TAG=pnpsink_vrcontrastive4_ki_lap \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
