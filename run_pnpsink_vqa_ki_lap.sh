#!/usr/bin/env bash

# PnPSinkToCounter (all three robots) + knowledge insulation on the LAP objective + a VQA term on
# the visual-robust renders.
#
# The VQA term asks "Where is the gripper?" of each embodiment render and scores the answer with the
# LM head. Every render of an instant carries the SAME answer, so the only way to be right for all
# six is to find the gripper regardless of which arm holds it.
#
# Why VQA rather than the existing EEF-state head: that head is an MLP on the vision tower, so it
# trains the tower alone. This runs vision -> connector -> text layers -> lm_head, which is what
# knowledge insulation needs -- the flow-matching gradient never reaches the VLM, and the LAP token
# objective saturates by ~15k steps, after which nothing else is teaching it.
#
# Sizing is set by measurement, not by preference. A VQA sample pushes one image through the whole
# SigLIP tower (1024 patch tokens, attention 1024^2), i.e. it costs about what a policy sample
# costs. Measured peak on one H100 with a policy batch of 48/rank:
#     12 frames x 6 views =  72 samples  ~68 GiB
#     16 frames x 6 views =  96 samples   80.5 GiB of 81.5 -- runs, no headroom
#     32 frames x 6 views = 192 samples   OOM
#
# The auxiliary batch is shuffled across episodes (same_episode_negatives=false). Same-episode
# batching exists to give the contrastive loss hard negatives; VQA has no negatives, so restricting
# a batch to one episode would only cost frame diversity.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_new_barx_ur5e/new_barx}"
VR_REPO_IDS="${VR_REPO_IDS:-UR5eOmron_PnPSinkToCounter/lerobot}"
VQA_WEIGHT="${VQA_WEIGHT:-1.0}"
VQA_FRAMES="${VQA_FRAMES:-12}"
KI_WEIGHT="${KI_WEIGHT:-1.0}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/pnpsink_vqa_ki_lap_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "PnPSinkToCounter + KI(LAP) + VQA (w=${VQA_WEIGHT}, ${VQA_FRAMES} frames x 6 views) -> ${LOG}"

EXTRA_TRAIN_ARGS=(
  --policy.knowledge_insulation=true
  --policy.ki_objective=lap
  --policy.ki_token_loss_weight="${KI_WEIGHT}"
  --dataset.visual_robust_repo_id="[${VR_REPO_IDS}]"
  --dataset.visual_robust_root="${VR_ROOT}"
  --dataset.visual_robust_front_prefixes=observation.images.
  --dataset.visual_robust_vqa_weight="${VQA_WEIGHT}"
  --dataset.visual_robust_vqa_batch_size="${VQA_FRAMES}"
  --dataset.visual_robust_vqa_position_resolution_cm=1.0
  --dataset.visual_robust_vqa_yaw_resolution_deg=5.0
  --dataset.visual_robust_batch_size="${VQA_FRAMES}"
  --dataset.visual_robust_max_views=6
  --dataset.visual_robust_random_views=false
  --dataset.visual_robust_same_episode_negatives=false
  --dataset.visual_robust_contrastive_weight=0.0
  --dataset.visual_robust_front_objective=none
  --dataset.visual_robust_num_workers="${VR_WORKERS:-6}"
  --dataset.visual_robust_cache_in_memory=false
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
STEPS="${STEPS:-50000}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29570}" \
JOB_TAG=pnpsink_vqa_ki_lap \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
