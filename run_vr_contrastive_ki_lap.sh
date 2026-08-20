#!/usr/bin/env bash

# Visual-robust contrastive as an auxiliary loss DURING policy training, together with knowledge
# insulation on the LAP (language) objective.
#
# Why the two combine: under knowledge insulation the flow-matching gradient never reaches the VLM,
# so the tower is trained only by what is left. The LAP token objective saturates -- content-word
# accuracy reaches 1.000 by ~15k steps and its cross-entropy falls to 0.003, after which it teaches
# nothing (CLAUDE.md section 8). The contrastive term does not saturate, so it keeps shaping the
# tower for the remaining 35k steps.
#
# Corpus: the original three-task mix -- Panda TurnOnSinkFaucet, IIWA PnPCounterToSink, UR5e
# PnPSinkToCounter. The visual-robust export renders exactly that mix, so the auxiliary batches
# match the policy batches, and the comparison points live there (KI lap 0.098, contrastive w=0.5
# centered gap +1.01).
#
# Batch 48 x 8 = 384 effective, matching every knowledge-insulation run.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_new_barx/new_barx}"
VR_REPO_IDS="${VR_REPO_IDS:-PandaOmron_TurnOnSinkFaucet/lerobot,IIWAOmron_PnPCounterToSink/lerobot,UR5eOmron_PnPSinkToCounter/lerobot}"
VR_WEIGHT="${VR_WEIGHT:-0.5}"
KI_WEIGHT="${KI_WEIGHT:-1.0}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/vrcontrastive_ki_lap_w${VR_WEIGHT}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "visual-robust contrastive (w=${VR_WEIGHT}) + knowledge insulation LAP -> ${LOG}"

EXTRA_TRAIN_ARGS=(
  --policy.knowledge_insulation=true
  --policy.ki_objective=lap
  --policy.ki_token_loss_weight="${KI_WEIGHT}"
  --dataset.visual_robust_repo_id="[${VR_REPO_IDS}]"
  --dataset.visual_robust_root="${VR_ROOT}"
  --dataset.visual_robust_contrastive_weight="${VR_WEIGHT}"
  --dataset.visual_robust_front_objective=contrastive
  --dataset.visual_robust_head_mode=none
  --dataset.visual_robust_max_views=3
  --dataset.visual_robust_random_views=false
  --dataset.visual_robust_encoder_chunk_size="${VR_CHUNK:-32}"
  --dataset.visual_robust_batch_size="${VR_BATCH:-8}"
  --dataset.visual_robust_num_workers="${VR_WORKERS:-8}"
  --dataset.visual_robust_cache_in_memory=true
  --dataset.visual_robust_same_episode_negatives=true
)
EXTRA_TRAIN_ARGS_STR="$(printf '%s\n' "${EXTRA_TRAIN_ARGS[@]}")"
export EXTRA_TRAIN_ARGS_STR

PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-900}" \
IIWA_EPISODES="${IIWA_EPISODES:-1000}" \
UR5E_EPISODES="${UR5E_EPISODES:-1000}" \
USE_PANDA_HUMAN=false NORMALIZE_TASK_LANGUAGE=true \
SOURCE_PANDA_MG="${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot" \
SOURCE_IIWA="${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot" \
SOURCE_UR5E="${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot" \
DATASET_ROOT="${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000" \
CAMERAS=observation.images.robot0_agentview_right USE_WRIST_CAM=false \
FREEZE_VISION_ENCODER=false \
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" BATCH_SIZE="${BATCH_SIZE:-48}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
STEPS="${STEPS:-50000}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29560}" \
JOB_TAG="vrcontrastive_ki_lap_w${VR_WEIGHT}" \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
