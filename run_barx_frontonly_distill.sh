#!/usr/bin/env bash

# barx front-camera-only policy, vision tower trained but its OUTPUTS pulled toward a frozen copy of
# the pretrained tower:  loss = policy_loss + VISION_DISTILL_WEIGHT * mean((f_student - f_teacher)^2)
#
# The feature-space sibling of run_barx_frontonly_l2sp.sh. Both anchor to the same pretrained tower;
# they differ in where the anchor bites:
#
#   L2-SP     penalises  sum_i (w_i - w0_i)^2   -- where the tower sits in PARAMETER space
#   distill   penalises  mean (f - f0)^2        -- what the tower COMPUTES on these images
#
# They are not interchangeable. A tower can drift a long way in weight space while its function on
# this narrow data distribution barely changes (many directions in parameter space are unused by
# these images), and conversely small weight changes concentrated in the directions this data excites
# can change the features a lot. Only the feature-space version constrains the thing the policy
# actually reads.
#
# Distillation is on the per-PATCH tokens, not the mean-pooled vector, deliberately: the alignment
# objective's failure on this project was precisely that it constrained a pooled statistic (its
# cosine hit 0.9999) while the 1024 tokens the policy consumes went untouched, leaving the action
# loss identical to baseline.
#
# Cost: one extra no-grad forward of the 86.4M tower per step + 0.32 GiB for its weights. The student
# side is captured by a forward hook from the policy's own forward, so it adds no student compute.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
WEIGHT="${VISION_DISTILL_WEIGHT:?set VISION_DISTILL_WEIGHT}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/barx_frontonly_distill_w${WEIGHT}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "barx front-only + feature distillation (weight ${WEIGHT}) -> ${LOG}"

PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-900}" \
IIWA_EPISODES="${IIWA_EPISODES:-1000}" \
UR5E_EPISODES="${UR5E_EPISODES:-1000}" \
USE_PANDA_HUMAN="${USE_PANDA_HUMAN:-false}" \
NORMALIZE_TASK_LANGUAGE="${NORMALIZE_TASK_LANGUAGE:-true}" \
SOURCE_PANDA_MG="${SOURCE_PANDA_MG:-${BARX}/PandaOmron/pretrain/TurnOnSinkFaucet/lerobot}" \
SOURCE_IIWA="${SOURCE_IIWA:-${BARX}/IIWAOmron/pretrain/PnPCounterToSink/lerobot}" \
SOURCE_UR5E="${SOURCE_UR5E:-${BARX}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot}" \
DATASET_ROOT="${DATASET_ROOT}" \
CAMERAS="${FRONT_CAM}" \
USE_WRIST_CAM="${USE_WRIST_CAM:-false}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
NUM_WORKERS="${NUM_WORKERS:-12}" \
STEPS="${STEPS:-50000}" \
SAVE_FREQ="${SAVE_FREQ:-10000}" \
WANDB_MODE="${WANDB_MODE:-online}" \
FREEZE_VISION_ENCODER=false \
VISION_DISTILL_WEIGHT="${WEIGHT}" \
GPU_IDS="${GPU_IDS:-}" \
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29540}" \
JOB_TAG="${JOB_TAG:-barx_frontonly_distill_w${WEIGHT}}" \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "distill w=${WEIGHT} finished OK -- log: ${LOG}"
else
  echo "distill w=${WEIGHT} FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
