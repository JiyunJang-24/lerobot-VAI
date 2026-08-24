#!/usr/bin/env bash

# PnPSinkToCounter + knowledge insulation (LAP), single tower, FROZEN.
#
# Two conditions launched from this one script, distinguished only by which weights the frozen
# tower starts from:
#
#   VISION_TOWER unset   the stock SmolVLM2 SigLIP tower, frozen as-is
#   VISION_TOWER set     a pre-trained tower (e.g. the 6-view contrastive+EEF one), frozen as-is
#
# Both are single-encoder -- no auxiliary/fusion tower, unlike run_pnpsink_dualencoder_ki_lap.sh.
# The comparison this is for: does starting the frozen tower from an embodiment-invariant
# pre-training beat just freezing the stock tower (which already has a data point: pnpsink_ki_lap
# used a trainable stock tower, and CLAUDE.md section 4 shows freezing alone costs ~25% action
# loss relative to fine-tuning on the original mixed-task corpus).
#
# A frozen tower has no backward graph, which is why this can afford a batch matching the §4
# baselines (64/rank) at the same effective size (384) the KI runs use, rather than needing to
# shrink like the trainable-aux run did.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
VISION_TOWER="${VISION_TOWER:-}"
TAG="${JOB_TAG:-pnpsink_frozen_vision_ki_lap}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/${TAG}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

if [[ -n "${VISION_TOWER}" && ! -f "${VISION_TOWER}" ]]; then
  echo "VISION_TOWER not found: ${VISION_TOWER}"
  exit 1
fi
echo "PnPSink + KI(LAP), frozen tower: ${VISION_TOWER:-stock SigLIP} -> ${LOG}"

EXTRA_TRAIN_ARGS=(
  --policy.knowledge_insulation=true
  --policy.ki_objective=lap
  --policy.ki_token_loss_weight=1.0
)
if [[ -n "${VISION_TOWER}" ]]; then
  EXTRA_TRAIN_ARGS+=(--policy.vision_encoder_path="${VISION_TOWER}")
fi
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
FREEZE_VISION_ENCODER=true \
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" BATCH_SIZE="${BATCH_SIZE:-64}" \
NUM_WORKERS="${NUM_WORKERS:-10}" \
STEPS="${STEPS:-50000}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29590}" \
JOB_TAG="${TAG}" \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
