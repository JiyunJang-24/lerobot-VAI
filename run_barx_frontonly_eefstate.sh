#!/usr/bin/env bash

# barx front-camera-only policy + an EEF-state regression head on the visual-robust views.
#
#   loss = policy_loss + VR_STATE_WEIGHT * MSE(MLP(visual_feature), eef_state)
#
# The auxiliary export stores ONE observation.state per frame -- fingertip xyz(3) + quat_xyzw(4) +
# gripper(1) -- and renders that same frame with three different robots. Every view therefore has an
# identical target, so the only way the head can win is to locate the fingertip regardless of which
# arm is holding it. That is the invariance pressure, supervised rather than contrastive.
#
# Why this is worth trying after contrastive/alignment:
#   alignment    "views should be similar"      -> satisfied at cos 0.9999 by collapsing everything;
#                                                  final action loss identical to baseline.
#   contrastive  "views similar, frames apart"  -> big representation change (gap +1.0) but the
#                                                  invariant it learns is unconstrained.
#   this         "views should decode to THIS pose" -> the invariant is pinned to a physical
#                                                  quantity, so it cannot be satisfied by a
#                                                  degenerate code that throws the scene away.
#
# Two data hazards handled in the loss (both verified on the real export):
#   * quaternion double cover -- 19 of 30 sampled frames store w < 0, so q and -q both appear for the
#     same rotation. Targets are canonicalised to w >= 0 first; without it the head is asked to
#     predict two different vectors for identical images.
#   * constant dimensions -- PandaOmron_TurnOnSinkFaucet never opens its gripper (std exactly 0 in
#     that tree). Pooling across trees rescues it here, but the normaliser still drops any dimension
#     whose pooled spread is below a floor rather than letting `std + 1e-8` turn it into |z| ~ 1e8.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${BARX:-${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa}"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_new_barx/new_barx}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/barx_frontonly_p900_i1000_u1000}"
FRONT_CAM="observation.images.robot0_agentview_right"
WEIGHT="${VR_STATE_WEIGHT:?set VR_STATE_WEIGHT}"
VR_REPO_IDS="${VR_REPO_IDS:-PandaOmron_TurnOnSinkFaucet/lerobot,IIWAOmron_PnPCounterToSink/lerobot,UR5eOmron_PnPSinkToCounter/lerobot}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/barx_frontonly_eefstate_w${WEIGHT}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

echo "barx front-only + EEF-state regression (weight ${WEIGHT}) -> ${LOG}"

EXTRA_TRAIN_ARGS=(
  --dataset.visual_robust_repo_id="[${VR_REPO_IDS}]"
  --dataset.visual_robust_root="${VR_ROOT}"
  --dataset.visual_robust_state_weight="${WEIGHT}"
  --dataset.visual_robust_state_head_hidden_dim="${VR_STATE_HIDDEN:-512}"
  --dataset.visual_robust_state_head_layers="${VR_STATE_LAYERS:-2}"
  --dataset.visual_robust_state_pool="${VR_STATE_POOL:-mean}"
  --dataset.visual_robust_state_policy_weight="${VR_STATE_POLICY_WEIGHT:-0.0}"
  --dataset.visual_robust_contrastive_weight=0.0
  --dataset.visual_robust_front_objective=none
  --dataset.visual_robust_head_mode="${VR_HEAD_MODE:-none}"
  --dataset.visual_robust_max_views="${VR_MAX_VIEWS:-3}"
  --dataset.visual_robust_random_views=false
  --dataset.visual_robust_encoder_chunk_size="${VR_CHUNK:-32}"
  --dataset.visual_robust_batch_size="${VR_BATCH:-8}"
  --dataset.visual_robust_num_workers="${VR_WORKERS:-8}"
  --dataset.visual_robust_cache_in_memory=true
  --dataset.visual_robust_same_episode_negatives=true
)
# A bash array cannot cross a process boundary; the child reads this back with mapfile.
EXTRA_TRAIN_ARGS_STR="$(printf '%s\n' "${EXTRA_TRAIN_ARGS[@]}")"
export EXTRA_TRAIN_ARGS_STR

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
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29580}" \
JOB_TAG="${JOB_TAG:-barx_frontonly_eefstate_${VR_STATE_POOL:-mean}_w${WEIGHT}${VR_STATE_POLICY_WEIGHT:+_pw${VR_STATE_POLICY_WEIGHT}}}" \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "EEF-state w=${WEIGHT} finished OK -- log: ${LOG}"
else
  echo "EEF-state w=${WEIGHT} FAILED exit ${status} -- see ${LOG}"
fi
exit "${status}"
