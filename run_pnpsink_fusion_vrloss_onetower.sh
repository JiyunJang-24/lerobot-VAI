#!/usr/bin/env bash

# PnPSinkToCounter + KI(LAP) + a 2-tower fusion where the visual-robust auxiliary loss (contrastive)
# is added DURING policy training and lands on only ONE of the two towers.
#
#   main   stock SigLIP, trainable, ALSO carries the live visual_robust_contrastive loss
#          (compute_visual_robust_contrastive_loss_multi always reads get_vlm_model().vision_model,
#          i.e. the main tower -- there is no code path that lets it see the aux tower instead)
#   aux    the contrastive+EEF 6-view tower, pre-trained offline, held FROZEN here
#
# Distinct from every earlier fusion run: those combined a tower that already has embodiment
# invariance baked in (frozen, from a separate pre-training stage) with a plain trainable tower and
# no live auxiliary loss. This instead asks whether invariance can be shaped directly on the
# trainable tower, live, while it is simultaneously being fused with an already-invariant partner.
#
# --dropout N runs the second condition instead: same setup with --policy.fusion_aux_dropout=N,
# which zeros the aux branch on that fraction of training steps so the fused feature sometimes
# equals the main tower alone -- guarding against the action loss learning to route around whatever
# tower carries the auxiliary loss (see CLAUDE.md / the fusion_aux_dropout config field for why).
#
# VR export: the 6-view PnPSinkToCounter render (visual_robust_new_barx_ur5e), matching the corpus
# this trains on (all three robots doing PnPSinkToCounter), restricted to the 6 front cameras via
# --dataset.visual_robust_include_views so the wrist views (same prefix) don't get pulled in as an
# extra "embodiment".

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BARX="${SCRIPT_DIR}/dataset_git/barx_panda_ur5e_iiwa"
AUX_TOWER="${AUX_TOWER:-${SCRIPT_DIR}/outputs/siglip_pretrain/both_cmean_eefattn_6views_b16/vision_tower.safetensors}"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_new_barx_ur5e/new_barx}"
VR_REPO_IDS="${VR_REPO_IDS:-UR5eOmron_PnPSinkToCounter/lerobot}"
VIEWS="${VIEWS:-UR5eOmron.robot0_agentview_right,PandaOmron.robot0_agentview_right,IIWAOmron.robot0_agentview_right,JacoOmron.robot0_agentview_right,PandaOmronPandaGripper.robot0_agentview_right,JacoOmronPandaGripper.robot0_agentview_right}"
VR_WEIGHT="${VR_WEIGHT:-0.5}"
VR_FRAMES="${VR_FRAMES:-16}"
AUX_DROPOUT="${AUX_DROPOUT:-0.0}"
JOB_TAG="${JOB_TAG:-pnpsink_fusion_vrloss_maintower_ki_lap}"
LOG="${LOG:-${SCRIPT_DIR}/outputs/logs/${JOB_TAG}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

if [[ ! -f "${AUX_TOWER}" ]]; then echo "aux tower not found: ${AUX_TOWER}"; exit 1; fi
echo "PnPSink fusion, VR contrastive on the MAIN tower (aux_dropout=${AUX_DROPOUT}) -> ${LOG}"

EXTRA_TRAIN_ARGS=(
  --policy.knowledge_insulation=true
  --policy.ki_objective=lap
  --policy.ki_token_loss_weight=1.0
  --policy.aux_vision_encoder_path="${AUX_TOWER}"
  --policy.freeze_aux_vision_encoder=true
  --policy.fusion_aux_dropout="${AUX_DROPOUT}"
  --dataset.visual_robust_repo_id="[${VR_REPO_IDS}]"
  --dataset.visual_robust_root="${VR_ROOT}"
  --dataset.visual_robust_front_prefixes=observation.images.
  --dataset.visual_robust_include_views="${VIEWS}"
  --dataset.visual_robust_contrastive_weight="${VR_WEIGHT}"
  --dataset.visual_robust_front_objective=contrastive
  --dataset.visual_robust_head_mode=none
  --dataset.visual_robust_batch_size="${VR_FRAMES}"
  --dataset.visual_robust_max_views=6
  --dataset.visual_robust_random_views=false
  --dataset.visual_robust_same_episode_negatives=true
  --dataset.visual_robust_encoder_chunk_size=48
  --dataset.visual_robust_num_workers="${VR_WORKERS:-8}"
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
NUM_WORKERS="${NUM_WORKERS:-10}" \
STEPS="${STEPS:-50000}" SAVE_FREQ="${SAVE_FREQ:-10000}" MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29597}" \
JOB_TAG="${JOB_TAG}" \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
