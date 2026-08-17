#!/usr/bin/env bash

# Same robocasa_x PnP cross-embodiment run as run_robocasa_x_pnp_1000_mgonly.sh, plus the
# visual-robust FRONT contrastive loss on top of the action loss.
#
# What the auxiliary loss does: dataset_git/visual_robust_robocasa_x renders the *same* episode from
# three embodiments (Panda / UR5e / IIWA). The supervised-contrastive term pulls the vision
# encoder's features for all three renderings of a given frame together, and pushes different frames
# apart -- so the representation stops encoding "which robot is in the picture" and keeps encoding
# the scene. It is trained from its own dataloader, interleaved with the normal action batches.
#
# FRONT ONLY, as requested: VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT=0 disables the wrist branch (the
# trainer gates it on `weight > 0`), so only the agentview contrastive term is active. The wrist
# views are still present in the prepared dataset, so enabling that branch later is a weight change.
#
# Prerequisites (both handled by ./setup_visual_robust_x_dataset.sh):
#   1. dataset_git/visual_robust_robocasa_x/<task>/lerobot converted to v3.0
#   2. those trees reshaped to observation.image.<Emb> / observation.wrist_image.<Emb> keys
#
# The main robocasa_x prep (panda from mg only + task-language normalisation) is inherited from
# train_smolVLA_robocasa_x.sh via TRAIN_SCRIPT/EXTRA_TRAIN_ARGS rather than duplicated here.

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
BASE="${SCRIPT_DIR}/dataset_git/robocasa_x_atmoic"
VR_ROOT="${VR_ROOT:-${SCRIPT_DIR}/dataset_git/visual_robust_robocasa_x_bg}"
LOG="${SCRIPT_DIR}/outputs/logs/robocasa_x_pnp_visualrobust_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${SCRIPT_DIR}/outputs/logs"

# One entry per task tree; each contributes 3 front views (one per embodiment) of the same frame.
VR_REPO_IDS="${VR_REPO_IDS:-PickPlaceCounterToStove/lerobot,PickPlaceCounterToSink/lerobot,PickPlaceSinkToCounter/lerobot}"

# Auxiliary-loss knobs. Defaults follow train_smolVLA_visual_robust.sh except the wrist weight,
# which is 0 here.
VISUAL_ROBUST_BATCH_SIZE="${VISUAL_ROBUST_BATCH_SIZE:-8}"
VISUAL_ROBUST_CONTRASTIVE_WEIGHT="${VISUAL_ROBUST_CONTRASTIVE_WEIGHT:-0.5}"
VISUAL_ROBUST_TEMPERATURE="${VISUAL_ROBUST_TEMPERATURE:-0.1}"
# How many auxiliary views to encode per step. This is a *memory* budget, not the dataset's view
# count: every view is a full pass through the vision encoder, and the measured cost on this box is
# ~2.7 GB/view at VISUAL_ROBUST_BATCH_SIZE=8 (3 views -> 74.7 GB, 4 views -> 77.4 GB of 81.5 GB).
VISUAL_ROBUST_VIEW_BUDGET="${VISUAL_ROBUST_VIEW_BUDGET:-4}"

# One contrastive group per prefix. Default is a single "observation.image." group; the left/right
# export uses two so that left renders are only contrasted against left and right against right
# (mixing them would force the two viewpoints onto one representation, i.e. erase the viewpoint
# information rather than the robot/background nuisance the loss targets).
VISUAL_ROBUST_FRONT_PREFIXES="${VISUAL_ROBUST_FRONT_PREFIXES:-observation.image.}"

# Counted for the FIRST prefix, because max_views is applied per group -- with two groups the total
# encoded per step is max_views x number-of-prefixes.
VR_AVAILABLE_VIEWS="$(python -c "
import json,sys
info=json.load(open(sys.argv[1]))
pfx=sys.argv[2]
print(sum(1 for k,f in info['features'].items() if f.get('dtype')=='video' and k.startswith(pfx)))
" "${VR_ROOT}/${VR_REPO_IDS%%,*}/meta/info.json" "${VISUAL_ROBUST_FRONT_PREFIXES%%,*}" 2>/dev/null || echo 0)"
VR_NUM_GROUPS="$(awk -F',' '{print NF}' <<<"${VISUAL_ROBUST_FRONT_PREFIXES}")"

if [[ "${VR_AVAILABLE_VIEWS}" -lt 2 ]]; then
  echo "Could not determine front-view count from ${VR_ROOT} (got ${VR_AVAILABLE_VIEWS}); " \
       "the contrastive loss needs >= 2. Is the dataset prepared?" >&2
  exit 1
fi

if [[ -z "${VISUAL_ROBUST_MAX_VIEWS:-}" ]]; then
  if [[ "${VR_AVAILABLE_VIEWS}" -lt "${VISUAL_ROBUST_VIEW_BUDGET}" ]]; then
    VISUAL_ROBUST_MAX_VIEWS="${VR_AVAILABLE_VIEWS}"
  else
    VISUAL_ROBUST_MAX_VIEWS="${VISUAL_ROBUST_VIEW_BUDGET}"
  fi
fi

# Whenever the dataset holds more views than we can afford per step, the subset MUST be drawn at
# random. _select_visual_robust_image_keys() otherwise keeps the alphabetically-first keys, and these
# sort by embodiment: on the background-variation export (3 embodiments x 4 backgrounds = 12 front
# views) the first 4 are IIWAOmron.{dark,plain,warm,white}, i.e. a single robot -- the contrastive
# term would never see a cross-embodiment positive pair and would quietly train the wrong invariance.
# Random sampling costs nothing extra and covers all 12 views (and all pairings) over training.
if [[ -z "${VISUAL_ROBUST_RANDOM_VIEWS:-}" ]]; then
  if [[ "${VR_AVAILABLE_VIEWS}" -gt "${VISUAL_ROBUST_MAX_VIEWS}" ]]; then
    VISUAL_ROBUST_RANDOM_VIEWS=true
  else
    VISUAL_ROBUST_RANDOM_VIEWS=false
  fi
fi
VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT="${VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT:-0.0}"
VISUAL_ROBUST_ENCODER_CHUNK_SIZE="${VISUAL_ROBUST_ENCODER_CHUNK_SIZE:-32}"
VISUAL_ROBUST_NUM_WORKERS="${VISUAL_ROBUST_NUM_WORKERS:-8}"
# "none"        : contrastive straight on the mean-pooled vision backbone output (original).
# "adapter_mlp" : backbone frozen for this loss, features go through the VLM's own connector, are
#                 pooled, then an MLP head is contrasted -- so the head absorbs the invariance
#                 pressure instead of the backbone the policy depends on. The policy still trains
#                 the backbone through its own forward pass either way.
VISUAL_ROBUST_HEAD_MODE="${VISUAL_ROBUST_HEAD_MODE:-none}"
VISUAL_ROBUST_HEAD_HIDDEN_DIM="${VISUAL_ROBUST_HEAD_HIDDEN_DIM:-1024}"
VISUAL_ROBUST_HEAD_OUTPUT_DIM="${VISUAL_ROBUST_HEAD_OUTPUT_DIM:-256}"
VISUAL_ROBUST_HEAD_LAYERS="${VISUAL_ROBUST_HEAD_LAYERS:-3}"
# Only meaningful with head_mode=adapter_mlp. false lets the contrastive gradient reach the vision
# backbone through the head+connector; it costs ~17x more memory per auxiliary sample (the backbone
# activations for those views must be kept for backward), so lower VISUAL_ROBUST_BATCH_SIZE to suit.
VISUAL_ROBUST_FREEZE_BACKBONE="${VISUAL_ROBUST_FREEZE_BACKBONE:-true}"
# "contrastive": pull a frame's views together, push different frames apart (needs negatives).
# "alignment"  : positives only -- 1 - mean cosine similarity within each frame's view group.
VISUAL_ROBUST_FRONT_OBJECTIVE="${VISUAL_ROBUST_FRONT_OBJECTIVE:-contrastive}"
# Appended to the job name so variants (e.g. frontonly) get their own output dir and wandb run.
JOB_TAG_SUFFIX="${JOB_TAG_SUFFIX:-}"

for repo_id in ${VR_REPO_IDS//,/ }; do
  if [[ ! -f "${VR_ROOT}/${repo_id}/meta/info.json" ]]; then
    echo "visual robust dataset missing: ${VR_ROOT}/${repo_id}/meta/info.json" >&2
    echo "Run ./setup_visual_robust_x_dataset.sh first." >&2
    exit 1
  fi
done

echo "Visual robust root:      ${VR_ROOT}"
echo "Visual robust repo_ids:  ${VR_REPO_IDS}"
echo "Front groups:            ${VR_NUM_GROUPS}  (${VISUAL_ROBUST_FRONT_PREFIXES})"
echo "Front views per group:   ${VR_AVAILABLE_VIEWS}  (using ${VISUAL_ROBUST_MAX_VIEWS}/group/step, random=${VISUAL_ROBUST_RANDOM_VIEWS})"
echo "Encoded views per step:  $((VISUAL_ROBUST_MAX_VIEWS * VR_NUM_GROUPS))"
echo "Front objective:         ${VISUAL_ROBUST_FRONT_OBJECTIVE}  weight ${VISUAL_ROBUST_CONTRASTIVE_WEIGHT}$([[ "${VISUAL_ROBUST_FRONT_OBJECTIVE}" == "contrastive" ]] && echo " (temp ${VISUAL_ROBUST_TEMPERATURE})" || echo " (positives only, no negatives)")"
echo "Wrist alignment weight   ${VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT} (0 = front only)"
echo "Contrastive head mode:   ${VISUAL_ROBUST_HEAD_MODE}$([[ "${VISUAL_ROBUST_HEAD_MODE}" == "adapter_mlp" ]] && echo "  (backbone $([[ "${VISUAL_ROBUST_FREEZE_BACKBONE}" == "true" ]] && echo frozen || echo TRAINED) by this loss; connector+MLP ${VISUAL_ROBUST_HEAD_LAYERS}L h=${VISUAL_ROBUST_HEAD_HIDDEN_DIM} out=${VISUAL_ROBUST_HEAD_OUTPUT_DIM})")"
echo "Training PnP + visual robust -> ${LOG}"

EXTRA_TRAIN_ARGS=(
  --dataset.visual_robust_repo_id="[${VR_REPO_IDS}]"
  --dataset.visual_robust_root="${VR_ROOT}"
  --dataset.visual_robust_contrastive_weight="${VISUAL_ROBUST_CONTRASTIVE_WEIGHT}"
  --dataset.visual_robust_temperature="${VISUAL_ROBUST_TEMPERATURE}"
  --dataset.visual_robust_max_views="${VISUAL_ROBUST_MAX_VIEWS}"
  --dataset.visual_robust_random_views="${VISUAL_ROBUST_RANDOM_VIEWS}"
  --dataset.visual_robust_front_prefixes="${VISUAL_ROBUST_FRONT_PREFIXES}"
  --dataset.visual_robust_head_mode="${VISUAL_ROBUST_HEAD_MODE}"
  --dataset.visual_robust_head_hidden_dim="${VISUAL_ROBUST_HEAD_HIDDEN_DIM}"
  --dataset.visual_robust_head_output_dim="${VISUAL_ROBUST_HEAD_OUTPUT_DIM}"
  --dataset.visual_robust_head_layers="${VISUAL_ROBUST_HEAD_LAYERS}"
  --dataset.visual_robust_freeze_backbone="${VISUAL_ROBUST_FREEZE_BACKBONE}"
  --dataset.visual_robust_front_objective="${VISUAL_ROBUST_FRONT_OBJECTIVE}"
  --dataset.visual_robust_wrist_alignment_weight="${VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT}"
  --dataset.visual_robust_encoder_chunk_size="${VISUAL_ROBUST_ENCODER_CHUNK_SIZE}"
  --dataset.visual_robust_batch_size="${VISUAL_ROBUST_BATCH_SIZE}"
  --dataset.visual_robust_num_workers="${VISUAL_ROBUST_NUM_WORKERS}"
  --dataset.visual_robust_cache_in_memory=true
  --dataset.visual_robust_same_episode_negatives=true
)
# Hand these over as a newline-delimited string: a bash array cannot cross a process boundary, so
# exporting the array itself would deliver nothing and the run would quietly train with no
# visual-robust loss. train_smolVLA_robocasa_x.sh reads EXTRA_TRAIN_ARGS_STR back with mapfile.
EXTRA_TRAIN_ARGS_STR="$(printf '%s\n' "${EXTRA_TRAIN_ARGS[@]}")"
export EXTRA_TRAIN_ARGS_STR

# Job name spells out the settings that actually change the method, so runs are told apart from the
# output dir / wandb name alone: objective, its weight, auxiliary batch, head mode and whether the
# backbone is frozen by this loss.
VR_TAG="vr${VISUAL_ROBUST_FRONT_OBJECTIVE}_w${VISUAL_ROBUST_CONTRASTIVE_WEIGHT}_vb${VISUAL_ROBUST_BATCH_SIZE}"
if [[ "${VISUAL_ROBUST_HEAD_MODE}" != "none" ]]; then
  VR_TAG="${VR_TAG}_${VISUAL_ROBUST_HEAD_MODE}"
  VR_TAG="${VR_TAG}_$([[ "${VISUAL_ROBUST_FREEZE_BACKBONE}" == "true" ]] && echo bbfrozen || echo bbtrain)"
else
  VR_TAG="${VR_TAG}_nohead"
fi

# Episode counts and sources are overridable so this script can drive a different corpus (e.g. the
# barx export) without a fork. Defaults stay on the PnP mix it was written for.
PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-1000}" \
IIWA_EPISODES="${IIWA_EPISODES:-1000}" \
UR5E_EPISODES="${UR5E_EPISODES:-1000}" \
USE_PANDA_HUMAN="${USE_PANDA_HUMAN:-false}" \
NORMALIZE_TASK_LANGUAGE="${NORMALIZE_TASK_LANGUAGE:-true}" \
JOB_TAG="${JOB_TAG_BASE:-multi_task_pnp_mgonly_langnorm}_${VR_TAG}${JOB_TAG_SUFFIX:+_${JOB_TAG_SUFFIX}}" \
TRAIN_SCRIPT=src/lerobot/scripts/lerobot_train_with_visual_robust.py \
SOURCE_PANDA_MG="${SOURCE_PANDA_MG:-${BASE}/mg/PandaOmron/pretrain/PnPCounterToStove}" \
SOURCE_IIWA="${SOURCE_IIWA:-${BASE}/IIWAOmron/pretrain/PnPCounterToSink/lerobot}" \
SOURCE_UR5E="${SOURCE_UR5E:-${BASE}/UR5eOmron/pretrain/PnPSinkToCounter/lerobot}" \
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_x_pnp_mgonly_p1000_i1000_u1000}" \
CAMERAS="${CAMERAS:-observation.images.robot0_agentview_right observation.images.robot0_eye_in_hand}" \
USE_WRIST_CAM="${USE_WRIST_CAM:-true}" \
  "${SCRIPT_DIR}/train_smolVLA_robocasa_x.sh" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}

if [[ ${status} -eq 0 ]]; then
  echo "PnP + visual robust (front) finished OK -- log: ${LOG}"
else
  echo "PnP + visual robust (front) FAILED exit ${status} -- see ${LOG}"
fi
echo "free disk: $(df -h --output=avail "${SCRIPT_DIR}" | tail -1 | tr -d ' ')"
exit "${status}"
