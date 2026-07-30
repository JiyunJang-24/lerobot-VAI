#!/usr/bin/env bash

# Trains smolVLA on a cross-embodiment mix of RoboCasa TurnOnSinkFaucet demonstrations:
#   - panda_human: ALL Panda "human" teleop demonstrations
#   - panda_mg:    just enough Panda "mg" (MimicGen) demonstrations so that
#                  panda_human + panda_mg == PANDA_TOTAL_EPISODES (default 1000)
#   - iiwa:        ALL IIWA cross-embodiment demonstrations
#   - ur5e:        ALL UR5e cross-embodiment demonstrations
# using only the robot0_agentview_right and robot0_eye_in_hand cameras (robot0_agentview_left is
# dropped during data prep), same convention as train_smolVLA_robocasa.sh.
#
# Prerequisite: all four source dirs below must already be codebase_version "v3.0" -- run
# ./convert_robocasa_to_v30.sh once first (it converts RoboCasa's raw exports in place; this script
# never touches dataset format/version). This script only *subsets* those v3.0 sources (episode cap
# for the panda pair + camera selection for all four) via tools/prepare_robocasa_x_dataset.py,
# writing the smaller result to DATASET_ROOT/raw/{panda_human,panda_mg,iiwa,ur5e}, then points a
# normal bracketed `--dataset.repo_id=[...]` run at that directory.
#
# Also assumes `accelerate` (and this repo's other training deps) are already on PATH -- e.g. the
# `smolvla` conda env is active in your shell before running this script.
#
# Usage:
#   ./convert_robocasa_to_v30.sh   # once
#   ./train_smolVLA_robocasa_x.sh
# Override any of the exported vars below, e.g.:
#   PANDA_TOTAL_EPISODES=1000 GPU_IDS=4,5,6,7 ./train_smolVLA_robocasa_x.sh

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

# --- Data prep (subset only -- source datasets must already be v3.0) ---------------------------
ROBOCASA_ROOT="/root/Desktop/workspace/jiyun/robocasa/datasets"
SOURCE_PANDA_HUMAN="${SOURCE_PANDA_HUMAN:-${ROBOCASA_ROOT}/v1.0/pretrain/atomic/TurnOnSinkFaucet/20250819/lerobot}"
SOURCE_PANDA_MG="${SOURCE_PANDA_MG:-${ROBOCASA_ROOT}/v1.0/pretrain/atomic/TurnOnSinkFaucet/20250819/mg/demo/2025-08-21-12-24-03/lerobot}"
SOURCE_IIWA="${SOURCE_IIWA:-${ROBOCASA_ROOT}/cross_embodiment/lerobot_datasets/IIWAOmron_Robotiq85Gripper/TurnOnSinkFaucet/2026-07-24/lerobot}"
SOURCE_UR5E="${SOURCE_UR5E:-${ROBOCASA_ROOT}/cross_embodiment/lerobot_datasets/UR5eOmron_Robotiq85Gripper/TurnOnSinkFaucet/2026-07-24/lerobot}"
# panda_human (kept in full) + panda_mg (however many are needed) sum to this.
PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-1000}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_x}"
CAMERAS="${CAMERAS:-observation.images.robot0_agentview_right observation.images.robot0_eye_in_hand}"
# Set FORCE=true to rebuild DATASET_ROOT/raw/* even if it already exists -- needed when switching
# PANDA_TOTAL_EPISODES (e.g. a prior run built a smaller subset at this same DATASET_ROOT).
FORCE="${FORCE:-false}"

for pair in "SOURCE_PANDA_HUMAN:${SOURCE_PANDA_HUMAN}" "SOURCE_PANDA_MG:${SOURCE_PANDA_MG}" \
            "SOURCE_IIWA:${SOURCE_IIWA}" "SOURCE_UR5E:${SOURCE_UR5E}"; do
  name="${pair%%:*}"
  path="${pair#*:}"
  if [[ ! -d "${path}" ]]; then
    echo "${name} not found: ${path}" >&2
    exit 1
  fi
done

force_flag=()
if [[ "${FORCE}" == "true" ]]; then
  force_flag=(--force)
fi

# shellcheck disable=SC2086
python "${SCRIPT_DIR}/tools/prepare_robocasa_x_dataset.py" \
  --source-panda-human "${SOURCE_PANDA_HUMAN}" \
  --source-panda-mg "${SOURCE_PANDA_MG}" \
  --source-iiwa "${SOURCE_IIWA}" \
  --source-ur5e "${SOURCE_UR5E}" \
  --output-root "${DATASET_ROOT}" \
  --panda-total-episodes "${PANDA_TOTAL_EPISODES}" \
  --cameras ${CAMERAS} \
  "${force_flag[@]}"
prep_status=$?
if [[ ${prep_status} -ne 0 ]]; then
  echo "prepare_robocasa_x_dataset.py failed (exit ${prep_status}); aborting before training." >&2
  exit "${prep_status}"
fi

RAW_DATASET_DIR="${DATASET_ROOT}/raw"
for repo_id in panda_human panda_mg iiwa ur5e; do
  if [[ ! -f "${RAW_DATASET_DIR}/${repo_id}/meta/info.json" ]]; then
    echo "Expected dataset at ${RAW_DATASET_DIR}/${repo_id} but meta/info.json is missing." >&2
    exit 1
  fi
done

# --- Training ------------------------------------------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
CACHE_IN_MEMORY="${CACHE_IN_MEMORY:-false}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
# "torchcodec" (this project's default when the package is importable) needs system ffmpeg shared
# libs (libavutil.so.*) that this machine doesn't have -- decoding fails at runtime even though the
# python package imports fine. "pyav" bundles its own ffmpeg libs and was confirmed working here.
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
# Default 1e-4 is too strict for mg-derived subsets: dataset_tools' PyAV/SVT-AV1 re-encode of packed
# multi-episode video files (done when an episode subset doesn't align with the source's video-file
# boundaries) introduces ~1e-4 s of frame-timestamp drift by the end of a file, which trips the
# dataloader's hard assertion on the very last frame of a re-encoded episode. 1e-3 is still <2% of
# one frame at 20fps (0.05s) -- nowhere near large enough to pick the wrong neighboring frame.
TOLERANCE_S="${TOLERANCE_S:-1e-3}"
USE_WRIST_CAM="${USE_WRIST_CAM:-true}"
USE_STATE="${USE_STATE:-true}"
# "vanilla" (rather than "plucker_concat") is the safe default here since plucker embedding requires
# camera intrinsics this dataset's conversion pipeline hasn't been checked to provide -- override
# POLICY_VISUAL_CUE_MODE=plucker_concat if you've confirmed that's set up.
POLICY_VISUAL_CUE_MODE="${POLICY_VISUAL_CUE_MODE:-vanilla}"
# "online" logs to wandb.ai and requires `wandb login` (an API key) to have been run first in this
# environment -- set WANDB_MODE=offline (writes logs locally under the run's output dir only, no
# login needed) on machines without wandb credentials configured, e.g. for a quick smoke test.
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -z "${GPU_IDS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPU_IDS="0"
  fi
fi
NUM_GPUS="$(awk -F',' '{print NF}' <<<"${GPU_IDS}")"

echo "Dataset root: ${RAW_DATASET_DIR} (repo_ids: panda_human, panda_mg, iiwa, ur5e)"
echo "Cameras: ${CAMERAS}"
echo "GPU IDs: ${GPU_IDS} (${NUM_GPUS} process(es))"
echo "Per-GPU batch size: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"

accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_GPUS}" \
  --gpu_ids "${GPU_IDS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="[panda_human,panda_mg,iiwa,ur5e]" \
  --dataset.root="${RAW_DATASET_DIR}" \
  --tolerance_s="${TOLERANCE_S}" \
  --dataset.cache_in_memory="${CACHE_IN_MEMORY}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --dataset.use_wrist_cam="${USE_WRIST_CAM}" \
  --dataset.use_state="${USE_STATE}" \
  --policy.type="smolvla" \
  --policy.push_to_hub=false \
  --steps=100000 \
  --save_freq=5000 \
  --batch_size="${BATCH_SIZE}" \
  --wandb.enable=true \
  --wandb.project="robocasa_x_smolvla" \
  --wandb.disable_artifact=true \
  --wandb.entity="DynamicVLA" \
  --wandb.mode="${WANDB_MODE}" \
  --num_workers="${NUM_WORKERS}" \
  --dataloader_prefetch_factor="${PREFETCH_FACTOR}" \
  --dataloader_persistent_workers=true \
  --job_name="smolvla_robocasa_x_${PANDA_TOTAL_EPISODES}" \
  --policy.visual_cue_mode="${POLICY_VISUAL_CUE_MODE}" \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_smolvla
