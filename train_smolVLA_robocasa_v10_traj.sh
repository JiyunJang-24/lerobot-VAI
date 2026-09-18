#!/usr/bin/env bash

# Trains smolVLA on the robocasa_v10_crossembodiment "_traj" datasets (OpenDrawer/UR5eOmron,
# PickPlaceCounterToSink/PandaOmron, PreheatOven/IIWAOmron), using only the
# robot0_agentview_left + robot0_agentview_right cameras (robot0_eye_in_hand dropped during data
# prep, same convention as train_smolVLA_robocasa_x.sh -- see tools/prepare_robocasa_v10_traj_dataset.py).
#
# Prerequisite: the 3 source dirs under dataset_git/robocasa_v10_crossembodiment/*_traj must already
# be codebase_version "v3.0" -- run ./convert_robocasa_to_v30.sh <that traj dir> once per dataset
# first (this script never touches dataset format/version, only camera selection).
#
# Also assumes `accelerate` (and this repo's other training deps) are already on PATH -- e.g. the
# `smolvla` conda env is active in your shell before running this script.
#
# Usage:
#   ./train_smolVLA_robocasa_v10_traj.sh
# Override any of the exported vars below, e.g.:
#   BATCH_SIZE=16 STEPS=100000 ./train_smolVLA_robocasa_v10_traj.sh

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

# --- Dataloader performance -----------------------------------------------------------------------
# Ported verbatim from train_smolVLA_robocasa_x.sh (commit 077c222) -- torchcodec is ~11x faster
# than pyav on packed multi-episode videos but needs ffmpeg shared libs this machine lacks, so the
# shim below exposes PyAV's bundled ffmpeg 7 build under its plain sonames.
FFMPEG_SHIM_DIR="${SCRIPT_DIR}/.ffmpeg_shim"
if [[ ! -e "${FFMPEG_SHIM_DIR}/libavutil.so.59" ]]; then
  AV_LIBS="$(python -c 'import av, os; print(os.path.join(os.path.dirname(os.path.dirname(av.__file__)), "av.libs"))')"
  if [[ -d "${AV_LIBS}" ]]; then
    mkdir -p "${FFMPEG_SHIM_DIR}"
    for f in "${AV_LIBS}"/*.so*; do
      b="$(basename "$f")"
      soname="$(sed -E 's/^(lib[a-z0-9]+)-[0-9a-f]+\.so\.([0-9]+).*/\1.so.\2/' <<<"$b")"
      [[ "${soname}" != "$b" ]] && ln -sf "$f" "${FFMPEG_SHIM_DIR}/${soname}"
      ln -sf "$f" "${FFMPEG_SHIM_DIR}/${b}"
    done
    echo "Built ffmpeg shim for torchcodec at ${FFMPEG_SHIM_DIR}"
  else
    echo "WARNING: could not locate PyAV's bundled ffmpeg; torchcodec may fail (set VIDEO_BACKEND=pyav)" >&2
  fi
fi
export LD_LIBRARY_PATH="${FFMPEG_SHIM_DIR}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Data prep (camera selection + video re-encode; source dirs must already be v3.0) ------------
# Mirrors train_smolVLA_robocasa_x.sh's convention of always invoking its prep script rather than
# just checking a fixed output path exists: prepare_robocasa_v10_traj_dataset.py is itself
# idempotent (build_subset skips a dest that already has the expected episode count unless FORCE),
# so re-running this script is cheap and always reproduces the fixed-up (short-keyframe, common
# resolution) videos even after a from-scratch dataset_git checkout.
ROBOCASA_ROOT="${SCRIPT_DIR}/dataset_git/robocasa_v10_crossembodiment"
SOURCE_OPENDRAWER="${SOURCE_OPENDRAWER:-${ROBOCASA_ROOT}/OpenDrawer_UR5eOmron_traj}"
SOURCE_PICKPLACE="${SOURCE_PICKPLACE:-${ROBOCASA_ROOT}/PickPlaceCounterToSink_PandaOmron_traj}"
SOURCE_PREHEATOVEN="${SOURCE_PREHEATOVEN:-${ROBOCASA_ROOT}/PreheatOven_IIWAOmron_traj}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_v10_traj_lr}"
RAW_DATASET_DIR="${DATASET_ROOT}/raw"
REPO_IDS="${REPO_IDS:-opendrawer,pickplace,preheatoven}"
CAMERAS="${CAMERAS:-observation.images.robot0_agentview_left observation.images.robot0_agentview_right}"
TARGET_WIDTH="${TARGET_WIDTH:-256}"
TARGET_HEIGHT="${TARGET_HEIGHT:-256}"
GOP_SIZE="${GOP_SIZE:-1}"
# Set FORCE=true to rebuild RAW_DATASET_DIR/* even if it already exists.
FORCE="${FORCE:-false}"
force_flag=()
if [[ "${FORCE}" == "true" ]]; then
  force_flag=(--force)
fi

for pair in "SOURCE_OPENDRAWER:${SOURCE_OPENDRAWER}" "SOURCE_PICKPLACE:${SOURCE_PICKPLACE}" \
            "SOURCE_PREHEATOVEN:${SOURCE_PREHEATOVEN}"; do
  name="${pair%%:*}"
  path="${pair#*:}"
  if [[ ! -d "${path}" ]]; then
    echo "${name} not found: ${path}" >&2
    exit 1
  fi
done

python "${SCRIPT_DIR}/tools/prepare_robocasa_v10_traj_dataset.py" \
  --source "opendrawer:${SOURCE_OPENDRAWER}" \
  --source "pickplace:${SOURCE_PICKPLACE}" \
  --source "preheatoven:${SOURCE_PREHEATOVEN}" \
  --output-root "${DATASET_ROOT}" \
  --cameras ${CAMERAS} \
  --target-width "${TARGET_WIDTH}" \
  --target-height "${TARGET_HEIGHT}" \
  --gop-size "${GOP_SIZE}" \
  "${force_flag[@]}"
prep_status=$?
if [[ ${prep_status} -ne 0 ]]; then
  echo "prepare_robocasa_v10_traj_dataset.py failed (exit ${prep_status}); aborting before training." >&2
  exit "${prep_status}"
fi

for repo_id in ${REPO_IDS//,/ }; do
  if [[ ! -f "${RAW_DATASET_DIR}/${repo_id}/meta/info.json" ]]; then
    echo "Expected dataset at ${RAW_DATASET_DIR}/${repo_id} but meta/info.json is missing." >&2
    exit 1
  fi
done

# --- Training ------------------------------------------------------------------------------------
# This box's GPUs are 48 GB RTX A6000s (vs the 80 GB H100s train_smolVLA_robocasa_x.sh was tuned
# for), so batch size is scaled down proportionally from that script's 64/H100 default.
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
CACHE_IN_MEMORY="${CACHE_IN_MEMORY:-true}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
TOLERANCE_S="${TOLERANCE_S:-1e-3}"
USE_WRIST_CAM="${USE_WRIST_CAM:-false}"
USE_STATE="${USE_STATE:-true}"
POLICY_VISUAL_CUE_MODE="${POLICY_VISUAL_CUE_MODE:-vanilla}"
WANDB_MODE="${WANDB_MODE:-online}"
STEPS="${STEPS:-500000}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
JOB_TAG="${JOB_TAG:-}"
if [[ -n "${JOB_TAG}" ]]; then
  JOB_NAME="smolvla_robocasa_v10_traj_${JOB_TAG}"
else
  JOB_NAME="smolvla_robocasa_v10_traj"
fi
output_dir_flag=()
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  output_dir_flag=(--output_dir="${OUTPUT_DIR}")
fi

if [[ -z "${GPU_IDS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPU_IDS="0"
  fi
fi
NUM_GPUS="$(awk -F',' '{print NF}' <<<"${GPU_IDS}")"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/lerobot/scripts/lerobot_train.py}"
EXTRA_TRAIN_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS_STR:-}" ]]; then
  mapfile -t EXTRA_TRAIN_ARGS <<<"${EXTRA_TRAIN_ARGS_STR}"
fi

echo "Dataset root: ${RAW_DATASET_DIR} (repo_ids: ${REPO_IDS})"
echo "Job name: ${JOB_NAME}"
echo "GPU IDs: ${GPU_IDS} (${NUM_GPUS} process(es))"
echo "Per-GPU batch size: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Trainer: ${TRAIN_SCRIPT}"

accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_GPUS}" \
  --gpu_ids "${GPU_IDS}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29500}" \
  --mixed_precision "${MIXED_PRECISION}" \
  "${TRAIN_SCRIPT}" \
  --dataset.repo_id="[${REPO_IDS}]" \
  --dataset.root="${RAW_DATASET_DIR}" \
  --tolerance_s="${TOLERANCE_S}" \
  --dataset.cache_in_memory="${CACHE_IN_MEMORY}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --dataset.use_wrist_cam="${USE_WRIST_CAM}" \
  --dataset.use_state="${USE_STATE}" \
  --policy.type="smolvla" \
  --policy.push_to_hub=false \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --batch_size="${BATCH_SIZE}" \
  "${output_dir_flag[@]}" \
  --wandb.enable=true \
  --wandb.project="robocasa_v10_traj_smolvla" \
  --wandb.disable_artifact=true \
  --wandb.entity="${WANDB_ENTITY:-DynamicVLA}" \
  --wandb.mode="${WANDB_MODE}" \
  --num_workers="${NUM_WORKERS}" \
  --dataloader_prefetch_factor="${PREFETCH_FACTOR}" \
  --dataloader_persistent_workers=true \
  --job_name="${JOB_NAME}" \
  --policy.visual_cue_mode="${POLICY_VISUAL_CUE_MODE}" \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER:-false}" \
  --policy.train_expert_only=false \
  "${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"}"
# Training checkpoints will be saved under: lerobot-VAI/outputs/train/202x-xx-xx/xx-xx-xx_smolvla_robocasa_v10_traj
