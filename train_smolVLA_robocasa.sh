#!/usr/bin/env bash

# Trains smolVLA on a RoboCasa atomic task, combining:
#   - ALL of the "human" teleop demonstrations, and
#   - just enough "mg" (MimicGen) demonstrations to reach TOTAL_EPISODES total,
# using only the robot0_agentview_right and robot0_eye_in_hand cameras (robot0_agentview_left is
# dropped during data prep).
#
# Prerequisite: both source dirs below must already be codebase_version "v3.0" -- run
# ./convert_robocasa_to_v30.sh once first (it converts RoboCasa's raw exports in place; this script
# never touches dataset format/version). This script only *subsets* those v3.0 sources (episode cap
# + camera selection) via tools/prepare_robocasa_dataset.py, writing the smaller result to
# DATASET_ROOT/raw/{human,mg}, then points a normal bracketed `--dataset.repo_id=[human,mg]` run at
# that directory, same convention as the other train_*.sh scripts in this repo.
#
# Also assumes `accelerate` (and this repo's other training deps) are already on PATH -- e.g. the
# `smolvla` conda env is active in your shell before running this script.
#
# Usage:
#   ./convert_robocasa_to_v30.sh   # once
#   ./train_smolVLA_robocasa.sh
# Override any of the exported vars below, e.g.:
#   TOTAL_EPISODES=1000 SOURCE_MG=/path/to/other/mg/run/lerobot ./train_smolVLA_robocasa.sh

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

# --- Dataloader performance ----------------------------------------------------------------------
# This dataset packs ~350k frames into a single mp4 per camera with a ~180-frame keyframe interval.
# That makes random-access frame reads brutally expensive on the "pyav" backend, because
# decode_video_frames_torchvision() (a) builds a fresh VideoReader per sample -- ~185 ms just to
# index a file that large -- and (b) can only seek to keyframes, so it decodes ~108 frames to return
# one. Measured 214 ms per frame => ~27 s of CPU per 64-sample batch, which starved all 8 DDP ranks
# (one rank at a time sat at 0% GPU while the other 7 spun in the NCCL all-reduce).
#
# "torchcodec" seeks by frame index and reuses a cached decoder (VideoDecoderCache), so it avoids
# both costs. It was previously unusable here because it needs ffmpeg shared libs this machine
# doesn't have -- but PyAV ships its own ffmpeg 7 build, which is exactly the version torchcodec
# looks for. The shim below just exposes those bundled libs under their plain sonames
# (libavutil-<hash>.so.59.39.100 -> libavutil.so.59), so no system/conda install is needed.
# Verified pixel-identical to the pyav path (max abs diff 0.0 over 60 random frames).
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

# 8 ranks x N dataloader workers, each defaulting to 96 OpenMP threads, means thousands of threads
# fighting over 96 cores -- it made even a torch.stack of one 256x256 frame take ~16 ms. Pinning to
# 1 thread per worker cut the decode path from 80 ms to 18 ms per frame. All the real math is on
# GPU, so the main processes lose nothing by this either.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Data prep (subset only -- source datasets must already be v3.0) ---------------------------
# Relative to SCRIPT_DIR (like DATASET_ROOT below) rather than hardcoded to one machine's home dir,
# so this works unmodified on any server that has its own dataset_git/pretrain/... checked out.
ROBOCASA_TASK_ROOT="${ROBOCASA_TASK_ROOT:-${SCRIPT_DIR}/dataset_git/pretrain/atomic/TurnOnSinkFaucet/20250819}"
SOURCE_HUMAN="${SOURCE_HUMAN:-${ROBOCASA_TASK_ROOT}/lerobot}"
SOURCE_MG="${SOURCE_MG:-${ROBOCASA_TASK_ROOT}/mg/demo/2025-08-21-12-24-03/lerobot}"
TOTAL_EPISODES="${TOTAL_EPISODES:-2000}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_turnonsinkfaucet}"
CAMERAS="${CAMERAS:-observation.images.robot0_agentview_right observation.images.robot0_eye_in_hand}"
# Set FORCE=true to rebuild DATASET_ROOT/raw/{human,mg} even if they already exist -- needed when
# switching TOTAL_EPISODES (e.g. a prior run built a smaller subset at this same DATASET_ROOT).
# Defaults to false so re-running this script doesn't redo the (slow) subset+re-encode step; the
# prep script already hard-errors if the existing subset's episode count doesn't match this run.
FORCE="${FORCE:-false}"

if [[ ! -d "${SOURCE_HUMAN}" ]]; then
  echo "SOURCE_HUMAN not found: ${SOURCE_HUMAN}" >&2
  exit 1
fi
if [[ ! -d "${SOURCE_MG}" ]]; then
  echo "SOURCE_MG not found: ${SOURCE_MG}" >&2
  exit 1
fi

force_flag=()
if [[ "${FORCE}" == "true" ]]; then
  force_flag=(--force)
fi

# shellcheck disable=SC2086
python "${SCRIPT_DIR}/tools/prepare_robocasa_dataset.py" \
  --source-human "${SOURCE_HUMAN}" \
  --source-mg "${SOURCE_MG}" \
  --output-root "${DATASET_ROOT}" \
  --total-episodes "${TOTAL_EPISODES}" \
  --cameras ${CAMERAS} \
  "${force_flag[@]}"
prep_status=$?
if [[ ${prep_status} -ne 0 ]]; then
  echo "prepare_robocasa_dataset.py failed (exit ${prep_status}); aborting before training." >&2
  exit "${prep_status}"
fi

RAW_DATASET_DIR="${DATASET_ROOT}/raw"
for repo_id in human mg; do
  if [[ ! -f "${RAW_DATASET_DIR}/${repo_id}/meta/info.json" ]]; then
    echo "Expected dataset at ${RAW_DATASET_DIR}/${repo_id} but meta/info.json is missing." >&2
    exit 1
  fi
done

# --- Training ------------------------------------------------------------------------------------
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
# 96 cores / 8 ranks = 12 cores per rank, and with OMP_NUM_THREADS=1 each worker is single-threaded,
# so 12 workers per rank saturates the box exactly without oversubscribing it.
NUM_WORKERS="${NUM_WORKERS:-12}"
# 12 workers x 4 = 48 batches buffered per rank, plenty to absorb dataloader jitter. (Frames are
# float32 by the time they're queued, ~100 MB per batch, so this is ~5 GB/rank -- keep an eye on it
# if you raise either number.)
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
# Only caches the parquet (hf_dataset), not the videos -- so this is not what fixes the dataloader
# bottleneck. It's on because it's ~70 MB total and __getitem__ does 12 separate row lookups per
# sample for `past_states`.
CACHE_IN_MEMORY="${CACHE_IN_MEMORY:-true}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
# See the "Dataloader performance" block at the top: torchcodec is ~11x faster than pyav on this
# dataset (18 ms vs 214 ms per frame) and is made loadable by the ffmpeg shim built up there.
# Fall back to VIDEO_BACKEND=pyav if torchcodec ever fails to load.
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
# Default 1e-4 is too strict for mg: dataset_tools' PyAV/SVT-AV1 re-encode of packed multi-episode
# video files (done when an episode subset doesn't align with the source's video-file boundaries)
# introduces ~1e-4 s of frame-timestamp drift by the end of a file, which trips the dataloader's
# hard assertion on the very last frame of a re-encoded episode. 1e-3 is still <2% of one frame at
# 20fps (0.05s) -- nowhere near large enough to pick the wrong neighboring frame.
TOLERANCE_S="${TOLERANCE_S:-1e-3}"
# Neither remaining camera key contains "wrist" (RoboCasa names it robot0_eye_in_hand), so
# --dataset.use_wrist_cam is a no-op for this dataset either way -- left here only for parity with
# the other train_*.sh scripts.
USE_WRIST_CAM="${USE_WRIST_CAM:-true}"
USE_STATE="${USE_STATE:-true}"
# "vanilla" (rather than "plucker_concat", used in train_smolVLA_scaling.sh) is the safe default
# here since plucker embedding requires camera intrinsics this dataset's conversion pipeline hasn't
# been checked to provide -- override POLICY_VISUAL_CUE_MODE=plucker_concat if you've confirmed
# that's set up.
POLICY_VISUAL_CUE_MODE="${POLICY_VISUAL_CUE_MODE:-vanilla}"
# "online" logs to wandb.ai and requires `wandb login` (an API key) to have been run first in this
# environment -- set WANDB_MODE=offline (writes logs locally under the run's output dir only, no
# login needed) on machines without wandb credentials configured, e.g. for a quick smoke test.
WANDB_MODE="${WANDB_MODE:-online}"

# Comma-separated GPU indices to train on. Defaults to 4,5,6,7 -- change this line (or override
# with `GPU_IDS=0,1,2,3 ./train_smolVLA_robocasa.sh`) to use different GPUs.
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="$(awk -F',' '{print NF}' <<<"${GPU_IDS}")"

SAVE_FREQ="${SAVE_FREQ:-5000}"
# By default lerobot picks its own timestamped outputs/train/<date>/<time>_<job_name> directory.
# Set OUTPUT_DIR to pin it instead -- train_smolVLA_robocasa_sweep.sh relies on this so it knows
# exactly which directory to prune checkpoints in. lerobot refuses to start if the directory already
# exists (unless resuming), so it must be a fresh path.
output_dir_flag=()
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  output_dir_flag=(--output_dir="${OUTPUT_DIR}")
fi

echo "Dataset root: ${RAW_DATASET_DIR} (repo_ids: human, mg)"
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
  --dataset.repo_id="[human,mg]" \
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
  --wandb.project="robocasa_turnonsinkfaucet_smolvla" \
  --wandb.disable_artifact=true \
  --wandb.entity="DynamicVLA" \
  --wandb.mode="${WANDB_MODE}" \
  --num_workers="${NUM_WORKERS}" \
  --dataloader_prefetch_factor="${PREFETCH_FACTOR}" \
  --dataloader_persistent_workers=true \
  --job_name="smolvla_robocasa_turnonsinkfaucet_${TOTAL_EPISODES}" \
  --policy.visual_cue_mode="${POLICY_VISUAL_CUE_MODE}" \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_smolvla
