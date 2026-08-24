#!/usr/bin/env bash

# Trains smolVLA on a cross-embodiment mix of RoboCasa TurnOnSinkFaucet demonstrations, with an
# independently chosen episode count per robot:
#   - panda_human: ALL Panda "human" teleop demonstrations
#   - panda_mg:    just enough Panda "mg" (MimicGen) demonstrations so that
#                  panda_human + panda_mg == PANDA_TOTAL_EPISODES (default 1000)
#   - iiwa:        first IIWA_EPISODES IIWA demonstrations   (default 1000)
#   - ur5e:        first UR5E_EPISODES UR5e demonstrations   (default 1000)
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
#   PANDA_TOTAL_EPISODES=1000 IIWA_EPISODES=1000 UR5E_EPISODES=1000 ./train_smolVLA_robocasa_x.sh

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

# --- Dataloader performance -----------------------------------------------------------------------
# Ported verbatim from train_smolVLA_robocasa.sh (commit 077c222) -- see that script for the full
# writeup. Short version: these datasets pack hundreds of thousands of frames into one mp4 per
# camera, which makes random-access reads on the "pyav" backend ~11x slower than "torchcodec"
# (214 ms vs 18 ms per frame) and starves every DDP rank. torchcodec needs ffmpeg shared libs this
# machine lacks, so the shim below exposes PyAV's bundled ffmpeg 7 build under its plain sonames.
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
# Thousands of OpenMP threads fighting over the box made even a single-frame torch.stack take ~16 ms.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Data prep (subset only -- source datasets must already be v3.0) ---------------------------
# All four sources live under this repo's dataset_git/ tree (they were copied here from the original
# robocasa export box, which is why these are no longer /root/Desktop/... paths).
ROBOCASA_ROOT="${SCRIPT_DIR}/dataset_git"
# The Panda pair is taken from the already-prepared robocasa_sweep/ep1000 subset rather than the raw
# atomic export: it is v3.0, already reduced to the robot0_agentview_right + robot0_eye_in_hand
# camera pair, and already split 107 human + 893 mg = exactly PANDA_TOTAL_EPISODES (1000). Because
# the episode counts already match, the prep step below copies these two through untouched.
SOURCE_PANDA_HUMAN="${SOURCE_PANDA_HUMAN:-${ROBOCASA_ROOT}/robocasa_sweep/ep1000/raw/human}"
SOURCE_PANDA_MG="${SOURCE_PANDA_MG:-${ROBOCASA_ROOT}/robocasa_sweep/ep1000/raw/mg}"
# IIWA / UR5e come from the pre-converted cross_embodiment trees (already v3.0, same 256x256 cameras
# and same TurnOnSinkFaucet task as panda, but still carrying the third robot0_agentview_left camera
# that the prep step drops). The raw barx/mg trees are NOT usable here: they are HDF5, not LeRobot
# datasets, and carry a single `agentview_rgb` at 180x320, so they would need a full mujoco/robosuite
# re-render first.
SOURCE_IIWA="${SOURCE_IIWA:-${ROBOCASA_ROOT}/cross_embodiment/lerobot_datasets/IIWAOmron_Robotiq85Gripper/TurnOnSinkFaucet/2026-07-24/lerobot}"
SOURCE_UR5E="${SOURCE_UR5E:-${ROBOCASA_ROOT}/cross_embodiment/lerobot_datasets/UR5eOmron_Robotiq85Gripper/TurnOnSinkFaucet/2026-07-24/lerobot}"
# panda_human (kept in full) + panda_mg (however many are needed) sum to this.
PANDA_TOTAL_EPISODES="${PANDA_TOTAL_EPISODES:-1000}"
# How many episodes to take from each of the other two robots. These sources hold slightly under
# 1000 usable episodes each (iiwa 990, ur5e 997 at the time of writing), so asking for 1000 just
# takes all of them -- the prep script clamps and says so rather than failing.
IIWA_EPISODES="${IIWA_EPISODES:-1000}"
UR5E_EPISODES="${UR5E_EPISODES:-1000}"
DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset_git/robocasa_x}"
CAMERAS="${CAMERAS:-observation.images.robot0_agentview_right observation.images.robot0_eye_in_hand}"
# Set FORCE=true to rebuild DATASET_ROOT/raw/* even if it already exists -- needed when switching
# PANDA_TOTAL_EPISODES (e.g. a prior run built a smaller subset at this same DATASET_ROOT).
FORCE="${FORCE:-false}"

# Set USE_PANDA_HUMAN=false to build the whole panda share from panda_mg instead. Needed whenever
# panda_human is the only subset with mobile-base/torso motion in action dims 0,1,2,4: those dims
# are constant everywhere else, so one pooled MEAN_STD normalizer over all four subsets drives their
# std to ~1e-3 and rescales panda_human's handful of real values to |z| ~ 10^2, which then swamps the
# action loss (measured: 36% of total action MSE on the PnP mix, vs 0% on TurnOnSinkFaucet where no
# subset moved the base).
USE_PANDA_HUMAN="${USE_PANDA_HUMAN:-true}"
if [[ "${USE_PANDA_HUMAN}" == "true" ]]; then
  SOURCE_PAIRS=("SOURCE_PANDA_HUMAN:${SOURCE_PANDA_HUMAN}")
  REPO_IDS="panda_human,panda_mg,iiwa,ur5e"
  panda_human_flag=()
else
  SOURCE_PAIRS=()
  REPO_IDS="panda_mg,iiwa,ur5e"
  panda_human_flag=(--panda-human-episodes 0)
fi

for pair in "${SOURCE_PAIRS[@]+"${SOURCE_PAIRS[@]}"}" "SOURCE_PANDA_MG:${SOURCE_PANDA_MG}" \
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

# Set NORMALIZE_TASK_LANGUAGE=true to restyle every subset's instructions to one convention
# (leading capital + full stop). Only matters on mixes whose robots phrase instructions
# differently -- see the flag's help in tools/prepare_robocasa_x_dataset.py.
NORMALIZE_TASK_LANGUAGE="${NORMALIZE_TASK_LANGUAGE:-false}"
normalize_lang_flag=()
if [[ "${NORMALIZE_TASK_LANGUAGE}" == "true" ]]; then
  normalize_lang_flag=(--normalize-task-language)
fi

# shellcheck disable=SC2086
python "${SCRIPT_DIR}/tools/prepare_robocasa_x_dataset.py" \
  --source-panda-human "${SOURCE_PANDA_HUMAN}" \
  --source-panda-mg "${SOURCE_PANDA_MG}" \
  --source-iiwa "${SOURCE_IIWA}" \
  --source-ur5e "${SOURCE_UR5E}" \
  --output-root "${DATASET_ROOT}" \
  --panda-total-episodes "${PANDA_TOTAL_EPISODES}" \
  --iiwa-episodes "${IIWA_EPISODES}" \
  --ur5e-episodes "${UR5E_EPISODES}" \
  --cameras ${CAMERAS} \
  "${panda_human_flag[@]+"${panda_human_flag[@]}"}" \
  "${normalize_lang_flag[@]+"${normalize_lang_flag[@]}"}" \
  "${force_flag[@]}"
prep_status=$?
if [[ ${prep_status} -ne 0 ]]; then
  echo "prepare_robocasa_x_dataset.py failed (exit ${prep_status}); aborting before training." >&2
  exit "${prep_status}"
fi

RAW_DATASET_DIR="${DATASET_ROOT}/raw"
for repo_id in ${REPO_IDS//,/ }; do
  if [[ ! -f "${RAW_DATASET_DIR}/${repo_id}/meta/info.json" ]]; then
    echo "Expected dataset at ${RAW_DATASET_DIR}/${repo_id} but meta/info.json is missing." >&2
    exit 1
  fi
done

# --- Training ------------------------------------------------------------------------------------
# 64 per GPU is what the panda-only ep1000 run sustained on this box's 80 GB H100s (~64 GB resident,
# ~79% of GPU memory), so it leaves headroom for the extra embodiments' dataloader state.
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
# Caches the parquet (hf_dataset) only, not the videos -- cheap, and __getitem__ does many row
# lookups per sample for `past_states`.
CACHE_IN_MEMORY="${CACHE_IN_MEMORY:-true}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
# torchcodec is ~11x faster than pyav on these packed videos and is made loadable by the ffmpeg shim
# built at the top of this script. Fall back to VIDEO_BACKEND=pyav if it ever fails to load.
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
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
# Overridable so a smoke test can run a handful of steps (e.g. STEPS=60 SAVE_FREQ=50) instead of the
# full schedule.
STEPS="${STEPS:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
# Goes into --job_name (and therefore lerobot's default output dir + the wandb run name). Without
# it, runs are named only by their episode counts -- so a TurnOnSinkFaucet mix and a PnP mix with
# the same per-robot counts produce indistinguishable directories. Set it to whatever identifies
# the task/dataset behind the run, e.g. JOB_TAG=multi_task_pnp. Empty keeps the old naming.
JOB_TAG="${JOB_TAG:-}"
if [[ -n "${JOB_TAG}" ]]; then
  JOB_NAME="smolvla_robocasa_x_${JOB_TAG}_p${PANDA_TOTAL_EPISODES}_i${IIWA_EPISODES}_u${UR5E_EPISODES}"
else
  JOB_NAME="smolvla_robocasa_x_p${PANDA_TOTAL_EPISODES}_i${IIWA_EPISODES}_u${UR5E_EPISODES}"
fi
# By default lerobot picks its own timestamped outputs/train/<date>/<time>_<job_name> directory; set
# OUTPUT_DIR to pin it. lerobot refuses to start if that directory already exists (unless resuming).
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

# Which training entrypoint to launch, plus any extra flags to append. Defaults reproduce the plain
# run; train_smolVLA_robocasa_x_visual_robust.sh points these at the visual-robust trainer so it can
# reuse this script's data prep instead of duplicating it.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/lerobot/scripts/lerobot_train.py}"
# Extra flags arrive as a NEWLINE-DELIMITED string, not a bash array: arrays are not part of the
# process environment, so `export EXTRA_TRAIN_ARGS=(...)` in a caller reaches this script as
# nothing at all -- which would silently drop e.g. every --dataset.visual_robust_* flag and train
# with no auxiliary loss while looking perfectly healthy.
EXTRA_TRAIN_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS_STR:-}" ]]; then
  mapfile -t EXTRA_TRAIN_ARGS <<<"${EXTRA_TRAIN_ARGS_STR}"
fi

echo "Dataset root: ${RAW_DATASET_DIR} (repo_ids: ${REPO_IDS})"
echo "Episodes: panda=${PANDA_TOTAL_EPISODES} iiwa=${IIWA_EPISODES} ur5e=${UR5E_EPISODES}"
echo "Cameras: ${CAMERAS}"
echo "Job name: ${JOB_NAME}"
echo "GPU IDs: ${GPU_IDS} (${NUM_GPUS} process(es))"
echo "Per-GPU batch size: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Trainer: ${TRAIN_SCRIPT}"

# accelerate's rendezvous port. Two runs sharing the same GPUs (e.g. a frozen-encoder baseline
# alongside a live job) both try to bind 29500 and the second one dies with "address already in
# use", so a parallel launch must be given its own port.
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
  --wandb.project="robocasa_x_smolvla" \
  --wandb.disable_artifact=true \
  --wandb.entity="DynamicVLA" \
  --wandb.mode="${WANDB_MODE}" \
  --num_workers="${NUM_WORKERS}" \
  --dataloader_prefetch_factor="${PREFETCH_FACTOR}" \
  --dataloader_persistent_workers=true \
  --job_name="${JOB_NAME}" \
  --policy.visual_cue_mode="${POLICY_VISUAL_CUE_MODE}" \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder="${FREEZE_VISION_ENCODER:-false}" \
  --policy.train_expert_only=false \
  --dataset.vision_l2sp_weight="${VISION_L2SP_WEIGHT:-0.0}" \
  --dataset.vision_l2sp_scope="${VISION_L2SP_SCOPE:-vision}" \
  --dataset.vision_distill_weight="${VISION_DISTILL_WEIGHT:-0.0}" \
  "${EXTRA_TRAIN_ARGS[@]+"${EXTRA_TRAIN_ARGS[@]}"}"
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_smolvla
