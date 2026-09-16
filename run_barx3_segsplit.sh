#!/usr/bin/env bash
# barx3 robot/scene split: does separating the robot from the scene help, and should the two
# halves share a vision encoder or get one each?
#
#   observation.images.robot   the front frame with everything but the robot blacked out
#   observation.images.scene   the front frame with the robot blacked out
#
# Both come from the segmentation videos shipped with barx3_segmentation, thresholded at 128
# (the maps are binary 0/255 through a lossy codec, so the in-between ids are edge ringing).
#
#   shared    policy.num_image_encoders=1  -- the stock SmolVLA path, one SigLIP for both images
#   separate  policy.num_image_encoders=2  -- a second tower, deep-copied from the first so both
#                                             start from the same pretrained weights (+86.4M params)
#
# This calls the trainer DIRECTLY rather than going through train_smolVLA_robocasa_x.sh, whose
# first act is to (re)build its dataset from the barx sources -- which would overwrite this one.
#
# The dataset physically contains ONLY these two image keys. That is deliberate: --dataset.use_wrist_cam
# in this repo drops `observation.wrist_image` and keys containing "wrist", so a key named
# `robot0_eye_in_hand` passes straight through it. Relying on that flag is how a "no wrist" run ends
# up training on the wrist anyway.
set -uo pipefail
cd "$(dirname "$0")"
source /home/gpuuser/miniforge3/etc/profile.d/conda.sh
conda activate smolvla

# torchcodec needs ffmpeg sonames this box does not ship; expose PyAV's bundled build under them.
SHIM="$(pwd)/.ffmpeg_shim"
if [[ ! -e "${SHIM}/libavutil.so.59" ]]; then
  AV_LIBS="$(python -c 'import av, os; print(os.path.join(os.path.dirname(os.path.dirname(av.__file__)), "av.libs"))')"
  mkdir -p "${SHIM}"
  for f in "${AV_LIBS}"/*.so*; do
    b="$(basename "$f")"
    soname="$(sed -E 's/^(lib[a-z0-9]+)-[0-9a-f]+\.so\.([0-9]+).*/\1.so.\2/' <<<"$b")"
    [[ "${soname}" != "$b" ]] && ln -sf "$f" "${SHIM}/${soname}"
    ln -sf "$f" "${SHIM}/${b}"
  done
fi
export LD_LIBRARY_PATH="${SHIM}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1

ROOT=${ROOT:-dataset_git/barx3_segsplit/raw}
STEPS=${STEPS:-50000}
SAVE_FREQ=${SAVE_FREQ:-10000}
BATCH=${BATCH:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
WANDB_MODE=${WANDB_MODE:-online}

run () {   # run <tag> <num_image_encoders> <gpu_ids> <port>
  local TAG=$1 ENC=$2 GPUS=$3 PORT=$4
  local N; N="$(awk -F',' '{print NF}' <<<"$GPUS")"
  echo "[$TAG] gpus=$GPUS encoders=$ENC batch=$BATCH effective=$((BATCH * N))"
  accelerate launch --multi_gpu --num_processes "$N" --gpu_ids "$GPUS" \
    --main_process_port "$PORT" --mixed_precision bf16 \
    src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id="[panda_mg,iiwa,ur5e]" \
    --dataset.root="$ROOT" \
    --tolerance_s=1e-3 \
    --dataset.cache_in_memory=true \
    --dataset.video_backend=torchcodec \
    --dataset.use_state=true \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --policy.load_vlm_weights=true \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.visual_cue_mode=vanilla \
    --policy.num_image_encoders="$ENC" \
    --steps="$STEPS" --save_freq="$SAVE_FREQ" --batch_size="$BATCH" \
    --output_dir="${OUT_ROOT:-outputs/barx3_segsplit}/$TAG" \
    --job_name="smolvla_barx3_segsplit_${TAG}" \
    --num_workers="$NUM_WORKERS" --dataloader_prefetch_factor=4 \
    --dataloader_persistent_workers=true \
    --wandb.enable=true --wandb.project="robocasa_x_smolvla" \
    --wandb.disable_artifact=true --wandb.entity="DynamicVLA" --wandb.mode="$WANDB_MODE" \
    > "${LOG_ROOT:-outputs/logs}/barx3_segsplit_${TAG}.log" 2>&1
  echo "[$TAG] exit $?"
}

mkdir -p outputs/logs
run shared   1 "${GPUS_A:-0,1}" 29810 &
A=$!
sleep 90
run separate 2 "${GPUS_B:-2,3}" 29820 &
B=$!
wait $A; wait $B
echo SEGSPLIT_DONE
