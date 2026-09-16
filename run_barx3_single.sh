#!/usr/bin/env bash
# One-image ablations on the segmentation split.
#
#   scene_only   observation.images.scene   kitchen with the robot blacked out
#   robot_only   observation.images.robot   robot on black
#
# Both use num_image_encoders=1 (one image, one tower) and batch 64 x 2 GPUs = effective 128,
# matching front/baseline and the segsplit pair, so all five are directly comparable.
#
# The unwanted stream is absent from the dataset rather than filtered at run time -- see
# tools/make_single_stream_dataset.py for why.
set -uo pipefail
cd "$(dirname "$0")"
source /home/gpuuser/miniforge3/etc/profile.d/conda.sh
conda activate smolvla
SHIM="$(pwd)/.ffmpeg_shim"; export LD_LIBRARY_PATH="${SHIM}:${LD_LIBRARY_PATH:-}"; export OMP_NUM_THREADS=1
STEPS=${STEPS:-50000}; SAVE_FREQ=${SAVE_FREQ:-10000}; BATCH=${BATCH:-64}
run () {  # run <tag> <dataset_root> <gpus> <port>
  local TAG=$1 ROOT=$2 GPUS=$3 PORT=$4
  local N; N="$(awk -F',' '{print NF}' <<<"$GPUS")"
  echo "[$TAG] gpus=$GPUS root=$ROOT batch=$BATCH effective=$((BATCH*N))"
  accelerate launch --multi_gpu --num_processes "$N" --gpu_ids "$GPUS" \
    --main_process_port "$PORT" --mixed_precision bf16 \
    src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id="[panda_mg,iiwa,ur5e]" --dataset.root="$ROOT" \
    --tolerance_s=1e-3 --dataset.cache_in_memory=true --dataset.video_backend=torchcodec \
    --dataset.use_state=true \
    --policy.type=smolvla --policy.push_to_hub=false --policy.load_vlm_weights=true \
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
    --policy.visual_cue_mode=vanilla --policy.num_image_encoders=1 \
    --steps="$STEPS" --save_freq="$SAVE_FREQ" --batch_size="$BATCH" \
    --output_dir="outputs/barx3_single/$TAG" --job_name="smolvla_barx3_${TAG}" \
    --num_workers=8 --dataloader_prefetch_factor=4 --dataloader_persistent_workers=true \
    --wandb.enable=true --wandb.project="robocasa_x_smolvla" --wandb.disable_artifact=true \
    --wandb.entity="DynamicVLA" --wandb.mode="${WANDB_MODE:-online}" \
    > "outputs/logs/barx3_${TAG}.log" 2>&1
  echo "[$TAG] exit $?"
}
mkdir -p outputs/logs
run scene_only dataset_git/barx3_scene_only/raw "${GPUS_A:-0,1}" 29910 &
A=$!
sleep 90
run robot_only dataset_git/barx3_robot_only/raw "${GPUS_B:-2,3}" 29920 &
B=$!
wait $A; wait $B
echo SINGLE_DONE
