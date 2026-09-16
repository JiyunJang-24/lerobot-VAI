#!/usr/bin/env bash
# Seed-2 replicate of the segsplit pair. Seed 0 gave shared 0.0637 vs separate 0.0618; this says
# whether that 0.0019 is a real difference or run-to-run noise. Everything else is identical to
# run_barx3_segsplit.sh -- same dataset, same batch, same steps.
#
# These occupy GPUs the segsplit+contrastive run is meant to use. They checkpoint every 10k, so
# stopping them when the regenerated segmentation arrives costs at most 10k steps.
set -uo pipefail
cd "$(dirname "$0")"
source /home/gpuuser/miniforge3/etc/profile.d/conda.sh
conda activate smolvla
SHIM="$(pwd)/.ffmpeg_shim"; export LD_LIBRARY_PATH="${SHIM}:${LD_LIBRARY_PATH:-}"; export OMP_NUM_THREADS=1
STEPS=${STEPS:-50000}; SAVE_FREQ=${SAVE_FREQ:-10000}; BATCH=${BATCH:-64}; SEED=${SEED:-1}
run () {  # run <tag> <encoders> <gpus> <port>
  local TAG=$1 ENC=$2 GPUS=$3 PORT=$4
  local N; N="$(awk -F',' '{print NF}' <<<"$GPUS")"
  echo "[$TAG] gpus=$GPUS encoders=$ENC seed=$SEED effective=$((BATCH*N))"
  accelerate launch --multi_gpu --num_processes "$N" --gpu_ids "$GPUS" \
    --main_process_port "$PORT" --mixed_precision bf16 \
    src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id="[panda_mg,iiwa,ur5e]" --dataset.root=dataset_git/barx3_segsplit/raw \
    --tolerance_s=1e-3 --dataset.cache_in_memory=true --dataset.video_backend=torchcodec \
    --dataset.use_state=true --seed="$SEED" \
    --policy.type=smolvla --policy.push_to_hub=false --policy.load_vlm_weights=true \
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
    --policy.visual_cue_mode=vanilla --policy.num_image_encoders="$ENC" \
    --steps="$STEPS" --save_freq="$SAVE_FREQ" --batch_size="$BATCH" \
    --output_dir="outputs/barx3_segsplit_seed${SEED}/$TAG" \
    --job_name="smolvla_barx3_segsplit_${TAG}_seed${SEED}" \
    --num_workers=8 --dataloader_prefetch_factor=4 --dataloader_persistent_workers=true \
    --wandb.enable=true --wandb.project="robocasa_x_smolvla" --wandb.disable_artifact=true \
    --wandb.entity="DynamicVLA" --wandb.mode="${WANDB_MODE:-online}" \
    > "outputs/logs/barx3_segsplit_${TAG}_seed${SEED}.log" 2>&1
  echo "[$TAG] exit $?"
}
mkdir -p outputs/logs
run shared   1 "${GPUS_A:-2,3}" 29930 &
A=$!
sleep 90
run separate 2 "${GPUS_B:-4,5}" 29940 &
B=$!
wait $A; wait $B
echo SEGSPLIT_SEED_DONE
