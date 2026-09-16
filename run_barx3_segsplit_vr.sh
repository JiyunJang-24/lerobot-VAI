#!/usr/bin/env bash
# segsplit + visual-robust contrastive, applied to the ROBOT image only.
#
# Policy inputs (both cut from the front camera; no wrist stream exists in the dataset):
#   index 0  observation.images.robot  -> tower 0   robot kept, rest black
#   index 1  observation.images.scene  -> tower 1   robot blacked out, kitchen kept
#
# The contrastive loss runs on ROBOT-SEGMENTED renders of the same frame from three embodiments
# (dataset_git/vr_seg_robot) and is pinned to tower 0 by name. Verified before launch: with
# encoder_idx=0 the scene tower receives grad on 0 of its 197 parameters, and the routing is
# symmetric (pinning to index 1 reverses it exactly).
#
# Direct baseline: outputs/barx3_segsplit/separate -- same structure, same batch, same steps,
# differing only in this loss.
set -uo pipefail
cd "$(dirname "$0")"
source /home/gpuuser/miniforge3/etc/profile.d/conda.sh
conda activate smolvla
SHIM="$(pwd)/.ffmpeg_shim"; export LD_LIBRARY_PATH="${SHIM}:${LD_LIBRARY_PATH:-}"; export OMP_NUM_THREADS=1
STEPS=${STEPS:-50000}; SAVE_FREQ=${SAVE_FREQ:-10000}; BATCH=${BATCH:-32}
GPUS=${GPUS:-4,5,6,7}; N="$(awk -F',' '{print NF}' <<<"$GPUS")"
VR_ROOT=${VR_ROOT:-dataset_git/vr_seg_robot}
VR_IDS="IIWAOmron_PnPCounterToSink/lerobot,PandaOmron_TurnOnSinkFaucet/lerobot,UR5eOmron_PnPSinkToCounter/lerobot"
mkdir -p outputs/logs
echo "gpus=$GPUS batch=$BATCH effective=$((BATCH*N)) vr_root=$VR_ROOT"
accelerate launch --multi_gpu --num_processes "$N" --gpu_ids "$GPUS" \
  --main_process_port "${PORT:-29950}" --mixed_precision bf16 \
  src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  --dataset.repo_id="[panda_mg,iiwa,ur5e]" --dataset.root=dataset_git/barx3_segsplit/raw \
  --tolerance_s=1e-3 --dataset.cache_in_memory=true --dataset.video_backend=torchcodec \
  --dataset.use_state=true \
  --policy.type=smolvla --policy.push_to_hub=false --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --policy.visual_cue_mode=vanilla --policy.num_image_encoders=2 \
  --dataset.visual_robust_repo_id="[${VR_IDS}]" \
  --dataset.visual_robust_root="${VR_ROOT}" \
  --dataset.visual_robust_target_image_key=observation.images.robot \
  --dataset.visual_robust_contrastive_weight=0.5 \
  --dataset.visual_robust_temperature=0.1 \
  --dataset.visual_robust_max_views=3 \
  --dataset.visual_robust_random_views=false \
  --dataset.visual_robust_front_prefixes="observation.image." \
  --dataset.visual_robust_head_mode=none \
  --dataset.visual_robust_front_objective=contrastive \
  --dataset.visual_robust_wrist_alignment_weight=0.0 \
  --dataset.visual_robust_encoder_chunk_size=32 \
  --dataset.visual_robust_batch_size=${VB:-32} \
  --dataset.visual_robust_num_workers=8 \
  --dataset.visual_robust_cache_in_memory=true \
  --dataset.visual_robust_same_episode_negatives=true \
  --steps="$STEPS" --save_freq="$SAVE_FREQ" --batch_size="$BATCH" \
  --output_dir="${OUT:-outputs/barx3_segsplit_vr}" --job_name="smolvla_barx3_segsplit_vr" \
  --num_workers=8 --dataloader_prefetch_factor=4 --dataloader_persistent_workers=true \
  --wandb.enable=true --wandb.project="robocasa_x_smolvla" --wandb.disable_artifact=true \
  --wandb.entity="DynamicVLA" --wandb.mode="${WANDB_MODE:-online}" \
  > "${LOG:-outputs/logs/barx3_segsplit_vr.log}" 2>&1
echo "exit $?"
