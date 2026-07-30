#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

CONDA_ENV="${CONDA_ENV:-smolvla}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
  if [[ -n "${CONDA_EXE:-}" && -f "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh" ]]; then
    source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found. Run 'conda activate ${CONDA_ENV}' first or install conda."
    exit 1
  fi

  conda activate "${CONDA_ENV}" || exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

BATCH_SIZE="${BATCH_SIZE:-12}"
VISUAL_ROBUST_BATCH_SIZE="${VISUAL_ROBUST_BATCH_SIZE:-8}"
VISUAL_ROBUST_CONTRASTIVE_WEIGHT="${VISUAL_ROBUST_CONTRASTIVE_WEIGHT:-0.5}"
VISUAL_ROBUST_TEMPERATURE="${VISUAL_ROBUST_TEMPERATURE:-0.1}"
VISUAL_ROBUST_MAX_VIEWS="${VISUAL_ROBUST_MAX_VIEWS:-5}"
VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT="${VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT:-0.5}"
VISUAL_ROBUST_WRIST_ALIGNMENT_MODE="${VISUAL_ROBUST_WRIST_ALIGNMENT_MODE:-width_continuous}"
VISUAL_ROBUST_WRIST_ALIGNMENT_MAX_VIEWS="${VISUAL_ROBUST_WRIST_ALIGNMENT_MAX_VIEWS:-${VISUAL_ROBUST_MAX_VIEWS}}"
VISUAL_ROBUST_WRIST_WIDTH_BIN_SIZE="${VISUAL_ROBUST_WRIST_WIDTH_BIN_SIZE:-0.01}"
VISUAL_ROBUST_WRIST_WIDTH_TEMPERATURE="${VISUAL_ROBUST_WRIST_WIDTH_TEMPERATURE:-0.1}"
VISUAL_ROBUST_WRIST_WIDTH_MIN="${VISUAL_ROBUST_WRIST_WIDTH_MIN:-0.0}"
VISUAL_ROBUST_WRIST_WIDTH_MAX="${VISUAL_ROBUST_WRIST_WIDTH_MAX:-0.08}"
VISUAL_ROBUST_WRIST_WIDTH_SIGMA="${VISUAL_ROBUST_WRIST_WIDTH_SIGMA:-0.2}"
VISUAL_ROBUST_ENCODER_CHUNK_SIZE="${VISUAL_ROBUST_ENCODER_CHUNK_SIZE:-32}"
LOG_FREQ="${LOG_FREQ:-10}"
STEPS="${STEPS:-50000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="$(awk -F',' '{print NF}' <<<"${GPU_IDS}")"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

echo "GPU IDs: ${GPU_IDS} (${NUM_GPUS} process(es))"
echo "Per-GPU batch size: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Per-GPU visual robust batch size: ${VISUAL_ROBUST_BATCH_SIZE}"
echo "Effective visual robust batch size: $((VISUAL_ROBUST_BATCH_SIZE * NUM_GPUS))"
echo "Visual robust front contrastive weight: ${VISUAL_ROBUST_CONTRASTIVE_WEIGHT}"
echo "Visual robust wrist alignment weight: ${VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT}"
echo "Visual robust wrist alignment mode: ${VISUAL_ROBUST_WRIST_ALIGNMENT_MODE}"
echo "Visual robust wrist width bin size: ${VISUAL_ROBUST_WRIST_WIDTH_BIN_SIZE}"
echo "Visual robust wrist width normalization: [${VISUAL_ROBUST_WRIST_WIDTH_MIN}, ${VISUAL_ROBUST_WRIST_WIDTH_MAX}]"
echo "Visual robust wrist width sigma: ${VISUAL_ROBUST_WRIST_WIDTH_SIGMA}"
echo "Mixed precision: ${MIXED_PRECISION}"

accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_GPUS}" \
  --gpu_ids "${GPU_IDS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  src/lerobot/scripts/lerobot_train_with_visual_robust.py \
  --dataset.repo_id=[iiwa_robotiq_50_01_0.0_0.0/v-1.000-1.000_num1,iiwa_robotiq_50_01_0.0_0.0/v-1.000-1.000_num2,panda_default_50_01_0.0_0.0/v-1.000-1.000_num1,panda_default_50_01_0.0_0.0/v-1.000-1.000_num2,panda_rethink_50_01_0.0_0.0/v-1.000-1.000_num3,panda_rethink_50_01_0.0_0.0/v-1.000-1.000_num9,ur5e_robotiq_50_01_0.0_0.0/v-1.000-1.000_num5,ur5e_robotiq_50_01_0.0_0.0/v-1.000-1.000_num9] \
  --dataset.root="/root/Desktop/workspace/jiyun/lerobot-VAI/dataset_git/visual_robust_ex01" \
  --dataset.visual_robust_repo_id=[visual_robust_ex3_multi_embodiment_goal_30_01_0.0_0.0/v-1.000-1.000_num1,visual_robust_ex3_multi_embodiment_goal_30_01_0.0_0.0/v-1.000-1.000_num5,visual_robust_ex3_multi_embodiment_goal_30_01_0.0_0.0/v-1.000-1.000_num9,visual_robust_ex3_multi_embodiment_object_30_01_0.0_0.0/v-1.000-1.000_num1,visual_robust_ex3_multi_embodiment_object_30_01_0.0_0.0/v-1.000-1.000_num2,visual_robust_ex3_multi_embodiment_object_30_01_0.0_0.0/v-1.000-1.000_num3,visual_robust_ex3_multi_embodiment_spatial_30_01_0.0_0.0/v-1.000-1.000_num1,visual_robust_ex3_multi_embodiment_spatial_30_01_0.0_0.0/v-1.000-1.000_num2,visual_robust_ex3_multi_embodiment_spatial_30_01_0.0_0.0/v-1.000-1.000_num3] \
  --dataset.visual_robust_root="/root/Desktop/workspace/jiyun/lerobot-VAI/dataset_git/visual_robust_ex03" \
  --dataset.visual_robust_contrastive_weight="${VISUAL_ROBUST_CONTRASTIVE_WEIGHT}" \
  --dataset.visual_robust_temperature="${VISUAL_ROBUST_TEMPERATURE}" \
  --dataset.visual_robust_max_views="${VISUAL_ROBUST_MAX_VIEWS}" \
  --dataset.visual_robust_wrist_alignment_weight="${VISUAL_ROBUST_WRIST_ALIGNMENT_WEIGHT}" \
  --dataset.visual_robust_wrist_alignment_mode="${VISUAL_ROBUST_WRIST_ALIGNMENT_MODE}" \
  --dataset.visual_robust_wrist_alignment_max_views="${VISUAL_ROBUST_WRIST_ALIGNMENT_MAX_VIEWS}" \
  --dataset.visual_robust_wrist_width_bin_size="${VISUAL_ROBUST_WRIST_WIDTH_BIN_SIZE}" \
  --dataset.visual_robust_wrist_width_temperature="${VISUAL_ROBUST_WRIST_WIDTH_TEMPERATURE}" \
  --dataset.visual_robust_wrist_width_min="${VISUAL_ROBUST_WRIST_WIDTH_MIN}" \
  --dataset.visual_robust_wrist_width_max="${VISUAL_ROBUST_WRIST_WIDTH_MAX}" \
  --dataset.visual_robust_wrist_width_sigma="${VISUAL_ROBUST_WRIST_WIDTH_SIGMA}" \
  --dataset.visual_robust_encoder_chunk_size="${VISUAL_ROBUST_ENCODER_CHUNK_SIZE}" \
  --dataset.visual_robust_batch_size="${VISUAL_ROBUST_BATCH_SIZE}" \
  --dataset.visual_robust_num_workers=8 \
  --dataset.visual_robust_cache_in_memory=true \
  --dataset.visual_robust_same_episode_negatives=true \
  --dataset.cache_in_memory=true \
  --dataset.use_wrist_cam=true \
  --dataset.use_state=false \
  --policy.type="smolvla" \
  --policy.push_to_hub=false \
  --steps="${STEPS}" \
  --log_freq="${LOG_FREQ}" \
  --save_freq=5000 \
  --batch_size="${BATCH_SIZE}" \
  --wandb.enable=true \
  --wandb.project="visual_robust_libero_smolvla" \
  --wandb.disable_artifact=true \
  --wandb.entity="DynamicVLA" \
  --num_workers=16 \
  --dataloader_prefetch_factor=8 \
  --dataloader_persistent_workers=true \
  --job_name="smolvla_vanilla_visual_robust_ex3_contrastive_front_wrist_width_loss_wo_state" \
  --policy.visual_cue_mode="vanilla" \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# --wandb.project=smolVLA_wrist_libero_goal \
