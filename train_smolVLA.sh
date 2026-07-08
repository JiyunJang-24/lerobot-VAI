#!/usr/bin/env bash

# IMPORTANT: Set `--dataset.root` to your lerobot-formatted dataset path.
# `--dataset.repo_id`: the two sub-datasets forming the islands; the example below uses camera positions 40%→40% and 60%→60% (Diffusion Policy diversity setting 1).
# Note: You may need to log in to Weights & Biases (wandb) if enabled.
#   --dataset.repo_id=[xyg_20_10_15.0_65.0/v-0.400-0.400_num1,xyg_20_10_15.0_65.0/v-0.600-0.600_num5] \
  # --dataset.repo_id=[xyg_10_10_0.0_0.0/v-1.000-1.000_num1,xyg_10_10_0.0_0.0/v-1.000-1.000_num5,xyg_10_10_45.0_45.0/v-1.000-1.000_num1,xyg_10_10_45.0_45.0/v-1.000-1.000_num5,xyg_10_10_90.0_90.0/v-1.000-1.000_num1,xyg_10_10_90.0_90.0/v-1.000-1.000_num5,xyg_10_10_135.0_135.0/v-1.000-1.000_num1,xyg_10_10_135.0_135.0/v-1.000-1.000_num5,xyg_10_10_225.0_225.0/v-1.000-1.000_num1,xyg_10_10_225.0_225.0/v-1.000-1.000_num5,xyg_10_10_270.0_270.0/v-1.000-1.000_num1,xyg_10_10_270.0_270.0/v-1.000-1.000_num5,xyg_10_10_315.0_315.0/v-1.000-1.000_num1,xyg_10_10_315.0_315.0/v-1.000-1.000_num5] \
# --dataset.repo_id=[v-1.000-1.000_num1,v-1.000-1.000_num2,v-1.000-1.000_num3,v-1.000-1.000_num4,v-1.000-1.000_num5,v-1.000-1.000_num6,v-1.000-1.000_num7,v-1.000-1.000_num8,v-1.000-1.000_num9,v-1.000-1.000_num10] \

# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# REPO_ROOT="${SCRIPT_DIR}/.."
# export PYTHONPATH="${REPO_ROOT}/third_party:${REPO_ROOT}/third_party/LIBERO/libero:${PYTHONPATH}"

# CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_train.py \
#   --dataset.repo_id=[v-1.000-1.000_num1,v-1.000-1.000_num2,v-1.000-1.000_num3,v-1.000-1.000_num4,v-1.000-1.000_num5,v-1.000-1.000_num6,v-1.000-1.000_num7,v-1.000-1.000_num8,v-1.000-1.000_num9,v-1.000-1.000_num10] \
#   --dataset.root="/root/Desktop/workspace/lerobot-VAI/dataset_git/libero_10_reproduce" \
#   --dataset.use_wrist_cam=true \
#   --dataset.use_state=true \
#   --policy.type="smolvla" \
#   --policy.push_to_hub=false \
#   --steps=100000 \
#   --save_freq=5000 \
#   --batch_size=64 \
#   --wandb.enable=true \
#   --wandb.project="libero_smolvla" \
#   --wandb.disable_artifact=true \
#   --wandb.entity="DynamicVLA" \
#   --num_workers=16 \
#   --job_name="smolvla_10_trace" \
#   --policy.visual_cue_mode="basis_concat" \
#   --policy.load_vlm_weights=true \
#   --policy.freeze_vision_encoder=false \
#   --policy.train_expert_only=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# --wandb.project=smolVLA_wrist_libero_goal \


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

# CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_train.py \
#   --dataset.repo_id=[libero_10/RMA_vla_50_01_330.0_330.0/v-1.000-1.000_num1,libero_goal/RMA_vla_50_01_15.0_15.0/v-1.000-1.000_num1,libero_object/RMA_vla_50_01_30.0_30.0/v-1.000-1.000_num1,libero_spatial/RMA_vla_50_01_0.0_0.0/v-1.000-1.000_num1,libero_spatial/RMA_vla_50_01_345.0_345.0/v-1.000-1.000_num3] \
#   --dataset.root="/root/Desktop/workspace/lerobot-VAI/dataset_git/RMA_ex02" \
#   --dataset.cache_in_memory=true \
#   --dataset.use_wrist_cam=false \
#   --policy.type="smolvla" \
#   --policy.push_to_hub=false \
#   --steps=50000 \
#   --save_freq=5000 \
#   --batch_size=64 \
#   --wandb.enable=true \
#   --wandb.project="RMA02_libero_smolvla" \
#   --wandb.disable_artifact=true \
#   --wandb.entity="DynamicVLA" \
#   --num_workers=64 \
#   --dataloader_prefetch_factor=8 \
#   --dataloader_persistent_workers=true \
#   --job_name="smolvla_vanilla_rma_02" \
#   --policy.visual_cue_mode="vanilla" \
#   --policy.load_vlm_weights=true \
#   --policy.freeze_vision_encoder=false \
#   --policy.train_expert_only=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# --wandb.project=smolVLA_wrist_libero_goal \

# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

# CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_train.py \
#   --dataset.repo_id=[libero_10/RMA_vla_50_01_330.0_330.0/v-1.000-1.000_num1,libero_goal/RMA_vla_50_01_15.0_15.0/v-1.000-1.000_num1,libero_object/RMA_vla_50_01_30.0_30.0/v-1.000-1.000_num1,libero_spatial/RMA_vla_50_01_0.0_0.0/v-1.000-1.000_num1,libero_spatial/RMA_vla_50_01_345.0_345.0/v-1.000-1.000_num3] \
#   --dataset.root="/root/Desktop/workspace/lerobot-VAI/dataset_git/RMA_ex02" \
#   --dataset.cache_in_memory=true \
#   --dataset.use_wrist_cam=false \
#   --policy.type="smolvla" \
#   --policy.push_to_hub=false \
#   --steps=50000 \
#   --save_freq=5000 \
#   --batch_size=64 \
#   --wandb.enable=true \
#   --wandb.project="RMA_libero_smolvla" \
#   --wandb.disable_artifact=true \
#   --wandb.entity="DynamicVLA" \
#   --num_workers=0 \
#   --dataloader_prefetch_factor=8 \
#   --dataloader_persistent_workers=true \
#   --job_name="smolvla_axisguide_rma_02" \
#   --policy.visual_cue_mode="basis_concat" \
#   --policy.load_vlm_weights=true \
#   --policy.freeze_vision_encoder=false \
#   --policy.train_expert_only=false
# # Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# # --wandb.project=smolVLA_wrist_libero_goal \

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

CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=[iiwa_robotiq_50_01_0.0_0.0/v-1.000-1.000_num1,iiwa_robotiq_50_01_0.0_0.0/v-1.000-1.000_num2,panda_default_50_01_0.0_0.0/v-1.000-1.000_num1,panda_default_50_01_0.0_0.0/v-1.000-1.000_num2,panda_rethink_50_01_0.0_0.0/v-1.000-1.000_num3,panda_rethink_50_01_0.0_0.0/v-1.000-1.000_num9,ur5e_robotiq_50_01_0.0_0.0/v-1.000-1.000_num5,ur5e_robotiq_50_01_0.0_0.0/v-1.000-1.000_num9] \
  --dataset.root="/root/Desktop/workspace/jiyun/lerobot-VAI/dataset_git/visual_robust_ex01" \
  --dataset.cache_in_memory=true \
  --dataset.use_wrist_cam=true \
  --dataset.use_state=false \
  --policy.type="smolvla" \
  --policy.push_to_hub=false \
  --steps=50000 \
  --save_freq=5000 \
  --batch_size=48 \
  --wandb.enable=true \
  --wandb.project="visual_robust_libero_smolvla" \
  --wandb.disable_artifact=true \
  --wandb.entity="DynamicVLA" \
  --num_workers=16 \
  --dataloader_prefetch_factor=8 \
  --dataloader_persistent_workers=true \
  --job_name="smolvla_vanilla_visual_robust_wrist_wo_state" \
  --policy.visual_cue_mode="vanilla" \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# --wandb.project=smolVLA_wrist_libero_goal \

# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

# CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_train.py \
#   --dataset.repo_id=[libero_10/RMA_vla_50_01_330.0_330.0/v-1.000-1.000_num2,libero_goal/RMA_vla_50_01_15.0_15.0/v-1.000-1.000_num1,libero_object/RMA_vla_50_01_30.0_30.0/v-1.000-1.000_num1,libero_spatial/RMA_vla_50_01_0.0_0.0/v-1.000-1.000_num1,libero_10/RMA_vla_50_01_345.0_345.0/v-1.000-1.000_num3] \
#   --dataset.root="/root/Desktop/workspace/jiyun/lerobot-VAI/dataset_git/RMA_ex02" \
#   --dataset.cache_in_memory=true \
#   --dataset.use_wrist_cam=false \
#   --dataset.use_state=false \
#   --policy.type="smolvla" \
#   --policy.push_to_hub=false \
#   --steps=50000 \
#   --save_freq=5000 \
#   --batch_size=64 \
#   --wandb.enable=true \
#   --wandb.project="RMA02_libero_smolvla" \
#   --wandb.disable_artifact=true \
#   --wandb.entity="DynamicVLA" \
#   --num_workers=64 \
#   --dataloader_prefetch_factor=8 \
#   --dataloader_persistent_workers=true \
#   --job_name="smolvla_kyc_rma_02_ex2_plucker_encoder_wo_state" \
#   --policy.visual_cue_mode="plucker_concat" \
#   --policy.load_vlm_weights=true \
#   --policy.freeze_vision_encoder=false \
#   --policy.train_expert_only=false
# # Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# # --wandb.project=smolVLA_wrist_libero_goal \


# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

# CUDA_VISIBLE_DEVICES=1 python src/lerobot/scripts/lerobot_train.py \
#   --dataset.repo_id=[libero_10/RMA_vla_50_01_330.0_330.0/v-1.000-1.000_num2,libero_goal/RMA_vla_50_01_15.0_15.0/v-1.000-1.000_num1,libero_object/RMA_vla_50_01_30.0_30.0/v-1.000-1.000_num1,libero_spatial/RMA_vla_50_01_0.0_0.0/v-1.000-1.000_num1,libero_10/RMA_vla_50_01_345.0_345.0/v-1.000-1.000_num3] \
#   --dataset.root="/root/Desktop/workspace/jiyun/lerobot-VAI/dataset_git/RMA_ex02" \
#   --dataset.cache_in_memory=true \
#   --dataset.use_wrist_cam=false \
#   --dataset.use_state=false \
#   --policy.type="smolvla" \
#   --policy.push_to_hub=false \
#   --steps=50000 \
#   --save_freq=5000 \
#   --batch_size=64 \
#   --wandb.enable=true \
#   --wandb.project="RMA02_libero_smolvla" \
#   --wandb.disable_artifact=true \
#   --wandb.entity="DynamicVLA" \
#   --num_workers=64 \
#   --dataloader_prefetch_factor=8 \
#   --dataloader_persistent_workers=true \
#   --job_name="smolvla_kyc_rma_02_ex2_plucker_wo_state" \
#   --policy.visual_cue_mode="plucker" \
#   --policy.load_vlm_weights=true \
#   --policy.freeze_vision_encoder=false \
#   --policy.train_expert_only=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# --wandb.project=smolVLA_wrist_libero_goal \

# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
# export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/third_party:${SCRIPT_DIR}/third_party/LIBERO/libero:${PYTHONPATH:-}"

# CUDA_VISIBLE_DEVICES=0 python src/lerobot/scripts/lerobot_train.py \
#   --dataset.repo_id=[libero_10/RMA_vla_50_01_330.0_330.0/v-1.000-1.000_num1,libero_goal/RMA_vla_50_01_15.0_15.0/v-1.000-1.000_num1,libero_object/RMA_vla_50_01_30.0_30.0/v-1.000-1.000_num1,libero_spatial/RMA_vla_50_01_0.0_0.0/v-1.000-1.000_num1,libero_spatial/RMA_vla_50_01_345.0_345.0/v-1.000-1.000_num3] \
#   --dataset.root="/root/Desktop/workspace/lerobot-VAI/dataset_git/RMA_ex02" \
#   --dataset.cache_in_memory=true \
#   --dataset.use_wrist_cam=true \
#   --policy.type="smolvla" \
#   --policy.push_to_hub=false \
#   --steps=50000 \
#   --save_freq=5000 \
#   --batch_size=64 \
#   --wandb.enable=true \
#   --wandb.project="RMA02_libero_smolvla" \
#   --wandb.disable_artifact=true \
#   --wandb.entity="DynamicVLA" \
#   --num_workers=64 \
#   --dataloader_prefetch_factor=8 \
#   --dataloader_persistent_workers=true \
#   --job_name="smolvla_axisguide_rma_02_wrist" \
#   --policy.visual_cue_mode="basis_concat" \
#   --policy.load_vlm_weights=true \
#   --policy.freeze_vision_encoder=false \
#   --policy.train_expert_only=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# --wandb.project=smolVLA_wrist_libero_goal \
