#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}/.."
export PYTHONPATH="${SCRIPT_DIR}/src:${REPO_ROOT}/LIBERO:${PYTHONPATH:-}"

# Keep dataloader workers from oversubscribing CPU threads while decoding images.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DATASET_ROOT="${SCRIPT_DIR}/dataset_git/libero_spatial_reproduce"
DATASET_REPO_ID="[v-1.000-1.000_num1,v-1.000-1.000_num2,v-1.000-1.000_num3,v-1.000-1.000_num4,v-1.000-1.000_num5,v-1.000-1.000_num6,v-1.000-1.000_num7,v-1.000-1.000_num8,v-1.000-1.000_num9,v-1.000-1.000_num10]"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" conda run --no-capture-output -n lerobot \
  python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.use_wrist_cam=true \
    --dataset.use_state=false \
    --policy.type="diffusion" \
    --policy.push_to_hub=false \
    --policy.image_goal_cond=true \
    --policy.n_obs_steps=1 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --steps=100000 \
    --save_freq=5000 \
    --batch_size=64 \
    --wandb.enable=true \
    --wandb.project="libero_diffusion" \
    --wandb.disable_artifact=true \
    --wandb.entity="DynamicVLA" \
    --num_workers=8 \
    --log_freq=1 \
    --job_name="diffusion_spatial_image_goal_wo_state" \
    "$@"
