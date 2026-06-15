#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/src:${REPO_ROOT}/third_party:${REPO_ROOT}/third_party/LIBERO/libero:${PYTHONPATH:-}"

ORIG_DATASET_ROOT="${SCRIPT_DIR}/dataset_git/libero_spatial_reproduce"
REWIND_DATASET_ROOT="${SCRIPT_DIR}/dataset_git/libero_spatial_reproduce_rewind_gripper"
COMBINED_DATASET_ROOT="${SCRIPT_DIR}/dataset_git/libero_spatial_reproduce_combined_with_rewind_gripper"
BASE_REPO_IDS=(
  v-1.000-1.000_num1
  v-1.000-1.000_num2
  v-1.000-1.000_num3
  v-1.000-1.000_num4
  v-1.000-1.000_num5
  v-1.000-1.000_num6
  v-1.000-1.000_num7
  v-1.000-1.000_num8
  v-1.000-1.000_num9
  v-1.000-1.000_num10
)

mkdir -p "${COMBINED_DATASET_ROOT}"
COMBINED_REPO_IDS=()
for repo_id in "${BASE_REPO_IDS[@]}"; do
  ln -sfn "${ORIG_DATASET_ROOT}/${repo_id}" "${COMBINED_DATASET_ROOT}/orig_${repo_id}"
  ln -sfn "${REWIND_DATASET_ROOT}/${repo_id}" "${COMBINED_DATASET_ROOT}/rewind_gripper_${repo_id}"
  COMBINED_REPO_IDS+=("orig_${repo_id}" "rewind_gripper_${repo_id}")
done

DATASET_REPO_ID="["
for repo_id in "${COMBINED_REPO_IDS[@]}"; do
  DATASET_REPO_ID+="${repo_id},"
done
DATASET_REPO_ID="${DATASET_REPO_ID%,}]"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" conda run --no-capture-output -n lerobot \
  python src/lerobot/scripts/lerobot_train.py \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${COMBINED_DATASET_ROOT}" \
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
    --num_workers=16 \
    --job_name="diffusion_spatial_image_goal_rewind_gripper_wo_state" \
    "$@"
