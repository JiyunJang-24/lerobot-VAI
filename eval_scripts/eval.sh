#!/bin/bash

VENV_PATH="venv-lerobot-eval"

if [ ! -d "${VENV_PATH}" ]; then
  echo "Error: ${VENV_PATH} virtual environment folder not found."
  exit 1
fi

source ${VENV_PATH}/bin/activate

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}/.."
export PYTHONPATH="${REPO_ROOT}/LIBERO:${PYTHONPATH}"

# Do avoid EGL device display error
export MUJOCO_GL=osmesa

# # Evaluate a policy on the LIBERO benchmark
# python3 /root/Desktop/workspace/jiyun/lerobot-VAI/src/lerobot/scripts/lerobot_eval.py \
#   --policy.path=lerobot/smolvla_base \
#   --policy.visual_cue_mode=vanilla \
#   --policy.code_mode=eval \
#   --env.type=libero \
#   --env.task=libero_spatial \
#   --eval.n_episodes=1 \
#   --eval.batch_size=1 \
#   --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
#   # (Below): For ours || (Above): For debugging
#   # --rename_map='{"observation.images.image": "observation.images.image", "observation.images.image2": "observation.images.wrist_image"}'

python3 /root/Desktop/workspace/jiyun/lerobot-VAI/src/lerobot/scripts/lerobot_eval.py \
  --policy.path=/root/Desktop/workspace/jiyun/lerobot-VAI/outputs/train/2026-01-18/03-22-52_smolvla_basis_concat_spatial_except_2_4_angle_from_0/checkpoints/100000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --rename_map='{"observation.images.image": "observation.image", "observation.images.image2": "observation.wrist_image"}'
  # (Below): For ours || (Above): For debugging
  # --rename_map='{"observation.images.image": "observation.images.image", "observation.images.image2": "observation.images.wrist_image"}'


