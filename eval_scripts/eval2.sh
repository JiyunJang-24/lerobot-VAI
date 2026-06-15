#!/bin/bash

VENV_PATH="venv-lerobot-eval"

if [ ! -d "${VENV_PATH}" ]; then
  echo "Error: ${VENV_PATH} virtual environment folder not found."
  exit 1
fi

source ${VENV_PATH}/bin/activate

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}/.."
export PYTHONPATH="${REPO_ROOT}/third_party:${REPO_ROOT}/third_party/LIBERO/libero:${PYTHONPATH}"

# Do avoid EGL device display error
export MUJOCO_GL=osmesa

# # Evaluate a policy on the LIBERO benchmark
(
python3 /data1/local/lerobot-VAI/src/lerobot/scripts/lerobot_eval.py \
  --policy.path=/data1/local/lerobot-VAI/outputs/train/2026-01-22/02-51-56_smolvla_vanilla_object_expert_only/checkpoints/100000/pretrained_model \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --job_name=smolvla_object_vanilla_expert_only_100000 \
  --rename_map='{"observation.images.image": "observation.image", "observation.images.image2": "observation.wrist_image"}'
) &
(
python3 /data1/local/lerobot-VAI/src/lerobot/scripts/lerobot_eval.py \
  --policy.path=/data1/local/lerobot-VAI/outputs/train/2026-01-22/02-52-08_smolvla_vanilla_10_expert_only/checkpoints/100000/pretrained_model \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --job_name=smolvla_10_vanilla_expert_only_100000 \
  --rename_map='{"observation.images.image": "observation.image", "observation.images.image2": "observation.wrist_image"}'
)