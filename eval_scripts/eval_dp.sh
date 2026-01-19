source .venv/bin/activate

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}/.."
export PYTHONPATH="${REPO_ROOT}/LIBERO:${PYTHONPATH}"

# Do avoid EGL device display error
export MUJOCO_GL=osmesa

python3 /root/Desktop/workspace/jiyun/lerobot-VAI/src/lerobot/scripts/lerobot_eval_for_dp.py \
  --policy.path=/root/Desktop/workspace/jiyun/lerobot-VAI/outputs/train/2026-01-18/03-22-52_smolvla_basis_concat_spatial_except_2_4_angle_from_0/checkpoints/100000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --job_name=smolvla_basis_concat_spatial_0_1_3_5_6_7_8_9_angle_from_0 \
  --rename_map='{"observation.images.image": "observation.image", "observation.images.image2": "observation.wrist_image"}'
  # (Below): For ours || (Above): For debugging
  # --rename_map='{"observation.images.image": "observation.images.image", "observation.images.image2": "observation.images.wrist_image"}'

python3 /root/Desktop/workspace/jiyun/lerobot-VAI/src/lerobot/scripts/lerobot_eval_for_dp.py \
  --policy.path=/root/Desktop/workspace/jiyun/lerobot-VAI/outputs/train/2026-01-18/03-23-25_smolvla_basis_spatial_except_2_4_angle_from_0/checkpoints/100000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --job_name=smolvla_basis_spatial_0_1_3_5_6_7_8_9_angle_from_0 \
  --rename_map='{"observation.images.image": "observation.image", "observation.images.image2": "observation.wrist_image"}'
  # (Below): For ours || (Above): For debugging
  # --rename_map='{"observation.images.image": "observation.images.image", "observation.images.image2": "observation.images.wrist_image"}'

python3 /root/Desktop/workspace/jiyun/lerobot-VAI/src/lerobot/scripts/lerobot_eval_for_dp.py \
  --policy.path=/root/Desktop/workspace/jiyun/lerobot-VAI/outputs/train/2026-01-18/03-22-03_smolvla_vanilla_spatial_except_2_4_angle_from_0/checkpoints/100000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --job_name=smolvla_vanilla_spatial_0_1_3_5_6_7_8_9_angle_from_0 \
  --rename_map='{"observation.images.image": "observation.image", "observation.images.image2": "observation.wrist_image"}'
  # (Below): For ours || (Above): For debugging
  # --rename_map='{"observation.images.image": "observation.images.image", "observation.images.image2": "observation.images.wrist_image"}'

python3 /root/Desktop/workspace/jiyun/lerobot-VAI/src/lerobot/scripts/lerobot_eval_for_dp.py \
  --policy.path=/root/Desktop/workspace/jiyun/lerobot-VAI/outputs/train/2026-01-18/03-37-11_smolvla_basis_vision_frozen_spatial_except_2_4_angle_from_0/checkpoints/100000/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=20 \
  --eval.batch_size=1 \
  --job_name=smolvla_basis_vision_frozen_spatial_0_1_3_5_6_7_8_9_angle_from_0 \
  --rename_map='{"observation.images.image": "observation.image", "observation.images.image2": "observation.wrist_image"}'
  # (Below): For ours || (Above): For debugging
  # --rename_map='{"observation.images.image": "observation.images.image", "observation.images.image2": "observation.images.wrist_image"}'





