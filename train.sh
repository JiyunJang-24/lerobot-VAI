# IMPORTANT: Set `--dataset.root` to your lerobot-formatted dataset path.
# `--dataset.repo_id`: the two sub-datasets forming the islands; the example below uses camera positions 40%→40% and 60%→60% (Diffusion Policy diversity setting 1).
# Note: You may need to log in to Weights & Biases (wandb) if enabled.
#   --dataset.repo_id=[xyg_20_10_15.0_65.0/v-0.400-0.400_num1,xyg_20_10_15.0_65.0/v-0.600-0.600_num5] \
  # --dataset.repo_id=[xyg_10_10_0.0_0.0/v-1.000-1.000_num1,xyg_10_10_0.0_0.0/v-1.000-1.000_num5,xyg_10_10_45.0_45.0/v-1.000-1.000_num1,xyg_10_10_45.0_45.0/v-1.000-1.000_num5,xyg_10_10_90.0_90.0/v-1.000-1.000_num1,xyg_10_10_90.0_90.0/v-1.000-1.000_num5,xyg_10_10_135.0_135.0/v-1.000-1.000_num1,xyg_10_10_135.0_135.0/v-1.000-1.000_num5,xyg_10_10_225.0_225.0/v-1.000-1.000_num1,xyg_10_10_225.0_225.0/v-1.000-1.000_num5,xyg_10_10_270.0_270.0/v-1.000-1.000_num1,xyg_10_10_270.0_270.0/v-1.000-1.000_num5,xyg_10_10_315.0_315.0/v-1.000-1.000_num1,xyg_10_10_315.0_315.0/v-1.000-1.000_num5] \

CUDA_VISIBLE_DEVICES=1 python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id="[v-1.000-1.000_num1]" \
  --dataset.root="/data1/local/lerobot-VAI/data/libero_spatial_lerobot_v30" \
  --policy.type="smolvla" \
  --policy.push_to_hub=false \
  --steps=30000 \
  --save_freq=5 \
  --batch_size=4 \
  --wandb.enable=false \
  --wandb.project="debug" \
  --wandb.disable_artifact=true \
  --wandb.entity="DynamicVLA" \
  --job_name="DP_wrist_goal_0_angle_from_0" \
  --num_workers=0 \
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# --wandb.project=smolVLA_wrist_libero_goal \