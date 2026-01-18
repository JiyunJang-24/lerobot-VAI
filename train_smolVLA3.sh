
# IMPORTANT: Set `--dataset.root` to your lerobot-formatted dataset path.
# `--dataset.repo_id`: the two sub-datasets forming the islands; the example below uses camera positions 40%→40% and 60%→60% (Diffusion Policy diversity setting 1).
# Note: You may need to log in to Weights & Biases (wandb) if enabled.
#   --dataset.repo_id=[xyg_20_10_15.0_65.0/v-0.400-0.400_num1,xyg_20_10_15.0_65.0/v-0.600-0.600_num5] \
  # --dataset.repo_id=[xyg_10_10_0.0_0.0/v-1.000-1.000_num1,xyg_10_10_0.0_0.0/v-1.000-1.000_num5,xyg_10_10_45.0_45.0/v-1.000-1.000_num1,xyg_10_10_45.0_45.0/v-1.000-1.000_num5,xyg_10_10_90.0_90.0/v-1.000-1.000_num1,xyg_10_10_90.0_90.0/v-1.000-1.000_num5,xyg_10_10_135.0_135.0/v-1.000-1.000_num1,xyg_10_10_135.0_135.0/v-1.000-1.000_num5,xyg_10_10_225.0_225.0/v-1.000-1.000_num1,xyg_10_10_225.0_225.0/v-1.000-1.000_num5,xyg_10_10_270.0_270.0/v-1.000-1.000_num1,xyg_10_10_270.0_270.0/v-1.000-1.000_num5,xyg_10_10_315.0_315.0/v-1.000-1.000_num1,xyg_10_10_315.0_315.0/v-1.000-1.000_num5] \

CUDA_VISIBLE_DEVICES=2 python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=[v-1.000-1.000_num1,v-1.000-1.000_num2,v-1.000-1.000_num4,v-1.000-1.000_num6,v-1.000-1.000_num7,v-1.000-1.000_num8,v-1.000-1.000_num9,v-1.000-1.000_num10] \
  --dataset.root="/home/kwonmc/jiyun/lerobot-VAI/dataset_git/libero_goal_no_noops_island_1_lerobot/DP_wrist_10_10_0.0_0.0/v3.0" \
  --policy.type="smolvla" \
  --policy.push_to_hub=false \
  --steps=100000 \
  --save_freq=5000 \
  --batch_size=64 \
  --wandb.enable=true \
  --wandb.project="libero_smolvla" \
  --wandb.disable_artifact=true \
  --wandb.entity="DynamicVLA" \
  --num_workers=16 \
  --job_name="smolvla_basis_spatial_except_2_4_angle_from_0" \
  --policy.visual_cue_mode="basis" \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion
# --wandb.project=smolVLA_wrist_libero_goal \
