#!/bin/bash

# 设置采取的策略网络
# export policy_type="diffusion"
export policy_type="act"

# panda_wristcam数据集
# export repo_id="MultiTask-v1/panda_wristcam"
# export root_dir="/data1/jibaixu/datasets/ManiSkill/MultiTask-v1/panda_wristcam"
# export job_name="panda_wristcam_${policy_type}"
# export output_dir="outputs/train/panda_wristcam_${policy_type}"

# xarm6_robotiq_wristcam数据集
export repo_id="MultiTask-v1/xarm6_robotiq_wristcam"
export root_dir="/data1/jibaixu/datasets/ManiSkill/MultiTask-v1/xarm6_robotiq_wristcam"
export job_name="xarm6_robotiq_wristcam_${policy_type}"
export output_dir="outputs/train/xarm6_robotiq_wristcam_${policy_type}"

# 设置 CUDA_VISIBLE_DEVICES 环境变量
export CUDA_VISIBLE_DEVICES=1

# 运行训练脚本
# Args:
#     wandb.disable_artifact: 是否禁用 WandB 远程存储checkpoints功能
#     job_name: 可作为wandb的name配置(run记录名称)
python -m lerobot.scripts.train \
    --dataset.repo_id="$repo_id" \
    --dataset.root="$root_dir" \
    --policy.type="$policy_type" \
    --policy.push_to_hub=False \
    --wandb.enable=True \
    --wandb.project="lerobot" \
    --wandb.disable_artifact=True \
    --job_name="$job_name" \
    --output_dir="$output_dir"
