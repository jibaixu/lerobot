#!/bin/bash

# 设置 CUDA_VISIBLE_DEVICES 环境变量
export CUDA_VISIBLE_DEVICES=5

# 设置训练步数
batch_size=16
steps=200_000
save_freq=50_000

# 设置采取的策略网络
policy_type="diffusion"
# policy_type="act"

# panda_wristcam数据集
repo_id="AllTasks-v2/panda_wristcam"
root_dir="/data1/jibaixu/datasets/ManiSkill/AllTasks-v2/panda_wristcam"
job_name="panda_wristcam_${policy_type}_${steps}_steps_b${batch_size}"
output_dir="/data1/jibaixu/checkpoints/AllTasks-v2/panda_wristcam_${policy_type}_${steps}_steps_b${batch_size}"

# xarm6_robotiq_wristcam数据集
# repo_id="AllTasks-v2/xarm6_robotiq_wristcam"
# root_dir="/data1/jibaixu/datasets/ManiSkill/AllTasks-v2/xarm6_robotiq_wristcam"
# job_name="xarm6_robotiq_wristcam_${policy_type}_${steps}_steps_b${batch_size}"
# output_dir="/data1/jibaixu/checkpoints/AllTasks-v2/xarm6_robotiq_wristcam_${policy_type}_${steps}_steps_b${batch_size}"

# 运行训练脚本
# Args:
#     wandb.disable_artifact: 是否禁用 WandB 远程存储 checkpoints 功能
#     job_name: 可作为 wandb 的 name 配置(run记录名称)
python -m lerobot.scripts.train \
    --dataset.repo_id="$repo_id" \
    --dataset.root="$root_dir" \
    --policy.type="$policy_type" \
    --policy.push_to_hub=False \
    --batch_size=$batch_size \
    --steps=$steps \
    --save_freq=$save_freq \
    --wandb.enable=True \
    --wandb.project="AllTasks-v2" \
    --wandb.disable_artifact=True \
    --job_name="$job_name" \
    --output_dir="$output_dir"
