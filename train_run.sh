#!/bin/bash

# 设置 CUDA_VISIBLE_DEVICES 环境变量
export CUDA_VISIBLE_DEVICES=3

# 设置本体类型和子任务
robot="panda_wristcam" # [panda_wristcam, widowxai_wristcam, xarm6_robotiq_wristcam, xarm7_robotiq_wristcam]
task="PullCubeTool-v1" # [PickCube-v1, PushCube-v1, StackCube-v1, PullCube-v1, PullCubeTool-v1, PlaceSphere-v1, LiftPegUpright-v1]

# 设置采取的策略网络
policy_type="diffusion"

# 设置训练步数
batch_size=64
steps=200_000
save_freq=20_000

# panda_wristcam数据集
repo_id="AllTasks-v3/${robot}"
root_dir="/data1/jibaixu/datasets/Boundless/lerobot/${robot}"
job_name="${robot}_${task}_${policy_type}_${steps}_steps_b${batch_size}"
output_dir="/data1/jibaixu/checkpoints/AllTasks-v3/${job_name}"

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
    --wandb.project="AllTasks-v3" \
    --wandb.disable_artifact=True \
    --job_name="$job_name" \
    --output_dir="$output_dir"
