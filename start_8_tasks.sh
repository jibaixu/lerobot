#!/bin/bash

version="v4"

# 训练参数
lr=3e-4
batch_size=64
steps=200_000
save_freq=20_000
policy_type="diffusion"

vision_backbone="resnet18" # ['resnet18', 'vit_b_16']
vbckpt_type="scratch" # ['scratch', 'pretrained']

declare -A vbckpt_paths
vbckpt_paths[resnet18]="/path/to/checkpoints/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
vbckpt_paths[vit_b_16]="/path/to/checkpoints/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth"

# 创建日志目录
log_dir="logs/train_8_tasks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

echo "=========================================="
echo "启动训练任务"
echo "日志目录: $log_dir"
echo "=========================================="

# 定义8个具体任务
tasks=(
    "robot:env:0"
)

# 启动所有任务
for task_info in "${tasks[@]}"; do
    IFS=':' read -r robot task gpu_id <<< "$task_info"
    
    repo_id="AllTasks-${version}/${robot}"
    root_dir="/path/to/data/AllTasks-${version}/${robot}"
    timestamp=$(date +%Y%m%d_%H%M%S)

    # 根据use_group_norm自动设置ckpt_type和pretrained_backbone_weights
    if [[ "$vbckpt_type" == "scratch" ]]; then
        use_group_norm=True
        pretrained_backbone_weights=""
    elif [[ "$vbckpt_type" == "pretrained" ]]; then
        use_group_norm=False
        pretrained_backbone_weights="${vbckpt_paths[$vision_backbone]}"
    fi

    job_name="${robot}_${task}_${policy_type}_${vision_backbone}_${vbckpt_type}_${steps}_steps_lr${lr}_b${batch_size}_${timestamp}"
    output_dir="outputs/train/AllTasks-${version}/${job_name}"
    log_file="$log_dir/${robot}_${task}_gpu${gpu_id}_${vision_backbone}_${vbckpt_type}_lr${lr}.log"
    
    echo "启动训练: $robot - $task (GPU $gpu_id)"
    
    # 检查数据集是否存在
    if [[ ! -d "$root_dir" ]]; then
        echo "❌ 数据集不存在: $root_dir"
        continue
    fi
    
    # 启动训练任务（后台运行）
    CUDA_VISIBLE_DEVICES=$gpu_id python -m lerobot.scripts.train \
        --dataset.repo_id="$repo_id" \
        --dataset.root="$root_dir" \
        --policy.type="$policy_type" \
        --policy.vision_backbone="$vision_backbone" \
        --policy.use_group_norm=$use_group_norm \
        --policy.pretrained_backbone_weights="$pretrained_backbone_weights" \
        --policy.optimizer_lr=$lr \
        --policy.push_to_hub=False \
        --batch_size=$batch_size \
        --steps=$steps \
        --save_freq=$save_freq \
        --wandb.enable=True \
        --wandb.project="AllTasks-${version}" \
        --wandb.mode="offline" \
        --wandb.disable_artifact=True \
        --job_name="$job_name" \
        --output_dir="$output_dir" \
        > "$log_file" 2>&1 &
    
    pid=$!
    echo "✓ 任务已启动: PID $pid, 日志: $log_file"
    
    # 短暂延迟
    sleep 2
done

echo ""
echo "=========================================="
echo "所有任务已启动完成！"
echo "=========================================="
