#!/bin/bash

# 评测参数
policy_type="diffusion"
num_per_task=50

# 评测任务列表（每项为: robot:env:gpu_id:ckpt_path）
tasks=(
    "panda_wristcam:PullCube-v1:2:/home/jibaixu/projects/lerobot/outputs/train/AllTasks-v3/panda_wristcam_PullCube-v1_diffusion_200_000_steps_b64_20250818_214122/checkpoints/040000/pretrained_model"
)

# 创建日志目录
log_dir="logs/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

echo "=========================================="
echo "启动评测任务"
echo "日志目录: $log_dir"
echo "=========================================="

for task_info in "${tasks[@]}"; do
    IFS=':' read -r robot env gpu_id ckpt_path <<< "$task_info"
    timestamp=$(date +%Y%m%d_%H%M%S)
    save_path="outputs/eval/AllTasks-v3/${robot}_${env}_${policy_type}_${timestamp}"
    log_file="$log_dir/${robot}_${env}_gpu${gpu_id}.log"

    echo "启动评测: $robot - $env (GPU $gpu_id)"
    CUDA_VISIBLE_DEVICES=$gpu_id python -m lerobot.scripts.eval_for_maniskill \
        --pretrained_policy_path="$ckpt_path" \
        --robot_uids="$robot" \
        --env_uids="$env" \
        --num_per_task=$num_per_task \
        --save_path="$save_path" \
        > "$log_file" 2>&1 &

    pid=$!
    echo "✓ 评测已启动: PID $pid, 日志: $log_file"
    sleep 2
done

echo ""
echo "=========================================="
echo "所有评测任务已启动完成！"
echo "=========================================="
