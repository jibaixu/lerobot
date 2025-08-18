#!/bin/bash

# 启动8个指定训练任务：widowxai的7个任务 + xarm6的第一个任务
# 4张卡，每张卡2个任务

# 训练参数
batch_size=64
steps=200_000
save_freq=20_000
policy_type="diffusion"

# 创建日志目录
log_dir="logs/train_8_tasks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

echo "=========================================="
echo "启动8个训练任务"
echo "日志目录: $log_dir"
echo "=========================================="

# 定义8个具体任务
tasks=(
    "widowxai_wristcam:PickCube-v1:0"
    "widowxai_wristcam:PushCube-v1:0"
    "widowxai_wristcam:StackCube-v1:1"
    "widowxai_wristcam:PullCube-v1:1"
    "widowxai_wristcam:PullCubeTool-v1:2"
    "widowxai_wristcam:PlaceSphere-v1:2"
    "widowxai_wristcam:LiftPegUpright-v1:3"
    "xarm6_robotiq_wristcam:PickCube-v1:3"
)

# 启动所有任务
for task_info in "${tasks[@]}"; do
    IFS=':' read -r robot task gpu_id <<< "$task_info"
    
    repo_id="AllTasks-v3/${robot}"
    root_dir="/data2/wts/jibaixu/lerobot/data/AllTasks-v3/${robot}"
    timestamp=$(date +%Y%m%d_%H%M%S)
    job_name="${robot}_${task}_${policy_type}_${steps}_steps_b${batch_size}_${timestamp}"
    output_dir="outputs/train/AllTasks-v3/${job_name}"
    log_file="$log_dir/${robot}_${task}_gpu${gpu_id}.log"
    
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
        --policy.push_to_hub=False \
        --batch_size=$batch_size \
        --steps=$steps \
        --save_freq=$save_freq \
        --wandb.enable=True \
        --wandb.project="AllTasks-v3" \
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
echo "所有8个任务已启动完成！"
echo ""
echo "GPU分配："
echo "  GPU 0: widowxai PickCube-v1, widowxai PushCube-v1"
echo "  GPU 1: widowxai StackCube-v1, widowxai PullCube-v1"
echo "  GPU 2: widowxai PullCubeTool-v1, widowxai PlaceSphere-v1"
echo "  GPU 3: widowxai LiftPegUpright-v1, xarm6 PickCube-v1"
echo ""
echo "监控命令："
echo "  watch -n 5 'ps aux | grep lerobot.scripts.train | grep -v grep'"
echo "  tail -f $log_dir/*.log"
echo "  nvidia-smi"
echo "=========================================="