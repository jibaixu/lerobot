#!/bin/bash

version="v4"

# 评测参数
policy_type="diffusion"
num_per_task=50

# 仅保留显卡编号
tasks=(
    "0:/path/to/output/AllTasks-version/robot_env_policy_type_vision_backbone_vbckpt_type_step_timestamp/checkpoints/step/pretrained_model"
)

# 创建日志目录
log_dir="logs/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

echo "=========================================="
echo "启动评测任务"
echo "日志目录: $log_dir"
echo "=========================================="

for task_info in "${tasks[@]}"; do
    IFS=':' read -r gpu_id ckpt_path <<< "$task_info"

    # 自动识别robot、env、step
        # 识别 robot
        if [[ "$ckpt_path" =~ /all ]]; then
            robot="all"
        else
            robot=$(echo "$ckpt_path" | grep -oE '(panda_wristcam|widowxai_wristcam|xarm6_robotiq_wristcam|xarm7_robotiq_wristcam)')
        fi
        # 识别 env
        if [[ "$ckpt_path" =~ _all_ ]]; then
            env="all"
        else
            env=$(echo "$ckpt_path" | grep -oE '(PickCube-v1|PushCube-v1|StackCube-v1|PullCube-v1|PullCubeTool-v1|PlaceSphere-v1|LiftPegUpright-v1)')
        fi
    vision_backbone=$(echo "$ckpt_path" | grep -oE '(resnet18|vitb16)')
    # vbckpt_type（仅在/checkpoints前的路径部分匹配）
    vbckpt_type=$(echo "$ckpt_path" | sed 's|/checkpoints/.*||' | grep -oE '(scratch|pretrained)')
    # step: checkpoints/040000/pretrained_model
    step=$(echo "$ckpt_path" | grep -oE 'checkpoints/[0-9]+' | awk -F'/' '{print $2}')

    timestamp=$(date +%Y%m%d_%H%M%S)
    save_path="outputs/eval/AllTasks-${version}/${robot}_${env}_${policy_type}_${vision_backbone}_${vbckpt_type}_${step}_${timestamp}"
    log_file="$log_dir/${robot}_${env}_${vision_backbone}_${vbckpt_type}_${step}_gpu${gpu_id}.log"

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
