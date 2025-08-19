"""Validation of maniskill for different tasks

"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional
import gymnasium as gym
import numpy as np
import torch
import tyro
import math
import json
import os

from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
import mani_skill.examples.benchmarking.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper # import benchmark env code
from gymnasium.vector.async_vector_env import AsyncVectorEnv

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy


BENCHMARK_ROBOTS = ["panda_wristcam", "widowxai_wristcam", "xarm6_robotiq_wristcam", "xarm7_robotiq_wristcam"]
BENCHMARK_ENVS = ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PullCubeTool-v1", "PlaceSphere-v1", "LiftPegUpright-v1",]
ENV_MAXSTEP_MAP = {
    "PickCube-v1": 500,
    "PushCube-v1": 500,
    "StackCube-v1": 500,
    "PullCube-v1": 500,
    "PullCubeTool-v1": 800,
    "PlaceSphere-v1": 500,
    "LiftPegUpright-v1": 700,
}


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@dataclass
class EvalConfig:
    pretrained_policy_path: str = "/home/jibaixu/projects/lerobot/outputs/train/panda_diffusion_pullcube/checkpoints/020000/pretrained_model"
    resize_size: int = 224
    replan_steps: int = 5
    """Environment ID"""
    robot_uids: str = "panda_wristcam"   # ["panda_wristcam", "widowxai_wristcam", "xarm6_robotiq_wristcam", "xarm7_robotiq_wristcam"]
    env_uids: str = "PullCubeTool-v1"    # ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1", "PullCubeTool-v1", "PlaceSphere-v1", "LiftPegUpright-v1"]
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_ee_delta_pose"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    cpu_sim: bool = True
    """Whether to use the CPU or GPU simulation"""
    seed: int = 0
    save_example_image: bool = False
    control_freq: Optional[int] = 60
    sim_freq: Optional[int] = 120
    num_cams: Optional[int] = None
    """Number of cameras. Only used by benchmark environments"""
    cam_width: Optional[int] = None
    """Width of cameras. Only used by benchmark environments"""
    cam_height: Optional[int] = None
    """Height of cameras. Only used by benchmark environments"""
    render_mode: str = "rgb_array"
    """Which set of cameras/sensors to render for video saving. 'cameras' value will save a video showing all sensor/camera data in the observation, e.g. rgb and depth. 'rgb_array' value will show a higher quality render of the environment running."""
    save_video: bool = True
    """Whether to save videos"""
    save_results: Optional[str] = None
    """Path to save results to. Should be path/to/results.csv"""
    save_path: str = "outputs/eval/AllTasks-v3/panda_wristcam_pullcubetool-v1"
    shader: str = "default"
    num_per_task: int = 50

    def __post_init__(self):
        assert self.robot_uids in BENCHMARK_ROBOTS, f"{self.robot_uids} is not a valid robot uid."
        assert self.env_uids in BENCHMARK_ENVS, f"{self.env_uids} is not a valid env uid."
        assert self.cpu_sim, "CPU simulation is required for evaluation."


def main(args: EvalConfig):
    os.makedirs(args.save_path, exist_ok=True)
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq
    
    if "diffusion" in args.pretrained_policy_path:
        policy = DiffusionPolicy.from_pretrained(args.pretrained_policy_path)
    elif "act" in args.pretrained_policy_path:
        policy = ACTPolicy.from_pretrained(args.pretrained_policy_path)

    kwargs = dict()

    total_successes = 0.0
    success_dict = {
        "pretrained_policy_path": args.pretrained_policy_path,
        "robot_uids": args.robot_uids,
        "env_uids": args.env_uids if args.env_uids else BENCHMARK_ENVS,
        "num_per_task": ENV_MAXSTEP_MAP,
        "env_success_rate": {},
        "total_success_rate": 0.0,
    }

    # 根据task参数决定评估哪些任务
    if args.env_uids:
        eval_envs = [args.env_uids]
    else:
        eval_envs = BENCHMARK_ENVS

    for env_id in eval_envs:
        if args.cpu_sim:
            def make_env():
                def _init():
                    env = gym.make(env_id,
                                obs_mode=args.obs_mode,
                                sim_config=sim_config,
                                robot_uids=args.robot_uids,
                                sensor_configs=dict(shader_pack=args.shader),
                                human_render_camera_configs=dict(shader_pack=args.shader),
                                viewer_camera_configs=dict(shader_pack=args.shader),
                                render_mode=args.render_mode,
                                control_mode=args.control_mode,
                                **kwargs)
                    env = CPUGymWrapper(env, )
                    return env
                return _init
            # mac os system does not work with forkserver when using visual observations
            env = AsyncVectorEnv([make_env() for _ in range(num_envs)], context="forkserver" if sys.platform == "darwin" else None) if args.num_envs > 1 else make_env()()
            base_env = make_env()().unwrapped

        base_env.print_sim_details()
        
        task_successes = 0.0
        for seed in range(args.num_per_task):
            images = []
            video_nrows = int(np.sqrt(num_envs))
            with torch.inference_mode():
                env.reset(seed=seed+2025)
                env.step(env.action_space.sample())  # warmup step
                obs, info = env.reset(seed=seed+2025)
                if args.save_video:
                    images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())
                    # images.append(obs["sensor_data"]["third_view_camera"]["rgb"].cpu().numpy())

                step_length = ENV_MAXSTEP_MAP[env_id]
                N = step_length     # LeRobot 中的 policy 内部维护的 queue 队列会自己完成 replan

                with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
                    for i in range(N):
                        if args.cpu_sim:
                            img = np.ascontiguousarray(obs["sensor_data"]["third_view_camera"]["rgb"])
                            wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"])
                            state = np.expand_dims(
                                        np.concatenate(
                                                (
                                                    obs["extra"]["tcp_pose"],
                                                    obs["agent"]["qpos"][-1:],
                                                )
                                            ),
                                        axis=0,
                                    )
                            
                            # Prepare observation for the policy running in Pytorch
                            img         = torch.from_numpy(img)
                            wrist_img   = torch.from_numpy(wrist_img)
                            state       = torch.from_numpy(state)

                            # Convert to float32 with image from channel first in [0,255]
                            # to channel last in [0,1]
                            img = (img.to(torch.float32) / 255).permute(2, 0, 1)
                            wrist_img = (wrist_img.to(torch.float32) / 255).permute(2, 0, 1)
                            state = state.to(torch.float32)

                            # Send data tensors from CPU to GPU
                            device = "cuda"
                            img         = img.to(device, non_blocking=True)
                            wrist_img   = wrist_img.to(device, non_blocking=True)
                            state       = state.to(device, non_blocking=True)

                            # Add extra (empty) batch dimension, required to forward the policy
                            img = img.unsqueeze(0)
                            wrist_img = wrist_img.unsqueeze(0)

                            element = {
                                "observation.images.image": img,
                                "observation.images.wrist_image": wrist_img,
                                "observation.state": state,
                            }

                        action = policy.select_action(element)
                        numpy_action = action.squeeze(0).to("cpu").numpy()

                        print(f"Step {i+1}/{N}, action: {numpy_action}")
                        if args.robot_uids == "widowxai_wristcam":     # widowxai 本体在训练时动作维是7，但在评估时模拟环境中需要8维
                            numpy_action = np.append(numpy_action, numpy_action[-1])

                        obs, rew, terminated, truncated, info = env.step(numpy_action)
                        if args.save_video:
                            images.append(np.expand_dims(env.render(), axis=0)) if args.cpu_sim else images.append(env.render().cpu().numpy())
                            # images.append(obs["sensor_data"]["third_view_camera"]["rgb"].cpu().numpy())
                        terminated = terminated if args.cpu_sim else terminated.item()
                        if terminated:
                            task_successes += 1
                            total_successes += 1

                        if terminated:
                            break
                profiler.log_stats("env.step")

                if args.save_video:
                    images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
                    images_to_video(
                        images,
                        output_dir=args.save_path,
                        video_name=f"{env_id}-{seed}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}--success={terminated}",
                        fps=30,
                    )
                    del images
        env.close()
        print(f"Task Success Rate: {task_successes / args.num_per_task}")
        success_dict["env_success_rate"][env_id] = task_successes / args.num_per_task

    print(f"Total Success Rate: {total_successes / (args.num_per_task * len(eval_envs))}")
    success_dict['total_success_rate'] = total_successes / (args.num_per_task * len(eval_envs))
    with open(f"{args.save_path}/success_dict.json", "w") as f:
        json.dump(success_dict, f)
    

if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
