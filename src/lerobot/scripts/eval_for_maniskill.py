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

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import mani_skill.examples.benchmarking.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper # import benchmark env code
from gymnasium.vector.async_vector_env import AsyncVectorEnv

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy


os.environ['CUDA_VISIBLE_DEVICES'] = "4"

BENCHMARK_ENVS = ["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1",]
INDEX = 3


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
    # TODO
    pretrained_policy_path = "/data1/jibaixu/checkpoints/AllTasks-v2/panda_wristcam_diffusion_200_000_steps_b64/checkpoints/100000/pretrained_model"
    resize_size: int = 224
    replan_steps: int = 5
    # env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = BENCHMARK_ENVS[INDEX]
    """Environment ID"""
    # TODO
    robot_uids = "panda_wristcam"   # ["panda_wristcam", "xarm6_robotiq_wristcam"]
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
    # TODO
    save_path: str = "/home/jibaixu/projects/lerobot/outputs/eval/panda_wristcam_diffusion_100_000_steps_b64"
    shader: str = "default"
    num_per_task: int = 50
    max_step_length: int = 500


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
        "num_per_task": args.num_per_task,
        "max_step_length": args.max_step_length,
        "env_success_rate": {},
        "total_success_rate": 0.0,
    }
    for env_id in BENCHMARK_ENVS:
        if not args.cpu_sim:
            env = gym.make(
                env_id,
                num_envs=num_envs,
                obs_mode=args.obs_mode,
                robot_uids=args.robot_uids,
                sensor_configs=dict(shader_pack=args.shader),
                human_render_camera_configs=dict(shader_pack=args.shader),
                viewer_camera_configs=dict(shader_pack=args.shader),
                render_mode=args.render_mode,
                control_mode=args.control_mode,
                sim_config=sim_config,
                **kwargs
            )
            if isinstance(env.action_space, gym.spaces.Dict):
                env = FlattenActionSpaceWrapper(env)
            base_env: BaseEnv = env.unwrapped
        else:
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
                if env_id == "PickCube-v1":
                    task_description = "Pick up the cube."
                    step_length = args.max_step_length
                elif env_id == "PushCube-v1":
                    task_description = "Push the cube to the target position."
                    step_length = args.max_step_length
                elif env_id == "StackCube-v1":
                    task_description = "Stack the cube on top of the other cube."
                    step_length = args.max_step_length
                else:
                    task_description = "Pull the cube to the target position."
                    step_length = args.max_step_length
                # N = step_length // args.replan_steps
                N = step_length     # LeRobot 中的 policy 内部维护的 queue 队列会自己完成 replan
                # N = 100
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
                        else:
                            #TODO: no changing to adjust dp or act with gpu_sim
                            img = np.ascontiguousarray(obs["sensor_data"]["third_view_camera"]["rgb"].cpu().numpy())
                            wrist_img = np.ascontiguousarray(obs["sensor_data"]["hand_camera"]["rgb"].cpu().numpy())
                            element = {
                                    "observation.images.image": img,
                                    "observation.images.wrist_image": wrist_img,
                                    "observation.state": np.expand_dims(
                                        np.concatenate(
                                                (
                                                    obs["extra"]["tcp_pose"][0],
                                                    obs["agent"]["qpos"][0, -1:],
                                                )
                                            ),
                                        axis=0,
                                    ),
                                    # "annotation.human.task_description": [task_description],
                            }

                        action = policy.select_action(element)
                        # action[action[:, -1] == 0, -1] = -1   # 训练时输入的 action 最后一维就是 1 或 -1
                        # pred_action = pred_action[:args.replan_steps]
                        numpy_action = action.squeeze(0).to("cpu").numpy()

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
    print(f"Total Success Rate: {total_successes / (args.num_per_task * len(BENCHMARK_ENVS))}")
    success_dict['total_success_rate'] = total_successes / (args.num_per_task * len(BENCHMARK_ENVS))
    with open(f"{args.save_path}/success_dict.json", "w") as f:
        json.dump(success_dict, f)
    

if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
