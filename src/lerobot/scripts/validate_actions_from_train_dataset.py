import os
import tyro
import time
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def plot_prediction_vs_groundtruth(gt: np.ndarray, pred: np.ndarray, save_path: str):
    """
    将真实值和推理值随时间变化的关系画图并保存。
    
    参数:
        gt (np.ndarray): shape 为 (n, d) 的真实值
        pred (np.ndarray): shape 为 (n, d) 的推理值
        save_path (str): 图像保存路径，如 'output/pred_vs_gt.png'
    """
    assert gt.shape == pred.shape, "gt 和 pred 的 shape 必须相同"
    n, d = gt.shape

    # 创建输出目录（如有必要）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建图像
    fig, axes = plt.subplots(d, 1, figsize=(10, 2 * d), sharex=True)

    for i in range(d):
        axes[i].plot(range(n), gt[:, i], label='Ground Truth', color='blue', linestyle='-')
        axes[i].plot(range(n), pred[:, i], label='Prediction', color='red', linestyle='--')
        axes[i].set_ylabel(f'Dim {i+1}')
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    axes[-1].set_xlabel('Time Step')

    plt.suptitle('Ground Truth vs Prediction Over Time (All Dimensions)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图像
    plt.savefig(save_path, dpi=300)
    plt.close()


@dataclass
class EvalDatasetConfig:
    N: int = 500
    #################################################################################################################
    # Model parameters
    #################################################################################################################
    policy_pretrained_path: str = "/data1/jibaixu/checkpoints/AllTasks-v2/panda_wristcam_diffusion_100_000_steps/checkpoints/100000/pretrained_model"

    #################################################################################################################
    # Dataset parameters
    #################################################################################################################
    dataset_root_path: str = "/data1/jibaixu/datasets/ManiSkill/AllTasks-v2/panda_wristcam"

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)
    output_dir: str = "outputs/eval_dataset/AllTasks-v2/panda_wristcam"


def eval_alltasks_v2(cfg: EvalDatasetConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    # Set random seed
    np.random.seed(cfg.seed)

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------ step 1: load dataset ------------
    dataset = LeRobotDataset(
        cfg.dataset_root_path,
    )

    # ------------ step 2: load pretrained policy ------------
    policy = DiffusionPolicy.from_pretrained(cfg.policy_pretrained_path)

    pred_actions = []
    actions = []
    for i in range(cfg.N):
        data = dataset[i]

        img         = data['observation.images.image'].unsqueeze(0)
        wrist_img   = data['observation.images.wrist_image'].unsqueeze(0)
        state       = data['observation.state'].unsqueeze(0)

        img         = img.to(device, non_blocking=True)
        wrist_img   = wrist_img.to(device, non_blocking=True)
        state       = state.to(device, non_blocking=True)

        element = {
            "observation.images.image": img,
            "observation.images.wrist_image": wrist_img,
            "observation.state": state,
        }
        pred_action = policy.select_action(element).squeeze(0).cpu().numpy()
        action = data['action'].numpy()
        
        pred_actions.append(pred_action)
        actions.append(action)
        
        # pred_action[pred_action[:, -1] == 0, -1] = -1
        # loss = F.mse_loss(action, pred_action, reduction="none").sum()/(pred_action.shape[0]*pred_action.shape[1])
        # print(f"loss: {loss.mean()}")
        time.sleep(1)
    
    actions = np.stack(actions, axis=0)
    pred_actions = np.stack(pred_actions, axis=0)
    plot_prediction_vs_groundtruth(actions, pred_actions, save_path=os.path.join(cfg.output_dir, f"compare_N{cfg.N}.png"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_alltasks_v2)
