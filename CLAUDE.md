# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a state-of-the-art machine learning library for real-world robotics in PyTorch. It provides models, datasets, and tools for imitation learning and reinforcement learning on robotic tasks. The library supports both simulation environments and real-world robot control.

## Development Commands

### Installation and Setup
```bash
# Clone repository
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Create virtual environment (Python 3.10 required)
conda create -y -n lerobot python=3.10
conda activate lerobot

# Install ffmpeg
conda install ffmpeg -c conda-forge

# Install LeRobot in development mode
pip install -e .

# Install with specific simulation environments
pip install -e ".[aloha, pusht, xarm]"

# For development with tests and linting
pip install -e ".[dev, test]"
```

### Common Development Tasks
```bash
# Run tests
pytest tests/

# Run end-to-end tests (requires GPU for full testing)
make test-end-to-end DEVICE=cuda  # or DEVICE=cpu

# Code formatting and linting (using ruff)
ruff check src/
ruff format src/

# Type checking
# No explicit type checker configured - use ruff for general linting

# Build documentation
pip install -e ".[docs]"
doc-builder build lerobot docs/source/ --build_dir ~/tmp/test-build
doc-builder preview lerobot docs/source/  # Preview at localhost:3000
```

### Training and Evaluation
```bash
# Train a policy
python -m lerobot.scripts.train \
    --policy.type=diffusion \
    --dataset.repo_id=lerobot/pusht \
    --batch_size=64 \
    --steps=200000 \
    --wandb.enable=true

# Train with custom configuration
./train_run.sh  # Uses predefined training configuration

# Evaluate a trained policy
python -m lerobot.scripts.eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.n_episodes=10

# Evaluate local checkpoint
python -m lerobot.scripts.eval \
    --policy.path=outputs/train/path/to/checkpoints/last/pretrained_model
```

### Dataset Operations
```bash
# Visualize dataset
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht \
    --episode-index 0

# Visualize local dataset
python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir \
    --local-files-only 1 \
    --episode-index 0

# Load and inspect dataset in Python
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('lerobot/pusht')
print(f'Dataset length: {len(dataset)}')
print(f'First sample keys: {list(dataset[0].keys())}')
"
```

## Architecture Overview

### Core Components

**Policies** (`src/lerobot/policies/`):
- **ACT** (`act/`): Action Chunking Transformer for bimanual manipulation
- **Diffusion** (`diffusion/`): Diffusion-based policy learning
- **TDMPC** (`tdmpc/`): Temporal Difference Model Predictive Control
- **VQ-BeT** (`vqbet/`): Vector Quantized Behavior Transformer
- **SmolVLA** (`smolvla/`): Small Vision-Language-Action model
- **Pi0** (`pi0/`): Transformer-based policy model

**Datasets** (`src/lerobot/datasets/`):
- `lerobot_dataset.py`: Core dataset class with temporal indexing support
- `transforms.py`: Image and data transformations
- `utils.py`: Dataset utilities and statistics computation
- Supports both simulation datasets and real-world robot demonstrations

**Robots** (`src/lerobot/robots/`):
- **Koch**: Bimanual manipulation robot
- **SO100/SO101**: Follower robot configurations  
- **LeKiwi**: Mobile manipulation platform
- **Stretch3**: Hello Robot Stretch platform
- **ViperX**: Robot arm configuration

**Environments** (`src/lerobot/envs/`):
- Integration with Gymnasium environments
- Support for ALOHA, PushT, and XArm simulation environments
- Real-world robot environment wrappers

### Configuration System

LeRobot uses a structured configuration system with:
- **Default configs** in `src/lerobot/configs/`
- **YAML-based policy configurations** for reproducible training
- **Draccus** for command-line argument parsing and config management
- **Environment-specific configs** for robot hardware integration

### Key Data Flow

1. **Dataset Loading**: `LeRobotDataset` loads data from HuggingFace Hub or local storage
2. **Temporal Indexing**: Supports `delta_timestamps` for multi-frame observations
3. **Policy Training**: Policies consume observation-action pairs with temporal context
4. **Environment Integration**: Trained policies can be deployed in simulation or real robots

### Testing Infrastructure

- **Unit tests** in `tests/` directory using pytest
- **End-to-end tests** via Makefile for full training/evaluation pipelines
- **Simulation tests** for policy training on different environments
- **Hardware tests** for robot motor and camera integration

## File Organization Patterns

```
src/lerobot/
├── policies/           # All policy implementations
│   ├── [policy_name]/
│   │   ├── configuration_[policy_name].py  # Config dataclass
│   │   └── modeling_[policy_name].py       # Model implementation
├── datasets/           # Dataset loading and processing
├── robots/             # Robot hardware interfaces
├── envs/              # Environment wrappers
├── scripts/           # Training, evaluation, and utility scripts
├── utils/             # Shared utilities
└── configs/           # Configuration files
```

## Development Guidelines

### Adding New Policies
1. Create directory in `src/lerobot/policies/[policy_name]/`
2. Implement `configuration_[policy_name].py` with config dataclass
3. Implement `modeling_[policy_name].py` with model class
4. Update `available_policies` in `src/lerobot/__init__.py`
5. Add tests in `tests/policies/`

### Adding New Environments
1. Update `available_tasks_per_env` in `src/lerobot/__init__.py`
2. Add environment wrapper in `src/lerobot/envs/`
3. Update policy compatibility in `available_policies_per_env`

### Dataset Format
LeRobot uses a specific dataset format with:
- **Parquet files** for episode data in `data/chunk-*/`
- **MP4 videos** for camera observations in `videos/`
- **Metadata** in `meta/` (info.json, tasks.jsonl, episodes.jsonl, stats.json)
- **Temporal indexing** support for multi-frame observations

## Important Notes

### Dependencies
- **Python 3.10+** required
- **PyTorch 2.2.1+** for model training
- **FFmpeg** for video processing (install via conda)
- **Rerun SDK 0.21-0.22** for dataset visualization
- Optional robot-specific dependencies (dynamixel, realsense, etc.)

### GPU Support
- Most policies benefit from GPU acceleration
- Use `--policy.device=cuda` for training and evaluation
- Memory requirements vary by policy (ACT ~4GB, Diffusion ~8GB)

### WandB Integration
- Login with `wandb login` for experiment tracking
- Enable with `--wandb.enable=true` in training commands
- Supports offline mode with `--wandb.mode=offline`

### Known Issues
- ManiSkill compatibility requires specific rerun-sdk version constraints
- Some policy configs may have outdated `local_files_only` fields in cached configs
- FFmpeg libsvtav1 encoder compatibility varies by platform