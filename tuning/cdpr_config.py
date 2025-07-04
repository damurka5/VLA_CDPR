from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class CDPRFinetuneConfig:
    # Simulation and Dataset
    mujoco_model_path: str = "cdpr_model.xml"         # Path to MuJoCo model
    dataset_dir: Path = Path("dataset")               # Directory containing MuJoCo frames
    prompt_file: str = "prompts.json"                # File with natural language prompts
    shuffle_buffer_size: int = 100_000
    
    # RL Algorithm Selection
    rl_algorithm: str = "PPO"                        # Options: "PPO", "SAC", "DDPG", "TD3"
    use_l1_regression: bool = True                    # For continuous action spaces
    use_diffusion: bool = False                       # Alternative action modeling
    
    # RL Hyperparameters
    gamma: float = 0.99                              # Discount factor
    tau: float = 0.005                               # Target network update rate
    lr_actor: float = 3e-4                           # Actor learning rate
    lr_critic: float = 3e-4                          # Critic learning rate
    batch_size: int = 128                            # Batch size for RL updates
    buffer_size: int = 1_000_000                     # Replay buffer size
    exploration_noise: float = 0.1                   # Action noise for exploration
    policy_freq: int = 2                             # Policy update frequency (TD3)
    
    # Training Configuration
    num_episodes: int = 1000                         # Total training episodes
    max_steps_per_episode: int = 200                 # Max steps per episode
    eval_freq: int = 50                              # Evaluation frequency
    num_eval_episodes: int = 10                      # Number of evaluation episodes
    save_freq: int = 100                             # Model saving frequency
    
    # OpenVLA Configuration
    vla_path: str = "openvla/openvla-7b"
    use_lora: bool = True
    lora_rank: int = 32
    merge_lora_during_training: bool = False
    
    # Visualization and Logging
    render: bool = True                              # Render during training
    wandb_project: str = "cdpr_openvla_rl"          # WandB project name
    log_video: bool = True                          # Log evaluation videos