# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from easydict import EasyDict

from .va_arms_cfg import va_arms_cfg


va_arms_train_cfg = EasyDict(__name__="Config: VA arms train")
va_arms_train_cfg.update(va_arms_cfg)

# dataset root containing meta/data/videos/latents/empty_emb.pt
va_arms_train_cfg.dataset_path = os.environ.get("ARMS_LEROBOT_PATH", "/path/to/arms_lerobot")
va_arms_train_cfg.empty_emb_path = os.path.join(va_arms_train_cfg.dataset_path, "empty_emb.pt")

# logging / io
va_arms_train_cfg.enable_wandb = False
# NOTE: After FSDP/CUDA init, fork-based DataLoader workers can deadlock on some stacks.
# Keep this small (0 is safest for "first run green"); increase only if you validate spawn/fork safety.
va_arms_train_cfg.load_worker = 0

# Dataset construction uses a multiprocessing Pool inside `MultiLatentLeRobotDataset`.
# Default used to be 128 workers which looks like "hundreds of train.py processes" in pgrep.
va_arms_train_cfg.dataset_init_workers = 16
va_arms_train_cfg.save_interval = 200
va_arms_train_cfg.gc_interval = 50
va_arms_train_cfg.cfg_prob = 0.1

# training parameters (MI300X 单卡可以适当加大 batch/accum)
va_arms_train_cfg.learning_rate = 1e-5
va_arms_train_cfg.beta1 = 0.9
va_arms_train_cfg.beta2 = 0.95
va_arms_train_cfg.weight_decay = 1e-1
va_arms_train_cfg.warmup_steps = 50
va_arms_train_cfg.batch_size = 1
va_arms_train_cfg.gradient_accumulation_steps = 8
va_arms_train_cfg.num_steps = 10000

# loss weights (两阶段训练用)
# 阶段1（动作优先）：latent_loss_weight=0, action_loss_weight=1
# 阶段2（联合）：latent_loss_weight=1, action_loss_weight=1
va_arms_train_cfg.latent_loss_weight = 0.0
va_arms_train_cfg.action_loss_weight = 1.0

