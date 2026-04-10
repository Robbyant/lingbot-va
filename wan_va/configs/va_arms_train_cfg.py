# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from easydict import EasyDict

from .va_arms_cfg import va_arms_cfg

va_arms_train_cfg = EasyDict(__name__="Config: VA arms train")
va_arms_train_cfg.update(va_arms_cfg)

va_arms_train_cfg.dataset_path = "./prepared_arms"
va_arms_train_cfg.empty_emb_path = os.path.join(va_arms_train_cfg.dataset_path, "empty_emb.pt")

# Default to off: server environments often don't have WANDB_* configured.
va_arms_train_cfg.enable_wandb = False
va_arms_train_cfg.load_worker = 8
va_arms_train_cfg.save_interval = 200
va_arms_train_cfg.gc_interval = 50
va_arms_train_cfg.cfg_prob = 0.1

# Training parameters (start conservative)
va_arms_train_cfg.learning_rate = 1e-5
va_arms_train_cfg.beta1 = 0.9
va_arms_train_cfg.beta2 = 0.95
va_arms_train_cfg.weight_decay = 1e-1
va_arms_train_cfg.warmup_steps = 10
va_arms_train_cfg.batch_size = 1
va_arms_train_cfg.gradient_accumulation_steps = 10
va_arms_train_cfg.num_steps = 5000

