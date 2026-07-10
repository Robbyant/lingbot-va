# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robomme_train_cfg import va_robomme_train_cfg

va_robomme_debug_cfg = EasyDict(__name__="Config: VA RoboMME debug")
va_robomme_debug_cfg.update(va_robomme_train_cfg)

# === Paths (HPC3) ===
va_robomme_debug_cfg.wan22_pretrained_model_name_or_path = "/data/user/czi447/lingbot-va-model"
va_robomme_debug_cfg.dataset_path = "/data/user/czi447/robomme_lerobot"
va_robomme_debug_cfg.empty_emb_path = "/data/user/czi447/robomme_lerobot/empty_emb.pt"
va_robomme_debug_cfg.save_root = "/data/user/czi447/output/robomme_debug"

# === Debug: minimal steps, no wandb ===
va_robomme_debug_cfg.num_steps = 5
va_robomme_debug_cfg.save_interval = 999999
va_robomme_debug_cfg.gradient_accumulation_steps = 1
va_robomme_debug_cfg.enable_wandb = False
va_robomme_debug_cfg.load_worker = 2
