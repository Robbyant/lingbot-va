# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg
import os

va_libero_train_cfg = EasyDict(__name__='Config: VA libero train')
va_libero_train_cfg.update(va_libero_cfg)

# Use LeRobot latent dataset (default in train.py when dataset_format != "arms").
va_libero_train_cfg.dataset_format = "lerobot"

# Parent directory that *contains* one or more LeRobot dataset roots (each has meta/info.json).
# Latents must live under <dataset_repo>/latents/... mirroring videos (see README).
va_libero_train_cfg.dataset_path = "/path/to/lerobot_datasets_parent"
# Any one repo root is fine; create e.g. `torch.zeros_like(sample_text_emb)` once you have a .pth.
va_libero_train_cfg.empty_emb_path = "/path/to/some_lerobot_repo/empty_emb.pt"

va_libero_train_cfg.wan22_pretrained_model_name_or_path = "/root/checkpoints/lingbot-va-base"

va_libero_train_cfg.enable_wandb = True
# Single-GPU / ROCm: avoid DataLoader workers loading CUDA tensors in forked children.
va_libero_train_cfg.load_worker = 0
va_libero_train_cfg.save_interval = 200
va_libero_train_cfg.gc_interval = 50
va_libero_train_cfg.cfg_prob = 0.1

# Training parameters
va_libero_train_cfg.learning_rate = 1e-5
va_libero_train_cfg.beta1 = 0.9
va_libero_train_cfg.beta2 = 0.95
va_libero_train_cfg.weight_decay = 1e-1
va_libero_train_cfg.warmup_steps = 10
va_libero_train_cfg.batch_size = 1 
va_libero_train_cfg.gradient_accumulation_steps = 10
va_libero_train_cfg.num_steps = 5000