# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_arms_cfg = EasyDict(__name__="Config: VA arms (dual-arm, single-cam)")
va_arms_cfg.update(va_shared_cfg)

# dataset_format is used by wan_va/train.py to select dataset implementation.
va_arms_cfg.dataset_format = "arms"

# Single front-facing camera in ./arms videos.
va_arms_cfg.env_type = "arms"
va_arms_cfg.obs_cam_keys = ["observation.images.cam_high"]

# Video size used for VAE latent extraction / training.
va_arms_cfg.height = 256
va_arms_cfg.width = 256

# Transformer temporal settings (match your data / sampling later if needed).
va_arms_cfg.attn_window = 72
va_arms_cfg.frame_chunk_size = 4

# Training-time clip length control.
# ArmsLatentDataset will, by default, randomly crop a fixed number of *latent frames*
# from each episode to keep sequence length bounded (helps avoid OOM on long episodes).
# Set to None / 0 to disable cropping and use full episodes.
va_arms_cfg.train_latent_frames = 24

# FlowMatch schedulers
va_arms_cfg.snr_shift = 5.0
va_arms_cfg.action_snr_shift = 1.0

# Action format follows repo README "30 dims" standard. For release we mostly fill joint channels.
va_arms_cfg.action_dim = 30
va_arms_cfg.action_per_frame = 4

# Use dual-arm channels (16 total) like robotwin, but semantics come from your mapping.
# We keep the same idea: 7 + 1 + 7 + 1 = 16 channels selected from 30-dim action.
# Here we use JOINT channels: left joints [14:21), right joints [21:28), and grippers [28,29].
va_arms_cfg.used_action_channel_ids = list(range(14, 21)) + [28] + list(range(21, 28)) + [29]
inverse_used_action_channel_ids = [len(va_arms_cfg.used_action_channel_ids)] * va_arms_cfg.action_dim
for i, j in enumerate(va_arms_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_arms_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# Placeholder stats; ArmsLatentDataset will prefer <dataset_path>/norm_stat.json if present.
va_arms_cfg.action_norm_method = "quantiles"
va_arms_cfg.norm_stat = {"q01": [0.0] * 30, "q99": [1.0] * 30}

