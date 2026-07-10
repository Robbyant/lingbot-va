# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg


va_arms_cfg = EasyDict(__name__="Config: VA arms")
va_arms_cfg.update(va_shared_cfg)

# will be overridden by CLI / script
va_arms_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_VA_BASE_PATH", "/path/to/lingbot-va-base"
)

# inference window / chunking
# 16G 显存优先保证可跑通：更小窗口
va_arms_cfg.attn_window = 16
# 16G 显存优先保证可跑通：更小 chunk
va_arms_cfg.frame_chunk_size = 2
va_arms_cfg.env_type = "none"

# We encode videos to 256x256 latents for training; match inference resize.
va_arms_cfg.height = 256
va_arms_cfg.width = 256

# lingbot-va-base 的 transformer action_embedder 固定吃 30 维动作 token。
# arms 实际只有 26 维；我们在脚本侧把输入 state/action padding 到 29 维，
# 再通过 VA_Server.preprocess_action 内部的 +1 padding 变成 30 维喂给模型。
va_arms_cfg.action_dim = 30

# one action vector per frame
va_arms_cfg.action_per_frame = 1

# single front camera
va_arms_cfg.obs_cam_keys = ["observation.images.front"]

va_arms_cfg.guidance_scale = 1
va_arms_cfg.action_guidance_scale = 1

va_arms_cfg.num_inference_steps = 10
va_arms_cfg.video_exec_step = -1
# 更少扩散步数（先跑通，再逐步加）
va_arms_cfg.action_num_inference_steps = 10

va_arms_cfg.snr_shift = 5.0
va_arms_cfg.action_snr_shift = 1.0

# 我们把 26 维（双臂关节+手指）映射到 30 维动作空间：
#   [EEF_L7, EEF_R7, joints_L7, joints_R7, grip_L1, grip_R1]
# 无 URDF 时 EEF 维度会被置 0 并在 loss mask 中忽略，因此 used_action_channel_ids 只包含 joints+gripper。
va_arms_cfg.used_action_channel_ids = list(range(14, 30))
va_arms_cfg.inverse_used_action_channel_ids = list(range(va_arms_cfg.action_dim))

va_arms_cfg.action_norm_method = "quantiles"

# norm_stat 会被脚本从 <dataset_root>/norm_stat.json 覆盖；这里给默认占位（30 维）
va_arms_cfg.norm_stat = {"q01": [0.0] * va_arms_cfg.action_dim, "q99": [1.0] * va_arms_cfg.action_dim}

