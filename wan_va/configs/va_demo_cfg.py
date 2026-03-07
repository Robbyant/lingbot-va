# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
import torch
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_demo_cfg = EasyDict(__name__='Config: VA demo')
va_demo_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

va_demo_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_VA_MODEL_PATH", "/path/to/pretrained/model"
)

va_demo_cfg.attn_window = 30
va_demo_cfg.frame_chunk_size = 4
va_demo_cfg.env_type = 'none'

va_demo_cfg.height = 256
va_demo_cfg.width = 256
va_demo_cfg.action_dim = 30
va_demo_cfg.action_per_frame = 8
va_demo_cfg.obs_cam_keys = [
    'observation.images.top', 'observation.images.wrist'
]
va_demo_cfg.guidance_scale = 5
va_demo_cfg.action_guidance_scale = 1

va_demo_cfg.num_inference_steps = 5
va_demo_cfg.video_exec_step = -1
va_demo_cfg.action_num_inference_steps = 10

va_demo_cfg.snr_shift = 5.0
va_demo_cfg.action_snr_shift = 1.0

va_demo_cfg.used_action_channel_ids = list(range(0, 5)) + list(range(28, 29))
inverse_used_action_channel_ids = [len(va_demo_cfg.used_action_channel_ids)
                                   ] * va_demo_cfg.action_dim
for i, j in enumerate(va_demo_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_demo_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_demo_cfg.action_norm_method = 'quantiles'
va_demo_cfg.norm_stat = {
    "q01": [
        -90.60303497314453,
        -98.73043060302734,
        -79.9008560180664,
        48.95470428466797,
        -32.794578552246094,
    ] + [0.] * 23 + [0.8250824809074402, 0],
    "q99": [
        71.735107421875,
        65.89081573486328,
        92.87967681884766,
        100.0,
        22.784151077270508,
    ] + [0.] * 23 + [100.0, 0],
}
