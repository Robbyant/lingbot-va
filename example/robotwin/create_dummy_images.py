#!/usr/bin/env python3
"""生成 robotwin i2va 所需的占位首帧图像，便于先跑通推理流程。"""
import os

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("请先安装: pip install Pillow numpy")
    raise

# 与 va_robotwin_cfg.obs_cam_keys 一致
OBS_CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
# robotwin 主视角尺寸
HEIGHT, WIDTH = 256, 320

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(script_dir, exist_ok=True)
    for i, key in enumerate(OBS_CAM_KEYS):
        # 简单渐变图，避免全 0 导致潜在数值问题
        arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        arr[:, :, 0] = 30 + i * 60
        arr[:, :, 1] = 60 + i * 40
        arr[:, :, 2] = 90 + i * 30
        path = os.path.join(script_dir, f"{key}.png")
        Image.fromarray(arr).save(path)
        print(f"Written: {path}")
    print("Done. 可用 script/run_i2va_single_gpu.sh 跑 i2va 推理。")

if __name__ == "__main__":
    main()
