# 本目录用于 Image-to-Video-Action (i2va) 推理的「首帧图像」输入。
# 使用 robotwin_i2av 配置时，需要以下 3 个 PNG 文件（与 obs_cam_keys 对应）：
#
#   observation.images.cam_high.png
#   observation.images.cam_left_wrist.png
#   observation.images.cam_right_wrist.png
#
# 图像尺寸会被代码自动 resize（如 256x320），用任意尺寸的 RGB 图即可。
#
# 生成占位图（便于先跑通流程）：
#   python create_dummy_images.py
