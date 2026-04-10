
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg

va_libero_i2va_cfg = EasyDict(__name__='Config: VA libero i2va')
va_libero_i2va_cfg.update(va_libero_cfg)

va_libero_i2va_cfg.input_img_path = 'example/libero'
va_libero_i2va_cfg.num_chunks_to_infer = 10
va_libero_i2va_cfg.prompt = "put both the alphabet soup and the tomato sauce in the basket"
va_libero_i2va_cfg.infer_mode = 'i2va'
