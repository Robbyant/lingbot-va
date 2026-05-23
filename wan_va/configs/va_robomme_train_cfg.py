# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg

va_robomme_train_cfg = EasyDict(__name__="Config: VA RoboMME train")
va_robomme_train_cfg.update(va_libero_cfg)

# === Paths ===
va_robomme_train_cfg.wan22_pretrained_model_name_or_path = "/hpc2hdd/home/czi447/lingbot-va-model"
va_robomme_train_cfg.dataset_path = "/hpc2hdd/home/czi447/robomme_lerobot"
va_robomme_train_cfg.empty_emb_path = "/hpc2hdd/home/czi447/robomme_lerobot/empty_emb.pt"
va_robomme_train_cfg.save_root = "/hpc2hdd/home/czi447/output/robomme_train"

# === Camera keys ===
va_robomme_train_cfg.obs_cam_keys = [
    "observation.images.image",
    "observation.images.wrist_image",
]

# === Action space (7-dim eef action, same as LIBERO) ===
va_robomme_train_cfg.action_dim = 30
va_robomme_train_cfg.used_action_channel_ids = list(range(0, 7))
inverse_used_action_channel_ids = [7] * 30  # default to padding index
for i, j in enumerate(range(7)):
    inverse_used_action_channel_ids[j] = i
va_robomme_train_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# === Normalization stats (from robomme assets/norm_stats.json, actions q01/q99) ===
va_robomme_train_cfg.norm_stat = {
    "q01": [
        -0.18040079298019407,
        -0.3191569724082947,
        -0.13117732753753664,
        -0.5344759382843971,
        -0.12701646758317942,
        -0.2923886525869369,
        -0.4027193263053894,
    ] + [0.] * 23,
    "q99": [
        0.19587624197006226,
        0.3774391655921935,
        0.11749427890777586,
        0.4144394028544425,
        0.12551680266857157,
        0.5218227294564248,
        0.4166772034645081,
    ] + [0.] * 23,
}

# === Logging ===
va_robomme_train_cfg.enable_wandb = False

# === Data loading ===
va_robomme_train_cfg.load_worker = 8
va_robomme_train_cfg.save_interval = 200
va_robomme_train_cfg.gc_interval = 50
va_robomme_train_cfg.cfg_prob = 0.1

# === Training parameters ===
va_robomme_train_cfg.learning_rate = 1e-5
va_robomme_train_cfg.beta1 = 0.9
va_robomme_train_cfg.beta2 = 0.95
va_robomme_train_cfg.weight_decay = 1e-1
va_robomme_train_cfg.warmup_steps = 10
va_robomme_train_cfg.batch_size = 1
va_robomme_train_cfg.gradient_accumulation_steps = 10
va_robomme_train_cfg.num_steps = 20000
