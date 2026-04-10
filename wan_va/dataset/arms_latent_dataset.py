# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange


@dataclass(frozen=True)
class ArmsSample:
    episode_index: int
    start_frame: int
    end_frame: int
    action_text: str


def _load_norm_stat(dataset_root: Path, fallback_norm_stat: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Priority:
    1) <dataset_root>/norm_stat.json (written by prep script)
    2) config.norm_stat
    """
    stat_path = dataset_root / "norm_stat.json"
    if stat_path.exists():
        with open(stat_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        q01 = np.array(d["q01"], dtype=float)[None]
        q99 = np.array(d["q99"], dtype=float)[None]
        return q01, q99

    q01 = np.array(fallback_norm_stat["q01"], dtype=float)[None]
    q99 = np.array(fallback_norm_stat["q99"], dtype=float)[None]
    return q01, q99


class ArmsLatentDataset(torch.utils.data.Dataset):
    """
    Dataset for the repo-local ./arms data after running scripts/prepare_arms_dataset.py.

    Expected directory:
      dataset_path/
        meta/episodes.jsonl
        videos/chunk-000/observation.images.cam_high/episode_000000.mp4
        actions/episode_000000.npy            # [T, 30] float32 (mapped to 30-dim standard)
        latents/chunk-000/observation.images.cam_high/episode_000000_0_T.pth  (optional; required for training)
        empty_emb.pt
        norm_stat.json (optional)
    """

    def __init__(self, config):
        self.config = config
        self.root = Path(config.dataset_path)
        self.meta_path = self.root / "meta" / "episodes.jsonl"
        assert self.meta_path.exists(), f"episodes.jsonl not found: {self.meta_path}"

        self.used_video_keys = list(config.obs_cam_keys)
        assert len(self.used_video_keys) == 1, "release dataset expects a single camera key"

        empty_emb_path = Path(config.empty_emb_path)
        if not empty_emb_path.exists():
            # Create a compatible empty embedding from the first latent file's text_emb.
            # This avoids requiring users to manually provide empty_emb.pt.
            latent_dir = self.root / "latents" / "chunk-000" / self.used_video_keys[0]
            first_pth = next(iter(sorted(latent_dir.glob("episode_*.pth"))), None)
            if first_pth is None:
                raise FileNotFoundError(
                    f"empty_emb.pt not found at {empty_emb_path} and no latent files under {latent_dir} "
                    "to infer embedding shape. Please run latent extraction first."
                )
            sample = torch.load(first_pth, weights_only=False)
            text_emb = sample.get("text_emb", None)
            if text_emb is None:
                raise KeyError(f"'text_emb' missing in latent file: {first_pth}")
            empty_emb_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.zeros_like(text_emb), empty_emb_path)
        self.empty_emb = torch.load(empty_emb_path, weights_only=False)
        self.cfg_prob = getattr(config, "cfg_prob", 0.0)

        self.q01, self.q99 = _load_norm_stat(self.root, config.norm_stat)

        self.latent_path = self.root / "latents"
        self.actions_path = self.root / "actions"

        self.samples: list[ArmsSample] = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                ep = int(d["episode_index"])
                tasks = d.get("tasks", [])
                action_config = d.get("action_config", [])
                if not action_config:
                    # single segment fallback
                    self.samples.append(
                        ArmsSample(ep, 0, int(d["length"]), tasks[0] if tasks else "")
                    )
                else:
                    for acfg in action_config:
                        self.samples.append(
                            ArmsSample(
                                ep,
                                int(acfg["start_frame"]),
                                int(acfg["end_frame"]),
                                str(acfg.get("action_text", tasks[0] if tasks else "")),
                            )
                        )

        # inverse_used_action_channel_ids is defined on config; keep behavior consistent with LeRobot loader.
        self.inverse_used_action_channel_ids = np.asarray(config.inverse_used_action_channel_ids, dtype=int)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_latent_segment(self, episode_index: int, start_frame: int, end_frame: int) -> dict:
        # We keep the same naming convention as README: episode_{idx}_{start}_{end}.pth
        episode_chunk = 0
        key = self.used_video_keys[0]
        latent_file = (
            self.latent_path
            / f"chunk-{episode_chunk:03d}"
            / key
            / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        )
        assert latent_file.exists(), (
            f"latent file not found: {latent_file}\n"
            "You need to extract Wan2.2 VAE latents into dataset_root/latents/ mirroring videos/."
        )
        return torch.load(latent_file, weights_only=False, map_location="cpu")

    def _load_actions(self, episode_index: int) -> np.ndarray:
        ap = self.actions_path / f"episode_{episode_index:06d}.npy"
        assert ap.exists(), f"actions file not found: {ap}"
        a = np.load(ap)
        assert a.ndim == 2 and a.shape[1] == 30, f"actions must be [T, 30], got {a.shape}"
        return a

    def _action_post_process(self, local_start_frame: int, local_end_frame: int, latent_frame_ids, action: np.ndarray):
        # Keep same logic as lerobot_latent_dataset.py
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]

        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode="constant", constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4
        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype="bool")
        assert action.shape[0] == required_action_num

        # Extra mask channel, same as existing pipeline.
        action_paded = np.pad(action, ((0, 0), (0, 1)), mode="constant", constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode="constant", constant_values=0)

        action_aligned = action_paded[:, self.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0

        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx % len(self.samples)]
        latent_dict = self._load_latent_segment(s.episode_index, s.start_frame, s.end_frame)

        # Latent dict fields follow README table.
        latent = latent_dict["latent"]
        latent_num_frames = int(latent_dict["latent_num_frames"])
        latent_height = int(latent_dict["latent_height"])
        latent_width = int(latent_dict["latent_width"])
        frame_ids = latent_dict["frame_ids"]

        lat = rearrange(latent, "(f h w) c -> f h w c", f=latent_num_frames, h=latent_height, w=latent_width)
        lat = lat.permute(3, 0, 1, 2)  # C,F,H,W

        text_emb = latent_dict["text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        actions_full = self._load_actions(s.episode_index)
        actions_seg = actions_full[s.start_frame:s.end_frame]
        actions_aligned, actions_mask = self._action_post_process(s.start_frame, s.end_frame, frame_ids, actions_seg)

        return {
            "latents": lat,
            "text_emb": text_emb,
            "actions": actions_aligned,
            "actions_mask": actions_mask,
        }

