"""
为 arms_lerobot 提取视频 latents（Wan2.2 VAE encoder），输出到：
  arms_lerobot/latents/chunk-000/observation.images.front/episode_XXXXXX_0_T.pth

该 .pth 文件的字段尽量对齐 README “Extract Latents” 部分：
  - latent: Tensor [N, C] (bfloat16/float16)
  - latent_num_frames: int
  - latent_height: int
  - latent_width: int
  - video_num_frames: int
  - video_height / video_width: int
  - text_emb: Tensor [L, D]
  - text: str
  - frame_ids: list[int]
  - start_frame / end_frame: int（end_frame 为 exclusive）
  - fps / ori_fps: int/float

依赖：
  - torch, diffusers, transformers, opencv-python, pyarrow（只用来读 episodes.jsonl 的话不需要）

运行建议（示例）：
  conda activate gmr
  python arms/extract_latents_arms_lerobot.py \
    --dataset-root arms_lerobot \
    --wan22-path /path/to/wan2.2 \
    --device cuda:0 \
    --dtype bfloat16 \
    --target-size 256 256 \
    --max-episodes 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diffusers import AutoencoderKLWan


def load_vae(vae_path: str, torch_dtype: torch.dtype, device: torch.device):
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch_dtype)
    return vae.to(device)


def patchify(x: torch.Tensor, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    @torch.no_grad()
    def encode_chunk(self, x_chunk: torch.Tensor):
        if hasattr(self.vae.config, "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc


def read_video_all_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frames.append(frame_bgr)
    cap.release()
    return frames, fps


def sample_frames(frames: List[np.ndarray], stride: int) -> Tuple[List[np.ndarray], List[int]]:
    if stride <= 0:
        stride = 1
    idxs = list(range(0, len(frames), stride))
    out = [frames[i] for i in idxs]
    return out, idxs


def preprocess_frames_to_tensor(
    frames_bgr: List[np.ndarray],
    target_h: int,
    target_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # frames_bgr: list of HWC uint8
    rgb = [
        cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        for f in frames_bgr
    ]
    arr = np.stack(rgb, axis=0)  # [F, H, W, 3]
    # resize
    if arr.shape[1] != target_h or arr.shape[2] != target_w:
        resized = []
        for f in arr:
            resized.append(cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA))
        arr = np.stack(resized, axis=0)
    # to torch: [1,3,F,H,W] float in [-1,1]
    x = torch.from_numpy(arr).to(device=device)
    x = x.permute(3, 0, 1, 2).contiguous().float()  # [3,F,H,W]
    x = (x / 255.0) * 2.0 - 1.0
    x = x.unsqueeze(0).to(dtype=dtype)
    return x


def make_empty_text_emb(max_sequence_length: int, text_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((max_sequence_length, text_dim), device=device, dtype=dtype)


@torch.no_grad()
def encode_video_to_latent_flat(
    vae,
    video_tensor: torch.Tensor,  # [1,3,F,H,W]
    chunk_frames: int = 0,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    使用 diffusers 的 `AutoencoderKLWan.encode` 直接编码。

    说明：
    - `AutoencoderKLWan.encode` 内部会处理其需要的 patchify / 3D 因果卷积缓存。
    - 为避免不同版本 diffusers 的 streaming API 兼容问题，这里默认整段编码。
    """
    _ = chunk_frames  # keep CLI arg for backward compatibility
    enc = vae.encode(video_tensor)
    mu = enc.latent_dist.mean  # [1, z_dim, F', H', W']

    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    mu_norm = (mu.float() - latents_mean.float()) * (1.0 / latents_std.float())
    mu_norm = mu_norm.to(mu.dtype)

    _, c, f, h, w = mu_norm.shape
    flat = mu_norm.permute(0, 2, 3, 4, 1).reshape(-1, c)  # [F*H*W, C]
    return flat, f, h, w


def load_episodes_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def ensure_latent_dir(dataset_root: Path, video_rel_dir: Path) -> Path:
    # video_rel_dir like: videos/chunk-000/observation.images.front
    latent_dir = dataset_root / "latents" / "chunk-000" / video_rel_dir.name
    latent_dir.mkdir(parents=True, exist_ok=True)
    return latent_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument(
        "--out-root",
        type=str,
        default="",
        help="可选：latents 输出根目录（默认写到 <dataset-root>/latents）。会自动创建 chunk-000/observation.images.front/",
    )
    ap.add_argument("--wan22-path", type=str, required=True, help="包含 vae/tokenizer/text_encoder/transformer 的目录（取其中 vae/tokenizer/text_encoder）")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    ap.add_argument("--target-size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    ap.add_argument("--stride", type=int, default=1, help="从原视频每 stride 帧取一帧")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument(
        "--text-mode",
        type=str,
        default="empty",
        choices=["empty", "encode"],
        help="empty: 不加载 text encoder，写全零 text_emb；encode: 用 tokenizer+text_encoder 编码指令文本",
    )
    ap.add_argument("--text-dim", type=int, default=4096, help="text-mode=empty 时使用的 embedding 维度")
    ap.add_argument("--max-episodes", type=int, default=-1, help="调试用，<=0 表示全量")
    ap.add_argument("--skip-existing", action="store_true", help="若 latent 文件已存在则跳过")
    ap.add_argument("--vae-chunk-frames", type=int, default=8, help="VAE 编码按时间分块大小（避免 OOM）")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root) if str(args.out_root).strip() else (dataset_root / "latents")
    out_root.mkdir(parents=True, exist_ok=True)
    wan22_path = Path(args.wan22_path)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    target_h, target_w = int(args.target_size[0]), int(args.target_size[1])

    # load VAE: support local directory with subfolder `vae/` or HF repo id
    if wan22_path.exists():
        vae = load_vae(str(wan22_path / "vae"), torch_dtype=dtype, device=device)
        is_local = True
    else:
        # treat as HuggingFace repo id
        try:
            vae = AutoencoderKLWan.from_pretrained(args.wan22_path, subfolder="vae", torch_dtype=dtype).to(device)
        except Exception:
            # fallback: repo root itself is the VAE
            vae = AutoencoderKLWan.from_pretrained(args.wan22_path, torch_dtype=dtype).to(device)
        is_local = False
    # text embedding
    if args.text_mode == "encode":
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # local import to keep deps optional
        from transformers import T5TokenizerFast, UMT5EncoderModel

        if is_local:
            tokenizer = T5TokenizerFast.from_pretrained(str(wan22_path / "tokenizer"))
            text_encoder = UMT5EncoderModel.from_pretrained(str(wan22_path / "text_encoder"), torch_dtype=dtype).to(device)
        else:
            tokenizer = T5TokenizerFast.from_pretrained(args.wan22_path, subfolder="tokenizer")
            text_encoder = UMT5EncoderModel.from_pretrained(args.wan22_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
        text_encoder.eval()
    else:
        tokenizer = None
        text_encoder = None
        # also dump empty_emb.pt for later training configs
        empty_emb = make_empty_text_emb(args.max_seq_len, args.text_dim, device=device, dtype=dtype).cpu()
        torch.save(empty_emb, out_root / "empty_emb.pt")

    episodes = load_episodes_jsonl(dataset_root / "meta" / "episodes.jsonl")

    video_dir = dataset_root / "videos" / "chunk-000" / "observation.images.front"
    latent_dir = out_root / "chunk-000" / "observation.images.front"
    latent_dir.mkdir(parents=True, exist_ok=True)

    n = len(episodes) if args.max_episodes <= 0 else min(len(episodes), int(args.max_episodes))
    for i in range(n):
        ep = episodes[i]
        episode_index = int(ep["episode_index"])
        task_text = ep["tasks"][0]
        start_frame = int(ep["action_config"][0]["start_frame"])
        end_frame = int(ep["action_config"][0]["end_frame"])  # exclusive

        video_path = video_dir / f"episode_{episode_index:06d}.mp4"
        out_path = latent_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        if args.skip_existing and out_path.exists():
            continue

        frames_all, ori_fps = read_video_all_frames(video_path)
        frames, frame_ids = sample_frames(frames_all, args.stride)
        if len(frames) == 0:
            raise RuntimeError(f"Empty video after sampling: {video_path}")

        video_tensor = preprocess_frames_to_tensor(frames, target_h, target_w, device=device, dtype=dtype)

        latent_flat, latent_f, latent_h, latent_w = encode_video_to_latent_flat(
            vae=vae,
            video_tensor=video_tensor,
            chunk_frames=int(args.vae_chunk_frames),
        )

        if args.text_mode == "encode":
            prompt = prompt_clean(task_text)
            text_inputs = tokenizer(
                [prompt],
                padding="max_length",
                max_length=args.max_seq_len,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            attn_mask = text_inputs.attention_mask.to(device)
            text_emb = text_encoder(text_input_ids, attn_mask).last_hidden_state[0].to(dtype=dtype)
        else:
            text_emb = make_empty_text_emb(args.max_seq_len, args.text_dim, device=device, dtype=dtype)

        payload = {
            "latent": latent_flat.to(dtype),
            "latent_num_frames": int(latent_f),
            "latent_height": int(latent_h),
            "latent_width": int(latent_w),
            "video_num_frames": int(len(frames)),
            "video_height": int(target_h),
            "video_width": int(target_w),
            "text_emb": text_emb.to(dtype),
            "text": task_text,
            "frame_ids": [int(x) for x in frame_ids],
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "fps": float(ori_fps / max(1, args.stride)) if ori_fps else 0.0,
            "ori_fps": float(ori_fps) if ori_fps else 0.0,
        }

        torch.save(payload, out_path)

    print(f"Done. Wrote latents to: {latent_dir}")


if __name__ == "__main__":
    main()

