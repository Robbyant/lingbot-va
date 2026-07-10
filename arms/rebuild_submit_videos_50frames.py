"""
为提交目录重建 video.mp4，使其与 sample_result 对齐：50 帧、30fps。

输入：
  - pred_root: run_arms_test_infer_latents_only.py 的输出（含 pred_latents.pt）
  - submit_root: 最终提交目录（每个 episode 子目录已经有 action/joint/instruction）

做法：
  - 从 pred_latents.pt 读取 normalized latents（latent 帧数=51）
  - 用 VAE 解码得到视频（通常会上采样到 ~200 帧）
  - 将解码视频均匀采样到 50 帧，再导出为 mp4（30fps）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=str, required=True)
    ap.add_argument("--submit-root", type=str, required=True)
    ap.add_argument("--vae-root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--target-frames", type=int, default=50)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--target-size", type=int, nargs=2, default=[720, 1280], metavar=("H", "W"))
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    submit_root = Path(args.submit_root)
    vae_root = Path(args.vae_root)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    vae_path = vae_root / "vae" if (vae_root / "vae").exists() else vae_root
    vae = AutoencoderKLWan.from_pretrained(str(vae_path), torch_dtype=dtype).to(device).eval()
    video_processor = VideoProcessor(vae_scale_factor=1)

    for ep_dir in sorted([p for p in pred_root.iterdir() if p.is_dir()]):
        ep_id = ep_dir.name
        if ep_id == "real":
            continue

        pt_path = ep_dir / "pred_latents.pt"
        if not pt_path.exists():
            continue

        out_dir = submit_root / ep_id
        out_mp4 = out_dir / "video.mp4"
        if not out_dir.exists():
            continue
        if args.skip_existing and out_mp4.exists():
            continue

        payload = torch.load(pt_path, map_location="cpu")
        latents = payload["latents"].to(device=device, dtype=dtype)  # normalized [1,48,T,16,16]

        latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
        latents_denorm = latents / latents_std + latents_mean

        video = vae.decode(latents_denorm, return_dict=False)[0]
        video = video_processor.postprocess_video(video, output_type="np")[0]  # [T,H,W,3]

        T = int(video.shape[0])
        target = int(args.target_frames)
        if T != target:
            # evenly sample to target frames
            idx = np.linspace(0, max(0, T - 1), num=target)
            idx = np.round(idx).astype(np.int64)
            idx = np.clip(idx, 0, max(0, T - 1))
            video = video[idx]

        # resize to target resolution (GT is 1280x720)
        target_h, target_w = int(args.target_size[0]), int(args.target_size[1])
        if int(video.shape[1]) != target_h or int(video.shape[2]) != target_w:
            resized = []
            for fr in video:
                resized.append(cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_LINEAR))
            video = np.stack(resized, axis=0)

        export_to_video(video, str(out_mp4), fps=int(args.fps))

    print(f"Done. Rebuilt submit videos in: {submit_root}")


if __name__ == "__main__":
    main()

