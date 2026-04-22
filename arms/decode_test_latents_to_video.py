"""
把 run_arms_test_infer_latents_only.py 产出的 pred_latents.pt 解码成 video.mp4。

只加载 VAE（不加载 transformer），所以显存占用低，能在 GPU 上快速解码。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor


def _probe_video(path: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return w, h, fps


def _count_target_frames(ep_dir: Path) -> int | None:
    # Prefer action.txt line count (matches submission expectation)
    action_path = ep_dir / "action.txt"
    if not action_path.exists():
        return None
    n_lines = len(action_path.read_text(encoding="utf-8").splitlines())
    if n_lines <= 1:
        return None
    return n_lines - 1


def _resample_video_np(video: np.ndarray, target_frames: int) -> np.ndarray:
    # video: [F,H,W,C] float/uint8
    if target_frames <= 0:
        raise ValueError("target_frames must be > 0")
    f = int(video.shape[0])
    if f == target_frames:
        return video
    if f <= 1:
        return np.repeat(video, target_frames, axis=0)

    # Linear interpolation in time
    t_src = np.linspace(0.0, 1.0, num=f, endpoint=True, dtype=np.float32)
    t_tgt = np.linspace(0.0, 1.0, num=target_frames, endpoint=True, dtype=np.float32)
    out = np.empty((target_frames, *video.shape[1:]), dtype=video.dtype)
    for i, t in enumerate(t_tgt):
        j = int(np.searchsorted(t_src, t, side="right") - 1)
        j = max(0, min(j, f - 2))
        t0, t1 = float(t_src[j]), float(t_src[j + 1])
        w = 0.0 if t1 == t0 else (float(t) - t0) / (t1 - t0)
        out[i] = (1.0 - w) * video[j] + w * video[j + 1]
    return out


def _match_length(video: np.ndarray, target_frames: int) -> np.ndarray:
    """Prefer trimming over interpolation to preserve sharpness."""
    f = int(video.shape[0])
    if f == target_frames:
        return video
    if f > target_frames:
        return video[:target_frames]
    # f < target_frames
    if f <= 1:
        return np.repeat(video, target_frames, axis=0)
    # If short, interpolate (less harmful than upscaling everything)
    return _resample_video_np(video, target_frames)


def _resize_video_np(video: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if target_w <= 0 or target_h <= 0:
        return video
    f, h, w, c = video.shape
    if w == target_w and h == target_h:
        return video
    out = np.empty((f, target_h, target_w, c), dtype=video.dtype)
    for i in range(f):
        out[i] = cv2.resize(video[i], (target_w, target_h), interpolation=cv2.INTER_AREA)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=str, required=True, help="包含各 episode 子目录的输出根目录")
    ap.add_argument("--test-root", type=str, default=None, help="可选：原始 test 根目录（用于对齐 fps/分辨率）")
    ap.add_argument("--vae-root", type=str, required=True, help="models/lingbot-va-base 或其 vae 子目录")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--force-frames", type=int, default=None, help="可选：强制输出帧数（提交通常要求 50）")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    test_root = Path(args.test_root) if args.test_root else None
    vae_root = Path(args.vae_root)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    vae_path = vae_root / "vae" if (vae_root / "vae").exists() else vae_root
    vae = AutoencoderKLWan.from_pretrained(str(vae_path), torch_dtype=dtype).to(device).eval()
    video_processor = VideoProcessor(vae_scale_factor=1)

    ep_dirs = sorted([p for p in pred_root.iterdir() if p.is_dir()])
    for ep_dir in ep_dirs:
        pt_path = ep_dir / "pred_latents.pt"
        out_mp4 = ep_dir / "video.mp4"
        if not pt_path.exists():
            continue
        if args.skip_existing and out_mp4.exists():
            continue

        payload = torch.load(pt_path, map_location="cpu")
        latents = payload["latents"].to(device=device, dtype=dtype)  # [1,48,51,16,16] normalized

        latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, vae.config.z_dim, 1, 1, 1)
        latents_denorm = latents / latents_std + latents_mean

        video = vae.decode(latents_denorm, return_dict=False)[0]
        video = video_processor.postprocess_video(video, output_type="np")[0]

        # Align to submission expectations: same FPS/resolution as original test video, and frame count as action.txt
        target_frames = _count_target_frames(ep_dir) or int(video.shape[0])
        # Submission usually expects 50 future frames (README). Some scripts output 51 steps (80..130 inclusive).
        if args.force_frames is not None and args.force_frames > 0:
            target_frames = int(args.force_frames)
        elif target_frames == 51:
            target_frames = 50
        target_fps = float(payload.get("fps", 10) or 10)
        target_w = target_h = 0
        if test_root is not None:
            src_video = test_root / ep_dir.name / "video.mp4"
            if src_video.exists():
                target_w, target_h, src_fps = _probe_video(src_video)
                if src_fps > 0:
                    target_fps = src_fps

        video = video.astype(np.float32)
        video = _match_length(video, target_frames)
        video = _resize_video_np(video, target_w=target_w, target_h=target_h)
        # export_to_video expects uint8 [0,255]
        vmax = float(video.max()) if video.size else 0.0
        if vmax <= 1.0:
            video = video * 255.0
        elif vmax <= 2.0:
            # In case the range is [0,2] (rare), map to [0,255]
            video = (video / 2.0) * 255.0
        video = np.clip(video, 0, 255).astype(np.uint8)
        export_to_video(video, str(out_mp4), fps=int(round(target_fps)))

    print(f"Done. Decoded videos to: {pred_root}")


if __name__ == "__main__":
    main()

