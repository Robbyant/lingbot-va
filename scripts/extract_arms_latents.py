# Extract Wan2.2 VAE latents for prepared_arms dataset.
#
# Input:
#   prepared_arms/meta/episodes.jsonl
#   prepared_arms/videos/chunk-000/observation.images.cam_high/episode_000000.mp4
#
# Output:
#   prepared_arms/latents/chunk-000/observation.images.cam_high/episode_000000_0_T.pth
#
# Each .pth is a dict matching repo README (latent, latent_num_frames, frame_ids, text_emb, ...).
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange

# Allow running as a script from repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from wan_va.modules.utils import WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_vae, patchify


def _read_video_rgb(path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {path}")
    return np.stack(frames, axis=0), float(fps)


@torch.no_grad()
def _encode_video_to_latent(
    vae,
    streaming_vae: WanVAEStreamingWrapper,
    frames_rgb: np.ndarray,
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
    streaming: bool,
) -> torch.Tensor:
    """
    Returns normalized mean latents: [1, C, F, H', W'] (H'/W' ~ height//16,width//16).
    """
    # frames_rgb: [F, H, W, 3] uint8
    x = torch.from_numpy(frames_rgb).float().permute(3, 0, 1, 2)  # 3,F,H,W
    x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    x = (x / 255.0) * 2.0 - 1.0
    x = x.unsqueeze(0).to(device=device, dtype=dtype)  # 1,3,F,H,W

    F = x.shape[2]
    streaming_vae.clear_cache()
    if streaming:
        # For very long videos only: chunk over time and keep VAE cache.
        outs = []
        for st in range(0, F, chunk_size):
            x_chunk = x[:, :, st : st + chunk_size]
            enc_out = streaming_vae.encode_chunk(x_chunk)
            outs.append(enc_out)
        enc_out = torch.cat(outs, dim=2)
    else:
        # Default: encode the full clip in one call.
        # This matches how wan_va_server.py encodes an observation history and avoids temporal shape pitfalls.
    used_stride = 1
    try:
        enc_out = streaming_vae.encode_chunk(x)
    except RuntimeError as e:
        # Pragmatic fallback: if the WAN VAE encoder hits temporal shape mismatch for long clips,
        # downsample frames by 2 (typical 30fps -> 15fps) and retry.
        msg = str(e)
        if "must match the size of tensor" in msg and "at non-singleton dimension 2" in msg:
            used_stride = 2
            x = x[:, :, ::2].contiguous()
            enc_out = streaming_vae.encode_chunk(x)
        else:
            raise

    mu, _logvar = torch.chunk(enc_out, 2, dim=1)
    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    mu_norm = ((mu.float() - latents_mean.float()) * (1.0 / latents_std.float())).to(mu.dtype)
    # Crop back to original frame count (streaming may alter length).
    return mu_norm[:, :, :F], used_stride


@torch.no_grad()
def _encode_text(text_encoder, tokenizer, text: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tokens = tokenizer([text], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    out = text_encoder(**tokens).last_hidden_state  # [1, L, D]
    return out.to(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default="./prepared_arms", help="prepared_arms root")
    ap.add_argument("--ckpt-dir", type=str, required=True, help="Wan2.2 checkpoint dir containing vae/, tokenizer/, text_encoder/")
    ap.add_argument("--device", type=str, default="cuda", help="torch device (e.g. cuda)")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=2, help="frames per streaming VAE chunk (AutoencoderKLWan streaming is stable with 2)")
    ap.add_argument("--streaming", action="store_true", help="Use streaming VAE encode (only needed for very long videos).")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    root = Path(args.dataset_root)
    meta_path = root / "meta" / "episodes.jsonl"
    video_dir = root / "videos" / "chunk-000" / "observation.images.cam_high"
    out_dir = root / "latents" / "chunk-000" / "observation.images.cam_high"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.ckpt_dir)
    vae = load_vae(str(ckpt / "vae"), torch_dtype=dtype, torch_device=device)
    streaming_vae = WanVAEStreamingWrapper(vae)
    tokenizer = load_tokenizer(str(ckpt / "tokenizer"))
    text_encoder = load_text_encoder(str(ckpt / "text_encoder"), torch_dtype=dtype, torch_device=device)

    lines = meta_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        ep = int(d["episode_index"])
        instruction = d["tasks"][0] if d.get("tasks") else ""
        length = int(d["length"])
        start_frame = 0
        end_frame = length

        video_path = video_dir / f"episode_{ep:06d}.mp4"
        frames_rgb, ori_fps = _read_video_rgb(video_path)
        if frames_rgb.shape[0] != length:
            # allow minor mismatch, but keep end_frame consistent with actual frames
            end_frame = min(end_frame, frames_rgb.shape[0])
            frames_rgb = frames_rgb[:end_frame]

        lat, used_stride = _encode_video_to_latent(
            vae,
            streaming_vae,
            frames_rgb,
            height=args.height,
            width=args.width,
            device=device,
            dtype=dtype,
            chunk_size=args.chunk_size,
            streaming=args.streaming,
        )  # [1,C,F,h,w]

        # Flatten to [N,C] as repo README expects.
        lat_fhwc = lat[0].permute(1, 2, 3, 0).contiguous()  # F,h,w,C
        latent_num_frames, latent_height, latent_width = lat_fhwc.shape[:3]
        latent_flat = rearrange(lat_fhwc, "f h w c -> (f h w) c").to(torch.bfloat16)

        text_emb = _encode_text(text_encoder, tokenizer, instruction, device=device, dtype=torch.bfloat16)[0]
        # When we fallback to 2x downsample in VAE encode, frame_ids should still reflect the sampled frames.
        # For simplicity we always use every frame here; if fallback triggers, the latent's frame_ids will be subsampled.
        frame_ids = list(range(start_frame, end_frame, used_stride))

        out = {
            "latent": latent_flat,
            "latent_num_frames": int(latent_num_frames),
            "latent_height": int(latent_height),
            "latent_width": int(latent_width),
            "video_num_frames": int(frames_rgb.shape[0]),
            "video_height": int(frames_rgb.shape[1]),
            "video_width": int(frames_rgb.shape[2]),
            "text_emb": text_emb,
            "text": instruction,
            "frame_ids": frame_ids,
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "fps": int(round(ori_fps)) if ori_fps > 0 else 30,
            "ori_fps": float(ori_fps),
        }

        out_path = out_dir / f"episode_{ep:06d}_{start_frame}_{end_frame}.pth"
        torch.save(out, out_path)
        print(f"✅ wrote {out_path}")


if __name__ == "__main__":
    main()

