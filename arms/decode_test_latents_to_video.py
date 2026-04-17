"""
把 run_arms_test_infer_latents_only.py 产出的 pred_latents.pt 解码成 video.mp4。

只加载 VAE（不加载 transformer），所以显存占用低，能在 GPU 上快速解码。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=str, required=True, help="包含各 episode 子目录的输出根目录")
    ap.add_argument("--vae-root", type=str, required=True, help="models/lingbot-va-base 或其 vae 子目录")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
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
        export_to_video(video, str(out_mp4), fps=int(payload.get("fps", 10)))

    print(f"Done. Decoded videos to: {pred_root}")


if __name__ == "__main__":
    main()

