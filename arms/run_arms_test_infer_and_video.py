"""
对 arms/test/ 批量推理：
- 输入：每个 episode 目录包含 video.mp4（16帧）、instruction.txt、joint.txt（16行）
- 输出：每个 episode 输出 action.txt/joint.txt（80-130 共 51 行）以及预测 video.mp4

说明：
- 采用 latent 推理：先用 lingbot-va-base 的 VAE 把 16 帧编码成 48通道 latents（时域下采样=4）
- KV cache 预填充：用 4 个 latent 帧 + 16 步 joint history（padding 到 30 维动作空间）
- rollout：调用 VA_Server._infer 取回 (actions, latents)，同时生成视频
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict
from tqdm import tqdm

# allow running as a standalone script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

from wan_va.configs.va_arms_cfg import va_arms_cfg
from wan_va.wan_va_server import VA_Server


def _read_video_frames_rgb(video_path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def _frames_to_tensor(frames_rgb: List[np.ndarray], target_h: int, target_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # -> [1,3,F,H,W], value in [-1,1]
    if len(frames_rgb) == 0:
        raise RuntimeError("Empty frames")
    arr = []
    for fr in frames_rgb:
        fr = cv2.resize(fr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        arr.append(fr.astype(np.float32))
    x = np.stack(arr, axis=0)  # [F,H,W,3]
    x = torch.from_numpy(x).permute(3, 0, 1, 2).contiguous()  # [3,F,H,W]
    x = (x / 255.0) * 2.0 - 1.0
    return x.unsqueeze(0).to(device=device, dtype=dtype)


@torch.no_grad()
def _encode_video_to_latents_norm(server: VA_Server, frames_rgb: List[np.ndarray]) -> torch.Tensor:
    """
    返回 normalized latents: [1,48,F',H',W']，与 VA_Server._encode_obs 一致的归一化方式。
    """
    device = server.device
    dtype = server.dtype

    video_tensor = _frames_to_tensor(frames_rgb, server.job_config.height, server.job_config.width, device=device, dtype=dtype)

    # enable_offload 下 VAE 可能在 CPU；这里临时搬到 GPU 编码更快
    vae_was_on_cpu = next(server.vae.parameters()).device.type == "cpu"
    if vae_was_on_cpu:
        server.vae = server.vae.to(device).to(dtype)

    enc = server.vae.encode(video_tensor)
    mu = enc.latent_dist.mean  # [1,48,4,16,16]

    latents_mean = torch.tensor(server.vae.config.latents_mean, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(server.vae.config.latents_std, device=mu.device, dtype=mu.dtype).view(1, -1, 1, 1, 1)
    mu_norm = (mu.float() - latents_mean.float()) * (1.0 / latents_std.float())
    mu_norm = mu_norm.to(dtype)

    if vae_was_on_cpu and server.enable_offload:
        server.vae = server.vae.to("cpu")
        torch.cuda.empty_cache()

    return mu_norm


def _load_history_joint_cf1(test_ep_dir: Path, history_len: int) -> np.ndarray:
    df = pd.read_csv(test_ep_dir / "joint.txt")
    # drop first index col
    data = df.iloc[:, 1:].to_numpy(dtype=np.float32)  # [T,26]
    hist = data[: min(history_len, data.shape[0])]
    state_cf1 = hist.T[:, :, None]  # [26,F,1]
    # pad to 29 dims so preprocess_action 的 +1 padding 变成 30 dims
    state_cf1 = np.pad(state_cf1, ((0, 3), (0, 0), (0, 0)), mode="constant", constant_values=0.0)
    return state_cf1


def _write_csv_like_sample(path: Path, header_cols: List[str], idxs: List[int], data_t26: np.ndarray) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Unnamed: 0"] + header_cols)
        for i, t in enumerate(idxs):
            row = [int(t)] + [float(x) for x in data_t26[i].tolist()]
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", type=str, default="arms/test")
    ap.add_argument("--model-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, default="arms/test_generated")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--history-len", type=int, default=16)
    ap.add_argument("--start-idx", type=int, default=80)
    ap.add_argument("--end-idx", type=int, default=130)
    ap.add_argument("--max-episodes", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    test_root = Path(args.test_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # header columns from any action.txt (same schema)
    any_action = next(iter(sorted(test_root.glob("*/action.txt"))))
    header = any_action.read_text(encoding="utf-8").splitlines()[0].strip().split(",")
    header_cols = header[1:]  # drop Unnamed: 0

    # norm_stat from arms_lerobot (train stats)
    norm_stat_path = Path("arms_lerobot/norm_stat.json")
    if not norm_stat_path.exists():
        raise FileNotFoundError("需要先生成 arms_lerobot/norm_stat.json（用 compute_arms_norm_stat.py）")
    norm_stat = __import__("json").loads(norm_stat_path.read_text(encoding="utf-8"))
    for k in ["q01", "q99"]:
        if k in norm_stat and len(norm_stat[k]) < 30:
            pad_val = 0.0 if k == "q01" else 1.0
            norm_stat[k] = list(norm_stat[k]) + [pad_val] * (30 - len(norm_stat[k]))

    cfg = EasyDict(va_arms_cfg)
    cfg.wan22_pretrained_model_name_or_path = str(Path(args.model_root).resolve())
    cfg.save_root = str(out_root.resolve())
    cfg.enable_offload = True
    cfg.param_dtype = torch.float16
    cfg.local_rank = int(str(args.device).split(":")[-1]) if "cuda" in args.device else 0
    cfg.norm_stat = norm_stat

    if "cuda" in args.device:
        torch.cuda.empty_cache()
    server = VA_Server(cfg)
    server.video_processor = VideoProcessor(vae_scale_factor=1)

    ep_dirs = sorted([p for p in test_root.iterdir() if p.is_dir()])
    if args.max_episodes > 0:
        ep_dirs = ep_dirs[: int(args.max_episodes)]

    idxs = list(range(int(args.start_idx), int(args.end_idx) + 1))
    need_steps = len(idxs)

    for ep_dir in ep_dirs:
        ep_id = ep_dir.name
        t0 = time.time()
        try:
            ep_out = out_root / ep_id
            if args.skip_existing and (ep_out / "action.txt").exists() and (ep_out / "joint.txt").exists() and (ep_out / "video.mp4").exists():
                print(f"skip {ep_id}", flush=True)
                continue

            prompt = (ep_dir / "instruction.txt").read_text(encoding="utf-8").strip()
            video_path = ep_dir / "video.mp4"
            frames_rgb, fps = _read_video_frames_rgb(video_path)
            if len(frames_rgb) == 0:
                raise RuntimeError(f"Empty video: {video_path}")

            # reset and prepare caches
            server.infer({"reset": True, "prompt": prompt})

            # encode 16 frames -> latents (normalized)
            latents_hist_full = _encode_video_to_latents_norm(server, frames_rgb)  # [1,48,4,16,16]
            hist_latent_frames = int(np.ceil(min(args.history_len, len(frames_rgb)) / 4.0))
            latents_hist = latents_hist_full[:, :, :hist_latent_frames]

            server.init_latent = latents_hist[:, :, :1].to(server.dtype)
            latent_model_input = latents_hist[:, :, 1:].to(server.dtype) if hist_latent_frames > 1 else None

            state_cf1 = _load_history_joint_cf1(ep_dir, history_len=int(args.history_len))
            action_model_input = server.preprocess_action(state_cf1).to(device=server.device, dtype=server.dtype)

            server.transformer.clear_pred_cache(server.cache_name)
            server.frame_st_id = 0
            if server.init_latent is not None and latent_model_input is not None:
                latent_full = torch.cat([server.init_latent, latent_model_input], dim=2)
            else:
                latent_full = server.init_latent if latent_model_input is None else latent_model_input

            input_dict = server._prepare_latent_input(latent_full, action_model_input, frame_st_id=server.frame_st_id)
            with torch.no_grad():
                server.transformer(
                    server._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                    update_cache=2,
                    cache_name=server.cache_name,
                    action_mode=False,
                )
                server.transformer(
                    server._repeat_input_for_cfg(input_dict["action_res_lst"]),
                    update_cache=2,
                    cache_name=server.cache_name,
                    action_mode=True,
                )
            server.frame_st_id += int(latent_full.shape[2])

            # rollout actions + latents
            action_chunks: List[np.ndarray] = []
            latent_chunks: List[torch.Tensor] = []
            while sum(x.shape[0] for x in action_chunks) < need_steps:
                actions_chunk, latents_chunk = server._infer({"obs": [], "state": state_cf1}, frame_st_id=server.frame_st_id)
                a = np.asarray(actions_chunk, dtype=np.float32)
                # actions from server are [C, frame_chunk, 1]
                while a.ndim > 2 and a.shape[-1] == 1:
                    a = a[..., 0]
                if a.ndim == 2 and a.shape[0] == 26 and a.shape[1] == int(cfg.frame_chunk_size):
                    a = a.T
                if a.ndim != 2:
                    raise RuntimeError(f"Unexpected action chunk shape: {a.shape}")
                if a.shape[1] > 26:
                    a = a[:, :26]
                action_chunks.append(a)
                latent_chunks.append(latents_chunk.detach().to("cpu"))
                server.frame_st_id += int(cfg.frame_chunk_size)

            action_out = np.concatenate(action_chunks, axis=0)[:need_steps]
            joint_out = action_out.copy()

            # decode video (51 frames, take first need_steps frames)
            pred_latents = torch.cat(latent_chunks, dim=2)[:, :, :need_steps]
            # decode on CPU to avoid OOM (transformer already occupies most GPU memory)
            if next(server.vae.parameters()).device.type != "cpu":
                server.vae = server.vae.to("cpu")
            decoded_video = server.decode_one_video(pred_latents.to("cpu", dtype=server.dtype), "np")[0]

            ep_out.mkdir(parents=True, exist_ok=True)
            (ep_out / "instruction.txt").write_text(prompt + "\n", encoding="utf-8")
            _write_csv_like_sample(ep_out / "action.txt", header_cols, idxs, action_out)
            _write_csv_like_sample(ep_out / "joint.txt", header_cols, idxs, joint_out)
            export_to_video(decoded_video, str(ep_out / "video.mp4"), fps=10)

            print(f"done {ep_id} in {time.time()-t0:.1f}s (in_fps={fps})", flush=True)
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR {ep_id}: {repr(e)}", flush=True)
            torch.cuda.empty_cache()
            continue

    print(f"All done. Output: {out_root}")


if __name__ == "__main__":
    main()

