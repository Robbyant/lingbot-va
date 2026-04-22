"""
对 arms/test/ 批量推理（不解码视频）：
- 输入：每个 episode 目录包含 video.mp4（16帧）、instruction.txt、joint.txt（16行）
- 输出：每个 episode 输出
  - action.txt / joint.txt（80-130 共 51 行）
  - pred_latents.pt（预测视频 latents，供后续单独解码为 video.mp4）

这样把“推理”和“视频解码”拆开，可以避免显存 OOM（transformer + VAE 同时上 GPU）
并且比 CPU 解码视频快很多。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict

# allow running as a standalone script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    device = server.device
    dtype = server.dtype
    video_tensor = _frames_to_tensor(frames_rgb, server.job_config.height, server.job_config.width, device=device, dtype=dtype)

    # temporarily move VAE to GPU for encoding
    vae_was_on_cpu = next(server.vae.parameters()).device.type == "cpu"
    if vae_was_on_cpu:
        server.vae = server.vae.to(device).to(dtype)

    enc = server.vae.encode(video_tensor)
    mu = enc.latent_dist.mean
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
    data = df.iloc[:, 1:].to_numpy(dtype=np.float32)  # [T,26]
    hist = data[: min(history_len, data.shape[0])]
    state_cf1 = hist.T[:, :, None]  # [26,F,1]
    state_cf1 = np.pad(state_cf1, ((0, 3), (0, 0), (0, 0)), mode="constant", constant_values=0.0)  # -> [29,F,1]
    return state_cf1


def _load_history_joint_26(test_ep_dir: Path, history_len: int) -> np.ndarray:
    df = pd.read_csv(test_ep_dir / "joint.txt")
    data = df.iloc[:, 1:].to_numpy(dtype=np.float32)  # [T,26]
    return data[: min(history_len, data.shape[0])]


def _apply_finger_hold_then_grasp(
    out_t26: np.ndarray,
    hist_t26: np.ndarray,
    grasp_steps: int,
) -> np.ndarray:
    """
    Scheme A:
    - finger dims (14:26) hold the last observed pose for most steps
    - last `grasp_steps` steps switch to a "closed" template (max over history)
    """
    if out_t26.shape[1] != 26:
        return out_t26
    if hist_t26.size == 0:
        return out_t26

    grasp_steps = int(max(0, min(grasp_steps, out_t26.shape[0])))
    fingers_last = hist_t26[-1, 14:26].copy()
    fingers_closed = hist_t26[:, 14:26].max(axis=0).copy()

    out = out_t26.copy()
    out[:, 14:26] = fingers_last[None]
    if grasp_steps > 0:
        out[-grasp_steps:, 14:26] = fingers_closed[None]
    return out


def _write_csv_like_sample(path: Path, header_cols: List[str], idxs: List[int], data_t26: np.ndarray) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Unnamed: 0"] + header_cols)
        for i, t in enumerate(idxs):
            writer.writerow([int(t)] + [float(x) for x in data_t26[i].tolist()])


def _action16_to_action26(action_cf: np.ndarray, hist_t26: np.ndarray) -> np.ndarray:
    """
    VA_Server.postprocess_action returns only used_action_channel_ids (16 dims) for arms config:
      - 14 joint dims (left7+right7)
      - 2 gripper dims (not the 12 finger dims in ARMS csv schema)

    For submission/action.txt schema (26 dims), we:
      - take the first 14 dims as joints
      - fill finger dims (14:26) using history template (handled later by _apply_finger_hold_then_grasp)
    """
    if action_cf.ndim != 2:
        return action_cf
    if action_cf.shape[1] != 16:
        return action_cf
    out = np.zeros((action_cf.shape[0], 26), dtype=np.float32)
    out[:, :14] = action_cf[:, :14].astype(np.float32)
    if hist_t26.size > 0 and hist_t26.shape[1] == 26:
        out[:, 14:26] = hist_t26[-1, 14:26][None].astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", type=str, default="arms/test")
    ap.add_argument("--model-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, default="arms/test_generated_latents")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--history-len", type=int, default=16)
    ap.add_argument("--start-idx", type=int, default=80)
    ap.add_argument("--end-idx", type=int, default=130)
    ap.add_argument("--max-episodes", type=int, default=-1)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--overwrite-existing", action="store_true", help="重跑并覆盖已存在的输出（action/joint/pred_latents）")
    ap.add_argument("--no-pred-latents", action="store_true", help="不保存 pred_latents.pt（加速，且不做 latents 拼接）")
    ap.add_argument(
        "--start-from-test-end",
        action="store_true",
        help="从 arms/test/<ep>/action.txt 的最后一行 idx+1 开始连续预测，并输出 --predict-steps 行（覆盖 --start-idx/--end-idx）",
    )
    ap.add_argument("--predict-steps", type=int, default=51, help="--start-from-test-end 时预测多少步（默认 51）")
    ap.add_argument("--grasp-steps", type=int, default=10, help="方案A：最后多少步把手指切到闭合姿态")
    args = ap.parse_args()

    test_root = Path(args.test_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # header columns from any action.txt (same schema)
    any_action = next(iter(sorted(test_root.glob("*/action.txt"))))
    header = any_action.read_text(encoding="utf-8").splitlines()[0].strip().split(",")
    header_cols = header[1:]

    # norm_stat from arms_lerobot (train stats)
    norm_stat_path = Path("arms_lerobot/norm_stat.json")
    if not norm_stat_path.exists():
        raise FileNotFoundError("需要先生成 arms_lerobot/norm_stat.json（用 compute_arms_norm_stat.py）")
    norm_stat = json.loads(norm_stat_path.read_text(encoding="utf-8"))
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

    ep_dirs = sorted([p for p in test_root.iterdir() if p.is_dir()])
    if args.max_episodes > 0:
        ep_dirs = ep_dirs[: int(args.max_episodes)]

    fixed_start = int(args.start_idx)
    fixed_end = int(args.end_idx)
    fixed_idxs = list(range(fixed_start, fixed_end + 1))
    fixed_need_steps = len(fixed_idxs)

    for ep_dir in ep_dirs:
        ep_id = ep_dir.name
        t0 = time.time()
        try:
            ep_out = out_root / ep_id
            if (
                args.skip_existing
                and not args.overwrite_existing
                and (ep_out / "action.txt").exists()
                and (ep_out / "joint.txt").exists()
                and (args.no_pred_latents or (ep_out / "pred_latents.pt").exists())
            ):
                print(f"skip {ep_id}", flush=True)
                continue

            prompt = (ep_dir / "instruction.txt").read_text(encoding="utf-8").strip()
            frames_rgb, _fps = _read_video_frames_rgb(ep_dir / "video.mp4")
            if len(frames_rgb) == 0:
                raise RuntimeError("Empty video")

            server.infer({"reset": True, "prompt": prompt})

            latents_hist_full = _encode_video_to_latents_norm(server, frames_rgb)  # [1,48,4,16,16]
            hist_latent_frames = int(np.ceil(min(args.history_len, len(frames_rgb)) / 4.0))
            latents_hist = latents_hist_full[:, :, :hist_latent_frames]

            server.init_latent = latents_hist[:, :, :1].to(server.dtype)
            latent_model_input = latents_hist[:, :, 1:].to(server.dtype) if hist_latent_frames > 1 else None

            state_cf1 = _load_history_joint_cf1(ep_dir, history_len=int(args.history_len))
            hist_26 = _load_history_joint_26(ep_dir, history_len=int(args.history_len))
            action_model_input = server.preprocess_action(state_cf1).to(device=server.device, dtype=server.dtype)

            if args.start_from_test_end:
                df_hist = pd.read_csv(ep_dir / "action.txt")
                if df_hist.shape[0] == 0:
                    raise RuntimeError("Empty test action.txt")
                last_idx = int(df_hist.iloc[-1, 0])
                start_idx = last_idx + 1
                need_steps = int(args.predict_steps)
                idxs = list(range(start_idx, start_idx + need_steps))
            else:
                idxs = fixed_idxs
                need_steps = fixed_need_steps

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

            action_chunks: List[np.ndarray] = []
            latent_chunks: List[torch.Tensor] = []
            while sum(x.shape[0] for x in action_chunks) < need_steps:
                actions_chunk, latents_chunk = server._infer({"obs": [], "state": state_cf1}, frame_st_id=server.frame_st_id)
                a = np.asarray(actions_chunk, dtype=np.float32)
                while a.ndim > 2 and a.shape[-1] == 1:
                    a = a[..., 0]
                # Expected shapes:
                # - (16, F) from VA_Server.postprocess_action (C,F)
                # - (26, F) from older pipelines
                if a.ndim == 2 and a.shape[0] in (16, 26) and a.shape[1] == int(cfg.frame_chunk_size):
                    a = a.T
                # If action is still channel-first, transpose.
                if a.ndim == 2 and a.shape[0] in (16, 26) and a.shape[1] != int(cfg.frame_chunk_size):
                    # already (T,C) or something else; keep
                    pass

                # Convert 16-d outputs to 26-d csv schema
                if a.ndim == 2 and a.shape[1] == 16:
                    a = _action16_to_action26(a, hist_26)

                if a.ndim == 2 and a.shape[1] > 26:
                    a = a[:, :26]
                action_chunks.append(a)
                if not args.no_pred_latents:
                    latent_chunks.append(latents_chunk.detach().to("cpu"))
                server.frame_st_id += int(cfg.frame_chunk_size)

            action_out = np.concatenate(action_chunks, axis=0)[:need_steps]
            if action_out.ndim != 2 or action_out.shape[1] != 26:
                raise RuntimeError(f"Bad action_out shape {getattr(action_out,'shape',None)}, expected (T,26)")
            # Scheme A: fingers hold then grasp
            action_out = _apply_finger_hold_then_grasp(action_out, hist_26, grasp_steps=int(args.grasp_steps))
            joint_out = action_out.copy()

            ep_out.mkdir(parents=True, exist_ok=True)
            (ep_out / "instruction.txt").write_text(prompt + "\n", encoding="utf-8")
            _write_csv_like_sample(ep_out / "action.txt", header_cols, idxs, action_out)
            _write_csv_like_sample(ep_out / "joint.txt", header_cols, idxs, joint_out)
            if not args.no_pred_latents:
                pred_latents = torch.cat(latent_chunks, dim=2)[:, :, :need_steps].contiguous()  # [1,48,T,16,16]
                torch.save({"latents": pred_latents, "fps": 10}, ep_out / "pred_latents.pt")

            print(f"done {ep_id} in {time.time()-t0:.1f}s", flush=True)
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR {ep_id}: {repr(e)}", flush=True)
            torch.cuda.empty_cache()
            continue

    print(f"All done. Output: {out_root}")


if __name__ == "__main__":
    main()

