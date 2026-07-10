"""
用 lingbot-va-base（本地模型目录）对 arms_lerobot 做推理，并按样例 80–130（51 行）落盘。

当前实现目标：把“能跑通模型推理 → 拿到动作序列 → 写成 action.txt/joint.txt”这条链路打通。
joint 的输出默认直接复制 action（同维度），你后续可以替换为更合理的 joint 预测/解算。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict
from einops import rearrange

# allow running as a standalone script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.wan_va_server import VA_Server
from wan_va.configs.va_arms_cfg import va_arms_cfg

from arms.generate_arms_and_dump import read_manifest, write_csv_like_sample


def load_episode_prompt(dataset_root: Path, episode_index: int) -> str:
    line = (dataset_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[episode_index]
    return json.loads(line)["tasks"][0]


def load_episode_action_range(dataset_root: Path, episode_index: int) -> tuple[int, int]:
    line = (dataset_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[episode_index]
    ep = json.loads(line)
    cfg0 = ep["action_config"][0]
    return int(cfg0["start_frame"]), int(cfg0["end_frame"])


def load_latents_5d(latent_path: Path, device: torch.device) -> torch.Tensor:
    payload = torch.load(latent_path, map_location="cpu")
    latent_flat: torch.Tensor = payload["latent"]  # [N, C]
    f = int(payload["latent_num_frames"])
    h = int(payload["latent_height"])
    w = int(payload["latent_width"])
    c = int(latent_flat.shape[-1])
    latents = rearrange(latent_flat, "(f h w) c -> 1 c f h w", f=f, h=h, w=w, c=c)
    return latents.to(device)


def load_video_frames(video_path: Path, frame_ids: List[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frames = []
    for idx in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read frame {idx} from {video_path}")
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def build_obs(front_frames_rgb: List[np.ndarray], state_seq: np.ndarray, prompt: str) -> Dict:
    obs_seq = [{"observation.images.front": fr.astype(np.uint8)} for fr in front_frames_rgb]
    # VA_Server.preprocess_action expects numpy array shaped [C, F, H]
    # We store state_seq as [F, C] -> [C, F, 1]
    state_cf1 = state_seq.astype(np.float32).T[:, :, None]
    return {"obs": obs_seq, "state": state_cf1, "prompt": prompt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--model-root", type=str, required=True, help="本地 lingbot-va-base 目录")
    ap.add_argument("--latents-root", type=str, default="", help="可选：优先从这里读取 latents（例如 <dataset-root>/latents_lingbot）")
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--history-len", type=int, default=16)
    ap.add_argument("--start-idx", type=int, default=80)
    ap.add_argument("--end-idx", type=int, default=130)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--debug-shapes", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    manifest = read_manifest(dataset_root)
    episode_index = int(args.episode_index)
    episode_id = manifest["episode_id_map"][str(episode_index)]
    cols = manifest["columns"]["action"]

    # load norm stat
    norm_stat = json.loads((dataset_root / "norm_stat.json").read_text(encoding="utf-8"))
    # pad norm stat to 30 dims (lingbot-va-base expects 30)
    for k in ["q01", "q99"]:
        if k in norm_stat and len(norm_stat[k]) < 30:
            pad_val = 0.0 if k == "q01" else 1.0
            norm_stat[k] = list(norm_stat[k]) + [pad_val] * (30 - len(norm_stat[k]))

    # configure model
    cfg = EasyDict(va_arms_cfg)
    cfg.wan22_pretrained_model_name_or_path = str(Path(args.model_root).resolve())
    cfg.save_root = str(Path(args.out_root).resolve())
    # 16G 显存下建议 offload VAE/text_encoder 到 CPU，仅 transformer 在 GPU
    cfg.enable_offload = True
    cfg.param_dtype = torch.float16
    cfg.local_rank = int(str(args.device).split(":")[-1]) if "cuda" in args.device else 0
    cfg.norm_stat = norm_stat

    # init server
    if "cuda" in args.device:
        torch.cuda.empty_cache()
    server = VA_Server(cfg)
    device = server.device

    # reset with prompt
    prompt = load_episode_prompt(dataset_root, episode_index)
    server.infer({"reset": True, "prompt": prompt})

    # load episode state and frames for history
    parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    df = pd.read_parquet(parquet_path, columns=["observation.state"])
    state = np.stack(df["observation.state"].to_list()).astype(np.float32)  # [T, 26]

    # --- latent-based KV cache prefill (skip slow CPU VAE encoding) ---
    start_frame, end_frame = load_episode_action_range(dataset_root, episode_index)
    latents_root = Path(args.latents_root) if str(args.latents_root).strip() else (dataset_root / "latents")
    latent_path = latents_root / "chunk-000" / "observation.images.front" / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
    if not latent_path.exists():
        raise FileNotFoundError(
            f"找不到 latent 文件：{latent_path}\n"
            f"请先用 `arms/extract_latents_arms_lerobot.py`（wan22-path 指向 models/lingbot-va-base）提取到该目录。"
        )

    # VAE temporal downsample is 4 for lingbot-va-base
    hist_len = min(int(args.history_len), int(state.shape[0]))
    hist_latent_frames = int(np.ceil(hist_len / 4.0))
    latents_5d = load_latents_5d(latent_path, device=device)  # [1,48,F',16,16]
    if latents_5d.shape[2] < hist_latent_frames:
        raise RuntimeError(f"latent 帧数不足：need {hist_latent_frames}, got {latents_5d.shape[2]}")
    latents_hist = latents_5d[:, :, :hist_latent_frames]

    # init_latent = first latent frame, remaining go through kv cache
    server.init_latent = latents_hist[:, :, :1].to(server.dtype)
    latent_model_input = latents_hist[:, :, 1:].to(server.dtype) if hist_latent_frames > 1 else None

    # action/state history still uses original frames
    state_hist = state[:hist_len]
    state_cf1 = state_hist.astype(np.float32).T[:, :, None]  # [26,F,1]
    # pad to 29 dims so VA_Server.preprocess_action 的 +1 padding 变成 30 dims
    state_cf1 = np.pad(state_cf1, ((0, 3), (0, 0), (0, 0)), mode="constant", constant_values=0.0)
    action_model_input = server.preprocess_action(state_cf1).to(device=device, dtype=server.dtype)

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

    # rollout: this repository's server produces one chunk of actions at a time.
    # Here we call infer repeatedly until we have enough timesteps to write 80-130.
    actions_pred: List[np.ndarray] = []
    max_calls = 100
    for _ in range(max_calls):
        out = server.infer({"obs": [], "state": state_cf1})
        if "action" in out:
            if args.debug_shapes:
                a = out["action"]
                print("infer action.shape =", getattr(a, "shape", None))
            a = np.asarray(out["action"], dtype=np.float32)
            # server returns (C, frame_chunk, 1) in our config
            while a.ndim > 2 and a.shape[-1] == 1:
                a = a[..., 0]
            if a.ndim == 2 and a.shape[0] == 26 and a.shape[1] == cfg.frame_chunk_size:
                a = a.T  # -> (frame_chunk, 26)
            actions_pred.append(a)
        if sum(x.shape[0] for x in actions_pred) >= (args.end_idx - args.start_idx + 1):
            break

    if not actions_pred:
        raise RuntimeError("No action returned from server inference.")

    action_out = np.concatenate([np.asarray(x, dtype=np.float32) for x in actions_pred], axis=0)
    # normalize shape to [T, 26]
    if action_out.ndim == 1:
        action_out = action_out[None]
    while action_out.ndim > 2 and action_out.shape[-1] == 1:
        action_out = action_out[..., 0]
    # sometimes returned as [C, T]
    if action_out.ndim == 2 and action_out.shape[0] in (26, 29, 30) and action_out.shape[1] != 26:
        action_out = action_out.T
    if action_out.ndim != 2:
        raise RuntimeError(f"Unexpected action_out.ndim={action_out.ndim}, shape={action_out.shape}")
    if action_out.shape[1] > 26:
        action_out = action_out[:, :26]
    if action_out.shape[1] != 26:
        raise RuntimeError(f"Unexpected action_out shape={action_out.shape}, expected (*,26)")

    # pad/truncate to 51 rows
    idxs = list(range(args.start_idx, args.end_idx + 1))
    need = len(idxs)
    if action_out.shape[0] < need:
        pad = np.repeat(action_out[-1:], repeats=need - action_out.shape[0], axis=0)
        action_out = np.concatenate([action_out, pad], axis=0)
    action_out = action_out[:need]
    joint_out = action_out.copy()

    out_dir = Path(args.out_root) / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "instruction.txt").write_text(prompt + "\n", encoding="utf-8")
    write_csv_like_sample(out_dir / "action.txt", cols, idxs, action_out)
    write_csv_like_sample(out_dir / "joint.txt", cols, idxs, joint_out)

    print(f"Done. Wrote: {out_dir}")


if __name__ == "__main__":
    main()

