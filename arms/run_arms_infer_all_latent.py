"""
批量跑 arms_lerobot 的 latent 推理，并按比赛格式落盘到一个目录。

特点：
- 逐 episode 串行推理，避免显存/CPU 被并发打爆
- 每个 episode 复用同一个 VA_Server（只 reset prompt + 重建 kv cache）
- 出错会记录到 errors.jsonl，继续下一个 episode

示例：
  /home/landscape-layton-ljw/miniconda3/envs/gmr/bin/python arms/run_arms_infer_all_latent.py \
    --dataset-root arms_lerobot \
    --model-root models/lingbot-va-base \
    --latents-root arms_lerobot/latents_lingbot \
    --out-root arms/generated_samples_latent_all \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

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


def _load_episode_meta(dataset_root: Path, episode_index: int) -> Tuple[str, int, int]:
    line = (dataset_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[episode_index]
    ep = json.loads(line)
    prompt = ep["tasks"][0]
    cfg0 = ep["action_config"][0]
    return prompt, int(cfg0["start_frame"]), int(cfg0["end_frame"])


def _load_latents_5d(latent_path: Path, device: torch.device) -> torch.Tensor:
    payload = torch.load(latent_path, map_location="cpu")
    latent_flat: torch.Tensor = payload["latent"]  # [N, C]
    f = int(payload["latent_num_frames"])
    h = int(payload["latent_height"])
    w = int(payload["latent_width"])
    c = int(latent_flat.shape[-1])
    latents = rearrange(latent_flat, "(f h w) c -> 1 c f h w", f=f, h=h, w=w, c=c)
    return latents.to(device)


def _pad_norm_stat_to_30(norm_stat: Dict) -> Dict:
    out = dict(norm_stat)
    for k in ["q01", "q99"]:
        if k in out and len(out[k]) < 30:
            pad_val = 0.0 if k == "q01" else 1.0
            out[k] = list(out[k]) + [pad_val] * (30 - len(out[k]))
    return out


def _state_to_cf1_padded_29(state_t26: np.ndarray) -> np.ndarray:
    # input: [T,26] -> [29,T,1]
    state_cf1 = state_t26.astype(np.float32).T[:, :, None]
    state_cf1 = np.pad(state_cf1, ((0, 3), (0, 0), (0, 0)), mode="constant", constant_values=0.0)
    return state_cf1


def _actions_to_t26(actions_any: np.ndarray, frame_chunk_size: int) -> np.ndarray:
    a = np.asarray(actions_any, dtype=np.float32)
    while a.ndim > 2 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim == 2 and a.shape[0] == 26 and a.shape[1] == frame_chunk_size:
        a = a.T
    if a.ndim == 2 and a.shape[0] in (26, 29, 30) and a.shape[1] != 26:
        a = a.T
    if a.ndim != 2:
        raise RuntimeError(f"Unexpected action chunk shape: {a.shape}")
    if a.shape[1] > 26:
        a = a[:, :26]
    if a.shape[1] != 26:
        raise RuntimeError(f"Unexpected action chunk shape after trim: {a.shape}")
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--model-root", type=str, required=True)
    ap.add_argument("--latents-root", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--history-len", type=int, default=16)
    ap.add_argument("--start-idx", type=int, default=80)
    ap.add_argument("--end-idx", type=int, default=130)
    ap.add_argument("--max-episodes", type=int, default=-1, help="调试用，<=0 表示全量")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    latents_root = Path(args.latents_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(dataset_root)
    cols = manifest["columns"]["action"]

    norm_stat = json.loads((dataset_root / "norm_stat.json").read_text(encoding="utf-8"))
    norm_stat = _pad_norm_stat_to_30(norm_stat)

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
    device = server.device

    episodes = (dataset_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    n = len(episodes) if args.max_episodes <= 0 else min(len(episodes), int(args.max_episodes))

    errors_path = out_root / "errors.jsonl"
    if errors_path.exists():
        errors_path.unlink()

    for episode_index in range(n):
        t0 = time.time()
        try:
            episode_id = manifest["episode_id_map"][str(episode_index)]
            prompt, start_frame, end_frame = _load_episode_meta(dataset_root, episode_index)
            print(f"[{episode_index+1}/{n}] start {episode_id}", flush=True)

            # reset prompt
            server.infer({"reset": True, "prompt": prompt})

            # state
            parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
            df = pd.read_parquet(parquet_path, columns=["observation.state"])
            state = np.stack(df["observation.state"].to_list()).astype(np.float32)  # [T, 26]

            hist_len = min(int(args.history_len), int(state.shape[0]))
            hist_latent_frames = int(np.ceil(hist_len / 4.0))

            latent_path = latents_root / "chunk-000" / "observation.images.front" / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            if not latent_path.exists():
                raise FileNotFoundError(str(latent_path))

            latents_5d = _load_latents_5d(latent_path, device=device)
            if latents_5d.shape[2] < hist_latent_frames:
                raise RuntimeError(f"latent frames insufficient: need {hist_latent_frames}, got {latents_5d.shape[2]}")
            latents_hist = latents_5d[:, :, :hist_latent_frames]

            server.init_latent = latents_hist[:, :, :1].to(server.dtype)
            latent_model_input = latents_hist[:, :, 1:].to(server.dtype) if hist_latent_frames > 1 else None

            state_cf1 = _state_to_cf1_padded_29(state[:hist_len])
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

            # rollout actions
            need = int(args.end_idx) - int(args.start_idx) + 1
            chunks: List[np.ndarray] = []
            for _ in range(1000):
                out = server.infer({"obs": [], "state": state_cf1})
                if "action" in out:
                    chunks.append(_actions_to_t26(out["action"], frame_chunk_size=int(cfg.frame_chunk_size)))
                if sum(x.shape[0] for x in chunks) >= need:
                    break
            if not chunks:
                raise RuntimeError("No action returned.")

            action_out = np.concatenate(chunks, axis=0)[:need]
            idxs = list(range(int(args.start_idx), int(args.end_idx) + 1))
            if action_out.shape[0] < need:
                pad = np.repeat(action_out[-1:], repeats=need - action_out.shape[0], axis=0)
                action_out = np.concatenate([action_out, pad], axis=0)

            joint_out = action_out.copy()

            ep_out_dir = out_root / episode_id
            ep_out_dir.mkdir(parents=True, exist_ok=True)
            (ep_out_dir / "instruction.txt").write_text(prompt + "\n", encoding="utf-8")
            write_csv_like_sample(ep_out_dir / "action.txt", cols, idxs, action_out)
            write_csv_like_sample(ep_out_dir / "joint.txt", cols, idxs, joint_out)
            print(f"[{episode_index+1}/{n}] done {episode_id} in {time.time()-t0:.1f}s", flush=True)

        except Exception as e:
            rec = {
                "episode_index": episode_index,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{episode_index+1}/{n}] ERROR in {time.time()-t0:.1f}s: {rec['error']}", flush=True)
            continue

    print(f"Done. Wrote results to: {out_root}")
    if errors_path.exists():
        print(f"Errors (if any): {errors_path}")


if __name__ == "__main__":
    main()

