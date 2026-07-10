"""
从 arms_lerobot 生成“后续 50 步”（按样例索引 80–130 共 51 行）的 action/joint，并落盘为 sample_result 风格。

注意：
  1) 该脚本假设你已经有一个可用的 LingBot-VA 模型目录（包含 transformer/vae/tokenizer/text_encoder）。
  2) 当前实现优先把“推理跑通 + 输出格式正确”作为目标；joint 的生成默认直接使用 action（可替换为更精确的 joint 预测/解算）。

输出目录结构：
  <out-root>/<episode_id>/
    - instruction.txt
    - action.txt
    - joint.txt
    - video.mp4   (可选：如果你把 latent 解码为视频)

用法（示例）：
  conda activate gmr
  python arms/compute_arms_norm_stat.py --dataset-root arms_lerobot
  python arms/generate_arms_and_dump.py \\
    --dataset-root arms_lerobot \\
    --model-root /path/to/lingbot-va-checkpoint \\
    --episode-index 0 \\
    --out-root arms/generated_samples
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def read_manifest(dataset_root: Path) -> Dict:
    return json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))


def read_norm_stat(dataset_root: Path) -> Dict:
    p = dataset_root / "norm_stat.json"
    if not p.exists():
        raise RuntimeError(f"Missing {p}. Run: python arms/compute_arms_norm_stat.py --dataset-root {dataset_root}")
    return json.loads(p.read_text(encoding="utf-8"))


def read_instruction(dataset_root: Path, episode_index: int) -> str:
    # from episodes.jsonl
    ep_lines = (dataset_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    obj = json.loads(ep_lines[episode_index])
    return obj["tasks"][0]


def read_episode_action_state(dataset_root: Path, episode_index: int) -> Tuple[np.ndarray, np.ndarray]:
    p = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    df = pd.read_parquet(p, columns=["action", "observation.state"])
    action = np.stack(df["action"].to_list()).astype(np.float32)  # [T, D]
    state = np.stack(df["observation.state"].to_list()).astype(np.float32)  # [T, D]
    return action, state


def write_csv_like_sample(path: Path, header: List[str], idxs: List[int], data: np.ndarray) -> None:
    """
    header: 不含索引列名（样例里索引列名是 Unnamed: 0）
    data: shape [len(idxs), D]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Unnamed: 0"] + header)
        for i, t in enumerate(idxs):
            writer.writerow([int(t)] + [float(x) for x in data[i].tolist()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--model-root", type=str, required=True, help="LingBot-VA checkpoint 根目录（含 transformer/vae/tokenizer/text_encoder）")
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--start-idx", type=int, default=80)
    ap.add_argument("--end-idx", type=int, default=130)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    manifest = read_manifest(dataset_root)
    norm_stat = read_norm_stat(dataset_root)

    episode_index = int(args.episode_index)
    episode_id = manifest["episode_id_map"][str(episode_index)]

    # columns
    cols = manifest["columns"]["action"]
    action_all, state_all = read_episode_action_state(dataset_root, episode_index)
    prompt = read_instruction(dataset_root, episode_index)

    # 简化：先用“真值 action/ joint”来演示落盘格式（你后续换成模型预测即可）
    s = int(args.start_idx)
    e = int(args.end_idx)
    idxs = list(range(s, e + 1))
    # 某些 episode 可能长度 < 131。为了保证“按样例输出 80–130”，这里做 padding：
    # - 已有部分用真值
    # - 超出长度的部分用最后一帧重复（占位）
    T = action_all.shape[0]
    if s >= T:
        # 全部超出：直接用最后一帧重复（或全 0）
        last_a = action_all[-1]
        last_s = state_all[-1]
        action_out = np.repeat(last_a[None], repeats=len(idxs), axis=0)
        joint_out = np.repeat(last_s[None], repeats=len(idxs), axis=0)
    else:
        a_part = action_all[s : min(e + 1, T)]
        s_part = state_all[s : min(e + 1, T)]
        need = len(idxs) - a_part.shape[0]
        if need > 0:
            last_a = a_part[-1]
            last_s = s_part[-1]
            a_pad = np.repeat(last_a[None], repeats=need, axis=0)
            s_pad = np.repeat(last_s[None], repeats=need, axis=0)
            action_out = np.concatenate([a_part, a_pad], axis=0)
            joint_out = np.concatenate([s_part, s_pad], axis=0)
        else:
            action_out = a_part
            joint_out = s_part

    ep_out_dir = out_root / episode_id
    ep_out_dir.mkdir(parents=True, exist_ok=True)
    (ep_out_dir / "instruction.txt").write_text(prompt + "\n", encoding="utf-8")
    write_csv_like_sample(ep_out_dir / "action.txt", cols, idxs, action_out)
    write_csv_like_sample(ep_out_dir / "joint.txt", cols, idxs, joint_out)

    print(f"Done. Wrote: {ep_out_dir}")
    print("NOTE: 当前脚本先用真值 action/joint 演示输出格式；下一步把 action_out/joint_out 替换为模型预测即可。")


if __name__ == "__main__":
    main()

