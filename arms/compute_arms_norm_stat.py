"""
计算 arms_lerobot 的 action 分位数归一化统计量（q01/q99）。

输出：
  <dataset_root>/norm_stat.json

用法：
  conda activate gmr
  python arms/compute_arms_norm_stat.py --dataset-root arms_lerobot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--q-low", type=float, default=0.01)
    ap.add_argument("--q-high", type=float, default=0.99)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    data_dir = dataset_root / "data" / "chunk-000"
    files = sorted(data_dir.glob("episode_*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files under {data_dir}")

    # collect all action rows
    acts = []
    for p in files:
        df = pd.read_parquet(p, columns=["action"])
        a = np.stack(df["action"].to_list()).astype(np.float32)  # [T, D]
        acts.append(a)
    acts_all = np.concatenate(acts, axis=0)  # [N, D]

    q01 = np.quantile(acts_all, args.q_low, axis=0).tolist()
    q99 = np.quantile(acts_all, args.q_high, axis=0).tolist()

    out = {"q01": q01, "q99": q99}
    out_path = dataset_root / "norm_stat.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print("action_dim:", acts_all.shape[1], "rows:", acts_all.shape[0])


if __name__ == "__main__":
    main()

