"""
把 arms 数据集转换成 LeRobot 风格目录结构，尽量贴近本仓库 README/LeRobot 的期望：

输入（每条轨迹一个文件夹）：
  <arms_root>/<episode_id>/
    - video.mp4
    - instruction.txt
    - joint.txt
    - action.txt

输出（LeRobot 数据集目录）：
  <out_root>/
    - meta/episodes.jsonl
    - videos/chunk-000/observation.images.front/episode_000000.mp4
    - data/chunk-000/episode_000000.parquet

说明：
  - 本脚本保持 arms 的动作/状态维度不变：CSV 去掉第一列索引后是 D=26。
  - parquet schema 采用与现有 LeRobot 数据类似的“向量列”：
      action: float32[D]
      observation.state: float32[D]
    以及索引/时间戳列。
  - 该脚本不会提取 latents（.pth）。latents 可以后续用 Wan2.2 VAE 批处理提取。

运行示例（建议在有 pyarrow 的环境，比如 conda env gmr）：
  conda activate gmr
  python arms/convert_arms_to_lerobot.py --arms-root arms/train --out-root arms_lerobot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import cv2  # optional, for fps
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class ArmsEpisodePaths:
    episode_dir: Path
    video_path: Path
    instruction_path: Path
    joint_path: Path
    action_path: Path


def _read_single_line_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_csv_with_first_index_col(path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    返回：
      data: [T, D] float32（不含第一列索引）
      columns: D 列名（不含第一列索引）
      index: [T] int（第一列索引）
    """
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if len(r) > 0 and any(x.strip() for x in r)]

    columns = header[1:]
    idx = np.asarray([int(float(r[0])) for r in rows], dtype=np.int32)
    data = np.asarray([[float(x) for x in r[1:]] for r in rows], dtype=np.float32)
    return data, columns, idx


def _get_video_fps(video_path: Path) -> float | None:
    if cv2 is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps if fps > 0 else None


def _discover_arms_episodes(arms_root: Path) -> List[ArmsEpisodePaths]:
    episodes: List[ArmsEpisodePaths] = []
    for child in sorted(arms_root.iterdir()):
        if not child.is_dir():
            continue
        video = child / "video.mp4"
        instr = child / "instruction.txt"
        joint = child / "joint.txt"
        action = child / "action.txt"
        if video.exists() and instr.exists() and joint.exists() and action.exists():
            episodes.append(
                ArmsEpisodePaths(
                    episode_dir=child,
                    video_path=video,
                    instruction_path=instr,
                    joint_path=joint,
                    action_path=action,
                )
            )
    return episodes


def _ensure_dirs(out_root: Path) -> Dict[str, Path]:
    meta = out_root / "meta"
    videos = out_root / "videos" / "chunk-000" / "observation.images.front"
    data = out_root / "data" / "chunk-000"
    for p in (meta, videos, data):
        p.mkdir(parents=True, exist_ok=True)
    return {"meta": meta, "videos": videos, "data": data}


def _write_episode_parquet(
    out_parquet: Path,
    episode_index: int,
    task_index: int,
    action: np.ndarray,  # [T, D]
    state: np.ndarray,  # [T, D]
    frame_index: np.ndarray,  # [T]
    fps: float | None,
) -> None:
    T, D = action.shape
    if state.shape != (T, D):
        raise ValueError(f"state shape {state.shape} != action shape {action.shape}")

    # timestamps: seconds; if fps unknown, store 0..T-1
    if fps is None or fps <= 0:
        ts = frame_index.astype(np.float32)
    else:
        ts = frame_index.astype(np.float32) / np.float32(fps)

    # Use FixedSizeList for vector columns (consistent with how we read other datasets).
    action_arr = pa.FixedSizeListArray.from_arrays(pa.array(action.reshape(-1), type=pa.float32()), D)
    state_arr = pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1), type=pa.float32()), D)

    table = pa.table(
        {
            "episode_index": pa.array(np.full((T,), episode_index, dtype=np.int32)),
            "index": pa.array(np.arange(T, dtype=np.int32)),
            "frame_index": pa.array(frame_index.astype(np.int32)),
            "task_index": pa.array(np.full((T,), task_index, dtype=np.int32)),
            "timestamp": pa.array(ts.astype(np.float32)),
            "action": action_arr,
            "observation.state": state_arr,
        }
    )
    pq.write_table(table, out_parquet)


def convert(arms_root: Path, out_root: Path) -> None:
    eps = _discover_arms_episodes(arms_root)
    if not eps:
        raise RuntimeError(f"No valid episodes found under {arms_root}")

    dirs = _ensure_dirs(out_root)
    episodes_jsonl = dirs["meta"] / "episodes.jsonl"

    # overwrite episodes.jsonl
    if episodes_jsonl.exists():
        episodes_jsonl.unlink()

    id_map: Dict[int, str] = {}
    action_columns_ref: List[str] | None = None
    joint_columns_ref: List[str] | None = None

    for episode_index, ep in enumerate(eps):
        task = _read_single_line_text(ep.instruction_path)
        state, joint_cols, state_idx = _read_csv_with_first_index_col(ep.joint_path)
        action, action_cols, action_idx = _read_csv_with_first_index_col(ep.action_path)

        # sanity checks
        if state.shape[0] != action.shape[0]:
            raise ValueError(f"{ep.episode_dir}: joint/action length mismatch")
        if not np.array_equal(state_idx, action_idx):
            raise ValueError(f"{ep.episode_dir}: joint/action time index mismatch")

        if joint_columns_ref is None:
            joint_columns_ref = joint_cols
        elif joint_cols != joint_columns_ref:
            raise ValueError(f"{ep.episode_dir}: joint columns differ from first episode")

        if action_columns_ref is None:
            action_columns_ref = action_cols
        elif action_cols != action_columns_ref:
            raise ValueError(f"{ep.episode_dir}: action columns differ from first episode")

        T = int(state.shape[0])
        fps = _get_video_fps(ep.video_path)

        # write parquet
        out_parquet = dirs["data"] / f"episode_{episode_index:06d}.parquet"
        _write_episode_parquet(
            out_parquet=out_parquet,
            episode_index=episode_index,
            task_index=0,
            action=action,
            state=state,
            frame_index=state_idx,
            fps=fps,
        )

        # copy video
        out_video = dirs["videos"] / f"episode_{episode_index:06d}.mp4"
        if not out_video.exists():
            shutil.copy2(ep.video_path, out_video)

        # write episodes.jsonl line
        line = {
            "episode_index": episode_index,
            "tasks": [task],
            "length": T,
            "action_config": [
                {
                    "start_frame": int(state_idx[0]),
                    "end_frame": int(state_idx[-1]) + 1,  # end_frame is exclusive in many tools
                    "action_text": task,
                    "skill": "",
                }
            ],
        }
        with episodes_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        id_map[episode_index] = ep.episode_dir.name

    # write a small manifest for debugging
    manifest = {
        "arms_root": str(arms_root),
        "num_episodes": len(eps),
        "action_dim": len(action_columns_ref or []),
        "state_dim": len(joint_columns_ref or []),
        "columns": {
            "action": action_columns_ref,
            "state": joint_columns_ref,
        },
        "episode_id_map": id_map,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-root", type=str, default="arms/train", help="arms/train 目录")
    ap.add_argument("--out-root", type=str, default="arms_lerobot", help="输出 LeRobot 数据集目录")
    args = ap.parse_args()

    convert(Path(args.arms_root), Path(args.out_root))
    print(f"Done. Wrote LeRobot dataset to: {args.out_root}")


if __name__ == "__main__":
    main()

