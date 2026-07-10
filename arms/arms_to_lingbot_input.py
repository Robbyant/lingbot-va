import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ArmsEpisode:
    root: Path
    video_path: Path
    instruction_path: Path
    joint_path: Path
    action_path: Path


def load_arms_episode(episode_dir: str | Path) -> ArmsEpisode:
    root = Path(episode_dir)
    return ArmsEpisode(
        root=root,
        video_path=root / "video.mp4",
        instruction_path=root / "instruction.txt",
        joint_path=root / "joint.txt",
        action_path=root / "action.txt",
    )


def _read_single_line_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_csv_with_first_index_col(path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    arms 的 joint/action 文件是 CSV：
    - 第一列是时间索引（表头有时为空或 Unnamed:0）
    - 后面才是真正的关节/手指维度

    返回：
    - data: shape [T, D]，不含索引列
    - columns: D 个维度对应的列名（不含索引列）
    """
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if len(r) > 0]

    # drop the first index column
    columns = header[1:]
    data = np.asarray([[float(x) for x in r[1:]] for r in rows], dtype=np.float32)
    return data, columns


def decode_video_frames(
    video_path: Path,
    frame_indices: List[int],
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: List[np.ndarray] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            cap.release()
            raise RuntimeError(f"Cannot read frame={idx} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def build_obs_dict_from_arms(
    episode: ArmsEpisode,
    history_len: int = 16,
    start_frame: int = 0,
    cam_key: str = "front",
) -> Dict[str, Any]:
    """
    生成一个“类似 wan_va_server.py 期待的 obs dict”的最小结构：
    - obs['obs']  : List[Dict[str, np.ndarray]]，每个元素是一个时间步，各相机一张 HWC RGB uint8
    - obs['state']: np.ndarray，历史 state 序列（这里直接用 joint 的 D 维向量序列）
    - obs['action_history']: np.ndarray，历史 action 序列（用于你自己做 KV cache / 条件化时用）
    - obs['prompt']: 指令文本

    注意：
    - 这个脚本只负责“把 arms 数据读出来并组织好”，不直接调用 LingBot-VA 的 server。
    - state/action 的 shape 与下游你改的 loader/adapter 相关，这里先提供最原始的 [T, D]。
    """
    prompt = _read_single_line_text(episode.instruction_path)
    joint, joint_cols = _read_csv_with_first_index_col(episode.joint_path)
    action, action_cols = _read_csv_with_first_index_col(episode.action_path)

    T = joint.shape[0]
    if action.shape[0] != T:
        raise ValueError(f"joint/action length mismatch: joint={T}, action={action.shape[0]}")

    end = min(start_frame + history_len, T)
    frame_ids = list(range(start_frame, end))

    frames = decode_video_frames(episode.video_path, frame_ids)
    obs_seq = [{cam_key: f.astype(np.uint8)} for f in frames]

    obs: Dict[str, Any] = {
        "obs": obs_seq,
        "state": joint[start_frame:end],
        "action_history": action[start_frame:end],
        "prompt": prompt,
        "meta": {
            "joint_columns": joint_cols,
            "action_columns": action_cols,
            "start_frame": start_frame,
            "history_len": history_len,
        },
    }
    return obs


if __name__ == "__main__":
    ep = load_arms_episode("arms/train/1_1")
    obs = build_obs_dict_from_arms(ep, history_len=16, start_frame=0, cam_key="front")
    print("prompt:", obs["prompt"])
    print("obs len:", len(obs["obs"]), "frame shape:", obs["obs"][0]["front"].shape)
    print("state shape:", obs["state"].shape, "action_history shape:", obs["action_history"].shape)
    print("action_dim:", obs["action_history"].shape[-1])
