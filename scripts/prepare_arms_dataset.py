# Prepare repo-local ./arms (dual-arm single-cam) into a training-ready folder.
#
# Output structure (dataset_root):
#   meta/episodes.jsonl
#   videos/chunk-000/observation.images.cam_high/episode_000000.mp4
#   actions/episode_000000.npy         # float32 [T,30]
#   norm_stat.json                      # q01/q99 over 30 dims
#
# NOTE: You still need to extract Wan2.2 VAE latents into:
#   latents/chunk-000/observation.images.cam_high/episode_000000_0_T.pth
#
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


LEFT_JOINT_COLS = [f"idx{13+i}_left_arm_joint{i+1}_position" for i in range(7)]
RIGHT_JOINT_COLS = [f"idx{20+i}_right_arm_joint{i+1}_position" for i in range(7)]


def _read_csv_rows(path: Path) -> tuple[list[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        rows = []
        for r in reader:
            if not r:
                continue
            rows.append([float(x) for x in r])
    arr = np.asarray(rows, dtype=np.float32)
    return header, arr


def _map_release_action_to_30(header: list[str], arr: np.ndarray) -> np.ndarray:
    """
    Build 30-dim action following repo README standard:
      [left_eef(7), right_eef(7), left_joints(7), right_joints(7), left_gripper(1), right_gripper(1)]

    Arms data provides joint positions for both arms (and many finger joints).
    We fill joint channels and set eef/grippers to 0 by default.
    """
    col_to_idx = {c: i for i, c in enumerate(header)}

    out = np.zeros((arr.shape[0], 30), dtype=np.float32)

    # left joints -> dims 14..20
    for k, col in enumerate(LEFT_JOINT_COLS):
        if col in col_to_idx:
            out[:, 14 + k] = arr[:, col_to_idx[col]]
        else:
            raise KeyError(f"Missing column in action.txt: {col}")

    # right joints -> dims 21..27
    for k, col in enumerate(RIGHT_JOINT_COLS):
        if col in col_to_idx:
            out[:, 21 + k] = arr[:, col_to_idx[col]]
        else:
            raise KeyError(f"Missing column in action.txt: {col}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-root", type=str, default="./arms", help="Path to arms/ directory")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Which split to prepare")
    ap.add_argument("--out", type=str, default="./prepared_arms", help="Output dataset root")
    args = ap.parse_args()

    arms_root = Path(args.arms_root)
    split_root = arms_root / args.split
    out_root = Path(args.out)

    episodes = sorted([p for p in split_root.iterdir() if p.is_dir()])
    assert episodes, f"No episode folders under: {split_root}"

    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    video_dir = out_root / "videos" / "chunk-000" / "observation.images.cam_high"
    video_dir.mkdir(parents=True, exist_ok=True)
    actions_dir = out_root / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    instructions_dir = out_root / "instructions"
    instructions_dir.mkdir(parents=True, exist_ok=True)

    ep_jsonl = out_root / "meta" / "episodes.jsonl"
    all_actions = []

    with open(ep_jsonl, "w", encoding="utf-8") as f:
        for ep_idx, ep in enumerate(episodes):
            action_txt = ep / "action.txt"
            instr_txt = ep / "instruction.txt"
            video_mp4 = ep / "video.mp4"

            header, arr = _read_csv_rows(action_txt)
            actions30 = _map_release_action_to_30(header, arr)
            np.save(actions_dir / f"episode_{ep_idx:06d}.npy", actions30)
            all_actions.append(actions30)

            # copy / link video
            dst_video = video_dir / f"episode_{ep_idx:06d}.mp4"
            if not dst_video.exists():
                # prefer hardlink when possible
                try:
                    os.link(video_mp4, dst_video)
                except OSError:
                    import shutil
                    shutil.copy2(video_mp4, dst_video)

            instruction = instr_txt.read_text(encoding="utf-8").strip()
            (instructions_dir / f"episode_{ep_idx:06d}.txt").write_text(instruction + "\n", encoding="utf-8")
            length = int(actions30.shape[0])

            line = {
                "episode_index": ep_idx,
                "tasks": [instruction],
                "length": length,
                "action_config": [
                    {"start_frame": 0, "end_frame": length, "action_text": instruction}
                ],
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    all_actions = np.concatenate(all_actions, axis=0)
    q01 = np.quantile(all_actions, 0.01, axis=0).astype(float).tolist()
    q99 = np.quantile(all_actions, 0.99, axis=0).astype(float).tolist()
    with open(out_root / "norm_stat.json", "w", encoding="utf-8") as f:
        json.dump({"q01": q01, "q99": q99}, f, ensure_ascii=False, indent=2)

    print(f"Prepared {len(episodes)} episodes to: {out_root}")
    print("Next: extract Wan2.2 VAE latents into out_root/latents/ mirroring videos/.")


if __name__ == "__main__":
    main()

