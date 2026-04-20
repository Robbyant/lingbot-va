# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import json
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result


def _ensure_tasks_jsonl(dataset_root: Path) -> None:
    """
    LeRobotDatasetMetadata expects `meta/tasks.jsonl`.
    Some custom exports only ship `meta/episodes.jsonl`; synthesize tasks from it.
    """
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        return

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(
            f"Missing {tasks_path} and cannot synthesize without {episodes_path}"
        )

    tasks: list[str] = []
    seen: set[str] = set()
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = None
            if isinstance(obj.get("tasks"), list) and obj["tasks"]:
                t = str(obj["tasks"][0])
            elif isinstance(obj.get("task"), str):
                t = obj["task"]
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            tasks.append(t)

    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("w", encoding="utf-8") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}, ensure_ascii=False) + "\n")


def _load_task_to_index(dataset_root: Path) -> dict[str, int]:
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(f"Missing {tasks_path}; run training after tasks.jsonl exists")
    m: dict[str, int] = {}
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            m[str(obj["task"])] = int(obj["task_index"])
    return m


def _stat_1d(x: "np.ndarray") -> dict:
    # x: [T, D]
    if x.size == 0:
        d = int(x.shape[1]) if x.ndim == 2 else 0
        z = [0.0] * d
        o = [1.0] * d
        return {
            "min": z,
            "max": o,
            "mean": z,
            "std": z,
            "count": [0],
        }
    mn = np.min(x, axis=0).astype(float).tolist()
    mx = np.max(x, axis=0).astype(float).tolist()
    mu = np.mean(x, axis=0).astype(float).tolist()
    sd = np.std(x, axis=0).astype(float).tolist()
    return {"min": mn, "max": mx, "mean": mu, "std": sd, "count": [int(x.shape[0])]}


def _dummy_video_stat(count: int) -> dict:
    # Shape matches common LeRobot exports (3 nested levels), values are placeholders.
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.1]], [[0.1]], [[0.1]]],
        "count": [int(count)],
    }


def _ensure_episodes_stats_jsonl(dataset_root: Path) -> None:
    """
    LeRobotDatasetMetadata expects `meta/episodes_stats.jsonl` for v2.1 datasets.
    If missing, compute lightweight stats from parquet + episodes.jsonl.
    """
    out_path = dataset_root / "meta" / "episodes_stats.jsonl"
    if out_path.exists():
        return

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 30) or 30)

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing {episodes_path}")

    task_to_idx = _load_task_to_index(dataset_root)

    # global row index base (approximate LeRobot's global `index`)
    cum_rows = 0
    lines_out: list[str] = []

    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            ep_idx = int(ep["episode_index"])
            length = int(ep["length"])
            tasks = ep.get("tasks") or []
            task_str = str(tasks[0]) if tasks else ""
            task_index = int(task_to_idx[task_str]) if task_str in task_to_idx else 0

            pq = dataset_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
            if not pq.exists():
                raise FileNotFoundError(f"Missing parquet for episode {ep_idx}: {pq}")

            # Optional dependency: pandas/pyarrow are already required by lerobot workflows.
            import pandas as pd  # local import to keep import graph lighter

            df = pd.read_parquet(pq, columns=["action", "observation.state"])
            act = np.stack(df["action"].to_numpy()).astype(np.float32)
            st = np.stack(df["observation.state"].to_numpy()).astype(np.float32)

            T = int(act.shape[0])
            if T != length:
                # Keep going, but prefer parquet truth for stats.
                length = T

            idx = np.arange(cum_rows, cum_rows + length, dtype=np.int64)[:, None]
            ep_col = np.full((length, 1), ep_idx, dtype=np.int64)
            fi = np.arange(0, length, dtype=np.int64)[:, None]
            ti = np.full((length, 1), task_index, dtype=np.int64)
            ts = (fi.astype(np.float32) / float(fps))

            stats = {
                "episode_index": _stat_1d(ep_col.astype(np.float32)),
                "index": _stat_1d(idx.astype(np.float32)),
                "frame_index": _stat_1d(fi.astype(np.float32)),
                "task_index": _stat_1d(ti.astype(np.float32)),
                "timestamp": _stat_1d(ts),
                "action": _stat_1d(act),
                "observation.state": _stat_1d(st),
                # We don't decode mp4 here; latent training doesn't need accurate video stats.
                "observation.images.front": _dummy_video_stat(length),
            }

            lines_out.append(json.dumps({"episode_index": ep_idx, "stats": stats}, ensure_ascii=False))
            cum_rows += length

    out_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    # Always resolve dataset_path to an absolute path. Relative paths like
    # "./arms_lerobot" would otherwise be treated as an invalid HF repo id.
    dataset_root = Path(config.dataset_path).expanduser().resolve()
    repo_list = recursive_find_file(str(dataset_root), 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    for root in repo_list:
        _ensure_tasks_jsonl(Path(root))
        _ensure_episodes_stats_jsonl(Path(root))
    # Use spawn context to avoid fork-related crashes/hangs with torch + GPU init.
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_init_worker) as pool:
        datasets_out_lst = pool.map(construct_func, repo_list)
                
    return datasets_out_lst

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=None,
    ):
        if num_init_worker is None:
            num_init_worker = int(getattr(config, "dataset_init_workers", 16) or 16)
        cpu = os.cpu_count() or 1
        # Avoid spawning hundreds of short-lived workers during dataset init.
        num_init_worker = int(max(1, min(int(num_init_worker), max(1, cpu))))
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        # `repo_id` here is the local dataset root directory passed from
        # `recursive_find_file(...)`. LeRobotDatasetMetadata expects a HF-style
        # repo id string (no slashes), while the on-disk dataset lives at
        # `dataset_root`.
        dataset_root = Path(repo_id).expanduser().resolve()
        _ensure_tasks_jsonl(dataset_root)
        _ensure_episodes_stats_jsonl(dataset_root)

        self.repo_id = dataset_root.name  # e.g. "arms_lerobot"
        self.root = dataset_root
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = dataset_root / 'latents'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self.parse_meta()

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num

        # If dataset provides 26-dim (dual-arm joints 14 + fingers 12), map to
        # 30-dim (EEF_L7 + EEF_R7 + joints_L7 + joints_R7 + grip_L1 + grip_R1).
        # Without URDF we cannot compute EEF; we keep EEF dims at 0 and mask them out.
        # Finger joints are aggregated into a single gripper scalar per hand (mean).
        if int(action.shape[1]) == 26 and int(getattr(self.config, "action_dim", 26)) == 30:
            # action layout (26):
            #   0:7  left arm joints
            #   7:14 right arm joints
            #   14:20 left fingers (6)
            #   20:26 right fingers (6)
            left_j = action[:, 0:7]
            right_j = action[:, 7:14]
            left_f = action[:, 14:20]
            right_f = action[:, 20:26]

            grip_l = left_f.mean(axis=1, keepdims=True)
            grip_r = right_f.mean(axis=1, keepdims=True)

            eef_zeros = np.zeros((action.shape[0], 14), dtype=action.dtype)
            action = np.concatenate([eef_zeros, left_j, right_j, grip_l, grip_r], axis=1)  # [T,30]

            # mask: EEF dims invalid (0), joints+gripper valid (1)
            eef_mask = np.zeros((action_mask.shape[0], 14), dtype=bool)
            jr_mask = np.ones((action_mask.shape[0], 16), dtype=bool)  # 14 joints + 2 grippers
            action_mask = np.concatenate([eef_mask, jr_mask], axis=1)


        # Align action dim to model action_dim (lingbot-va-base expects 30).
        # Some datasets store 26-dim actions; pad with zeros to config.action_dim.
        target_action_dim = int(getattr(self.config, "action_dim", action.shape[1]))
        pad_dim = max(0, target_action_dim - int(action.shape[1]))
        action_paded = np.pad(action, ((0, 0), (0, pad_dim)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, pad_dim)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
