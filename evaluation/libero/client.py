import os
import sys


def _apply_mujoco_gl_argv():
    """MuJoCo reads MUJOCO_GL at native init; must run before importing libero/mujoco."""
    i = 0
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--mujoco-gl" and i + 1 < len(sys.argv):
            os.environ["MUJOCO_GL"] = sys.argv[i + 1]
            i += 2
            continue
        if a.startswith("--mujoco-gl="):
            os.environ["MUJOCO_GL"] = a.partition("=")[2]
        i += 1


_apply_mujoco_gl_argv()

import numpy as np
import torch

# PyTorch 2.6+ defaults torch.load(..., weights_only=True). LIBERO benchmark init *.pt
# files are trusted pickles (numpy + tensors); allow full unpickle for those loads.
_torch_load_orig = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        try:
            return _torch_load_orig(*args, weights_only=False, **kwargs)
        except TypeError:
            return _torch_load_orig(*args, **kwargs)
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load_compat

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
import argparse
from libero.libero import benchmark
import time
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from tqdm import tqdm
import json
import imageio
import cv2


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [_stack_frame(obs, video_names, target_size) for obs in real_obs_list]

    path = Path(save_path)
    try:
        imageio.mimsave(str(path), final_frames, fps=fps)
        print(f"✅ Video saved to: {path}")
    except ValueError as e:
        msg = str(e).lower()
        if path.suffix.lower() == ".mp4" and (
            "backend" in msg or "ffmpeg" in msg or "ffm" in msg or "wI" in str(e)
        ):
            gif_path = path.with_suffix(".gif")
            print(
                "⚠️ 无法写入 MP4：缺少 imageio 的 FFMPEG 插件。请先执行：pip install 'imageio[ffmpeg]'"
                f"\n    本次已改用 GIF：{gif_path}"
            )
            imageio.mimsave(str(gif_path), final_frames, fps=fps)
            print(f"✅ Video saved to: {gif_path}")
        else:
            raise


def construct_single_env(env_args):
    last_exc = None
    for _ in range(5):
        try:
            return OffScreenRenderEnv(**env_args)
        except Exception as e:
            last_exc = e
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
    raise RuntimeError(
        "OffScreenRenderEnv failed after 5 attempts. Headless EGL often needs a working GPU GL stack "
        "(e.g. amdgpu_dri / EGL PLATFORM_DEVICE). On minimal cloud images, use CPU software rendering: "
        "`sudo apt install -y libosmesa6` then `export MUJOCO_GL=osmesa` or "
        "`python3 -m evaluation.libero.client --mujoco-gl osmesa ...`. "
        "See ROCM_LIBERO_SETUP.md §5.1."
    ) from last_exc


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


# Robosuite / LIBERO 常见本体观测键（存在则写入 trajectory npz）
_PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_gripper_qpos",
    "robot0_eef_pos",
    "robot0_eef_quat",
)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _proprio_from_raw(raw_obs):
    """从 env 原始 obs 中取出关节 / 末端等向量，便于落盘。"""
    out = {}
    if raw_obs is None:
        return out
    for k in _PROPRIO_KEYS:
        if k in raw_obs:
            out[k] = np.asarray(raw_obs[k], dtype=np.float32).reshape(-1).copy()
    return out


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done, obs


def _stack_frame(obs, video_names, target_size):
    return np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)


def run_one(
    model,
    libero_benchmark,
    task_idx,
    out_dir,
    episode_idx,
):
    vnames = ["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    action_rows = []
    proprio_rows = {k: [] for k in _PROPRIO_KEYS}
    policy_chunks = []
    done = False
    first = True
    while cur_env.env.timestep < 800:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = _to_numpy(ret["action"])
        policy_chunks.append(np.asarray(action, dtype=np.float32).copy())

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = _to_numpy(action[:, i, j]).reshape(-1)
                action_rows.append(ee_action.astype(np.float32, copy=False))
                observes, done, raw_obs = env_one_step(cur_env, ee_action)
                prop = _proprio_from_raw(raw_obs)
                for k in proprio_rows:
                    if k in prop:
                        proprio_rows[k].append(prop[k])
                if done:
                    break
                if (j + 1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    key_frame_list.append(observes)

            if done:
                break

        first = False

        if done:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    out_file = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}" / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)
    episode_base = out_file.parent / out_file.stem

    if full_obs_list:
        png_dir = Path(str(episode_base) + "_png")
        png_dir.mkdir(parents=True, exist_ok=True)
        fo = full_obs_list[0]
        base_h, width_base = fo[vnames[0]].shape[:2]
        target_size = (width_base, base_h)
        for fi, obs in enumerate(full_obs_list):
            frame = _stack_frame(obs, vnames, target_size)
            cv2.imwrite(
                str(png_dir / f"frame_{fi:06d}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
        print(f"✅ PNG 关键帧: {png_dir} ({len(full_obs_list)} 帧)")

    if action_rows:
        n = len(action_rows)
        for k, rows in proprio_rows.items():
            if rows and len(rows) != n:
                raise RuntimeError(
                    f"proprio / action 步数不一致: {k} len={len(rows)}, actions len={n}"
                )
        traj = {"actions": np.stack(action_rows, axis=0)}
        for k, rows in proprio_rows.items():
            if rows:
                traj[k] = np.stack(rows, axis=0)
        if policy_chunks:
            traj["policy_chunks"] = np.empty(len(policy_chunks), dtype=object)
            for i, c in enumerate(policy_chunks):
                traj["policy_chunks"][i] = c
        npz_path = episode_base.with_suffix(".npz")
        np.savez_compressed(npz_path, **traj)
        msg = f"actions {traj['actions'].shape}"
        if len(traj) > 1:
            msg += " +" + ",".join(f" {k}{traj[k].shape}" for k in traj if k != "actions")
        print(f"✅ 轨迹 npz: {npz_path} ({msg})")

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=vnames,
    )

    cur_env.close()
    return done


def run(
    libero_benchmark,
    port,
    out_dir,
    test_num,
    task_range=None,
):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)

    video_save_root_dict = None

    episode_list = range(test_num)
    for task_idx in progress_bar:
        if video_save_root_dict is not None and task_idx in video_save_root_dict:
            video_save_list = os.listdir(os.path.join(out_dir, libero_benchmark, video_save_root_dict[task_idx]))
            video_states = [1 for file in video_save_list if file.split('_')[1].split('.')[0] == 'True']
            succ_num = float(len(video_states))
            episode_list = range(len(video_save_list), test_num)
        else:
            succ_num = 0.

        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(model, libero_benchmark, task_idx, out_dir, episode_idx)
            succ_num += res_i
            succ_rate = succ_num / (episode_idx + 1)
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {episode_idx + 1}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "succ_num": succ_num,
                        "total_num": float(episode_idx + 1),
                        "succ_rate": succ_rate,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--mujoco-gl",
        type=str,
        default=None,
        choices=["egl", "glfw", "osmesa"],
        help="Set MUJOCO_GL before MuJoCo loads (must match early argv parse; use when EGL is unavailable).",
    )
    args = parser.parse_args()
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    kw = vars(args)
    kw.pop("mujoco_gl", None)
    run(**kw)
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()