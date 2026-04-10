## MI300X / ROCm 7.2 跑通 LingBot-VA + LIBERO（Server/Client）指南

```bash
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
sudo apt update
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
#
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
#
reboot
#
pip3 install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
apt update && apt install -y git-lfs
git lfs install


```
适用场景：
- **AMD Instinct MI300X**（ROCm 7.2）
- **远端 Ubuntu 22.04**（SSH 上去跑）
- **无显示器/无 GUI**，需要 **MuJoCo + robosuite 离屏渲染（EGL）**
- 目标是先把 `evaluation/libero` 的 **server/client 流程跑通**

这份指南重点解决你遇到的两个坑：
- **谨慎安装 `lerobot`**：在 ROCm 环境中不建议让 `pip` 自动解析/升级它的依赖链，以免把 ROCm 的 torch 组合替换掉。推荐在装好 ROCm torch 后使用 `pip install --no-deps lerobot==0.3.3`，并把额外依赖（如 `scipy`、`wandb`）单独安装。
- LIBERO 仿真依赖链较长，按下面“一次装齐最小集合”做。
- 如果你选择装 `flash-attn`：**在 AMD/ROCm 上应走 Triton 后端**（aiter JIT），并避免“在错误目录执行 pip install .”把项目打成 `UNKNOWN` 包。

---

## 0. 强烈建议：用独立 venv（避免把系统 Python 搞崩）

在远端服务器执行：

```bash
python3 -m venv ~/venvs/lingbot-va
source ~/venvs/lingbot-va/bin/activate
python -m pip install -U pip
```

后面所有 `pip` / `python` 都在这个 venv 里执行（不再用系统的 `pip`）。

---

## 1) 安装 ROCm 版 PyTorch（按你的 ROCm 版本选）

你机器显示 ROCm 7.2（`amd-smi`），但 PyTorch 官方 wheel 可能不提供 rocm7.2 的 index。
如果你平台/镜像已经自带可用的 torch（ROCm），你可以跳过此步；否则按平台文档安装 ROCm PyTorch。

检查 torch 是否可用：

```bash
python -c "import torch; print(torch.__version__); print('hip:', torch.version.hip); print('cuda available:', torch.cuda.is_available())"
```

> 注意：ROCm 下 `torch.cuda.is_available()` 也可能为 True（PyTorch 沿用了 cuda API 名称）。

---

## 2) 安装 LingBot-VA 运行所需 Python 包（不要装 lerobot）

在 `~/lingbot-va` 目录，激活 venv 后执行：

```bash
pip install websockets msgpack opencv-python "imageio[ffmpeg]" matplotlib ftfy easydict einops tqdm
pip install "diffusers==0.36.0" "transformers==4.55.2" accelerate
```

### 2.1 关于 flash-attn
在 AMD/ROCm 环境下，`flash-attn` 可能没有对应内核或会回退实现。**能跑通优先**，不必强行追求 flash-attn CUDA 内核。

如果你确实想在 MI300X / ROCm 7.2 上安装 `flash-attn`（用于 `"attn_mode": "flashattn"`），推荐按 FlashAttention 官方说明走 **Triton AMD 后端**（ROCm 6.0+，7.2 也适用）：
- 需要依赖：`packaging psutil ninja`
- 需要在 **flash-attention 仓库目录**执行安装（不要在 `lingbot-va/` 根目录执行 `pip install .`，否则会出现打包成 `UNKNOWN` 的情况）

示例（在 venv 内）：

```bash
pip install packaging psutil ninja
cd ~
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
git submodule update --init --recursive

# 启用 Triton AMD 后端
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .

# 可选：打开 autotune（首次运行会预热更久）
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"
```

验证：

```bash
python3 -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)"
```

> 你可能会看到 `flash_attn_2_cuda not found, falling back to Triton implementation`，这在 AMD 上通常是正常的（代表走 Triton/aiter 路线）。

---

## 3) 安装 LIBERO（benchmark 源码版）

不要 `pip install libero`（PyPI 上同名包不对，且版本元数据有问题）。

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/LIBERO
pip install -e ~/LIBERO
python3 -c "from libero.libero import benchmark; print('libero import ok')"
```

首次安装会引导你生成 `~/.libero/config.yaml`（数据路径随便填一个能写的目录即可）。

---

## 3.1（可选）下载数据：HuggingFace / Google Drive / tgz 解压

### A) HuggingFace 下载 checkpoints（推荐 huggingface-cli）

```bash
pip install -U huggingface_hub
mkdir -p ~/lingbot-va/checkpoints
cd ~/lingbot-va/checkpoints
huggingface-cli download --repo-type model robbyant/lingbot-va-base --local-dir lingbot-va-base
```

国内可选镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### B) Google Drive 下载（gdown）

```bash
pip install -U gdown
```

注意：带 `&` 的 URL 一定要加引号，否则 shell 会把它拆成后台任务，最终只下载到一个几 KB 的 HTML 页面。

```bash
gdown --fuzzy "https://drive.google.com/file/d/1QGNkvsb1hlRmRkKCgFlyWitv17sRuagS/view"
```

### C) `.tgz` 解压

```bash
mkdir -p ~/lingbot-va/data/libero
tar -xzvf libero_10.tgz -C ~/lingbot-va/data/libero
```

## 4) 安装 LIBERO 仿真依赖（robosuite + mujoco + bddl + 其他）

### 4.1 系统 EGL/Mesa 依赖（无头渲染需要）

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgles2 libgl1-mesa-dri libgl1-mesa-glx mesa-utils
```

### 4.2 Python 依赖

```bash
pip install "mujoco==3.1.5" "robosuite==1.4.1"
pip install PyOpenGL PyOpenGL-accelerate
pip install bddl cloudpickle gymnasium
pip install "imageio[ffmpeg]" opencv-python tqdm
pip install gym==0.26.2
```

> 说明：LIBERO 的 `venv.py` 里写的是 `import gym`，所以需要装 `gym`（即使你也装了 `gymnasium`）。

---

## 5) 配置 MuJoCo 无头渲染（EGL）

每次跑 client 前都建议先 export：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

验证 robosuite+mujoco 是否能 import（会有一些 warning，能打印 ok 就行）：

```bash
python3 -c "import robosuite, mujoco; print('robosuite+mujoco ok')"
```

如果报 `eglQueryString` 类错误，通常是环境变量没生效或 EGL/Mesa 组件没装齐。

### 5.1 EGL 仍失败（`amdgpu_dri` / `PLATFORM_DEVICE` / `EGL_BAD_DISPLAY`）

部分云主机 / 容器装了 ROCm 计算栈，但**没有**完整的 OpenGL/EGL 用户态驱动（日志里常见 `failed to open amdgpu_dri.so`、`Cannot initialize a EGL device display`）。此时 **EGL 离屏不可用**，可改用 **OSMesa 软件光栅**（较慢，但能跑通仿真与录屏）：

```bash
sudo apt-get update
sudo apt-get install -y libosmesa6
export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM   # 若曾设为 egl，先取消，避免 PyOpenGL 仍走 EGL
```

或直接让 client 在 **导入 MuJoCo 之前** 设好后端（等价于上面 `export`）：

```bash
python3 -m evaluation.libero.client \
  --mujoco-gl osmesa \
  --libero-benchmark libero_10 \
  --port 29056 \
  --test-num 1 \
  --task-range 0 1 \
  --out-dir outputs/libero
```

> `--mujoco-gl` 必须在进程启动时生效；本仓库在 `client.py` 最前面解析该参数，无需改 LIBERO/robosuite 源码。

---

## 6) 配置 LingBot-VA checkpoints 路径 + 推理 attn_mode

### 6.1 改 checkpoints 路径
编辑 `wan_va/configs/va_libero_cfg.py`：

- `wan22_pretrained_model_name_or_path = "./checkpoints/libero-va-base"`

改成你的本地模型目录（建议用正斜杠），例如：

`/root/lingbot-va/lingbot-va-base`

该目录下必须包含：
`vae/ tokenizer/ text_encoder/ transformer/`

### 6.2 改 attn_mode（推理必须）
编辑：
`<模型目录>/transformer/config.json`

把 `"attn_mode"` 设为：
- `"torch"` 或 `"flashattn"`

不要用 `"flex"`（训练用，推理会报错）。

---

## 7) 启动 LIBERO Server / Client（两个终端/两个 tmux pane）

### 7.1 终端 A：启动 server

```bash
cd ~/lingbot-va
source ~/venvs/lingbot-va/bin/activate
bash evaluation/libero/launch_server.sh
```

看到类似：
`server listening on 0.0.0.0:29056`
说明启动成功。

### 7.2 终端 B：启动 client

```bash
cd ~/lingbot-va
source ~/venvs/lingbot-va/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 先用最小任务量跑通流程（推荐）
python3 -m evaluation.libero.client \
  --libero-benchmark libero_10 \
  --port 29056 \
  --test-num 1 \
  --task-range 0 1 \
  --out-dir outputs/libero
```

跑通后再把 `--test-num`、`--task-range` 调大。

> 如果你的系统没有 `python` 命令（只有 `python3`），请改用 `python3 -m ...`，或安装 `python-is-python3`：
>
> ```bash
> sudo apt-get install -y python-is-python3
> ```

输出视频在：
`outputs/libero/.../*.mp4`

---

## 7.3 Client 输出落盘说明（MP4 / PNG / NPZ：action + joint）

`evaluation/libero/client.py` 会按 episode 落盘到 `--out-dir` 指定的根目录（默认 `outputs/libero`，相对路径是相对于你启动命令时的当前目录）。

### 7.3.1 输出目录结构

每个 episode 的输出路径形如：

`{out_dir}/{libero_benchmark}/{task_idx}_{prompt(空格->下划线)}/`

在该目录下会生成（同一个 episode 前缀为 `{episode_idx}_{done}`，其中 done 为 True/False）：

- **视频**：`{episode_idx}_{done}.mp4`
  - 若写 mp4 失败（通常是缺 FFMPEG 后端），会自动回退写成同名 `.gif`
- **关键帧 PNG**：`{episode_idx}_{done}_png/frame_000000.png` ...（与视频帧一致，左右相机横向拼接）
- **轨迹 NPZ**：`{episode_idx}_{done}.npz`

另有每个 task 的成功率统计：

`{out_dir}/{libero_benchmark}_{task_idx}.json`（仅 `succ_num/total_num/succ_rate`）

### 7.3.2 `npz` 里保存了哪些量

`{episode}.npz` 中的主要键：

- `actions`: 每个 env step 实际执行的动作向量，形状通常为 `(T, act_dim)`
- `robot0_joint_pos`, `robot0_joint_vel`, `robot0_gripper_qpos`, `robot0_eef_pos`, `robot0_eef_quat`
  - 仅当 LIBERO/robosuite 的原始 `obs` 中存在对应键时才会写入
  - 行数会与 `actions` 的 \(T\) 对齐（逐步记录）
- `policy_chunks`: 策略每次 `model.infer(...)` 返回的整块 action（为了复盘 chunk 级输出），长度为策略前向次数
  - 读取时需要 `allow_pickle=True`

本地快速检查（示例）：

```bash
python3 - <<'PY'
import numpy as np
p = "outputs/libero/<...>/<episode>.npz"
d = np.load(p, allow_pickle=True)
print("keys:", d.files)
for k in d.files:
    v = d[k]
    print(k, v.dtype, v.shape)
PY
```

### 7.3.3 远程写不出 MP4 的处理

若出现 “无法写入 MP4 / backend / ffmpeg” 相关报错，优先安装：

```bash
pip install "imageio[ffmpeg]"
```

即使 MP4 临时写不了，也可以只用 PNG 在本地合成视频（在 `*_png/` 目录中执行）：

```bash
ffmpeg -y -framerate 60 -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4
```

### 7.3.4 常见日志提示含义

你可能会看到：

`[info] using task orders [0, 1, 2, ..., 9]`

这表示 benchmark 内部的默认任务顺序（例如 `libero_10` 通常是 10 个任务 0~9）。实际运行哪些任务仍由 `--task-range start end` 决定（常见为左闭右开，即 `0 1` 只跑 task 0）。

---

## 8) 重要：不要安装 lerobot（否则很容易把 ROCm 环境弄崩）

本仓库的 `evaluation/libero/client.py` 我们已改为使用标准库 `json` 写结果文件，
因此 **不再需要 `lerobot`**。

如果你已经装过 `lerobot` 并导致出现：
- `No module named 'flash_attn_2_cuda'`

这通常意味着环境被换成了 CUDA/NVIDIA 组合。
最稳的恢复方式是：**丢弃当前环境，重新建一个干净 venv**，按本指南重装最小依赖。

