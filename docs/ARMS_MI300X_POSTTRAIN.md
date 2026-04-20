# Arms 数据 + MI300X 服务器 Post-training / 训练排障指南

本文把从「把工程与数据搬到服务器」到「`arms_train` 能真正开训」这条链路里**高频踩坑**整理成一份可执行清单。默认你在服务器上的工程路径类似：

- 代码：`/root/lingbot-va`
- 数据集：`/root/lingbot-va/arms_lerobot`（注意：本仓库 `.gitignore` 忽略了 `arms_lerobot/`，**数据集不会随 git 推送**）
- Base 模型：`/root/lingbot-va/models/lingbot-va-base`（HuggingFace：`robbyant/lingbot-va-base`）
- Wan Diffusers（如需本地 VAE/权重目录）：`/root/lingbot-va/models/Wan2.2-Animate-14B-Diffusers`（示例 HF：`Wan-AI/Wan2.2-Animate-14B-Diffusers`）

---

## 1) 代码：GitHub `git pull` 或 rsync

### 推荐：GitHub 拉代码

```bash
git clone https://github.com/<你的组织>/<你的仓库>.git
cd lingbot-va
git checkout arms-ROCM-posttraining
git pull
```

### 备选：从本机 rsync 整个工程（注意 exclude 大目录）

常用排除：`models/`、`example/`、`assets/`、`libero_10/`、压缩包等。

---

## 2) 数据：`arms_lerobot` 单独同步（不要指望 git）

`arms_lerobot/` 在本仓库 `.gitignore` 中，因此请用 `rsync`/`scp` 同步到服务器固定路径，例如：

```bash
rsync -avP --partial --info=progress2 \
  /path/to/local/arms_lerobot/ \
  root@<SERVER_IP>:/root/lingbot-va/arms_lerobot/
```

### 训练走 latents 时，至少要包含

- `meta/episodes.jsonl`
- `data/chunk-000/episode_*.parquet`
- `latents/chunk-000/<camera_key>/*.pth`
- `empty_emb.pt`、`norm_stat.json`（你当前流程里会用到）
- `manifest.json`（如果你们工具链依赖）

`videos/` 是否必须取决于你是否还要走“读 mp4”的路径；**latents 训练通常不强依赖 videos**。

---

## 3) 模型权重：服务器下载（HF / 镜像）

### `lingbot-va-base`

仓库页：`https://huggingface.co/robbyant/lingbot-va-base`

```bash
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT="https://hf-mirror.com"   # 不能直连 huggingface.co 时用镜像；能直连则改成 https://huggingface.co

huggingface-cli download "robbyant/lingbot-va-base" \
  --local-dir "/root/lingbot-va/models/lingbot-va-base" \
  --local-dir-use-symlinks False
```

### Wan Diffusers（示例）

```bash
huggingface-cli download "Wan-AI/Wan2.2-Animate-14B-Diffusers" \
  --local-dir "/root/lingbot-va/models/Wan2.2-Animate-14B-Diffusers" \
  --include "vae/*"
```

---

## 4) Python 依赖：不要只装 torch

你至少会遇到过这些 import 缺失（按报错逐个装）：

```bash
python3 -m pip install -U wandb datasets jsonlines av
```

`av`（PyAV）即使训练主要读 latents，`lerobot` 在 import 阶段也会加载视频工具链，**没装 `av` 会直接起不来**。

> 若 `pip install av` 失败，优先用系统包补齐 ffmpeg 相关开发库后再装（不同发行版包名略有差异）。

---

## 5) 配置：三条路径必须对齐

1) `wan_va/configs/va_arms_cfg.py`

- `wan22_pretrained_model_name_or_path = "/root/lingbot-va/models/lingbot-va-base"`

2) `wan_va/configs/va_arms_train_cfg.py`

- `dataset_path = "/root/lingbot-va/arms_lerobot"`（**不要用 `./arms_lerobot`**）
- `empty_emb_path` 会拼接在 `dataset_path` 下，确保 `empty_emb.pt` 存在

3) 启动参数

```bash
torchrun --standalone --nproc_per_node=1 wan_va/train.py \
  --config-name arms_train \
  --save-root /root/lingbot-va/outputs/arms_train
```

---

## 6) W&B：不要用不存在的 CLI 参数

`wan_va/train.py` **不支持** `--logger wandb` 这类参数；是否启用看配置：

- `wan_va/configs/va_arms_train_cfg.py` → `enable_wandb`

若开启 W&B，需要环境变量（示例）：

```bash
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="..."
export WANDB_TEAM_NAME="..."
export WANDB_PROJECT="..."
```

---

## 7) LeRobot 元数据：你这次训练失败的“关键三连”

`lerobot==0.3.x` 的 `LeRobotDatasetMetadata` 会读取本地 `meta/` 下多份文件。自定义导出的 `arms_lerobot` 往往缺其中几份，表现为：

- `num_samples=0`（数据集根本没被注册进来）
- `HFValidationError: repo id ... '/root/...'`（异常分支把本地路径误当成 HF repo id）

### 7.1 必须有：`meta/info.json`

用于声明 `codebase_version`、`fps`、`features`、数据/视频路径模板等。

### 7.2 必须有：`meta/tasks.jsonl`

格式示例（每行一个 JSON）：

```json
{"task_index": 0, "task": "Pick up ..."}
```

### 7.3 必须有：`meta/episodes_stats.jsonl`

这是本次排障里最后一个阻塞点：`LeRobotDatasetMetadata` 会加载 episode 级统计；缺失会触发异常分支。

> 本仓库已在 `wan_va/dataset/lerobot_latent_dataset.py` 增加兜底：缺文件时尝试从 `episodes.jsonl` + `data/chunk-000/*.parquet` 自动生成 `tasks.jsonl` / `episodes_stats.jsonl`，并把传给 `LeRobotDatasetMetadata` 的 `repo_id` 固定为目录名（如 `arms_lerobot`），避免绝对路径被当成 HF repo id。

---

## 8) SSH / rsync：`Permission denied (publickey)`

这表示目标机只允许公钥登录。你需要：

- 在发起端生成 `ssh-keygen`
- 把公钥追加到目标机 `~/.ssh/authorized_keys`
- 或指定已有私钥：`-i /path/to/key.pem`

---

## 9) 推荐启动方式（与上游 README 一致）

上游 README 的范式是：

```bash
NGPU=1 CONFIG_NAME='arms_train' bash script/run_va_posttrain.sh
```

注意：本仓库脚本里 `WANDB_*` 可能是占位符；若你不开 W&B，请先把 `enable_wandb=False`，避免训练启动先去走 W&B 初始化。

---

## 10) 快速自检命令（服务器上）

```bash
python3 - <<'PY'
from wan_va.configs import VA_CONFIGS
print("dataset_path:", VA_CONFIGS["arms_train"].dataset_path)
print("wan22_pretrained_model_name_or_path:", VA_CONFIGS["arms_train"].wan22_pretrained_model_name_or_path)
PY

ls -lah /root/lingbot-va/arms_lerobot/meta/info.json
ls -lah /root/lingbot-va/arms_lerobot/meta/tasks.jsonl
ls -lah /root/lingbot-va/arms_lerobot/meta/episodes_stats.jsonl
ls -1 /root/lingbot-va/arms_lerobot/latents/chunk-000/observation.images.front | head
```

---

## 11) 仍然失败时，最有效的信息

请贴三段信息（从下到上）：

1) `python3 - <<PY ... print(dataset_path) ... PY` 的输出  
2) `ls arms_lerobot/meta/` 的文件列表  
3) 第一次出现 `FileNotFoundError` / `HFValidationError` 的那一段 traceback

---

## 相关链接

- LingBot-VA base 模型卡：`https://huggingface.co/robbyant/lingbot-va-base`
