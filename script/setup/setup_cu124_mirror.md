# LingBot-VA 环境：国内镜像安装 PyTorch (CUDA 12.4)

阿里云 cu124 是**目录列表页**，不是 pip 的 simple index，不能用 `--index-url`，要用 **`--find-links`**。

## 1. 激活环境后安装（推荐）

**阿里云镜像（--find-links）：**
```bash
conda activate lingbot-va
pip install --upgrade pip
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu124/
```

**不指定版本（装镜像站里最新）：**
```bash
conda activate lingbot-va
pip install torch torchvision torchaudio \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu124/
```

**南京大学镜像（若支持 find-links 可试）：**
```bash
pip install torch torchvision torchaudio \
  --find-links https://mirror.nju.edu.cn/pytorch-wheels/cu124/
```

## 2. 用脚本时走国内镜像

```bash
cd /mnt/users/wangyuxuan-20250915/EAI/lingbot-va
# 默认已用阿里云 --find-links；可改镜像：
# export PYTORCH_MIRROR=https://mirror.nju.edu.cn/pytorch-wheels/cu124/
bash script/setup_env_cu124.sh
```

## 3. 若镜像没有 cu124

部分镜像只同步到 cu121，可改用 cu121 的包（需本机 CUDA 兼容）：
```bash
pip install torch torchvision torchaudio --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121
```

或从 PyTorch 官方页下载对应 `.whl` 后本地安装：
```bash
pip install /path/to/torch-xxx-cu124-*.whl
```
