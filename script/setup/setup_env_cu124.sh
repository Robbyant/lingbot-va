#!/usr/bin/env bash
# LingBot-VA 环境配置（兼容 CUDA 12.4，且不依赖 Anaconda 官方频道，无需接受 ToS）
# 从 lingbot-va 仓库根目录执行: bash script/setup_env_cu124.sh
# 可选: bash script/setup_env_cu124.sh <conda_env_name>

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${1:-lingbot-va}"
PYTHON_VERSION="3.10"

echo "=== LingBot-VA 环境配置 (CUDA 12.4) ==="
echo "仓库根目录: $REPO_ROOT"
echo "Conda 环境名: $CONDA_ENV"
echo ""

# 1. 仅用 conda-forge 创建环境，避免 Anaconda 官方频道 ToS
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo ">>> 环境 $CONDA_ENV 已存在，将复用并更新依赖"
else
    echo ">>> 创建 conda 环境: $CONDA_ENV (Python $PYTHON_VERSION, 仅 conda-forge)"
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y -c conda-forge --override-channels
fi

# 2. 安装 PyTorch (CUDA 12.4)：阿里云为目录列表页，须用 --find-links 而非 --index-url
PYTORCH_MIRROR="${PYTORCH_MIRROR:-https://mirrors.aliyun.com/pytorch-wheels/cu124/}"
echo ">>> 安装 PyTorch 2.6.0 (CUDA 12.4), 镜像: $PYTORCH_MIRROR"
conda run -n "$CONDA_ENV" pip install --upgrade pip
conda run -n "$CONDA_ENV" pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --find-links "$PYTORCH_MIRROR"

echo ">>> 安装基础依赖 (README)"
conda run -n "$CONDA_ENV" pip install \
    websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack \
    opencv-python matplotlib ftfy easydict

echo ">>> 安装 flash-attn (可能较慢)"
conda run -n "$CONDA_ENV" pip install flash-attn --no-build-isolation

echo ">>> 安装其余依赖"
conda run -n "$CONDA_ENV" pip install \
    numpy==1.26.4 tqdm "imageio[ffmpeg]" safetensors Pillow \
    lerobot==0.3.3 scipy wandb

echo ">>> 以可编辑方式安装当前项目"
conda run -n "$CONDA_ENV" pip install -e .

echo ""
echo "=== 环境就绪 ==="
echo "激活: conda activate $CONDA_ENV"
echo "可选: export LINGBOT_VA_MODEL_PATH=$REPO_ROOT/lingbot-va-base"
echo ""
