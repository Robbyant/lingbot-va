#!/usr/bin/env bash
# LingBot-VA 环境配置脚本
# 从 lingbot-va 仓库根目录执行: bash script/setup_env.sh
# 可选: bash script/setup_env.sh <conda_env_name>

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${1:-lingbot-va}"
PYTHON_VERSION="3.10.16"

echo "=== LingBot-VA 环境配置 ==="
echo "仓库根目录: $REPO_ROOT"
echo "Conda 环境名: $CONDA_ENV"
echo ""

# 1. 创建 conda 环境
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo ">>> 环境 $CONDA_ENV 已存在，将复用并更新依赖"
else
    echo ">>> 创建 conda 环境: $CONDA_ENV (Python $PYTHON_VERSION)"
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
fi

# 2. 激活后安装（用 conda run 保证在正确 env 下执行）
echo ">>> 安装 PyTorch 2.9.0 (CUDA 12.6)"
conda run -n "$CONDA_ENV" pip install --upgrade pip
conda run -n "$CONDA_ENV" pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu126

echo ">>> 安装基础依赖 (README)"
conda run -n "$CONDA_ENV" pip install \
    websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack \
    opencv-python matplotlib ftfy easydict

echo ">>> 安装 flash-attn (可能较慢)"
conda run -n "$CONDA_ENV" pip install flash-attn --no-build-isolation

echo ">>> 安装 requirements.txt 中的其余依赖"
conda run -n "$CONDA_ENV" pip install -r "$REPO_ROOT/requirements.txt"

echo ">>> 以可编辑方式安装当前项目"
conda run -n "$CONDA_ENV" pip install -e .

echo ""
echo "=== 环境就绪 ==="
echo "激活环境: conda activate $CONDA_ENV"
echo ""
echo "可选环境变量（按需设置）："
echo "  # RoboTwin 测评时指定测评仓库路径"
echo "  export ROBOTWIN_ROOT=/mnt/users/wangyuxuan-20250915/EAI/RoboTwin"
echo ""
echo "  # Post-training 训练时指定 LeRobot 数据集路径"
echo "  export LINGBOT_VA_DATASET_PATH=$REPO_ROOT/robotwin-clean-and-aug-lerobot"
echo "  # 若数据集尚未下载，可运行: python script/download/download_dataset.py"
echo ""
echo "  # 推理/测评时指定模型路径（若未用默认 lingbot-va-base）"
echo "  export LINGBOT_VA_MODEL_PATH=/path/to/your/model"
echo ""
