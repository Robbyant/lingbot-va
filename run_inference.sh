#!/usr/bin/env bash
# LingBot-VA 一键推理（自动激活 conda 环境 + 设置模型路径）
# 用法: bash run_inference.sh  或  bash script/
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

CONDA_BASE=""
if [[ -n "$CONDA_EXE" ]]; then
  CONDA_BASE="${CONDA_EXE%/bin/conda}"
fi
if [[ -z "$CONDA_BASE" ]]; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || true
fi
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  echo "未检测到 conda，请先安装 Miniconda/Anaconda 或在本机已激活 lingbot-va 的终端中直接执行:"
  echo "  export LINGBOT_VA_MODEL_PATH=$REPO_ROOT/lingbot-va-base"
  echo "  bash script/run_i2va_single_gpu.sh"
  exit 1
fi

echo ">>> 激活环境: lingbot-va"
conda activate lingbot-va

echo ">>> 模型路径: $REPO_ROOT/lingbot-va-base"
export LINGBOT_VA_MODEL_PATH="${LINGBOT_VA_MODEL_PATH:-$REPO_ROOT/lingbot-va-base}"

cd "$REPO_ROOT"
bash script/run_i2va_single_gpu.sh "$@"
