#!/usr/bin/env bash
# 从根本上解决 flash_attn 与 PyTorch 2.6.0+cu124 的 ABI 兼容：安装匹配的 wheel 或从源码用正确 ABI 编译
# 用法: bash script/install_flash_attn_cu124.sh [lingbot-va]
# 要求: 已安装 torch==2.6.0+cu124（且 torch._C._GLIBCXX_USE_CXX11_ABI == False）

set -e

CONDA_ENV="${1:-lingbot-va}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 初始化 conda
CONDA_BASE="${CONDA_EXE%/bin/conda}"
[[ -z "$CONDA_BASE" ]] && CONDA_BASE="$(conda info --base 2>/dev/null)" || true
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

echo "=== 检查当前 PyTorch 与 ABI ==="
conda activate "$CONDA_ENV"
python -c "
import torch
v = torch.__version__
cuda = torch.version.cuda
abi = torch._C._GLIBCXX_USE_CXX11_ABI
print(f'PyTorch: {v}, CUDA: {cuda}, CXX11_ABI: {abi}')
if not v.startswith('2.6') or cuda != '12.4':
    print('Warning: 本脚本针对 torch 2.6.x+cu124 与 cxx11abiFALSE。当前环境可能不匹配。')
if abi is not False:
    print('Warning: 官方 PyTorch wheel 通常为 CXX11_ABI=False。若用 cxx11abiTRUE 的 flash_attn 会报 undefined symbol。')
"

echo ""
echo ">>> 卸载已有 flash-attn（若存在）"
pip uninstall -y flash-attn 2>/dev/null || true

# 方案 1：使用社区预编译 wheel（torch2.6 + cu124 + cp310，manylinux 与官方 PyTorch ABI 一致）
# 来源: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases (v0.7.16)
# 2.7.4 在 issue #1783 中 2.7.4.post1 确认与 torch 2.6.0+cu124 兼容；mjun0812 的 2.7.4+cu124torch2.6 为同组合
WHEEL_URL_274="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu124torch2.6-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"

echo ""
echo ">>> 方案 1: 安装预编译 wheel (2.7.4+cu124torch2.6, cp310)"
if pip install --no-cache-dir "$WHEEL_URL_274"; then
  echo ""
  echo ">>> 验证 flash_attn 导入"
  if python -c "from flash_attn import flash_attn_func; print('flash_attn 导入成功')"; then
    echo ""
    echo "=== 安装成功。可将 transformer 的 attn_mode 设为 flashattn 以使用 Flash Attention。 ==="
    exit 0
  fi
  echo ">>> 预编译 wheel 导入失败（多为 ABI 不匹配），尝试方案 2"
  pip uninstall -y flash-attn 2>/dev/null || true
fi

# 方案 2：从源码编译，强制使用与 PyTorch 一致的旧 ABI（CXX11_ABI=0）
# 官方 PyTorch wheel 使用 _GLIBCXX_USE_CXX11_ABI=0，故设为 FALSE
echo ""
echo ">>> 方案 2: 从源码编译 flash-attn (FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE)"
export FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE"
export FLASH_ATTENTION_FORCE_BUILD="TRUE"
export MAX_JOBS="${MAX_JOBS:-4}"

pip install --no-cache-dir --no-build-isolation "flash-attn>=2.6.3,<2.8"

echo ""
echo ">>> 验证 flash_attn 导入"
if python -c "from flash_attn import flash_attn_func; print('flash_attn 导入成功')"; then
  echo ""
  echo "=== 安装成功。可将 transformer 的 attn_mode 设为 flashattn。 ==="
  exit 0
fi

echo ""
echo "=== 安装或导入仍失败。请检查: 1) CUDA/nvcc 可用 2) 与当前 PyTorch 版本完全一致。 ==="
exit 1
