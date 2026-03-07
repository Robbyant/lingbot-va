#!/usr/bin/env bash
# RoboTwin 2.0 一键下载脚本
# 功能：克隆 RoboTwin 仓库（若不存在）+ 下载并解压 **2.0 专用** assets，并写入资源路径配置。
#
# 与 1.0 区分：默认安装到 RoboTwin-2.0，避免和已有 RoboTwin/1.0 共用同一 assets 目录造成混用或覆盖。
# 若需同时保留 1.0：1.0 用例如 EAI/RoboTwin 或 EAI/RoboTwin-1.0，2.0 用本脚本默认 EAI/RoboTwin-2.0。
#
# 用法（任选其一）：
#   # 从 lingbot-va 仓库根目录执行（推荐）
#   bash script/download/download_robotwin2.sh
#   bash script/download/download_robotwin2.sh /path/to/RoboTwin-2.0
#
#   # 国内镜像（HuggingFace 不稳定时）
#   HF_ENDPOINT=https://hf-mirror.com bash script/download/download_robotwin2.sh
#   BENCH_ROOT=/path/to/RoboTwin-2.0 bash script/download/download_robotwin2.sh
#
# 环境变量：
#   BENCH_ROOT      RoboTwin 2.0 安装目录，默认: .../EAI/RoboTwin-2.0（与 1.0 分目录）
#   ROBOTWIN_COMMIT 使用的 git 提交，默认: 2eeec322（与官方测评一致）
#   HF_ENDPOINT     可选，国内可设 https://hf-mirror.com 加速 assets 下载

set -e

# 脚本所在目录 -> lingbot-va 仓库根
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BENCH_ROOT="${1:-${BENCH_ROOT:-/mnt/users/wangyuxuan-20250915/EAI/RoboTwin-2.0}}"
ROBOTWIN_COMMIT="${ROBOTWIN_COMMIT:-2eeec322}"
# 转为绝对路径（若目录已存在）
if [ -d "$BENCH_ROOT" ]; then
  BENCH_ROOT="$(cd "$BENCH_ROOT" && pwd)"
elif [ -d "$(dirname "$BENCH_ROOT")" ]; then
  true
else
  mkdir -p "$(dirname "$BENCH_ROOT")"
fi

echo "============= RoboTwin 2.0 下载 =============
  lingbot-va 仓库: $REPO_ROOT
  RoboTwin 目录:  $BENCH_ROOT
  git commit:     $ROBOTWIN_COMMIT
  HF_ENDPOINT:    ${HF_ENDPOINT:-（未设置）}
================================================"

# 1. 克隆 RoboTwin 仓库（若不存在）
if [ ! -d "$BENCH_ROOT" ] || [ ! -f "$BENCH_ROOT/script/_download_assets.sh" ]; then
  if [ ! -d "$BENCH_ROOT" ]; then
    echo ">>> 克隆 RoboTwin 到: $BENCH_ROOT"
    mkdir -p "$(dirname "$BENCH_ROOT")"
    git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$BENCH_ROOT"
  fi
  cd "$BENCH_ROOT"
  git fetch origin 2>/dev/null || true
  git checkout "$ROBOTWIN_COMMIT" 2>/dev/null || true
  cd - > /dev/null
  echo ">>> 仓库就绪: $BENCH_ROOT"
else
  echo ">>> 已存在 RoboTwin 目录，跳过 clone；如需指定 commit 可设 ROBOTWIN_COMMIT 后重新运行"
  cd "$BENCH_ROOT"
  git fetch origin 2>/dev/null || true
  git checkout "$ROBOTWIN_COMMIT" 2>/dev/null || true
  cd - > /dev/null
fi
# 确保之后使用绝对路径
BENCH_ROOT="$(cd "$BENCH_ROOT" && pwd)"

# 2. 下载并解压 assets（使用带重试与镜像的 Python 脚本）
echo ""
echo ">>> 下载 RoboTwin 2.0 assets（HuggingFace: TianxingChen/RoboTwin2.0）..."
if [ ! -f "$REPO_ROOT/script/download/download_robotwin_assets.py" ]; then
  echo "Error: 未找到 $REPO_ROOT/script/download/download_robotwin_assets.py，请从 lingbot-va 仓库根目录执行本脚本。"
  exit 1
fi

cd "$REPO_ROOT"
python script/download/download_robotwin_assets.py "$BENCH_ROOT"

echo ""
echo "============= 下载完成 =============
  RoboTwin 2.0: $BENCH_ROOT
  assets:       $BENCH_ROOT/assets（仅 2.0 资源，与 1.0 分目录不混用）

后续步骤（测评 / 运行仿真）：
  1) 安装系统与 Python 依赖（见 RoboTwin 文档）：
     sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
     cd $BENCH_ROOT && bash script/_install.sh
  2) 测评时指定 2.0 路径并运行 eval：
     export ROBOTWIN_ROOT=$BENCH_ROOT
     bash script/run_eval_robotwin.sh
=========================================="
