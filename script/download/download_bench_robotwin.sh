#!/usr/bin/bash
# RoboTwin 2.0 测评环境与资源下载（测评 bench 用）
# 使用完整路径，从仓库根目录执行: bash script/download_bench_robotwin.sh
#
# 说明：测评不是在“数据集文件”上跑，而是在 RoboTwin 仿真里跑 50 个 task。
#       需要先克隆 RoboTwin 仓库并下载其 assets，再按 README 启动 server + client。

set -e

# 测评 bench 根目录（RoboTwin 克隆位置）
BENCH_ROOT="${BENCH_ROOT:-/mnt/users/wangyuxuan-20250915/EAI/RoboTwin}"
ROBOTWIN_COMMIT="${ROBOTWIN_COMMIT:-2eeec322}"

echo "BENCH_ROOT (RoboTwin 克隆目录): $BENCH_ROOT"
echo "RoboTwin commit: $ROBOTWIN_COMMIT"
echo ""

# 1. 克隆 RoboTwin 并 checkout
if [ ! -d "$BENCH_ROOT" ]; then
    echo ">>> 克隆 RoboTwin 到 $BENCH_ROOT"
    git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$BENCH_ROOT"
    cd "$BENCH_ROOT"
    git checkout "$ROBOTWIN_COMMIT"
    cd - > /dev/null
else
    echo ">>> 目录已存在: $BENCH_ROOT，跳过 clone；如需重装请先删掉该目录"
    cd "$BENCH_ROOT"
    git fetch origin 2>/dev/null || true
    git checkout "$ROBOTWIN_COMMIT" 2>/dev/null || true
    cd - > /dev/null
fi

# 2. 下载 assets（测评必需）
echo ""
echo ">>> 下载 RoboTwin assets（测评场景与资源）"
cd "$BENCH_ROOT"
if [ -f "script/_download_assets.sh" ]; then
    if ! bash script/_download_assets.sh; then
        echo ""
        echo "官方脚本下载失败（多为 HuggingFace 连接中断）。请用带重试与镜像的脚本："
        echo "  cd $(dirname "$0")/.."
        echo "  HF_ENDPOINT=https://hf-mirror.com python script/download_robotwin_assets.py $BENCH_ROOT"
        exit 1
    fi
else
    echo "未找到 script/_download_assets.sh，请先完成 RoboTwin 安装（见 README 步骤 2–4）。"
    exit 1
fi
cd - > /dev/null

echo ""
echo "测评 bench 下载完成."
echo "RoboTwin 路径: $BENCH_ROOT"
echo ""
echo "后续步骤（需在 RoboTwin 文档中完成安装后再做）："
echo "  1) 安装依赖: sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools"
echo "  2) 按 README 修改 script/requirements.txt 和 script/_install.sh 后执行: bash script/_install.sh"
echo "  3) 测评时设置 RoboTwin 路径并启动 client:"
echo "     export ROBOTWIN_ROOT=$BENCH_ROOT"
echo "     # 在 lingbot-va 仓库根目录执行: bash evaluation/robotwin/launch_client.sh <save_root> <task_name>"
echo "  4) 先启动 LingBot-VA server，再在同上目录运行 launch_client.sh 进行测评"
