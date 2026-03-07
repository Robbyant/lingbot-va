#!/usr/bin/bash
# 单 GPU 运行 Image-to-Video-Action 推理（用于调试/本地跑通）
# 使用前请设置模型路径并准备首帧图像，见 INFERENCE.md
#
# 从仓库根目录执行: bash script/run_i2va_single_gpu.sh

set -e

NGPU=${NGPU:-1}
CONFIG_NAME=${CONFIG_NAME:-robotwin_i2av}
# 模型目录：需包含 vae / tokenizer / text_encoder / transformer 子目录
export LINGBOT_VA_MODEL_PATH=${LINGBOT_VA_MODEL_PATH:-"/path/to/pretrained/model"}

if [ "$LINGBOT_VA_MODEL_PATH" = "/path/to/pretrained/model" ]; then
    echo "Error: 请设置环境变量 LINGBOT_VA_MODEL_PATH 为已下载的 LingBot-VA 模型目录"
    echo "  export LINGBOT_VA_MODEL_PATH=/path/to/lingbot-va-base"
    echo "  并确保该目录下存在: vae/ tokenizer/ text_encoder/ transformer/"
    exit 1
fi

if [ ! -d "$LINGBOT_VA_MODEL_PATH/transformer" ]; then
    echo "Error: 未找到 $LINGBOT_VA_MODEL_PATH/transformer"
    echo "  请从 HuggingFace/ModelScope 下载 lingbot-va-base 并解压到该路径"
    exit 1
fi

# 推理前请将 transformer/config.json 中 attn_mode 设为 "torch" 或 "flashattn"（不能是 "flex"）
echo "Config: NGPU=$NGPU CONFIG_NAME=$CONFIG_NAME"
echo "Model:  $LINGBOT_VA_MODEL_PATH"
echo ""

export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/.."

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" python -m torch.distributed.run \
    --nproc_per_node="$NGPU" \
    --master_port=29501 \
    -m wan_va.wan_va_server --config-name "$CONFIG_NAME" "$@"
