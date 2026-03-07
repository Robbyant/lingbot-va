#!/usr/bin/env bash
# RoboTwin 2.0 测评脚本（LingBot-VA）
# 使用前：已下载 RoboTwin 及 assets 到 ROBOTWIN_ROOT（默认 EAI/RoboTwin）
# 重要：eval client 会启动 RoboTwin 仿真，当前 conda 环境需安装 RoboTwin 依赖（sapien、mplib 等），
#       否则会报 ModuleNotFoundError: No module named 'sapien'。见下方「依赖」说明。
#
# 用法（在 lingbot-va 仓库根目录执行）：
#   bash script/run_eval_robotwin.sh                    # 单任务 adjust_bottle，结果到 ./results
#   bash script/run_eval_robotwin.sh ./my_results        # 指定结果目录
#   bash script/run_eval_robotwin.sh ./out stack_bowls_three   # 指定任务名
#   bash script/run_eval_robotwin.sh ./out adjust_bottle 10    # 结果目录 + 任务 + test_num
# 测完全集（50 任务 × 100 条）：bash script/run_eval_robotwin_full.sh [save_root] [test_num]
# 是否渲染/保存预测视频（VAE 解码+对比视频，较耗时）：SAVE_VISUALIZATION=1 bash script/run_eval_robotwin.sh ...  # 默认 0 关闭
# 是否启用 offload（1=把 VAE 放 CPU 节省显存，0=VAE 常驻 GPU 加速 encode_obs）：ENABLE_OFFLOAD=0 bash script/run_eval_robotwin.sh ...

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 最先激活 lingbot-va 环境，保证后续 python/sapien 检查及 server/client 均在该环境下运行
CONDA_BASE=""
if [[ -n "$CONDA_EXE" ]]; then
  CONDA_BASE="${CONDA_EXE%/bin/conda}"
fi
if [[ -z "$CONDA_BASE" ]]; then
  CONDA_BASE="$(conda info --base 2>/dev/null)" || true
fi
if [[ -z "$CONDA_BASE" ]]; then
  for d in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3" "/opt/conda"; do
    if [[ -f "${d}/etc/profile.d/conda.sh" ]]; then
      CONDA_BASE="$d"
      break
    fi
  done
fi
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate lingbot-va
  echo ">>> 已自动激活 conda 环境: lingbot-va"
else
  echo "Error: 未检测到 conda 或无法找到 lingbot-va 环境。"
  echo "  请安装 conda 并创建环境: conda create -n lingbot-va python=3.x && conda activate lingbot-va"
  exit 1
fi

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/users/wangyuxuan-20250915/EAI/RoboTwin-2.0}"
export ROBOTWIN_ROOT

# 模型路径：优先 models/，其次仓库根下 posttrain，否则 base
if [ -n "$LINGBOT_VA_MODEL_PATH" ]; then
  :
elif [ -d "$REPO_ROOT/models/lingbot-va-posttrain-robotwin" ]; then
  export LINGBOT_VA_MODEL_PATH="$REPO_ROOT/models/lingbot-va-posttrain-robotwin"
elif [ -d "$REPO_ROOT/lingbot-va-posttrain-robotwin" ]; then
  export LINGBOT_VA_MODEL_PATH="$REPO_ROOT/lingbot-va-posttrain-robotwin"
else
  export LINGBOT_VA_MODEL_PATH="${LINGBOT_VA_MODEL_PATH:-$REPO_ROOT/lingbot-va-base}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"

# 可调参数：server 使用的 GPU 进程数（默认 1，多卡时需 server 支持）
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# 解析参数
save_root="${1:-./results}"
task_name="${2:-adjust_bottle}"
test_num="${3:-10}"
PORT="${PORT:-29056}"
MASTER_PORT="${MASTER_PORT:-29501}"
# 是否渲染并保存预测视频（0=关闭可加速，1=开启保存对比视频）
save_visualization="${SAVE_VISUALIZATION:-0}"
# 是否把 VAE/text encoder offload 到 CPU（默认 0，优先速度）
enable_offload="${ENABLE_OFFLOAD:-0}"
export ENABLE_OFFLOAD="$enable_offload"

# 实际推理为 LingBot-VA（WebsocketClientPolicy），结果目录与日志用 lingbot；deploy 配置沿用 RoboTwin 的 ACT 配置
policy_name=lingbot
policy_config="${POLICY_CONFIG:-ACT}"
task_config=demo_clean 
train_config_name=0
model_name=0
seed=0

echo "============= RoboTwin Eval (LingBot-VA) =============
  REPO_ROOT          = $REPO_ROOT
  ROBOTWIN_ROOT      = $ROBOTWIN_ROOT
  LINGBOT_VA_MODEL_PATH = $LINGBOT_VA_MODEL_PATH
  policy_name        = $policy_name (LingBot-VA)
  policy_config      = $policy_config (deploy yml)
  save_root          = $save_root
  task_name          = $task_name
  test_num           = $test_num
  PORT               = $PORT
  NPROC_PER_NODE     = $NPROC_PER_NODE
  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
  SAVE_VISUALIZATION = $save_visualization (0=关 1=开预测视频)
  ENABLE_OFFLOAD     = $enable_offload (0=VAE在GPU加速 1=offload到CPU省显存)
========================================================"

# 首次运行：更新 embodiment 配置中的资源路径（从 _tmp.yml 生成 .yml）
if [ -d "$ROBOTWIN_ROOT/assets/embodiments" ] && [ -f "$ROBOTWIN_ROOT/script/update_embodiment_config_path.py" ]; then
  if [ -n "$(find "$ROBOTWIN_ROOT/assets/embodiments" -name '*_tmp.yml' 2>/dev/null | head -1)" ]; then
    echo ">>> 更新 RoboTwin embodiment 配置路径..."
    (cd "$ROBOTWIN_ROOT" && python script/update_embodiment_config_path.py </dev/null) || true
  fi
fi

cd "$REPO_ROOT"
mkdir -p "$save_root"
server_log="${save_root}/server.log"

# 启动 LingBot-VA server（后台，输出写入 server.log，避免结束时的 SIGTERM 信息刷屏终端）
echo ">>> 启动 LingBot-VA server (port=$PORT, nproc_per_node=$NPROC_PER_NODE)，日志: $server_log ..."
PYTHONWARNINGS=ignore::UserWarning python -m torch.distributed.run \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  wan_va/wan_va_server.py \
  --config-name robotwin \
  --port "$PORT" \
  --save_root "$save_root" \
  >> "$server_log" 2>&1 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

echo ">>> 启动 eval client (task=$task_name, test_num=$test_num) ..."
PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi \
  --config "$ROBOTWIN_ROOT/policy/$policy_config/deploy_policy.yml" \
  --overrides \
  --task_name "$task_name" \
  --task_config "$task_config" \
  --train_config_name "$train_config_name" \
  --model_name "$model_name" \
  --ckpt_setting "$model_name" \
  --seed "$seed" \
  --policy_name "$policy_name" \
  --save_root "$save_root" \
  --video_guidance_scale 5 \
  --action_guidance_scale 1 \
  --test_num "$test_num" \
  --save_visualization "$save_visualization" \
  --port "$PORT"

echo ">>> Eval 结束，结果见: $save_root"
