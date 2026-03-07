#!/usr/bin/env bash
# 仅运行 RoboTwin eval client（需先在另一终端启动 LingBot-VA server）
# 用法（在 lingbot-va 仓库根目录执行）：
#   bash script/run_eval_robotwin_client_only.sh
#   bash script/run_eval_robotwin_client_only.sh ./results adjust_bottle 20
#   export PORT=29056; bash script/run_eval_robotwin_client_only.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/users/wangyuxuan-20250915/EAI/RoboTwin-2.0}"
export ROBOTWIN_ROOT

save_root="${1:-./results}"
task_name="${2:-adjust_bottle}"
test_num="${3:-100}"
PORT="${PORT:-29056}"

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=0

export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"

echo "Client only: ROBOTWIN_ROOT=$ROBOTWIN_ROOT save_root=$save_root task_name=$task_name test_num=$test_num port=$PORT"
cd "$REPO_ROOT"
mkdir -p "$save_root"

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi \
  --config "$ROBOTWIN_ROOT/policy/$policy_name/deploy_policy.yml" \
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
  --port "$PORT"
