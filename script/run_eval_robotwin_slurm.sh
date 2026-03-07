#!/usr/bin/env bash
#SBATCH -p a100_global
#SBATCH -N 1
#SBATCH --gres=gpu:4
# 多卡并行：改为 --gres=gpu:4 则起 4 个 server+4 个 client 并行跑 4 个任务（每卡一个）
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --job-name=robotwin_va
#SBATCH --time=48:00:00
#SBATCH --output=logs/robotwin_%x-%j.out
#SBATCH --error=logs/robotwin_%x-%j.err
#
# RoboTwin 2.0 测评脚本（LingBot-VA）- Slurm 版
# 用法（在 lingbot-va 仓库根目录执行）：
#   sbatch script/run_eval_robotwin_slurm.sh
#   sbatch script/run_eval_robotwin_slurm.sh ./my_results
#   sbatch script/run_eval_robotwin_slurm.sh ./out stack_bowls_three
#   sbatch script/run_eval_robotwin_slurm.sh ./out adjust_bottle 10
# 可选环境变量（提交前 export 或 sbatch 前设置）：
#   SAVE_ROOT, TASK_NAME, TEST_NUM, PORT, NPROC_PER_NODE, CUDA_VISIBLE_DEVICES, SLURM_GPUS
# 多卡并行：sbatch --gres=gpu:4 时起 4 个 server（每卡一个）+ 4 个 client 并行跑 4 个任务；
#   可通过 TASK_LIST="t1 t2 t3 t4" 指定任务，否则用前 4 个 from 50 任务列表。

set -e

NGPU="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/mnt/users/wangyuxuan-20250915/EAI/RoboTwin-2.0}"
export ROBOTWIN_ROOT

# 50 任务列表（多卡时取前 N 个，或由 TASK_LIST 指定）
ROBOTWIN_ALL_TASKS=(
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock
  click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug
  lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
  open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left
  place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket
  place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan
  place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand
  place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
  shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three
  stack_bowls_two stamp_seal turn_switch
)

# 模型路径（优先 models/ 下，其次仓库根下）
if [ -n "$LINGBOT_VA_MODEL_PATH" ]; then
  :
elif [ -d "$REPO_ROOT/models/lingbot-va-posttrain-robotwin" ]; then
  export LINGBOT_VA_MODEL_PATH="$REPO_ROOT/models/lingbot-va-posttrain-robotwin"
elif [ -d "$REPO_ROOT/lingbot-va-posttrain-robotwin" ]; then
  export LINGBOT_VA_MODEL_PATH="$REPO_ROOT/lingbot-va-posttrain-robotwin"
else
  export LINGBOT_VA_MODEL_PATH="${LINGBOT_VA_MODEL_PATH:-$REPO_ROOT/lingbot-va-base}"
fi
# 使用前 NGPU 张卡（未设 CUDA_VISIBLE_DEVICES 时；单卡即为 0）
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NGPU-1)))"
fi
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"

save_root="${SAVE_ROOT:-${1:-./results}}"
task_name="${TASK_NAME:-${2:-adjust_bottle}}"
test_num="${TEST_NUM:-${3:-100}}"
PORT="${PORT:-29056}"
MASTER_PORT="${MASTER_PORT:-29501}"

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=0

# 多卡时每卡 1 个 server（1 进程），单卡时也是 1 进程
NPROC_PER_NODE=1

echo "============= RoboTwin Eval (LingBot-VA) [Slurm] =============
  JOB_ID             = ${SLURM_JOB_ID:-N/A}
  NGPU               = $NGPU
  REPO_ROOT          = $REPO_ROOT
  ROBOTWIN_ROOT      = $ROBOTWIN_ROOT
  LINGBOT_VA_MODEL_PATH = $LINGBOT_VA_MODEL_PATH
  save_root          = $save_root
  task_name          = $task_name
  test_num           = $test_num
  PORT               = $PORT
  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
========================================================"

if [ ! -d "$ROBOTWIN_ROOT" ]; then
  echo "Error: ROBOTWIN_ROOT 不存在: $ROBOTWIN_ROOT"
  exit 1
fi

if [ ! -d "$ROBOTWIN_ROOT/assets" ]; then
  echo "Error: 未找到 $ROBOTWIN_ROOT/assets，请先下载 RoboTwin assets"
  exit 1
fi

if ! python -c "import sapien" 2>/dev/null; then
  echo "Error: 当前环境缺少 RoboTwin 仿真依赖 'sapien'."
  exit 1
fi

if [ -d "$ROBOTWIN_ROOT/assets/embodiments" ] && [ -f "$ROBOTWIN_ROOT/script/update_embodiment_config_path.py" ]; then
  if [ -n "$(find "$ROBOTWIN_ROOT/assets/embodiments" -name '*_tmp.yml' 2>/dev/null | head -1)" ]; then
    (cd "$ROBOTWIN_ROOT" && python script/update_embodiment_config_path.py </dev/null) || true
  fi
fi

CONDA_BASE=""
[[ -n "$CONDA_EXE" ]] && CONDA_BASE="${CONDA_EXE%/bin/conda}"
[[ -z "$CONDA_BASE" ]] && CONDA_BASE="$(conda info --base 2>/dev/null)" || true
if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate lingbot-va
else
  echo "Error: 未检测到 conda 或 lingbot-va 环境"
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p logs "$save_root"

if [ "$NGPU" -eq 1 ]; then
  # ---------- 单卡：1 个 server + 1 个 client ----------
  echo ">>> 启动 LingBot-VA server (单卡 port=$PORT) ..."
  PYTHONWARNINGS=ignore::UserWarning python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port "$PORT" \
    --save_root "$save_root" &
  SERVER_PID=$!
  trap "kill $SERVER_PID 2>/dev/null || true" EXIT

  sleep 15
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server 启动失败"
    exit 1
  fi

  echo ">>> 启动 eval client (task=$task_name, test_num=$test_num) ..."
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
else
  # ---------- 多卡：每卡 1 个 server + 1 个 client，并行跑 N 个任务 ----------
  if [ -n "$TASK_LIST" ]; then
    TASKS=($TASK_LIST)
  else
    TASKS=("${ROBOTWIN_ALL_TASKS[@]:0:$NGPU}")
  fi
  if [ ${#TASKS[@]} -lt "$NGPU" ]; then
    echo "Error: 任务数 ${#TASKS[@]} 小于 NGPU=$NGPU，请设置 TASK_LIST=\"t1 t2 ...\""
    exit 1
  fi

  SERVER_PIDS=()
  trap 'for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null || true; done' EXIT

  echo ">>> 启动 $NGPU 个 LingBot-VA server (ports $PORT..$((PORT+NGPU-1))) ..."
  for ((g=0; g<NGPU; g++)); do
    port_g=$((PORT + g))
    master_port_g=$((MASTER_PORT + g))
    export CUDA_VISIBLE_DEVICES=$g
    PYTHONWARNINGS=ignore::UserWarning python -m torch.distributed.run \
      --nproc_per_node 1 \
      --master_port "$master_port_g" \
      wan_va/wan_va_server.py \
      --config-name robotwin \
      --port "$port_g" \
      --save_root "$save_root" &
    SERVER_PIDS+=($!)
    sleep 2
  done

  sleep 15
  for ((g=0; g<NGPU; g++)); do
    if ! kill -0 "${SERVER_PIDS[$g]}" 2>/dev/null; then
      echo "Error: Server GPU $g 启动失败"
      exit 1
    fi
  done

  echo ">>> 启动 $NGPU 个 eval client 并行 (tasks: ${TASKS[*]}) ..."
  PIDS=()
  for ((g=0; g<NGPU; g++)); do
    task_g="${TASKS[$g]}"
    port_g=$((PORT + g))
    (
      export CUDA_VISIBLE_DEVICES=$g
      PYTHONWARNINGS=ignore::UserWarning \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi \
        --config "$ROBOTWIN_ROOT/policy/$policy_name/deploy_policy.yml" \
        --overrides \
        --task_name "$task_g" \
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
        --port "$port_g"
    ) &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait "$p" || true; done
fi

echo ">>> Eval 结束，结果见: $save_root"
