#!/usr/bin/env bash
# RoboTwin 2.0 完全集测评（50 个任务 × test_num）
# 支持单卡顺序跑或多卡并行：通过 NUM_GPUS 或第 3 个参数指定 GPU 数，>1 时每卡起一个 server，任务按卡分批并行。
# 与 run_eval_robotwin.sh 同环境要求。
#
# 用法（在 lingbot-va 仓库根目录执行）：
#   bash script/run_eval_robotwin_full.sh                    # 单卡顺序，结果 ./results，每任务 100 条
#   bash script/run_eval_robotwin_full.sh ./my_results        # 指定结果目录
#   bash script/run_eval_robotwin_full.sh ./my_results 100 4 # 结果目录 + test_num + 4 卡并行
#   NUM_GPUS=8 bash script/run_eval_robotwin_full.sh ./out   # 8 卡并行（50 任务分 8 批）
#   GPU_IDS=2,3,4,5 bash script/run_eval_robotwin_full.sh ./out 100 4  # 使用指定 GPU 4卡并行
#   GPU_IDS=1 bash script/run_eval_robotwin_full.sh ./out    # 单卡运行，使用 GPU 1
#
# 可选环境变量：ROBOTWIN_ROOT, LINGBOT_VA_MODEL_PATH, NUM_GPUS, GPU_IDS, START_PORT, MASTER_PORT_BASE

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 最先激活 lingbot-va 环境（与 run_eval_robotwin.sh 一致）
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

# 模型路径：优先 models/，其次仓库根下 posttrain，否则 base（与 run_eval_robotwin.sh 一致）
if [ -n "$LINGBOT_VA_MODEL_PATH" ]; then
  :
elif [ -d "$REPO_ROOT/models/lingbot-va-posttrain-robotwin" ]; then
  export LINGBOT_VA_MODEL_PATH="$REPO_ROOT/models/lingbot-va-posttrain-robotwin"
elif [ -d "$REPO_ROOT/lingbot-va-posttrain-robotwin" ]; then
  export LINGBOT_VA_MODEL_PATH="$REPO_ROOT/lingbot-va-posttrain-robotwin"
else
  export LINGBOT_VA_MODEL_PATH="${LINGBOT_VA_MODEL_PATH:-$REPO_ROOT/lingbot-va-base}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"

save_root="${1:-./results}"
test_num="${2:-100}"
num_gpus="${3:-${NUM_GPUS:-6}}"
num_gpus=$((num_gpus))
GPU_IDS="${GPU_IDS:-2,3,4,5,6,7}"
START_PORT="${START_PORT:-29556}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29661}"
PORT="${PORT:-29056}"
MASTER_PORT="${MASTER_PORT:-29501}"

# 实际推理为 LingBot-VA，结果目录与日志用 lingbot；deploy 配置沿用 RoboTwin 的 ACT
policy_name=lingbot
policy_config="${POLICY_CONFIG:-ACT}"
task_config=demo_clean
train_config_name=0
model_name=0
seed=0

# 50 个任务（与 evaluation/robotwin/calc_stat.py TASK_CLASS 一致）
# ROBOTWIN_ALL_TASKS=(
#   adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock
#   click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug
#   lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
#   open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left
#   place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket
#   place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan
#   place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand
#   place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
#   shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three
#   stack_bowls_two stamp_seal turn_switch
# )

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

echo "============= RoboTwin 完全集 Eval (LingBot-VA) =============
  REPO_ROOT          = $REPO_ROOT
  ROBOTWIN_ROOT      = $ROBOTWIN_ROOT
  LINGBOT_VA_MODEL_PATH = $LINGBOT_VA_MODEL_PATH
  policy_name        = $policy_name (LingBot-VA)
  policy_config      = $policy_config (deploy yml)
  save_root          = $save_root
  test_num per task  = $test_num (每任务条数)
  任务数              = ${#ROBOTWIN_ALL_TASKS[@]} (共 50 个任务)
  num_gpus           = $num_gpus
  GPU_IDS            = ${GPU_IDS:-自动按卡分配 (0,1,2...)}
  START_PORT         = $START_PORT
  PORT               = $PORT
  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
================================================================"

if [ ! -d "$ROBOTWIN_ROOT" ]; then
  echo "Error: ROBOTWIN_ROOT 不存在: $ROBOTWIN_ROOT"
  exit 1
fi

if [ ! -d "$ROBOTWIN_ROOT/assets" ]; then
  echo "Error: 未找到 $ROBOTWIN_ROOT/assets，请先下载 RoboTwin assets"
  echo "  例: cd $REPO_ROOT && python script/download/download_robotwin_assets.py $ROBOTWIN_ROOT"
  exit 1
fi

# 检查 RoboTwin 仿真依赖（client 会 import envs -> sapien）（与 run_eval_robotwin.sh 一致）
if ! python -c "import sapien" 2>/dev/null; then
  echo "Error: 当前环境缺少 RoboTwin 仿真依赖 'sapien'，导致 client 报错 ModuleNotFoundError."
  echo "  请在当前 conda 环境中安装 RoboTwin 依赖后再运行本脚本，例如："
  echo "    pip install sapien==3.0.0b1"
  echo "    pip install -r $ROBOTWIN_ROOT/script/requirements.txt"
  echo "  并按 RoboTwin 文档完成 script/_install.sh（pytorch3d、mplib 补丁、curobo 等）。"
  echo "  详见: $ROBOTWIN_ROOT/INSTALLATION.md 或 README"
  exit 1
fi

# 首次运行：更新 embodiment 配置中的资源路径（从 _tmp.yml 生成 .yml）
if [ -d "$ROBOTWIN_ROOT/assets/embodiments" ] && [ -f "$ROBOTWIN_ROOT/script/update_embodiment_config_path.py" ]; then
  if [ -n "$(find "$ROBOTWIN_ROOT/assets/embodiments" -name '*_tmp.yml' 2>/dev/null | head -1)" ]; then
    echo ">>> 更新 RoboTwin embodiment 配置路径..."
    (cd "$ROBOTWIN_ROOT" && python script/update_embodiment_config_path.py </dev/null) || true
  fi
fi

cd "$REPO_ROOT"
mkdir -p "$save_root"

if [ "$num_gpus" -eq 1 ]; then
  # ---------- 单卡：一个 server，顺序跑 50 个任务 ----------
  if [ -z "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  else
    gpu_single=$(echo "$GPU_IDS" | cut -d, -f1)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_single}"
  fi
  echo ">>> 启动 LingBot-VA server (单卡 GPU=$CUDA_VISIBLE_DEVICES, port=$PORT) ..."
  PYTHONWARNINGS=ignore::UserWarning python -m torch.distributed.run \
    --nproc_per_node "$num_gpus" \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port "$PORT" \
    --save_root "$save_root" &
  SERVER_PID=$!
  trap "kill $SERVER_PID 2>/dev/null || true" EXIT

  sleep 15
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server 启动失败，请检查日志（如 attn_mode 是否为 torch/flashattn）"
    exit 1
  fi

  total=${#ROBOTWIN_ALL_TASKS[@]}
  for i in "${!ROBOTWIN_ALL_TASKS[@]}"; do
    task_name="${ROBOTWIN_ALL_TASKS[$i]}"
    echo ""
    echo ">>> [$((i+1))/$total] 启动 eval client: task=$task_name, test_num=$test_num"
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
      --port "$PORT"
  done
else
  # ---------- 多卡：每卡一个 server，任务分批并行 ----------
  # 解析 GPU_IDS
  if [ -z "$GPU_IDS" ]; then
    GPU_ARRAY=()
    for ((i=0; i<num_gpus; i++)); do
      GPU_ARRAY+=($i)
    done
  else
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
  fi
  
  if [ ${#GPU_ARRAY[@]} -lt $num_gpus ]; then
    echo "Error: GPU_IDS 中的GPU数量 (${#GPU_ARRAY[@]}) 小于指定的 num_gpus ($num_gpus)"
    exit 1
  fi
  
  SERVER_PIDS=()
  trap 'for p in "${SERVER_PIDS[@]}"; do kill $p 2>/dev/null || true; done' EXIT

  echo ">>> 启动 $num_gpus 个 LingBot-VA server (GPU=${GPU_ARRAY[*]}, ports $START_PORT..$((START_PORT + num_gpus - 1))) ..."
  for ((g=0; g<num_gpus; g++)); do
    gpu_id=${GPU_ARRAY[$g]}
    port=$((START_PORT + g))
    master_port=$((MASTER_PORT_BASE + g))
    export CUDA_VISIBLE_DEVICES=$gpu_id
    PYTHONWARNINGS=ignore::UserWarning python -m torch.distributed.run \
      --nproc_per_node 1 \
      --master_port "$master_port" \
      wan_va/wan_va_server.py \
      --config-name robotwin \
      --port "$port" \
      --save_root "$save_root" &
    SERVER_PIDS+=($!)
    sleep 2
  done

  sleep 15
  for ((g=0; g<num_gpus; g++)); do
    if ! kill -0 "${SERVER_PIDS[$g]}" 2>/dev/null; then
      echo "Error: Server GPU $g 启动失败，请检查日志（如 attn_mode 是否为 torch/flashattn）"
      exit 1
    fi
  done

  total=${#ROBOTWIN_ALL_TASKS[@]}
  batch_time=$(date +%Y%m%d_%H%M%S)
  log_dir="${save_root}/logs"
  mkdir -p "$log_dir"

  for ((batch_start=0; batch_start<total; batch_start+=num_gpus)); do
    batch_num=$((batch_start / num_gpus + 1))
    batch_total=$(( (total + num_gpus - 1) / num_gpus ))
    end_idx=$((batch_start + num_gpus))
    [ "$end_idx" -gt "$total" ] && end_idx=$total
    echo ""
    echo ">>> 批次 $batch_num/$batch_total (任务 $((batch_start+1))..${end_idx}/$total)"
    PIDS=()
    for ((j=0; j<num_gpus; j++)); do
      idx=$((batch_start + j))
      [ "$idx" -ge "$total" ] && break
      task_name="${ROBOTWIN_ALL_TASKS[$idx]}"
      gpu_id=${GPU_ARRAY[$j]}
      port=$((START_PORT + j))
      log_file="${log_dir}/${task_name}_${batch_time}.log"
      (
        export CUDA_VISIBLE_DEVICES=$gpu_id
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
          --port "$port"
      ) > "$log_file" 2>&1 &
      PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait "$p" || true; done
  done
fi

echo ""
echo ">>> 完全集 Eval 结束，结果见: $save_root"
if [ "$num_gpus" -gt 1 ]; then
  echo "    各任务日志: $save_root/logs/"
fi
echo "    汇总成功率: python -m evaluation.robotwin.calc_stat $save_root/stseed-0/visualization"
