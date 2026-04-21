#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, libero_train

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# Auto resume from latest checkpoint (set RESUME=1).
if [ "${RESUME:-0}" = "1" ]; then
    overrides="--resume-latest ${overrides}"
fi

# Do not clobber user-provided WANDB_* env vars.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_TEAM_NAME="${WANDB_TEAM_NAME:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python3 -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
