#!/bin/bash

dataset_base_path=$1
metadata_path=$2
dataset_repeat=${3:-100}
num_epochs=${4:-10}
lr=${5:-2e-6}
resume=${6:-None}
GPU_NUM=${GPU_NUM:-7}

echo "Using GPU_NUM: $GPU_NUM"
echo "Starting video regression training with Accelerate..."
echo "Config: $CONFIG_PATH"

script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir/.."

if [ "$resume" != "None" ]; then
    resume_arg="--resume $resume"
else
    resume_arg=""
fi

CMD="
accelerate launch --config_file accelerate_config.yaml --num_processes $GPU_NUM ar_train.py --config_path configs/wan_causal_ode_finetune.yaml \
    --dataset_base_path $dataset_base_path \
    --metadata_path $metadata_path \
    --lr $lr \
    --num_epochs $num_epochs \
    --dataset_repeat $dataset_repeat \
    --no_wandb \
    $resume_arg
"
echo $CMD
eval $CMD