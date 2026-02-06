#!/bin/bash
CHECKPOINT_PATH=$1
PROMPTS_FILE=$2
SEED=${3:-0}
NUM_ROLLOUT=${4:-1}
NUM_OVERLAP_FRAMES=${5:-3}
BACKGROUND_IMAGE=${6:-None}
OUTPUT_DIR=${7:-None}

script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir/.."

CONFIG_PATH="configs/wan_causal_ode_finetune.yaml"

HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
TORCH_DTYPE="bfloat16"
LOG_LEVEL="INFO"



if [ "$BACKGROUND_IMAGE" != "None" ]; then
    bg_args="--background_image $BACKGROUND_IMAGE"
else
    bg_args=""
fi

if [ "$OUTPUT_DIR" != "None" ]; then
    output_args="--output_dir $OUTPUT_DIR"
else
    output_args=""
fi

echo "Starting inference"
CMD="
python ar_inference.py \
    --config_path "$CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --prompts_file "$PROMPTS_FILE" \
    --height $HEIGHT \
    --width $WIDTH \
    --num_frames $NUM_FRAMES \
    --seed $SEED \
    --fps $FPS \
    --torch_dtype "$TORCH_DTYPE" \
    --log_level "$LOG_LEVEL" \
    --num_rollout $NUM_ROLLOUT \
    --num_overlap_frames $NUM_OVERLAP_FRAMES \
    $bg_args \
    $output_args
"
echo $CMD
eval $CMD
echo "Inference completed"