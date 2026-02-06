#!/bin/bash

lora_ckpt=$1
PROMPTS_FILE=$2
input_image=$3
SEED=${4:-None}
NUM_STEPS=$5
CFG_SCALE=${6:-2.0}

OUTPUT_DIR=$(dirname "$lora_ckpt")
# OUTPUT_DIR="outputs/I2V_inference_lora_14B"
mkdir -p $OUTPUT_DIR

# Check if the prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file '$PROMPTS_FILE' not found!"
    echo "Usage: $0 <lora_ckpt> <prompts_file> <input_image> [seed] [num_steps] [cfg_scale]"
    echo "Example: $0 model.safetensors prompts.txt input.jpg 42 50 5.0"
    exit 1
fi

# Configuration
MODEL_ID_WITH_ORIGIN_PATHS="Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
HEIGHT=480  # Match training script
WIDTH=832   # Match training script
NUM_FRAMES=81


REMOVE_PREFIX="pipe.dit."  # Match training script
TORCH_DTYPE="bfloat16"     # Match training script



echo "Starting batch inference..."
echo "LoRA checkpoint: $lora_ckpt"
echo "Input image: $input_image"
echo "Prompts file: $PROMPTS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Resolution: ${WIDTH}x${HEIGHT}, Frames: $NUM_FRAMES"
echo "Model config: $MODEL_ID_WITH_ORIGIN_PATHS"

script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir/.."
if [ "$SEED" != "None" ]; then
    SEED_ARG="--seed $SEED"
else
    SEED_ARG=""
fi


CMD="python inference.py \
    --lora_ckpt \"$lora_ckpt\" \
    --lora_base_model \"dit\" \
    --lora_rank 32 \
    --lora_target_modules \"q,k,v,o,ffn.0,ffn.2\" \
    --input_image \"$input_image\" \
    --prompts_file \"$PROMPTS_FILE\" \
    --model_id_with_origin_paths \"$MODEL_ID_WITH_ORIGIN_PATHS\" \
    --output_dir \"$OUTPUT_DIR\" \
    --height $HEIGHT \
    --width $WIDTH \
    --num_frames $NUM_FRAMES \
    --cfg_scale $CFG_SCALE \
    --num_inference_steps $NUM_STEPS \
    --remove_prefix_in_ckpt \"$REMOVE_PREFIX\" \
    --torch_dtype \"$TORCH_DTYPE\" \
    $SEED_ARG \
    --log_level INFO"
    
echo "$CMD"
eval "$CMD"
