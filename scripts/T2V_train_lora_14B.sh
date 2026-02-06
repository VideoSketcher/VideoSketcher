dataset_path="$1"
metadata_path="$2"
dataset_repeat="$3"
epochs=$4
resume_from="$5"
if [ "$resume_from" != "None" ] && [ "$resume_from" != "" ]; then
    resume_arg="--resume $resume_from"
else
    resume_arg=""
fi

GPU_NUM=${GPU_NUM:-7}
echo "Using GPU_NUM: $GPU_NUM"
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir/.."


accelerate launch  \
  --config_file "./accelerate_config.yaml" \
  --num_processes $GPU_NUM \
  train.py \
  --dataset_base_path "$dataset_path" \
  --dataset_metadata_path "$metadata_path" \
  --height 480 \
  --width 832 \
  --dataset_repeat "$dataset_repeat" \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs $epochs \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --num_frames 81 \
  --experiment_dir "experiments/T2V-14B_lora" \
  --no_wandb \
  $resume_arg