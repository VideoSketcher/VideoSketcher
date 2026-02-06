import os
import sys
import time
import json
import argparse
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import torch
import numpy as np
import cv2
from PIL import Image
import shutil

# project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from utils.experiment_utils import link_slurm_logs, setup_slurm_log_copy_on_exit
from utils import extract_caption_keywords, setup_logging
from utils.prompt_refiner import PromptRefiner



def build_inference_suffix(seed: int, num_rollout: int, num_overlap_frames: int, background_image: str) -> str:
    """Build output directory suffix from inference parameters."""
    seed_suffix = f"seed{seed}" if seed is not None else ""
    long_suffix = f"_long{num_rollout}-{num_overlap_frames}" if num_rollout and num_rollout > 1 else ""
    background_suffix = f"_bg-{background_image.split('/')[-1].split('.')[0]}" if background_image else ""
    return f"{seed_suffix}{long_suffix}{background_suffix}"


def generate_output_paths(output_dir: str, prompt: str, checkpoint_name: str = None,
                         timestamp: str = None, seed: int = None, prompt_index: int = None) -> Dict[str, str]:
    """
    Generate output file paths using caption abbreviation for folder and file naming

    Args:
        output_dir: Base output directory
        prompt: Prompt text
        checkpoint_name: Checkpoint name
        timestamp: Timestamp (optional)
        seed: Random seed (optional)

    Returns:
        Dictionary containing various file paths
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    safe_folder_name = extract_caption_keywords(prompt)
    if prompt_index is not None:
        safe_folder_name = f"{prompt_index}_{safe_folder_name}"
    caption_dir = os.path.join(output_dir, safe_folder_name)
    os.makedirs(caption_dir, exist_ok=True)
    base_filename = safe_folder_name
    
    return {
        "video_path": os.path.join(caption_dir, f"{base_filename}.mp4"),
        "config_path": os.path.join(caption_dir, f"{base_filename}_config.json"),
        "log_path": os.path.join(caption_dir, f"{base_filename}_log.txt"),
        "folder_name": safe_folder_name,
        "caption_dir": caption_dir,
        "prefix": base_filename,
        "timestamp": timestamp
    }


def save_experiment_config(
    config_path: str,
    args: argparse.Namespace,
    config: Dict[str, Any],
    start_time: float,
    end_time: float,
    video_path: str,
    success: bool = True,
    error_message: str = None,
    prompt: Optional[str] = None
) -> None:
    """
    Save experiment configuration information

    Args:
        config_path: Config file save path
        args: Command line arguments
        config: Model configuration
        start_time: Start timestamp
        end_time: End timestamp
        video_path: Video file path
        success: Whether successful
        error_message: Error message (if any)
        prompt: Prompt text (optional)
    """
    
    
    if prompt is None:
        prompt = getattr(args, 'prompt', None)
    args_dict = dict(vars(args))
    args_dict["prompt"] = prompt
    # config_dict = OmegaConf.to_container(config, resolve=True)
    config_data = {
        "experiment_info": {
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "duration_seconds": round(end_time - start_time, 2),
            "success": success,
            "error_message": error_message,
            "output_video": os.path.basename(video_path)
        },
        "generation_parameters": {
            **args_dict,
            "config": config,
        },
        "model_configuration": {
            "config_path": args.config_path,
            "checkpoint_path": args.checkpoint_path,
            "model_name": config.get('model_name', 'unknown'),
            "generator_name": config.get('generator_name', config.get('model_name', 'unknown')),
            "num_frame_per_block": config.get('num_frame_per_block', 1),
            "warp_denoising_step": config.get('warp_denoising_step', False)
        },
        "system_parameters": {
            "device": args.device,
            "torch_dtype": str(args.torch_dtype),
            "log_level": args.log_level
        },
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Experiment config saved to: {config_path}")



def load_prompts_from_file(prompts_file: str) -> List[str]:
    """
    Load prompts list from file

    Args:
        prompts_file: Prompts file path

    Returns:
        List of prompts
    """
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError(f"No valid prompts found in file: {prompts_file}")

    logging.info(f"Loaded {len(prompts)} prompts from: {prompts_file}")
    return prompts


def validate_inputs(args) -> None:
    """Validate input parameters"""
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    if args.prompts_file and not os.path.exists(args.prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")

    if hasattr(args, 'background_image') and args.background_image and not os.path.exists(args.background_image):
        raise FileNotFoundError(f"Background image not found: {args.background_image}")

    if args.height <= 0 or args.width <= 0:
        raise ValueError("Height and width must be positive integers")

    if args.num_frames <= 0:
        raise ValueError("Number of frames must be positive")

    if hasattr(args, 'num_rollout') and args.num_rollout is not None:
        if args.num_rollout <= 0:
            raise ValueError("num_rollout must be positive")
        if not hasattr(args, 'num_overlap_frames') or args.num_overlap_frames is None:
            raise ValueError("num_overlap_frames is required when num_rollout is specified")
        if args.num_overlap_frames <= 0:
            raise ValueError("num_overlap_frames must be positive")


def load_checkpoint_and_initialize_pipeline(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16
) -> tuple:
    """
    Load checkpoint and initialize pipeline (executed once)

    Args:
        config_path: Config file path
        checkpoint_path: Checkpoint path
        device: Compute device
        torch_dtype: Data type

    Returns:
        (pipeline, config) tuple
    """
    torch.set_grad_enabled(False)

    logging.info(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    logging.info("Initializing inference pipeline...")
    pipeline = InferencePipeline(config, device=device)
    pipeline.to(device=device, dtype=torch_dtype)

    checkpoint_path = os.path.abspath(checkpoint_path)

    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)['generator']
    pipeline.generator.load_state_dict(state_dict, strict=True)
    logging.info("Checkpoint loaded successfully")

    return pipeline, config


def encode_vae(vae, videos: torch.Tensor) -> torch.Tensor:
    """VAE encoding function for long video generation"""
    device, dtype = videos[0].device, videos[0].dtype
    scale = [vae.mean.to(device=device, dtype=dtype),
             1.0 / vae.std.to(device=device, dtype=dtype)]
    output = [
        vae.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]
    output = torch.stack(output, dim=0)
    return output


def load_and_process_background_image(image_path: str, height: int = 480, width: int = 832, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Load and process background image, convert to format suitable for start_latents

    Args:
        image_path: Background image path
        height: Target height
        width: Target width
        device: Compute device
        dtype: Data type

    Returns:
        Processed image tensor, format [1, 3, H, W], value range [0, 1]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Background image not found: {image_path}")

    image = Image.open(image_path).convert('RGB')
    logging.info(f"Loaded background image: {image_path}, original size: {image.size}")

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    logging.info(f"Resized background image to: {width}x{height}")

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=dtype)

    logging.info(f"Background image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
    return image_tensor


def encode_background_to_latents(pipeline, background_image: torch.Tensor) -> torch.Tensor:
    """
    Encode background image to VAE latents format

    Args:
        pipeline: Inference pipeline
        background_image: Background image tensor [1, 3, H, W], value range [0, 1]

    Returns:
        Encoded latents tensor
    """
    background_image_scaled = background_image * 2.0 - 1.0
    background_image_scaled = background_image_scaled.unsqueeze(2)
    background_image_scaled = background_image_scaled.repeat(1, 1, 9, 1, 1)

    device, dtype = background_image.device, background_image.dtype
    scale = [pipeline.vae.mean.to(device=device, dtype=dtype),
             1.0 / pipeline.vae.std.to(device=device, dtype=dtype)]

    latents = pipeline.vae.model.encode(background_image_scaled, scale).permute(0, 2, 1, 3, 4)

    logging.info(f"Encoded background to latents shape: {latents.shape}, dtype: {latents.dtype}")
    return latents


def copy_background_image_to_output(background_image_path: str, output_dir: str) -> str:
    _, ext = os.path.splitext(background_image_path)
    output_path = os.path.join(output_dir, f"background_image{ext}")
    shutil.copy2(background_image_path, output_path)
    logging.info(f"Copied background image to: {output_path}")
    return output_path


def run_long_video_inference_with_pipeline(
    pipeline,
    prompt: str,
    num_rollout: int = 3,
    num_overlap_frames: int = 3,
    output_paths: Dict[str, str] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    fps: int = 15,
    background_image_path: Optional[str] = None,
    height: int = 480,
    width: int = 832
) -> bool:
    """
    Run long video autoregressive inference using pre-loaded pipeline

    Args:
        pipeline: Initialized InferencePipeline
        prompt: Text prompt
        num_rollout: Number of autoregressive rollouts
        num_overlap_frames: Number of overlap frames
        output_paths: Output path dictionary
        seed: Random seed
        device: Compute device
        torch_dtype: Data type
        fps: Frame rate
        background_image_path: Background image path (optional)
        height: Video height
        width: Video width

    Returns:
        Success status
    """
    output_path = output_paths["video_path"] if output_paths else "output_video.mp4"
    if background_image_path and output_paths:
        copy_background_image_to_output(background_image_path, output_paths["caption_dir"])

    status = True
    if os.path.exists(output_path):
        logging.info(f"Video already exists: {output_path}")
        status = "exist"
    else:
        if hasattr(pipeline, 'num_frame_per_block'):
            assert num_overlap_frames % pipeline.num_frame_per_block == 0, \
                "num_overlap_frames must be divisible by num_frame_per_block"

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            logging.info(f"Set random seed to: {seed}")

        logging.info("Starting long video autoregressive inference...")
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Rollout count: {num_rollout}")
        logging.info(f"Overlap frames: {num_overlap_frames}")

        start_latents = None
        if background_image_path:
            logging.info(f"Using background image for long video: {background_image_path}")
            background_image = load_and_process_background_image(
                background_image_path, height, width, device, torch_dtype
            )
            start_latents = encode_background_to_latents(pipeline, background_image)
            logging.info("Background image encoded to start_latents for long video generation")

        all_video = []

        for rollout_index in range(num_rollout):
            logging.info(f"Processing rollout {rollout_index + 1}/{num_rollout}")

            sampled_noise = torch.randn([1, 21, 16, 60, 104], device=device, dtype=torch_dtype)

            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=[prompt],
                return_latents=True,
                start_latents=start_latents
            )

            current_video = video[0].permute(0, 2, 3, 1).cpu().numpy()
            logging.info(f"Generated video segment shape: {current_video.shape}")

            if rollout_index < num_rollout - 1:
                start_frame = encode_vae(pipeline.vae, (
                    video[:, -4 * (num_overlap_frames - 1) - 1:-4 * (num_overlap_frames - 1), :] * 2.0 - 1.0
                ).transpose(2, 1).to(torch_dtype)).transpose(2, 1).to(torch_dtype)

                start_latents = torch.cat(
                    [start_frame, latents[:, -(num_overlap_frames - 1):]], dim=1
                )

                all_video.append(current_video[:-(4 * (num_overlap_frames - 1) + 1)])
            else:
                all_video.append(current_video)

        final_video = np.concatenate(all_video, axis=0)
        logging.info(f"Final concatenated video shape: {final_video.shape}")

        logging.info(f"Saving long video to: {output_path}")
        export_to_video(final_video, output_path, fps=fps)
        logging.info("Long video saved successfully")

    return status


def set_random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        logging.info(f"Set random seed to: {seed}")

def generate_noise(shape, device, dtype):
    return torch.randn(shape, device=device, dtype=dtype)

def run_inference_with_pipeline(
    pipeline,
    prompt: str,
    output_paths: Dict[str, str] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    fps: int = 15,
    num_rollout: Optional[int] = None,
    num_overlap_frames: Optional[int] = None,
    background_image_path: Optional[str] = None,
    height: int = 480,
    width: int = 832
) -> tuple:
    """
    Run inference using pre-loaded pipeline (supports both long video and normal inference)

    Args:
        pipeline: Initialized InferencePipeline
        prompt: Text prompt
        output_paths: Output path dictionary
        seed: Random seed
        device: Compute device
        torch_dtype: Data type
        fps: Frame rate
        num_rollout: Number of autoregressive rollouts (for long video)
        num_overlap_frames: Number of overlap frames (for long video)
        background_image_path: Background image path (optional)
        height: Video height
        width: Video width

    Returns:
        (success status, inference duration) tuple
    """
    output_path = output_paths["video_path"] if output_paths else "output_video.mp4"
    if background_image_path and output_paths:
        copy_background_image_to_output(background_image_path, output_paths["caption_dir"])
    if os.path.exists(output_path):
        logging.info(f"Video already exists: {output_path}")
        result, duration = "exist", 0
    else:
        set_random_seed(seed)
        if num_rollout and num_rollout > 1:
            result = run_long_video_inference_with_pipeline(
                pipeline, prompt, num_rollout, num_overlap_frames,
                output_paths, seed, device, torch_dtype, fps, background_image_path, height, width
            )
            duration = 0
        else:
            noise_shape = [1, 21, 16, 60, 104]
            sampled_noise = generate_noise(noise_shape, device, torch_dtype)
            logging.info(f"Generated noise tensor with shape: {sampled_noise.shape}")

            start_latents = None
            if background_image_path:
                logging.info(f"Using background image: {background_image_path}")
                background_image = load_and_process_background_image(
                    background_image_path, height, width, device, torch_dtype
                )
                start_latents = encode_background_to_latents(pipeline, background_image)
                logging.info("Background image encoded to start_latents")

            logging.info("Starting inference...")
            logging.info(f"Prompt: {prompt}")
            if start_latents is not None:
                logging.info("Using background image as starting point")
            start_time = time.time()
            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts=[prompt],
                start_latents=start_latents
            )[0].permute(0, 2, 3, 1).cpu().numpy()
            duration = time.time() - start_time
            logging.info(f"Inference completed. Generated video shape: {video.shape}")
            export_to_video(video, output_path, fps=fps)
            logging.info("Video saved successfully")
            result = True

    return result, duration



def run_batch_inference(
    prompts: List[str],
    args: argparse.Namespace,
    config: Dict[str, Any]
) -> None:
    """
    Run batch inference

    Args:
        prompts: List of prompts
        args: Command line arguments
        config: Model configuration
    """
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(args.checkpoint_path), f"{checkpoint_name}_inference")
    base_output_dir = os.path.abspath(base_output_dir)
    prompts_name = Path(args.prompts_file).stem
    suffix = build_inference_suffix(args.seed, args.num_rollout, args.num_overlap_frames, args.background_image)
    output_dir_parts = [base_output_dir, prompts_name]
    if suffix:
        output_dir_parts.append(suffix)
    output_dir = os.path.join(*output_dir_parts)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Starting batch inference with {len(prompts)} prompts")
    logging.info(f"Batch output directory: {output_dir}")
    if args.num_rollout:
        logging.info(f"Long video mode: {args.num_rollout} rollouts with {args.num_overlap_frames} overlap frames")

    success_count = 0
    failed_count = 0
    failed_prompts = []
    all_results = []

    torch_dtype = getattr(torch, args.torch_dtype)

    logging.info("Loading checkpoint and initializing pipeline...")
    pipeline, loaded_config = load_checkpoint_and_initialize_pipeline(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        torch_dtype=torch_dtype
    )
    logging.info("Checkpoint loaded successfully, starting batch inference...")
    if os.getenv("SLURM_JOB_ID"):
        link_slurm_logs(output_dir)
        setup_slurm_log_copy_on_exit(output_dir)
    for i, prompt in enumerate(prompts, start=1):
        logging.info(f"Processing prompt {i}/{len(prompts)}: {prompt}")

        output_paths = generate_output_paths(
            output_dir,
            prompt,
            checkpoint_name=checkpoint_name,
            seed=args.seed,
            prompt_index=i
        )

        start_time = time.time()

        success, duration = run_inference_with_pipeline(
            pipeline=pipeline,
            prompt=prompt,
            output_paths=output_paths,
            seed=args.seed,
            device=args.device,
            torch_dtype=torch_dtype,
            fps=args.fps,
            num_rollout=args.num_rollout,
            num_overlap_frames=args.num_overlap_frames,
            background_image_path=args.background_image,
            height=args.height,
            width=args.width
        )

        end_time = time.time()

        if success:
            success_count += 1
            duration = round(duration, 2) if not success == "exist" else "skipped (existed)"

            all_results.append({
                "index": i,
                "prompt": prompt,
                "success": True,
                "video_path": output_paths["video_path"],
                "folder_name": output_paths["folder_name"],
                "duration": duration,
                "seed": args.seed,
                "num_rollout": args.num_rollout,
                "num_overlap_frames": args.num_overlap_frames,
            })

            if duration != "skipped (existed)":
                save_experiment_config(
                    config_path=output_paths["config_path"],
                    args=args,
                    config=config,
                    start_time=start_time,
                    end_time=end_time,
                    video_path=output_paths["video_path"],
                    success=True,
                    error_message=None,
                    prompt=prompt
                )

            logging.info(f"✓ Video {i} generated successfully in {duration}s")
        else:
            failed_count += 1

        logging.info("---")

    summary_path = os.path.join(output_dir, "batch_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("CausVid Autoregressive Inference - Batch Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Config: {args.config_path}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Prompts File: {args.prompts_file}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"- Resolution: {args.width}x{args.height}\n")
        f.write(f"- Frames: {args.num_frames}\n")
        f.write(f"- FPS: {args.fps}\n")
        f.write(f"- Seed: {args.seed}\n")
        f.write(f"- Data Type: {args.torch_dtype}\n")
        f.write(f"- Device: {args.device}\n")
        if args.num_rollout:
            f.write(f"- Long Video: {args.num_rollout} rollouts, {args.num_overlap_frames} overlap frames\n")
        if args.background_image:
            f.write(f"- Background Image: {args.background_image}\n")
        f.write("\n")
        
        f.write("Results:\n")
        f.write(f"- Total prompts: {len(prompts)}\n")
        f.write(f"- Successful: {success_count}\n")
        f.write(f"- Failed: {failed_count}\n")
        f.write(f"- Success rate: {success_count / len(prompts) * 100:.1f}%\n\n")
        
        f.write("Prompts and Results:\n")
        for result in all_results:
            status = "✓" if result["success"] else "✗"
            f.write(f"{result['index']}. {status} {result['prompt']}\n")
            if result["success"]:
                f.write(f"   → Video: {os.path.basename(result['video_path'])}\n")
                f.write(f"   → Folder: {result['folder_name']}\n")
                f.write(f"   → Duration: {result['duration']}\n")
            else:
                f.write(f"   → Error: {result['error']}\n")
            if result.get("seed") is not None:
                f.write(f"   → Seed: {result['seed']}\n")
            if result.get("num_rollout"):
                f.write(f"   → Long Video: {result['num_rollout']} rollouts, {result['num_overlap_frames']} overlap\n")
            f.write("\n")
        
        if failed_prompts:
            f.write("Failed Prompts Details:\n")
            for idx, prompt, error in failed_prompts:
                f.write(f"{idx}. {prompt}\n")
                f.write(f"   Error: {error}\n\n")

    logging.info("Batch inference completed!")
    logging.info(f"Generated videos are in: {output_dir}")
    logging.info(f"Total: {len(prompts)}, Success: {success_count}, Failed: {failed_count}")
    logging.info(f"Success rate: {success_count / len(prompts) * 100:.1f}%")
    logging.info(f"Batch summary saved: {summary_path}")


def run_single_inference(args: argparse.Namespace, config_dict: Dict[str, Any]) -> None:
    prompt = args.prompt
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    output_dir = args.output_dir if args.output_dir else args.output_folder
    output_dir = os.path.abspath(output_dir)
    suffix = build_inference_suffix(args.seed, args.num_rollout, args.num_overlap_frames, args.background_image)
    if suffix:
        output_dir = os.path.join(output_dir, suffix)
    os.makedirs(output_dir, exist_ok=True)

    torch_dtype = getattr(torch, args.torch_dtype)
    output_paths = generate_output_paths(
        output_dir=output_dir,
        prompt=prompt,
        checkpoint_name=checkpoint_name,
        seed=args.seed,
    )

    if os.getenv("SLURM_JOB_ID"):
        link_slurm_logs(output_paths["caption_dir"])
        setup_slurm_log_copy_on_exit(output_paths["caption_dir"])

    pipeline, _ = load_checkpoint_and_initialize_pipeline(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        torch_dtype=torch_dtype
    )

    start_time = time.time()
    success, duration = run_inference_with_pipeline(
        pipeline=pipeline,
        prompt=prompt,
        output_paths=output_paths,
        seed=args.seed,
        device=args.device,
        torch_dtype=torch_dtype,
        fps=args.fps,
        num_rollout=args.num_rollout,
        num_overlap_frames=args.num_overlap_frames,
        background_image_path=args.background_image,
        height=args.height,
        width=args.width
    )
    end_time = time.time()

    if success:
        if success != "exist":
            save_experiment_config(
                config_path=output_paths["config_path"],
                args=args,
                config=config_dict,
                start_time=start_time,
                end_time=end_time,
                video_path=output_paths["video_path"],
                success=True,
                error_message=None,
                prompt=prompt
            )
            logging.info(f"Single inference completed in {duration:.2f}s")
        else:
            logging.info("Single inference skipped because output video already exists")
        logging.info(f"Video path: {output_paths['video_path']}")
    else:
        logging.error("Single inference failed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Comprehensive CausVid Autoregressive Inference Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


    parser.add_argument(
        "--config_path", 
        type=str, 
        default="configs/wan_causal_ode_finetune.yaml",
        help="Path to CausVid config file"
    )
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="../pretrained/VideoSketcher/VideoSketcher-models/VideoSketcher_AR/AR_1.3B.pt",
        help="Path to checkpoint .pt file"
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt", 
        type=str,
        help="Single text prompt for video generation"
    )
    prompt_group.add_argument(
        "--prompts_file", 
        type=str,
        help="Path to text file containing prompts (one per line) for batch inference"
    )

    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument(
        "--height", 
        type=int, 
        default=480,
        help="Video height (note: actual output resolution depends on model)"
    )
    gen_group.add_argument(
        "--width", 
        type=int, 
        default=832,
        help="Video width (note: actual output resolution depends on model)"
    )
    gen_group.add_argument(
        "--num_frames", 
        type=int, 
        default=81,
        help="Number of frames to generate (note: actual frames depend on model)"
    )
    gen_group.add_argument(
        "--seed", 
        type=int,
        help="Random seed for reproducible generation"
    )
    gen_group.add_argument(
        "--fps", 
        type=int, 
        default=15,
        help="Frames per second for output video"
    )

    long_group = parser.add_argument_group("Long Video Parameters")
    long_group.add_argument(
        "--num_rollout",
        default=1,
        type=int,
        help="Number of autoregressive rollouts for long video generation (default: disabled)"
    )
    long_group.add_argument(
        "--num_overlap_frames",
        type=int,
        default=3,
        help="Number of overlap frames between rollouts (required when num_rollout is specified)"
    )

    output_group = parser.add_argument_group("Output Parameters")
    output_group.add_argument(
        "--output_folder", 
        type=str, 
        default="experiments/causvid_ar_inference",
        help="Output directory path"
    )
    output_group.add_argument(
        "--output_dir",
        type=str,
        help="Custom base directory for inference outputs"
    )
    output_group.add_argument(
        "--use_batch_output", 
        action="store_true",
        help="Use batch output directory structure (for script usage)"
    )

    sys_group = parser.add_argument_group("System Parameters")
    sys_group.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run inference on"
    )
    sys_group.add_argument(
        "--torch_dtype", 
        type=str, 
        choices=["float16", "bfloat16", "float32"], 
        default="bfloat16",
        help="PyTorch data type"
    )

    parser.add_argument(
        "--log_level", 
        type=str, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level"
    )

    background_group = parser.add_argument_group("Background Image Parameters")
    background_group.add_argument(
        "--background_image",
        type=str,
        help="Path to background image to start generation from (e.g., blank canvas)"
    )

    parser.add_argument("--extend_prompt", action="store_true", help="Refine prompt via LLM before inference")
    parser.add_argument("--refine_backend", type=str, choices=["openai", "qwen"], default="qwen")

    args = parser.parse_args()

    return args


def log_inference_summary(args, config_dict, title, prompt_info, output_dir=None):
    logging.info("=" * 60)
    logging.info(title)
    logging.info("=" * 60)
    logging.info(f"Config: {args.config_path}")
    logging.info(f"Checkpoint: {args.checkpoint_path}")
    logging.info(f"Model: {config_dict.get('model_name', 'unknown')}")
    logging.info(f"Generator: {config_dict.get('generator_name', config_dict.get('model_name', 'unknown'))}")
    logging.info(prompt_info)
    logging.info(f"Seed: {args.seed}")
    logging.info(f"Device: {args.device}, dtype: {args.torch_dtype}")
    if args.num_rollout:
        logging.info(f"Long Video: {args.num_rollout} rollouts with {args.num_overlap_frames} overlap frames")
    if args.background_image:
        logging.info(f"Background Image: {args.background_image}")
    if output_dir is not None:
        logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 60)


def main():
    """Main function"""
    args = parse_args()

    setup_logging(args.log_level)
    validate_inputs(args)

    config = OmegaConf.load(args.config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)

    if args.extend_prompt:
        refiner = PromptRefiner(backend=args.refine_backend)
        refiner.warmup()

    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        if args.extend_prompt:
            prompts = [refiner.refine(p) for p in prompts]
            logging.info(f"Refined {len(prompts)} prompts: {prompts}")
        log_inference_summary(
            args=args,
            config_dict=config_dict,
            title="Autoregressive Batch Inference",
            prompt_info=f"Prompts: {len(prompts)} from {args.prompts_file}"
        )
        run_batch_inference(prompts, args, config_dict)

    else:
        prompt = args.prompt
        if args.extend_prompt:
            prompt = refiner.refine(prompt)
            logging.info(f"Refined prompt: {prompt}")
            args.prompt = prompt
        log_inference_summary(
            args=args,
            config_dict=config_dict,
            title="Autoregressive Single Inference",
            prompt_info=f"Prompt: {prompt}"
        )
        run_single_inference(args, config_dict)


if __name__ == "__main__":
    main()