import argparse
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List

import torch


def setup_logging(log_level: str) -> None:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    if log_level not in level_map:
        raise ValueError(f"Unsupported log level: {log_level}")
    logging.basicConfig(
        level=level_map[log_level],
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sanitize_filename(text: str, max_length: int = 60) -> str:
    safe_text = re.sub(r'[<>:"/\\|?*]', "_", text)
    safe_text = re.sub(r"\s+", "_", safe_text)
    safe_text = re.sub(r"_+", "_", safe_text)
    safe_text = safe_text.strip("_")
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length].rstrip("_")
    if not safe_text:
        safe_text = "untitled"
    return safe_text


def extract_caption_keywords(prompt: str, max_words: int = 20) -> str:
    stop_words = {
        "a", "an", "the", "of", "step", "by", "draw", "sketch", "process",
        "first", "then", "next", "finally", "and", "or", "for", "with", "in",
        "on", "after", "that", "one", "onebyone", "canvas",
    }
    words = re.findall(r"\b\w+\b", prompt.lower())
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    keywords = filtered_words[:max_words]
    if not keywords:
        return sanitize_filename(prompt, max_length=60)
    caption = "_".join(keywords)
    return sanitize_filename(caption, max_length=60)


def generate_output_paths(
    output_dir: str,
    prompt: str,
    seed: int = None,
    prompt_index: int = None,
) -> Dict[str, str]:
    safe_folder_name = extract_caption_keywords(prompt)
    if prompt_index is not None:
        safe_folder_name = f"{prompt_index}_{safe_folder_name}"
    caption_dir = os.path.join(output_dir, safe_folder_name)
    exist = os.path.exists(caption_dir)
    os.makedirs(caption_dir, exist_ok=True)
    file_prefix = safe_folder_name
    return {
        "video_path": os.path.join(caption_dir, f"{file_prefix}.mp4"),
        "config_path": os.path.join(caption_dir, f"{file_prefix}_config.json"),
        "log_path": os.path.join(caption_dir, f"{file_prefix}_log.txt"),
        "folder_name": safe_folder_name,
        "caption_dir": caption_dir,
        "prefix": file_prefix,
        "exist": exist,
        "seed": seed,
    }


def save_run_config(
    config_path: str,
    args: argparse.Namespace,
    model_configs: List[dict],
    start_time: float,
    end_time: float,
    video_path: str,
    prompt: str,
    prompt_index: int = None,
    success: bool = True,
    error_message: str = None,
) -> None:
    args_dict = dict(vars(args))
    args_dict["prompt"] = prompt
    config_data = {
        "experiment_info": {
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "duration_seconds": round(end_time - start_time, 2),
            "success": success,
            "error_message": error_message,
            "output_video": os.path.basename(video_path),
        },
        "generation_parameters": args_dict,
        "batch_inference_info": {
            "prompts_file": args.prompts_file,
            "prompt_index": prompt_index,
            "is_batch_inference": args.prompts_file is not None,
        },
        "model_configuration": {
            "checkpoint_path": args.checkpoint_path,
            "model_id_with_origin_paths": args.model_id_with_origin_paths,
            "model_config_json": args.model_config_json,
            "model_type": args.model_type,
            "remove_prefix_in_ckpt": args.remove_prefix_in_ckpt,
            "model_configs": model_configs,
        },
        "system_parameters": {
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "enable_vram_management": not args.no_vram_management,
            "tiled": not args.no_tiled,
            "tile_size": args.tile_size,
            "tile_stride": args.tile_stride,
            "log_level": args.log_level,
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

