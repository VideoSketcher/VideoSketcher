import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from pipeline.wan_video import WanVideoPipeline
from utils.io_utils import save_video
from pipeline.models.utils import load_state_dict
from utils.pipeline_config import ModelConfig
from utils.trainer_utils import DiffusionTrainingModule
from utils import setup_logging, generate_output_paths, save_run_config
from utils.prompt_refiner import PromptRefiner

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_model_id_with_origin_paths(model_id_with_origin_paths: str) -> List[ModelConfig]:
    model_configs = []
    entries = model_id_with_origin_paths.split(",")
    for entry in entries:
        if ":" in entry:
            model_id, origin_file_pattern = entry.split(":", 1)
            model_configs.append(
                ModelConfig(
                    model_id=model_id,
                    origin_file_pattern=origin_file_pattern,
                    offload_device="cpu",
                )
            )
        else:
            model_configs.append(
                ModelConfig(
                    model_id=entry,
                    origin_file_pattern="*.safetensors",
                    offload_device="cpu",
                )
            )
    return model_configs


def load_model_configs_from_json(config_path: str) -> List[ModelConfig]:
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    model_configs = []
    for config in configs:
        offload_device = "cpu"
        if "offload_device" in config:
            offload_device = config["offload_device"]
        offload_dtype = None
        if "offload_dtype" in config and config["offload_dtype"] is not None:
            offload_dtype = TORCH_DTYPE_MAP[config["offload_dtype"]]
        model_configs.append(
            ModelConfig(
                model_id=config["model_id"],
                origin_file_pattern=config["origin_file_pattern"],
                path=config["path"] if "path" in config else None,
                offload_device=offload_device,
                offload_dtype=offload_dtype,
            )
        )
    return model_configs


def get_default_model_configs(model_type: str) -> List[ModelConfig]:
    if model_type == "1.3B":
        return parse_model_id_with_origin_paths(
            "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,"
            "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,"
            "Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth"
        )
    if model_type == "14B":
        return parse_model_id_with_origin_paths(
            "Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,"
            "Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,"
            "Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth"
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str not in TORCH_DTYPE_MAP:
        raise ValueError(f"Unsupported torch dtype: {dtype_str}")
    return TORCH_DTYPE_MAP[dtype_str]


def load_image_if_provided(image_path: Optional[str]) -> Optional[Image.Image]:
    if image_path is None:
        return None
    image = Image.open(image_path).convert("RGB")
    logging.info("Loaded image: %s (size: %s)", image_path, image.size)
    return image


def load_prompts_from_file(prompts_file: str) -> List[str]:
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No valid prompts found in file: {prompts_file}")
    logging.info("Loaded %d prompts from: %s", len(prompts), prompts_file)
    return prompts


def _inject_lora(
    model,
    lora_ckpt: str,
    target_modules: List[str],
    lora_rank: int,
) -> Any:
    model = DiffusionTrainingModule().add_lora_to_model(
        model,
        target_modules=target_modules,
        lora_rank=lora_rank,
    )
    lora_state = load_state_dict(lora_ckpt)
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    return model


def load_model_and_checkpoint(
    base_model_configs: List[ModelConfig],
    checkpoint_path: Optional[str],
    device: str,
    torch_dtype: torch.dtype,
    enable_vram_management: bool,
    remove_prefix_in_ckpt: str,
    lora_ckpt: Optional[str],
    lora_ckpt_high: Optional[str],
    lora_ckpt_low: Optional[str],
    lora_base_model: Optional[str],
    lora_target_modules: str,
    lora_rank: int,
) -> WanVideoPipeline:
    logging.info("Loading base models...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=base_model_configs,
    )
    logging.info("Base models loaded successfully")

    if checkpoint_path:
        logging.info("Loading trained checkpoint from: %s", checkpoint_path)
        state_dict = load_state_dict(checkpoint_path)
        if remove_prefix_in_ckpt:
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith(remove_prefix_in_ckpt):
                    new_state_dict[key[len(remove_prefix_in_ckpt):]] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        pipe.dit.load_state_dict(state_dict)
        logging.info("Checkpoint loaded into DiT model successfully")

    is_dual_model = hasattr(pipe, "dit2") and pipe.dit2 is not None
    target_modules = lora_target_modules.split(",")

    if lora_ckpt_high and lora_ckpt_low:
        if not is_dual_model:
            raise ValueError("Dual LoRA requires a dual-model pipeline")
        logging.info("Loading dual LoRA for Wan2.2 model")
        pipe.dit = _inject_lora(pipe.dit, lora_ckpt_high, target_modules, lora_rank)
        pipe.dit2 = _inject_lora(pipe.dit2, lora_ckpt_low, target_modules, lora_rank)
    elif lora_ckpt:
        if is_dual_model and lora_base_model is None:
            lora_base_model = "dit"
        if lora_base_model == "dit":
            pipe.dit = _inject_lora(pipe.dit, lora_ckpt, target_modules, lora_rank)
        elif lora_base_model == "dit2":
            if not is_dual_model:
                raise ValueError("lora_base_model=dit2 requires a dual-model pipeline")
            pipe.dit2 = _inject_lora(pipe.dit2, lora_ckpt, target_modules, lora_rank)
        else:
            raise ValueError("lora_base_model must be 'dit' or 'dit2'")

    if enable_vram_management:
        pipe.enable_vram_management()
        logging.info("VRAM management enabled")
    return pipe


def run_inference_with_pipeline(
    pipe: WanVideoPipeline,
    prompt: str,
    negative_prompt: str,
    input_image: Optional[Image.Image],
    end_image: Optional[Image.Image],
    height: int,
    width: int,
    num_frames: int,
    seed: Optional[int],
    cfg_scale: float,
    num_inference_steps: int,
    output_paths: Dict[str, str],
    tiled: bool,
    tile_size: tuple,
    tile_stride: tuple,
) -> None:
    output_path = output_paths["video_path"]
    logging.info("Starting inference...")
    logging.info("Prompt: %s", prompt)
    logging.info("Resolution: %dx%d, Frames: %d", width, height, num_frames)
    logging.info("CFG Scale: %s, Steps: %s", cfg_scale, num_inference_steps)

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        input_image=input_image,
        end_image=end_image,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
        cfg_merge=cfg_scale != 1.0,
    )
    save_video(video, output_path, fps=16, quality=5)
    logging.info("Video saved to: %s", output_path)


def build_batch_output_dir(args: argparse.Namespace) -> str:
    checkpoint_name = ""
    if args.checkpoint_path is not None:
        checkpoint_name = os.path.basename(args.checkpoint_path)
    elif args.lora_ckpt_high is not None:
        checkpoint_name = os.path.basename(args.lora_ckpt_high)
    elif args.lora_ckpt is not None:
        checkpoint_name = os.path.basename(args.lora_ckpt)
    if checkpoint_name.endswith(".safetensors"):
        checkpoint_name = checkpoint_name[:-12]
    elif checkpoint_name.endswith(".pth"):
        checkpoint_name = checkpoint_name[:-4]
    prompt_name = os.path.basename(args.prompts_file).split(".")[0]
    input_image_name = os.path.basename(args.input_image).split(".")[0] if args.input_image else ""
    seed_name = f"seed{args.seed}"
    base_tag = "ori_model_inference" if checkpoint_name == "" else f"{checkpoint_name}_inference"
    batch_output_dir = os.path.join(
        args.output_dir,
        base_tag,
        prompt_name,
        input_image_name,
        f"{seed_name}_step{args.num_inference_steps}_cfg{args.cfg_scale}",
    )
    if args.output_dir_suffix:
        batch_output_dir = f"{batch_output_dir}_{args.output_dir_suffix}"
    os.makedirs(batch_output_dir, exist_ok=True)
    return batch_output_dir


def run_batch_inference(
    prompts: List[str],
    args: argparse.Namespace,
    model_configs: List[ModelConfig],
) -> None:
    batch_output_dir = build_batch_output_dir(args)
    prompt_str = "\n".join(prompts)
    logging.info(f"Inferencing for prompts: {prompt_str}")
    logging.info(f"Output directory: {batch_output_dir}")

    input_image = load_image_if_provided(args.input_image)
    end_image = load_image_if_provided(args.end_image)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)

    logging.info("Loading model and checkpoint...")
    model_load_start = time.time()
    pipe = load_model_and_checkpoint(
        checkpoint_path=args.checkpoint_path,
        base_model_configs=model_configs,
        device=args.device,
        torch_dtype=torch_dtype,
        enable_vram_management=not args.no_vram_management,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        lora_ckpt=args.lora_ckpt,
        lora_ckpt_high=args.lora_ckpt_high,
        lora_ckpt_low=args.lora_ckpt_low,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
    )
    model_load_time = time.time() - model_load_start
    logging.info("Model loaded in %.2fs", model_load_time)

    success_count = 0
    failed_count = 0
    results = []

    for i, prompt in enumerate(prompts):
        output_paths = generate_output_paths(batch_output_dir, prompt, seed=args.seed, prompt_index=i + 1)
        if not args.force and output_paths["exist"]:
            continue
        start_time = time.time()
        run_inference_with_pipeline(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            input_image=input_image,
            end_image=end_image,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            output_paths=output_paths,
            tiled=not args.no_tiled,
            tile_size=tuple(args.tile_size),
            tile_stride=tuple(args.tile_stride),
        )
        end_time = time.time()
        success_count += 1
        duration = round(end_time - start_time, 2)
        results.append(
            {
                "index": i + 1,
                "prompt": prompt,
                "video_path": output_paths["video_path"],
                "folder_name": output_paths["folder_name"],
                "duration": duration,
                "seed": args.seed,
            }
        )
        save_run_config(
            config_path=output_paths["config_path"],
            args=args,
            model_configs=[
                {
                    "model_id": config.model_id,
                    "origin_file_pattern": config.origin_file_pattern,
                    "path": config.path,
                }
                for config in model_configs
            ],
            start_time=start_time,
            end_time=end_time,
            video_path=output_paths["video_path"],
            prompt=prompt,
            prompt_index=i + 1,
        )
        logging.info("Video %d generated in %.2fs", i + 1, duration)

    summary_path = os.path.join(batch_output_dir, "batch_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_prompts": len(prompts),
        "successful": success_count,
        "failed": failed_count,
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info("Batch summary saved: %s", summary_path)


def run_single_inference(
    prompt: str,
    args: argparse.Namespace,
    model_configs: List[ModelConfig],
) -> None:
    output_paths = generate_output_paths(args.output_dir, prompt, seed=args.seed)
    logging.info(f"Inferencing for prompt: {prompt}")
    logging.info(f"Output directory: {output_paths['folder_name']}")
    input_image = load_image_if_provided(args.input_image)
    end_image = load_image_if_provided(args.end_image)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    pipe = load_model_and_checkpoint(
        checkpoint_path=args.checkpoint_path,
        base_model_configs=model_configs,
        device=args.device,
        torch_dtype=torch_dtype,
        enable_vram_management=not args.no_vram_management,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        lora_ckpt=args.lora_ckpt,
        lora_ckpt_high=args.lora_ckpt_high,
        lora_ckpt_low=args.lora_ckpt_low,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
    )
    start_time = time.time()
    run_inference_with_pipeline(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        end_image=end_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        output_paths=output_paths,
        tiled=not args.no_tiled,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
    )
    end_time = time.time()
    save_run_config(
        config_path=output_paths["config_path"],
        args=args,
        model_configs=[
            {
                "model_id": config.model_id,
                "origin_file_pattern": config.origin_file_pattern,
                "path": config.path,
            }
            for config in model_configs
        ],
        start_time=start_time,
        end_time=end_time,
        video_path=output_paths["video_path"],
        prompt=prompt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WAN video inference (public release)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Text prompt")
    prompt_group.add_argument("--prompts_file", type=str, help="Text file with prompts")

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_id_with_origin_paths", type=str, default="Wan-AI/Wan2.1-T2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-14B:Wan2.1_VAE.pth")
    model_group.add_argument("--model_config_json", type=str)
    model_group.add_argument(
        "--model_type",
        type=str,
        choices=["1.3B", "14B", "Wan2.2-T2V-A14B"],
        default="14B",
    )

    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_dir_suffix", type=str, default="")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.")

    lora_group = parser.add_argument_group("LoRA Parameters")
    lora_group.add_argument("--lora_ckpt", type=str, default=None)
    lora_group.add_argument("--lora_ckpt_high", type=str, default=None)
    lora_group.add_argument("--lora_ckpt_low", type=str, default=None)
    lora_group.add_argument("--lora_base_model", type=str, default="dit")
    lora_group.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    lora_group.add_argument("--lora_rank", type=int, default=32)

    input_group = parser.add_argument_group("Input Parameters")
    input_group.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    input_group.add_argument("--input_image", type=str)
    input_group.add_argument("--end_image", type=str)

    gen_group = parser.add_argument_group("Generation Parameters")
    gen_group.add_argument("--height", type=int, default=480)
    gen_group.add_argument("--width", type=int, default=832)
    gen_group.add_argument("--num_frames", type=int, default=81)
    gen_group.add_argument("--seed", type=int)
    gen_group.add_argument("--cfg_scale", type=float, default=5.0)
    gen_group.add_argument("--num_inference_steps", type=int, default=50)

    sys_group = parser.add_argument_group("System Parameters")
    sys_group.add_argument("--device", type=str, default="cuda")
    sys_group.add_argument("--torch_dtype", type=str, choices=list(TORCH_DTYPE_MAP.keys()), default="bfloat16")
    sys_group.add_argument("--no_vram_management", action="store_true")
    sys_group.add_argument("--no_tiled", action="store_true")
    sys_group.add_argument("--tile_size", type=int, nargs=2, default=[30, 52])
    sys_group.add_argument("--tile_stride", type=int, nargs=2, default=[15, 26])

    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--extend_prompt", action="store_true", help="Refine prompt via LLM before inference")
    parser.add_argument("--refine_backend", type=str, choices=["openai", "qwen"], default="qwen")

    args = parser.parse_args()
    if args.output_dir is None:
        if args.lora_ckpt:
            args.output_dir = os.path.dirname(args.lora_ckpt)
        elif args.lora_ckpt_high:
            args.output_dir = os.path.dirname(args.lora_ckpt_high)
        elif args.checkpoint_path:
            args.output_dir = os.path.dirname(args.checkpoint_path)
        else:
            raise ValueError("Please provide --output_dir")
    return args


_I2V_BRUSH_INSERT = ", using the color and style of the brush shown in the top-left corner,"


def _postprocess_refined_prompt(prompt: str, args: argparse.Namespace) -> str:
    """For I2V (input_image provided), insert brush style clause before 'following this drawing order'."""
    if args.input_image and _I2V_BRUSH_INSERT not in prompt:
        prompt = prompt.replace(", following this drawing order", f"{_I2V_BRUSH_INSERT} following this drawing order")
    return prompt


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    if args.lora_ckpt_high is not None and args.lora_ckpt_low is not None:
        if os.path.basename(args.lora_ckpt_high) != os.path.basename(args.lora_ckpt_low):
            raise ValueError("lora_ckpt_high and lora_ckpt_low must share the same filename")

    if args.model_id_with_origin_paths:
        model_configs = parse_model_id_with_origin_paths(args.model_id_with_origin_paths)
    elif args.model_config_json:
        model_configs = load_model_configs_from_json(args.model_config_json)
    else:
        model_configs = get_default_model_configs(args.model_type)

    if args.extend_prompt:
        refiner = PromptRefiner(backend=args.refine_backend)
        refiner.warmup()

    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        if args.extend_prompt:
            prompts = [_postprocess_refined_prompt(refiner.refine(p), args) for p in prompts]
            logging.info(f"Refined {len(prompts)} prompts: {prompts}")
        run_batch_inference(prompts, args, model_configs)
    else:
        prompt = args.prompt
        if args.extend_prompt:
            prompt = _postprocess_refined_prompt(refiner.refine(prompt), args)
            logging.info(f"Refined prompt: {prompt}")
        run_single_inference(prompt, args, model_configs)


if __name__ == "__main__":
    main()

