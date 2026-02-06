import logging
import os
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class PrecomputedWanLatentDataset(Dataset):
    def __init__(self, samples, repeat=1):
        if len(samples) == 0:
            raise ValueError("No samples available for precomputed dataset.")
        self.samples = samples
        self.repeat = max(1, int(repeat))

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        base_sample = self.samples[index % len(self.samples)]
        return {key: base_sample[key] for key in base_sample}


def precompute_wan_latents(dataset, model, args, accelerator=None):
    logging.info("Precomputing WAN latents before training.")
    extra_inputs = []
    if args.extra_inputs:
        extra_inputs = [item for item in args.extra_inputs.split(",") if item]
    dataset_args = {
        "dataset_base_path": args.dataset_base_path,
        "dataset_metadata_path": args.dataset_metadata_path,
        "heatmap_base_path": args.heatmap_base_path,
        "max_pixels": args.max_pixels,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "data_file_keys": args.data_file_keys,
        # "dataset_repeat": args.dataset_repeat,
        "name_map_path": args.name_map_path,
        "extra_inputs": args.extra_inputs,
    }
    dataset_repeat = args.dataset_repeat
    if args.dataset_metadata_path:
        metadata_name = os.path.splitext(os.path.basename(args.dataset_metadata_path))[0]
    else:
        metadata_name = os.path.basename(args.dataset_base_path.rstrip("/"))
    cache_dir = args.dataset_base_path if os.path.isdir(args.dataset_base_path) else os.path.dirname(args.dataset_base_path)
    cache_dir = cache_dir if cache_dir else "."
    cache_save_path = os.path.join(cache_dir, f"wan_latent_cache_{metadata_name}.pt")
    if os.path.exists(cache_save_path):
        dataset_cache = torch.load(cache_save_path, map_location="cpu", weights_only=False)

        args_match = True
        for key, value in dataset_args.items():
            if key not in dataset_cache["dataset_args"] or dataset_cache["dataset_args"][key] != value:
                args_match = False
                break
        if args_match:
            logging.info("Loaded WAN latent cache from %s", cache_save_path)
            cached_samples = dataset_cache["cached_samples"]
            if extra_inputs:
                sample = cached_samples[0]
                required_extras = {"input_image", "end_image", "reference_image", "vace_reference_image"}
                expected_extras = [item for item in extra_inputs if item in required_extras]
                missing_extras = [item for item in expected_extras if item not in sample]
                if missing_extras:
                    logging.info("WAN latent cache missing extra inputs %s. Recomputing latents.", missing_extras)
                else:
                    return PrecomputedWanLatentDataset(cached_samples, repeat=dataset_repeat)
            else:
                return PrecomputedWanLatentDataset(cached_samples, repeat=dataset_repeat)
        logging.info("WAN latent cache mismatch. Recomputing latents.")
    if torch.cuda.is_available():
        target_device = torch.device("cuda")
    else:
        target_device = model.pipe.device
    if accelerator is None:
        accelerator = Accelerator()
    process_index = accelerator.process_index
    num_processes = accelerator.num_processes
    is_main_process = accelerator.is_main_process
    progress_disabled = not is_main_process
    sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.dataset_num_workers,
        shuffle=False,
        collate_fn=lambda x: x[0],
    )
    cached_samples = []
    was_training = model.training
    pipe = model.pipe
    original_device = pipe.device
    original_dtype = pipe.torch_dtype
    vae = pipe.vae
    vae_params = list(vae.parameters())
    if vae_params:
        vae_original_device = vae_params[0].device
        vae_original_dtype = vae_params[0].dtype
    else:
        vae_original_device = original_device
        vae_original_dtype = original_dtype
    pipe.load_models_to_device(("vae",))
    if torch.cuda.is_available():
        vae.to(device=target_device, dtype=pipe.torch_dtype)
    vae_was_training = vae.training
    vae.eval()
    print(f"num_processes: {num_processes}, process_index: {process_index}")
    with torch.no_grad():
        for index, data in tqdm(enumerate(dataloader), desc="Encoding latents", leave=False, disable=progress_disabled):
            if num_processes > 1 and index % num_processes != process_index:
                continue
            if data is None or "video" not in data:
                continue
            video = data["video"]
            height = video[0].size[1]
            width = video[0].size[0]
            num_frames = len(video)
            preprocess_device = target_device if torch.cuda.is_available() else original_device
            video_tensor = pipe.preprocess_video(video, device=preprocess_device)
            latents = vae.encode(video_tensor, device=preprocess_device).to(dtype=pipe.torch_dtype, device=preprocess_device)
            cached_sample = {
                "precompute_latents": latents.detach().cpu(),
                "prompt": data["prompt"],
                "sample_id" : data["sample_id"],
            }
            if extra_inputs:
                for extra_input in extra_inputs:
                    if extra_input == "input_image":
                        if "input_image" in data:
                            cached_sample["input_image"] = data["input_image"]
                        elif video is not None:
                            cached_sample["input_image"] = video[0]
                    elif extra_input == "end_image":
                        if "end_image" in data:
                            cached_sample["end_image"] = data["end_image"]
                        elif video is not None:
                            cached_sample["end_image"] = video[-1]
                    elif extra_input in ("reference_image", "vace_reference_image"):
                        cached_sample[extra_input] = data[extra_input]
            # cached_sample.update(data)
            cached_samples.append(cached_sample)
    gathered_by_accelerator = False
    if num_processes > 1:
        os.makedirs(cache_dir, exist_ok=True)
        shard_path = os.path.join(cache_dir, f"wan_latent_cache_{metadata_name}.shard.{process_index}.pt")
        torch.save(cached_samples, shard_path)
        accelerator.wait_for_everyone()
        if is_main_process:
            merged_samples = []
            for rank in range(num_processes):
                rank_path = os.path.join(cache_dir, f"wan_latent_cache_{metadata_name}.shard.{rank}.pt")
                merged_samples.extend(torch.load(rank_path, map_location="cpu", weights_only=False))
            cached_samples = merged_samples
            for rank in range(num_processes):
                rank_path = os.path.join(cache_dir, f"wan_latent_cache_{metadata_name}.shard.{rank}.pt")
                os.remove(rank_path)
        accelerator.wait_for_everyone()
    print(f"Got {len(cached_samples)} cached samples after latent precomputation.")
    if vae_was_training:
        vae.train()
    else:
        vae.eval()
    if torch.cuda.is_available():
        vae.to(device=vae_original_device, dtype=vae_original_dtype)
    pipe.device = original_device
    pipe.torch_dtype = original_dtype
    if was_training:
        model.train()
    else:
        model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Latent precomputation finished. Cached %d unique samples.", len(cached_samples))
    dataset_cache = {
        "dataset_args": dataset_args,
        "cached_samples": cached_samples,
    }
    accelerator.wait_for_everyone()
    if is_main_process:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(dataset_cache, cache_save_path)
        logging.info("Saved WAN latent cache to %s", cache_save_path)
    accelerator.wait_for_everyone()
    if num_processes > 1 and not is_main_process and not gathered_by_accelerator:
        dataset_cache = torch.load(cache_save_path, map_location="cpu", weights_only=False)
        cached_samples = dataset_cache["cached_samples"]
    torch.cuda.empty_cache()
    return PrecomputedWanLatentDataset(cached_samples, repeat=dataset_repeat)

