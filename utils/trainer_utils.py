from typing import Any
import imageio, os, sys, torch, warnings, torchvision, argparse, json, signal
import torch.nn.functional as F
from utils.pipeline_config import ModelConfig
from pipeline.models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import logging
import re
from safetensors.torch import load_file
import yaml
import glob
import numpy as np

class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        heatmap_base_path=None,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        name_map_path=None,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames if args.num_frames >= 0 else None
            data_file_keys = args.data_file_keys.split(",")
            if hasattr(args, "heatmap_base_path"):
                heatmap_base_path = args.heatmap_base_path
            if repeat is None:
                repeat = args.dataset_repeat
            if name_map_path is None and hasattr(args, "name_map_path"):
                name_map_path = args.name_map_path

        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.heatmap_base_path = heatmap_base_path
        self.heatmap_spatial_downscale = 8

        self.dynamic_resolution = True
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")

        self.data = self._load_metadata(metadata_path, base_path)

        if name_map_path:
            print(f"Loading name map from {name_map_path}")
            with open(name_map_path, "r") as f:
                self.name_map = yaml.load(f, Loader=yaml.FullLoader)
            self.data = self.parse_style(self.data)
        if os.path.isfile(self.base_path):
            self.data = self.parse_style(self.data)
        else:
            print("No name map provided.")
            self.name_map = None

    def _load_metadata(self, metadata_path, base_path):
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            return [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        suffix = os.path.splitext(metadata_path)[1].lower()
        if suffix == ".json":
            with open(metadata_path, "r") as f:
                return json.load(f)
        if suffix == ".jsonl":
            records = []
            with open(metadata_path, "r") as f:
                for line in tqdm(f):
                    records.append(json.loads(line.strip()))
            return records
        metadata = pd.read_csv(metadata_path)
        return [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image: Any = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        # Support unlimited frames when num_frames is None or negative
        if self.num_frames is None or self.num_frames < 0:
            num_frames = int(reader.count_frames())
            # Still need to satisfy the time division factor constraint for model compatibility
            if num_frames > 1:
                while num_frames % self.time_division_factor != self.time_division_remainder:
                    num_frames -= 1
            return num_frames
        
        # Original behavior: limit to specified num_frames
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = [Image.fromarray(reader.get_data(frame_id)) for frame_id in range(num_frames)]
        target_height, target_width = self.get_height_width(frames[0])
        frames = [self.crop_and_resize(frame, target_height, target_width) for frame in frames]
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_heatmap(self, file_path):
        frames = self.load_video(file_path)
        frames = [frame.convert("L") for frame in frames]
        frames = self.transform_heatmap(frames)
        return frames
    
    
    def transform_heatmap(self, heatmap_frames):
        latent_scale = self.heatmap_spatial_downscale
        height, width = heatmap_frames[0].size[1], heatmap_frames[0].size[0]
        latent_height = max(1, height // latent_scale)
        latent_width = max(1, width // latent_scale)

        tensors = [torchvision.transforms.functional.to_tensor(frame) for frame in heatmap_frames]
        heatmap = torch.stack(tensors, dim=0)  # (F, 1, H, W)
        heatmap = F.interpolate(heatmap, size=(latent_height, latent_width), mode="bilinear", align_corners=False)
        heatmap = heatmap.permute(1, 0, 2, 3).contiguous()  # (1, T_latent, H_latent, W_latent)
        return heatmap
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None

    def parse_style(self, data):
        base_path = self.base_path
        new_data = []
        if os.path.isfile(base_path):
            with open(base_path, "r") as f:
                video_dirs = f.readlines()
            base_dir = os.path.dirname(os.path.abspath(base_path))
            video_dirs = [os.path.abspath(os.path.join(base_dir, video_dir.strip())) for video_dir in video_dirs]
        else:
            video_dirs = [os.path.abspath(p) for p in glob.glob(os.path.join(base_path))]

        print(f"video_dirs: {video_dirs}")

        for video_dir in video_dirs:
            if "styles_ignore" in os.listdir(video_dir):
                print(f"ignore {video_dir}")
                continue
            if hasattr(self, "name_map") and self.name_map is not None:
                color_key = os.path.basename(os.path.dirname(video_dir))
                style_key = os.path.basename(os.path.dirname(os.path.dirname(video_dir)))
                color_name = self.name_map["color"][color_key]
                style_name = self.name_map["brush"][style_key]

                for sample in data:
                    prompt = sample["prompt"].replace("{color}", color_name).replace("{brush}", style_name)
                    new_data.append({"video": os.path.join(video_dir, sample["video"]), "prompt": prompt})
            else:
                for sample in data:
                    new_data.append({"video": os.path.join(video_dir, sample["video"]), "prompt": sample["prompt"]})

        print(f"{len(new_data)} samples in total.")
        print(new_data[:2])
        return new_data



    def __getitem__(self, data_id):
        sample_id = data_id % len(self.data)
        raw = self.data[sample_id]
        data = raw.copy()
        video_rel_path = raw.get("video")
        if self.base_path is not None:
            for key in self.data_file_keys:
                if key in data:
                    path = data[key]
                    if not os.path.isabs(path):
                        path = os.path.join(self.base_path, path)
                    data[key] = self.load_data(path)
                    if data[key] is None:
                        warnings.warn(f"cannot load file {data[key]}.")
                        return None
        if self.heatmap_base_path is not None and video_rel_path is not None:
            heatmap_rel = raw.get("heatmap")
            if heatmap_rel is None:
                base_name = os.path.basename(video_rel_path)
                heatmap_rel = os.path.splitext(base_name)[0] + ".mp4"
            heatmap_path = heatmap_rel if os.path.isabs(heatmap_rel) else os.path.join(self.heatmap_base_path, heatmap_rel)
            data["heatmap"] = self.load_heatmap(heatmap_path)
        data["sample_id"] = sample_id
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    def export_trainable_state_dict(self, remove_prefix=None):
        # trainable_param_names = self.trainable_param_names()
        # state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        state_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}
        
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict


    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data


    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs


    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)

        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)




class EnhancedModelLogger(ModelLogger):
    """
    Enhanced model logger that extends the basic ModelLogger with WandB support
    and better loss tracking.

    Note: To track loss values, call log_loss(loss) before on_step_end() in your training loop.
    """

    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, wandb_run=None, visual_log_interval=50):
        super().__init__(output_path, remove_prefix_in_ckpt, state_dict_converter)
        self.wandb_run = wandb_run
        self.epoch_losses = []
        self.current_epoch_losses = []
        self.current_loss = None
        self.epoch_diffusion_losses = []
        self.current_epoch_diffusion_losses = []
        self.epoch_weighted_diffusion_losses = []
        self.current_epoch_weighted_diffusion_losses = []
        self.current_learning_rate = None
        self.visual_log_interval = visual_log_interval

    def log_loss(self, loss):
        """
        Record loss value for the current step.
        Call this method after computing loss and before on_step_end().
        """
        self.current_loss = loss

    def log_learning_rate(self, learning_rate):
        self.current_learning_rate = float(learning_rate)

    def on_step_end(self, accelerator, model, save_steps=None):
        """Called after each training step."""
        # Call parent's on_step_end to handle checkpoint saving
        super().on_step_end(accelerator, model, save_steps)

        # Log loss tracking if loss was recorded via log_loss()
        if self.current_loss is not None:
            # Convert loss to float if it's a tensor
            if hasattr(self.current_loss, 'item'):
                loss_value = self.current_loss.item()
            else:
                loss_value = float(self.current_loss)

            self.current_epoch_losses.append(loss_value)

            diffusion_loss_value = None
            weighted_diffusion_loss_value = None
            pipe = getattr(model, "pipe", None)
            if pipe is not None:
                diffusion_loss_tensor = getattr(pipe, "_last_diffusion_loss", None)
                if diffusion_loss_tensor is not None:
                    diffusion_loss_value = diffusion_loss_tensor.item() if hasattr(diffusion_loss_tensor, "item") else float(diffusion_loss_tensor)
                    self.current_epoch_diffusion_losses.append(diffusion_loss_value)
                weighted_diffusion_loss_tensor = getattr(pipe, "_last_weighted_diffusion_loss", None)
                if weighted_diffusion_loss_tensor is not None:
                    weighted_diffusion_loss_value = weighted_diffusion_loss_tensor.item() if hasattr(weighted_diffusion_loss_tensor, "item") else float(weighted_diffusion_loss_tensor)
                    self.current_epoch_weighted_diffusion_losses.append(weighted_diffusion_loss_value)

            # Log to WandB
            if self.wandb_run is not None:
                log_payload = {
                    "loss/step": loss_value,
                    "training/step": self.num_steps
                }
                if self.current_learning_rate is not None:
                    log_payload["lr/step"] = self.current_learning_rate
                if diffusion_loss_value is not None:
                    log_payload["loss/diffusion_step"] = diffusion_loss_value
                if weighted_diffusion_loss_value is not None:
                    log_payload["loss/diffusion_weighted_step"] = weighted_diffusion_loss_value
                self.wandb_run.log(log_payload)

            # Log every 100 steps
            if self.num_steps % 100 == 0:
                avg_loss = sum(self.current_epoch_losses[-100:]) / min(100, len(self.current_epoch_losses))
                message = f"Step {self.num_steps}: Loss = {loss_value:.6f}, Avg(100) = {avg_loss:.6f}"
                if self.current_learning_rate is not None:
                    message += f", LR = {self.current_learning_rate:.8f}"
                if diffusion_loss_value is not None:
                    window = self.current_epoch_diffusion_losses[-min(100, len(self.current_epoch_diffusion_losses)):]
                    diffusion_avg = sum(window) / len(window)
                    message += f", DiffLoss = {diffusion_loss_value:.6f}, DiffAvg(100) = {diffusion_avg:.6f}"
                logging.info(message)

            # Clear current loss after logging
            self.current_loss = None
            self.current_learning_rate = None
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        """Called at the end of each epoch."""
        # Calculate epoch statistics
        if self.current_epoch_losses:
            epoch_avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
            epoch_min_loss = min(self.current_epoch_losses)
            epoch_max_loss = max(self.current_epoch_losses)
            
            self.epoch_losses.append(epoch_avg_loss)
            
            logging.info(f"Epoch {epoch_id} completed:")
            logging.info(f"  Average Loss: {epoch_avg_loss:.6f}")
            logging.info(f"  Min Loss: {epoch_min_loss:.6f}")
            logging.info(f"  Max Loss: {epoch_max_loss:.6f}")
            logging.info(f"  Steps: {len(self.current_epoch_losses)}")

            diffusion_epoch_avg = None
            diffusion_epoch_min = None
            diffusion_epoch_max = None
            if self.current_epoch_diffusion_losses:
                diffusion_epoch_avg = sum(self.current_epoch_diffusion_losses) / len(self.current_epoch_diffusion_losses)
                diffusion_epoch_min = min(self.current_epoch_diffusion_losses)
                diffusion_epoch_max = max(self.current_epoch_diffusion_losses)
                self.epoch_diffusion_losses.append(diffusion_epoch_avg)
                logging.info(f"  Diffusion Avg Loss: {diffusion_epoch_avg:.6f}")
                logging.info(f"  Diffusion Min Loss: {diffusion_epoch_min:.6f}")
                logging.info(f"  Diffusion Max Loss: {diffusion_epoch_max:.6f}")

            weighted_diffusion_epoch_avg = None
            if self.current_epoch_weighted_diffusion_losses:
                weighted_diffusion_epoch_avg = sum(self.current_epoch_weighted_diffusion_losses) / len(self.current_epoch_weighted_diffusion_losses)
                self.epoch_weighted_diffusion_losses.append(weighted_diffusion_epoch_avg)

            # Log epoch stats to WandB
            if self.wandb_run is not None:
                # try:
                log_payload = {
                    "loss/epoch_avg": epoch_avg_loss,
                    "loss/epoch_min": epoch_min_loss,
                    "loss/epoch_max": epoch_max_loss,
                    "training/epoch": epoch_id,
                    "training/steps_per_epoch": len(self.current_epoch_losses)
                }
                if diffusion_epoch_avg is not None:
                    log_payload.update({
                        "loss/diffusion_epoch_avg": diffusion_epoch_avg,
                        "loss/diffusion_epoch_min": diffusion_epoch_min,
                        "loss/diffusion_epoch_max": diffusion_epoch_max,
                    })
                if weighted_diffusion_epoch_avg is not None:
                    log_payload["loss/diffusion_weighted_epoch_avg"] = weighted_diffusion_epoch_avg
                self.wandb_run.log(log_payload)
                # except Exception as e:
                #     logging.warning(f"Failed to log epoch stats to WandB: {e}")
            
            # Reset for next epoch
            self.current_epoch_losses = []
            self.current_epoch_diffusion_losses = []
            self.current_epoch_weighted_diffusion_losses = []
        
        # Save checkpoint (using parent class method)
        super().on_epoch_end(accelerator, model, epoch_id)
        
        # Log checkpoint save to WandB
        if self.wandb_run is not None:
            checkpoint_path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            # Get file size
            if os.path.exists(checkpoint_path):
                file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                self.wandb_run.log({
                    "training/checkpoint_size_mb": file_size_mb,
                    "training/checkpoint_saved": epoch_id
                })

                
    def get_training_summary(self):
        """Get a summary of training statistics."""
        if not self.epoch_losses:
            return "No training data available"
            
        summary = {
            "num_steps": self.num_steps,
            "total_epochs": len(self.epoch_losses),
            "final_loss": self.epoch_losses[-1],
            "best_loss": min(self.epoch_losses),
            "worst_loss": max(self.epoch_losses),
        }

        if self.epoch_diffusion_losses:
            summary.update({
                "diffusion_final_loss": self.epoch_diffusion_losses[-1],
                "diffusion_best_loss": min(self.epoch_diffusion_losses),
                "diffusion_worst_loss": max(self.epoch_diffusion_losses),
            })

        if self.epoch_weighted_diffusion_losses:
            summary.update({
                "diffusion_weighted_final_loss": self.epoch_weighted_diffusion_losses[-1],
                "diffusion_weighted_best_loss": min(self.epoch_weighted_diffusion_losses),
                "diffusion_weighted_worst_loss": max(self.epoch_weighted_diffusion_losses),
            })
        
        return summary 



def load_checkpoint_and_extract_info(checkpoint_path):
    """
    Load checkpoint and extract information about its contents.

    Args:
        checkpoint_path (str): Path to the checkpoint file

    Returns:
        tuple: (checkpoint_data, epoch_num, info_summary)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from: {checkpoint_path}")

    # Extract epoch number from filename
    epoch_num = None
    filename = os.path.basename(checkpoint_path)
    epoch_match = re.search(r'epoch-(\d+)', filename)
    if epoch_match:
        epoch_num = int(epoch_match.group(1))
        logging.info(f"Extracted epoch number from filename: {epoch_num}")

    # Load checkpoint data
    checkpoint_data = load_file(checkpoint_path)

    # Analyze checkpoint contents
    total_params = len(checkpoint_data)
    total_size = sum(param.numel() for param in checkpoint_data.values())

    # Group parameters by prefix to understand model structure
    prefixes = {}
    for key in checkpoint_data.keys():
        prefix = key.split('.')[0] if '.' in key else key
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)

    info_summary = {
        'checkpoint_path': checkpoint_path,
        'epoch_num': epoch_num,
        'total_parameters': total_params,
        'total_parameter_count': total_size,
        'parameter_groups': {prefix: len(params) for prefix, params in prefixes.items()},
        'parameter_keys': list(checkpoint_data.keys())[:10]  # First 10 keys for inspection
    }

    logging.info(f"Checkpoint loaded successfully:")
    logging.info(f"  - Total parameters: {total_params}")
    logging.info(f"  - Total parameter count: {total_size:,}")
    logging.info(f"  - Parameter groups: {info_summary['parameter_groups']}")
    if len(checkpoint_data.keys()) > 10:
        logging.info(f"  - Sample parameter keys: {info_summary['parameter_keys']} (and {total_params-10} more)")
    else:
        logging.info(f"  - All parameter keys: {list(checkpoint_data.keys())}")

    return checkpoint_data, epoch_num, info_summary

def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    resume_from_checkpoint: str = None,
    remove_prefix_in_ckpt: str = None,
):
    # Create optimizer and scheduler if not provided (for backward compatibility)
    if optimizer is None:
        assert False, "Please provide an optimizer."
        # optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler is None:
        assert False, "Please provide a scheduler."
        # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model,dataloader= accelerator.prepare(model,dataloader)

    handled_signals = (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2)
    shutdown_triggered = False

    def _handle_shutdown(signum, frame):
        nonlocal shutdown_triggered
        if shutdown_triggered:
            return
        shutdown_triggered = True
        logging.warning(f"Received termination signal {signum}, releasing distributed resources.")
        for _sig in handled_signals:
            signal.signal(_sig, signal.SIG_IGN)
        pgid = os.getpgid(os.getpid())
        os.killpg(pgid, signum)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if torch.cuda.is_available():
            accelerator.free_memory()
            torch.cuda.empty_cache()
        sys.exit(128 + signum)

    for _sig in handled_signals:
        signal.signal(_sig, _handle_shutdown)

    start_epoch = 0

    # Handle checkpoint resuming
    if resume_from_checkpoint is not None:
        checkpoint_data, checkpoint_epoch, _ = load_checkpoint_and_extract_info(resume_from_checkpoint)
        # add back the prefix
        if remove_prefix_in_ckpt is not None:
            # checkpoint_data = {remove_prefix_in_ckpt + key: value for key, value in checkpoint_data.items()}
            new_checkpoint_data = {}
            for key, value in checkpoint_data.items():
                if key.startswith("pipe."):
                    new_checkpoint_data[key] = value
                else:
                    new_key = remove_prefix_in_ckpt + key
                    new_checkpoint_data[new_key] = value
            checkpoint_data = new_checkpoint_data
        # Load model state
        logging.info(f"Loading model state from checkpoint: {resume_from_checkpoint}")
        # The checkpoint contains only trainable parameters, so we use load_state_dict with strict=False
        missing_keys, unexpected_keys = accelerator.unwrap_model(model).load_state_dict(checkpoint_data, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys when loading model state: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading model state: {unexpected_keys}")

        logging.info("Model state loaded successfully from checkpoint")

        # Set start epoch to continue from next epoch
        if checkpoint_epoch is not None:
            start_epoch = checkpoint_epoch + 1
            logging.info(f"Resuming training from epoch {start_epoch}")
        else:
            logging.warning("Could not determine epoch from checkpoint filename, starting from epoch 0")
    model = accelerator.unwrap_model(model)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    model_logger.on_epoch_end(accelerator, model, start_epoch - 1)
    for epoch_id in range(start_epoch, num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if hasattr(model_logger, "prepare_step"):
                    model_logger.prepare_step(accelerator.unwrap_model(model), sample_id=data["sample_id"])
                loss = model(data)
                accelerator.backward(loss) # step: optimizer.step() model parameter
                optimizer.step()
                if hasattr(model_logger, 'log_learning_rate'):
                    model_logger.log_learning_rate(optimizer.param_groups[0]["lr"])
                model_logger.log_loss(loss)
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
            torch.cuda.empty_cache()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)



def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--heatmap_base_path", type=str, default=None, help="Base path to load ground-truth heatmap videos.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix. Set to -1 to use all frames without limit.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--diffusion_loss_weight", type=float, default=1.0, help="Loss weight for diffusion supervision.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=8, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    # Resume training parameter
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from (e.g., experiments/.../ckpt/epoch-1.safetensors).")
    # WandB and experiment management (optional, can be overridden by specific training scripts)
    parser.add_argument("--wandb_project", type=str, default="WAN_training", help="WandB project name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--experiment_dir", type=str, default="experiments", help="Base directory for experiments")
    parser.add_argument("--name_map_path", type=str, default=None, help="Path to name map file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--visual_log_interval", type=int, default=50, help="Interval for visual logging")
    return parser

