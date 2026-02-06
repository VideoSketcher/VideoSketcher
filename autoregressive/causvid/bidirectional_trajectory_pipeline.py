from torch._tensor import Tensor


from causvid.models.model_interface import (
    InferencePipelineInterface,
    DiffusionModelInterface,
    TextEncoderInterface
)
from causvid.scheduler import SchedulerInterface
from typing import List
import torch


class BidirectionalInferenceWrapper(InferencePipelineInterface):
    def __init__(self, denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: DiffusionModelInterface, **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list

    def inference_with_trajectory(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        output_list = [noise]

        # initial point
        noisy_image_or_video = noise

        # use the last n-1 timesteps to simulate the generator's input
        for index, current_timestep in enumerate(self.denoising_step_list[:-1]):
            pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
            )  # [B, F, C, H, W]

            # TODO: Change backward simulation for causal video
            next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                noise.shape[:2], dtype=torch.long, device=noise.device)
            noisy_image_or_video = self.scheduler.add_noise(
                pred_image_or_video.flatten(0, 1),
                torch.randn_like(pred_image_or_video.flatten(0, 1)),
                next_timestep.flatten(0, 1)
            ).unflatten(0, noise.shape[:2])
            output_list.append(noisy_image_or_video)

        # [B, T, F, C, H, W]
        output: Tensor = torch.stack(output_list, dim=1)
        return output


class CausalInferenceWrapper(InferencePipelineInterface):
    def __init__(self, denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: DiffusionModelInterface,
                 num_frame_per_block: int = 3, **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        self.num_frame_per_block = num_frame_per_block
        self.frame_seq_length = getattr(generator, 'frame_seq_length', 1560)
        self.num_transformer_blocks = getattr(generator, 'num_transformer_blocks', 30)
        
        # Cache states
        self.kv_cache1 = None
        self.crossattn_cache = None

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, 32760, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 32760, 12, 128], dtype=dtype, device=device)
            })

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache

    def inference_with_trajectory(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        """
        Autoregressive inference with trajectory collection.
        
        Input:
            - noise: [B, F, C, H, W] input noise tensor
            - conditional_dict: dictionary containing conditional information
            
        Output:
            - output: [B, T, F, C, H, W] trajectory tensor where T is number of timesteps
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        # Initialize cache if needed
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # Reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False

        # Initialize trajectory collection
        # trajectory_collection[timestep_index][block_index] = block_output
        trajectory_collection = [[] for _ in range(len(self.denoising_step_list))]
        
        # Add initial noise to first timestep for all blocks
        num_blocks = num_frames // self.num_frame_per_block
        for block_index in range(num_blocks):
            block_noise = noise[:, block_index * self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]
            trajectory_collection[0].append(block_noise)

        # Outer loop: process each block sequentially
        for block_index in range(num_blocks):
            noisy_input = noise[:, block_index * self.num_frame_per_block:(block_index + 1) * self.num_frame_per_block]

            # Inner loop: spatial denoising for current block
            for timestep_index, current_timestep in enumerate(self.denoising_step_list[:-1]):
                # Set current timestep
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], device=noise.device, dtype=torch.int64) * current_timestep

                # Get prediction for current block at current timestep
                denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length
                )



                # if timestep_index < len(self.denoising_step_list) - 2:  # Not the last timestep
                next_timestep = self.denoising_step_list[timestep_index + 1]
                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones([batch_size], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])

                trajectory_collection[timestep_index + 1].append(noisy_input)

            # Update cache with final prediction (timestep=0)
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                current_end=(block_index + 1) * self.num_frame_per_block * self.frame_seq_length
            )

        # Convert trajectory collection to proper format [B, T, F, C, H, W]
        output_list = []
        for timestep_outputs in trajectory_collection:
            # Concatenate all blocks for this timestep
            timestep_tensor = torch.cat(timestep_outputs, dim=1)  # [B, F, C, H, W]
            output_list.append(timestep_tensor)
        
        # Stack along time dimension
        output = torch.stack(output_list, dim=1)  # [B, T, F, C, H, W]
        return output
