from causvid.ode_data.create_lmdb_iterative import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb


class TextDataset(Dataset):
    def __init__(self, data_path, repeat=1):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    self.texts.append(line.strip())
        self.repeat = repeat

    def __len__(self):
        return len(self.texts) * self.repeat

    def __getitem__(self, idx):
        # Handle repeat by using modulo to map back to original dataset
        return self.texts[idx % len(self.texts)]


class ODERegressionDataset(Dataset):
    def __init__(self, data_path, max_pair=int(1e8), repeat=1):
        self.data_dict = torch.load(data_path, weights_only=False)
        self.max_pair = max_pair
        self.repeat = repeat

    def __len__(self):
        return min(len(self.data_dict['prompts']), self.max_pair) * self.repeat

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        # Handle repeat by using modulo to map back to original dataset
        original_idx = idx % min(len(self.data_dict['prompts']), self.max_pair)
        return {
            "prompts": self.data_dict['prompts'][original_idx],
            "ode_latent": self.data_dict['latents'][original_idx].squeeze(0),
        }


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8), repeat: int = 1):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair
        self.repeat = repeat

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair) * self.repeat

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        # Handle repeat by using modulo to map back to original dataset
        original_idx = idx % min(self.latents_shape[0], self.max_pair)
        
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, original_idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, original_idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }
