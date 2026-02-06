import logging
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch


class PrecomputedCausVidLatentDataset(Dataset):
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


def precompute_causvid_latents(dataset, vae, device, dtype, dataset_repeat, num_workers):
    logging.info("Precomputing CausVid latents before training.")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda x: x[0],
    )
    cached_samples = []
    vae_was_training = vae.training
    vae_model_was_training = vae.model.training

    vae_params = list(vae.parameters())
    if vae_params:
        vae_original_device = vae_params[0].device
        vae_original_dtype = vae_params[0].dtype
    else:
        vae_original_device = device
        vae_original_dtype = dtype

    vae_model_params = list(vae.model.parameters())
    if vae_model_params:
        vae_model_device = vae_model_params[0].device
        vae_model_dtype = vae_model_params[0].dtype
    else:
        vae_model_device = device
        vae_model_dtype = dtype

    vae.eval()
    vae.model.eval()
    vae.model.to(device=device, dtype=dtype)
    vae.mean = vae.mean.to(device=device, dtype=dtype)
    vae.std = vae.std.to(device=device, dtype=dtype)

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Encoding latents", leave=False):
            video_tensor = data["video_tensor"].to(device=device, dtype=dtype)
            video_ctfw = video_tensor.permute(1, 0, 2, 3)
            latent_ctfw = vae.encode([video_ctfw])[0]
            latent_tchw = latent_ctfw.permute(1, 0, 2, 3).to(dtype=dtype)

            cached_samples.append(
                {
                    "video_latent": latent_tchw.cpu(),
                    "prompts": data["prompts"],
                    "video_path": data["video_path"],
                }
            )

    if vae_was_training:
        vae.train()
    else:
        vae.eval()

    if vae_model_was_training:
        vae.model.train()
    else:
        vae.model.eval()

    vae.model.to(device=vae_model_device, dtype=vae_model_dtype)
    vae.mean = vae.mean.to(device=vae_original_device, dtype=vae_original_dtype)
    vae.std = vae.std.to(device=vae_original_device, dtype=vae_original_dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Latent precomputation finished. Cached %d unique samples.", len(cached_samples))
    return PrecomputedCausVidLatentDataset(cached_samples, repeat=dataset_repeat)

