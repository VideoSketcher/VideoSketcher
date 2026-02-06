import glob
import os
import torch
from dataclasses import dataclass
from typing import Optional, Union
from huggingface_hub import snapshot_download


@dataclass
class ModelConfig:
    """Model configuration for loading Wan video models."""
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None
    local_model_path: str = "./pretrained"
    skip_download: bool = False

    def download_if_necessary(self, use_usp=False):
        """Download model from HuggingFace if not cached locally."""
        if self.path is None:
            if self.model_id is None:
                raise ValueError(f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`.""")
            
            skip_download = self.skip_download
            if use_usp:
                import torch.distributed as dist
                skip_download = skip_download or dist.get_rank() != 0
                
            # Check folder vs file pattern
            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.origin_file_pattern = ""
                allow_file_pattern = None
                is_folder = True
            elif isinstance(self.origin_file_pattern, str) and self.origin_file_pattern.endswith("/"):
                allow_file_pattern = self.origin_file_pattern + "*"
                is_folder = True
            else:
                allow_file_pattern = self.origin_file_pattern
                is_folder = False
            
            # Download from HuggingFace
            if self.local_model_path is None:
                self.local_model_path = "./models"
            if not skip_download:
                downloaded_files = glob.glob(self.origin_file_pattern, root_dir=os.path.join(self.local_model_path, self.model_id))
                if len(downloaded_files) == 0:
                    snapshot_download(
                        self.model_id,
                        local_dir=os.path.join(self.local_model_path, self.model_id),
                        allow_patterns=allow_file_pattern,
                        local_dir_use_symlinks=False,
                    )
            
            if use_usp:
                import torch.distributed as dist
                dist.barrier()
            
            # Find downloaded files
            downloaded_files = glob.glob(self.origin_file_pattern, root_dir=os.path.join(self.local_model_path, self.model_id))
            if len(downloaded_files) == 0:
                raise ValueError(f"Model download failed. model_id: {self.model_id}, origin_file_pattern: {self.origin_file_pattern}")
            
            if is_folder:
                self.path = os.path.join(self.local_model_path, self.model_id, self.origin_file_pattern)
            else:
                self.path = [os.path.join(self.local_model_path, self.model_id, file) for file in downloaded_files]
            
            print(f"Model path: {self.path}")

