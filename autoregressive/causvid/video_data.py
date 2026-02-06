import torch
import pandas as pd
import os
import imageio
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VideoDataset(Dataset):
    def __init__(self, dataset_base_path, metadata_path, 
                 num_frames=81, max_frames=None,
                 height=480, width=832, 
                 time_division_factor=4, time_division_remainder=1,
                 video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
                 repeat=1):
        """
        Video dataset for loading video files and corresponding prompts.
        
        Args:
            dataset_base_path: Base path containing video files
            metadata_path: Path to CSV file with video,prompt columns
            num_frames: Number of frames to load from each video
            max_frames: Maximum frames available in videos (for validation)
            height: Target height for video frames
            width: Target width for video frames
            time_division_factor: Temporal division factor for model compatibility
            time_division_remainder: Remainder for temporal division
            video_file_extension: Supported video file extensions
            repeat: Number of times to repeat the dataset per epoch
        """
        self.dataset_base_path = dataset_base_path
        self.num_frames = num_frames
        self.max_frames = max_frames
        self.height = height
        self.width = width
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        print(f"Loaded {len(self.metadata)} video entries from {metadata_path}")
        
        # Validate that all video files exist
        self._validate_files()
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _validate_files(self):
        """Validate that all video files exist"""
        valid_indices = []
        for idx, row in self.metadata.iterrows():
            video_path = os.path.join(self.dataset_base_path, row['video'])
            if os.path.exists(video_path):
                valid_indices.append(idx)
            else:
                print(f"Warning: Video file not found: {video_path}")
        
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        print(f"Found {len(self.metadata)} valid video files")
    
    def _get_adjusted_num_frames(self, total_frames):
        """Get number of frames satisfying division constraints"""
        if self.max_frames and total_frames > self.max_frames:
            total_frames = self.max_frames
            
        num_frames = min(self.num_frames, total_frames)
        
        # Ensure frames satisfy division constraints
        while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames -= 1
            
        return max(1, num_frames)  # At least 1 frame
    
    def _load_video_frames(self, video_path, num_frames_to_load):
        """Load video frames using imageio"""

        reader = imageio.get_reader(video_path)
        total_frames = int(reader.count_frames())

        # Adjust frame count
        actual_num_frames = self._get_adjusted_num_frames(total_frames)
        if actual_num_frames > total_frames:
            actual_num_frames = total_frames

        # Load frames
        frames = []
        frame_indices = torch.linspace(0, total_frames - 1, actual_num_frames).long()

        for frame_idx in frame_indices:
            frame = reader.get_data(frame_idx.item())
            frame = Image.fromarray(frame).convert('RGB')
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)

        reader.close()

        # Stack to tensor: [T, C, H, W]
        video_tensor = torch.stack(frames, dim=0)

        # Pad if necessary to reach target num_frames
        if video_tensor.shape[0] < self.num_frames:
            padding_frames = self.num_frames - video_tensor.shape[0]
            # Repeat last frame for padding
            last_frame = video_tensor[-1:].repeat(padding_frames, 1, 1, 1)
            video_tensor = torch.cat([video_tensor, last_frame], dim=0)

        return video_tensor
            

    
    def __len__(self):
        return len(self.metadata) * self.repeat
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'video': tensor of shape [num_frames, 3, height, width]
                - 'prompt': string prompt
                - 'video_path': path to video file
        """
        # Handle repeat by using modulo to map back to original dataset
        row = self.metadata.iloc[idx % len(self.metadata)]
        video_path = os.path.join(self.dataset_base_path, row['video'])
        prompt = row['prompt']
        
        # Load video frames
        video_tensor = self._load_video_frames(video_path, self.num_frames)
        
        return {
            'video': video_tensor,  # [T, C, H, W]
            'prompt': prompt,
            'video_path': video_path
        }


class VideoRegressionDataset(VideoDataset):
    """
    Extended video dataset compatible with CausVid training pipeline.
    Returns data in format expected by VideoRegression training.
    """
    
    def __getitem__(self, idx):
        """
        Returns data in format compatible with regression training:
            - 'video_tensor': [T, C, H, W] for VAE encoding
            - 'prompts': string (or list of strings for batch compatibility)
        """
        data = super().__getitem__(idx)
        
        return {
            'video_tensor': data['video'],  # [T, C, H, W]
            'prompts': data['prompt'],  # String prompt
            'video_path': data['video_path']
        } 