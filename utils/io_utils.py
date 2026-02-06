import os
import imageio
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open


def save_video(frames, save_path, fps=16, quality=9, ffmpeg_params=None):
    """Save frames as video file."""
    if fps != 16:
        raise NotImplementedError("We assume fps=16 to save currently.")
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


