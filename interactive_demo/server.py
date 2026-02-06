import argparse
import base64
import math
import threading
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import logging
import os

import numpy as np
import torch
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel, Field
import time
import uvicorn
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Paths / Python import setup
# ---------------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
MINIMAL_DIR = CURRENT_DIR.parent
if str(MINIMAL_DIR) not in sys.path:
    sys.path.append(str(MINIMAL_DIR))

from inference_demo import InferencePipeline  # noqa: E402
from utils.prompt_refiner import PromptRefiner  # noqa: E402


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _to_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_checkpoint_and_initialize_pipeline(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    Load config, initialize InferencePipeline, and load generator checkpoint.
    """
    torch.set_grad_enabled(False)

    logging.info(f"Loading config from: {config_path}")
    config = OmegaConf.load(config_path)

    logging.info("Initializing inference pipeline...")
    pipeline = InferencePipeline(config, device=device)
    pipeline.to(device=device, dtype=torch_dtype)

    checkpoint_path = os.path.abspath(checkpoint_path)

    logging.info("Loading checkpoint from: %s", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["generator"]
    pipeline.generator.load_state_dict(state_dict, strict=True)
    logging.info("Checkpoint loaded successfully")

    return pipeline, config


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """
    Per-session state for interactive use.
    """
    session_id: str
    prompt: str
    width: int
    height: int
    fps: int
    model_limit: int          # model limit in *latent* frames (aligned to block)
    user_limit: int           # user limit in *latent* frames (aligned to block)
    storage_dir: Path

    # Sliding-window history
    latents: Optional[torch.Tensor] = None      # tail latents [1, T_latent, C, H, W] (CPU)
    video_tail: Optional[np.ndarray] = None     # tail frames [T_tail, H, W, 3] in [0,1]

    # Bookkeeping
    total_frames: int = 0                       # total visible frames shown so far
    latent_frames: int = 0                      # total latent frames generated so far
    segment_index: int = 0
    last_frame_path: Optional[Path] = None

    def next_segment_path(self, kind: str) -> Path:
        """
        Return a path for the next segment MP4 and increment the index.
        """
        filename = f"{self.segment_index:04d}_{kind}.mp4"
        self.segment_index += 1
        return self.storage_dir / filename

    @property
    def public_dir(self) -> str:
        return f"/sessions/{self.session_id}"


# ---------------------------------------------------------------------------
# Core interactive engine
# ---------------------------------------------------------------------------

class InteractiveEngine:
    """
    Thin wrapper around InferencePipeline.

    Responsibilities:
      - Manage sessions.
      - Convert frames <-> latents.
      - Maintain a sliding window of the last N video frames per session.
      - Call pipeline.inference(noise, text_prompts, start_latents=tail_latents).
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str,
        torch_dtype: torch.dtype,
        model_limit: int,
        user_limit: int,
        overlap_frames: int,
        no_load_pipe: bool = False,
    ):
        self.device = device
        self.dtype = torch_dtype

        # All sessions keyed by session_id
        self.sessions: Dict[str, SessionState] = {}

        # Load model pipeline once
        logging.info("Loading model...")
        start_time = time.time()
        self.num_frame_per_block = 3
        if not no_load_pipe:
            self.pipeline, self.config = load_checkpoint_and_initialize_pipeline(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device=device,
                torch_dtype=torch_dtype,
            )
            logging.info("Finished loading model in %.2f seconds", time.time() - start_time)

            # Temporal block size of the model
            self.num_frame_per_block = self.pipeline.num_frame_per_block

        # Serialize access to the pipeline
        self.pipeline_lock = threading.Lock()

        # Latent spatial dimensions (fixed to model)
        self.latent_channels = 16
        self.latent_height = 60
        self.latent_width = 104


        # Root directory for all session outputs
        self.sessions_root = CURRENT_DIR / "sessions"
        self.sessions_root.mkdir(parents=True, exist_ok=True)

        # Mapping: latent frames -> video frames
        # First latent -> 1 frame, each subsequent latent -> +4 frames
        self.video_frames_per_latent = 4
        assert overlap_frames > 0
        assert overlap_frames % self.video_frames_per_latent == 1
        self.overlap_frames = overlap_frames

        # Hard cap on TOTAL latent frames per call (context + new)
        self.max_total_latents_per_call = 21  # -> at most 81 visible frames

        # Align model/user frame limits to multiples of block size
        aligned_model_limit = max(
            self.num_frame_per_block,
            (model_limit // self.num_frame_per_block) * self.num_frame_per_block,
        )
        aligned_user_limit = max(
            self.num_frame_per_block,
            (user_limit // self.num_frame_per_block) * self.num_frame_per_block,
        )
        self.model_limit = aligned_model_limit
        self.user_limit = aligned_user_limit

    def set_overlap_frames(self, overlap_frames: int) -> int:
        assert overlap_frames > 0
        assert overlap_frames % self.video_frames_per_latent == 1
        self.overlap_frames = overlap_frames
        return self.overlap_frames
    # ----------------------------------------------------------------------
    # Session management
    # ----------------------------------------------------------------------

    def create_session(self, prompt: str, width: int, height: int, fps: int) -> SessionState:
        """
        Create a new session directory and SessionState.
        """
        session_id = uuid.uuid4().hex
        storage_dir = self.sessions_root / session_id
        storage_dir.mkdir(parents=True, exist_ok=True)
        state = SessionState(
            session_id=session_id,
            prompt=prompt,
            width=width,
            height=height,
            fps=fps,
            model_limit=self.model_limit,
            user_limit=self.user_limit,
            storage_dir=storage_dir,
        )
        self.sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="session not found")
        return self.sessions[session_id]

    def update_session_prompt(self, session: SessionState, new_prompt: str) -> None:
        """
        Update the text prompt for this session. The pipeline re-encodes text
        internally on each inference call, so no cache management is needed here.
        """
        new_prompt = new_prompt.strip()
        if not new_prompt or new_prompt == session.prompt:
            return
        session.prompt = new_prompt

    # ----------------------------------------------------------------------
    # Temporal mapping utilities (video frames <-> latent frames)
    # ----------------------------------------------------------------------

    def _aligned_frames(self, frames: int) -> int:
        """
        Align a latent frame count to a multiple of num_frame_per_block,
        not exceeding model_limit.
        """
        aligned = frames - (frames % self.num_frame_per_block)
        if aligned == 0:
            aligned = self.num_frame_per_block
        return min(aligned, self.model_limit)

    def _latent_to_video_frames(self, latent_frames: int) -> int:
        """
        Map a number of latent frames to the corresponding number of video frames.
        """
        if latent_frames <= 0:
            return 0
        if latent_frames == 1:
            return 1
        return 1 + (latent_frames - 1) * self.video_frames_per_latent

    def _video_to_latent_frames(self, session: SessionState, video_frames: int) -> int:
        """
        Map desired video frames -> latent frames, then align to block size.
        """
        current_video_frames = session.total_frames
        if video_frames <= 1:
            desired = 1
        elif current_video_frames == 0:
            desired = 1 + math.ceil((video_frames - 1) / self.video_frames_per_latent)
        else:
            desired = video_frames // self.video_frames_per_latent
        return self._aligned_frames(desired)

    def _latent_increment_to_video_frames(self, start_latents: int, added_latents: int) -> int:
        """
        Given an initial latent count + newly added latents, how many extra
        visible frames did we get?
        """
        before = self._latent_to_video_frames(start_latents)
        after = self._latent_to_video_frames(start_latents + added_latents)
        return after - before

    # ----------------------------------------------------------------------
    # Noise preparation
    # ----------------------------------------------------------------------

    def _prepare_noise(self, total_frames: int, new_frames: int, seed: Optional[int]) -> torch.Tensor:
        """
        Allocate noise for total_frames latent frames, where the last new_frames
        are random and the prefix (if any) is zeros.
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        noise = torch.zeros(
            (1, total_frames, self.latent_channels, self.latent_height, self.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        if new_frames > 0:
            noise[:, total_frames - new_frames :] = torch.randn(
                (1, new_frames, self.latent_channels, self.latent_height, self.latent_width),
                device=self.device,
                dtype=self.dtype,
            )
        return noise

    # ----------------------------------------------------------------------
    # Video export helpers
    # ----------------------------------------------------------------------

    def _video_chunk_to_path(self, session: SessionState, frames_np: np.ndarray, kind: str) -> str:
        """
        Save a [T, H, W, 3] in [0,1] NumPy array as MP4 and update last_frame.
        """
        if frames_np.size == 0:
            # Nothing to save
            return ""

        chunk_path = session.next_segment_path(kind)
        export_to_video(frames_np, str(chunk_path), fps=session.fps)

        last_frame = (frames_np[-1] * 255).astype(np.uint8)
        last_frame_path = session.storage_dir / "last_frame.png"
        Image.fromarray(last_frame).save(last_frame_path)
        session.last_frame_path = last_frame_path

        return f"{session.public_dir}/{chunk_path.name}"

    def _segment_payload(self, session: SessionState, segment_url: str, frames_added: int) -> Dict[str, object]:
        """
        Uniform JSON payload for model/sketch segments.
        """
        return {
            "segment_url": segment_url,
            "frames": frames_added,
            "total_frames": session.total_frames,
            "last_frame_url": f"{session.public_dir}/last_frame.png",
        }

    # ----------------------------------------------------------------------
    # Frames <-> latents
    # ----------------------------------------------------------------------

    def _decode_base64_frames(self, frames: List[str], width: int, height: int) -> np.ndarray:
        """
        Decode a list of base64 data URLs into [T, H, W, 3] in [0,1].
        """
        if not frames:
            return np.empty((0, height, width, 3), dtype=np.float32)

        scale = 1.0 / 255.0

        def decode_one(frame: str) -> np.ndarray:
            content = frame.split(",", 1)[1] if "," in frame else frame
            raw = base64.b64decode(content)
            with Image.open(BytesIO(raw)) as image:
                rgb = image.convert("RGB").resize((width, height), Image.BILINEAR)
                return np.asarray(rgb, dtype=np.float32) * scale

        workers = min(8, len(frames))
        if workers == 1:
            decoded = [decode_one(frame) for frame in frames]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                decoded = list(pool.map(decode_one, frames))
        return np.stack(decoded, axis=0)

    def _encode_pixel_to_latents(self, pixel: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel video [1, T, 3, H, W] into latents [1, T_latent, C, H', W'].
        """
        vae = self.pipeline.vae
        device = pixel.device
        dtype = pixel.dtype
        scale = [
            vae.mean.to(device=device, dtype=dtype),
            1.0 / vae.std.to(device=device, dtype=dtype),
        ]
        latents = vae.model.encode(pixel, scale).float()
        # Move time dimension to second axis: [B, T, C, H, W]
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    def _frames_to_latents(self, frames_np: np.ndarray) -> torch.Tensor:
        """
        Convert [T, H, W, 3] in [0,1] -> latents [1, T_latent, C, H, W].
        """
        if frames_np.size == 0:
            return torch.empty(1, 0, self.latent_channels, self.latent_height, self.latent_width)

        pixel = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # [T, 3, H, W]
        pixel = pixel.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # [1, T, 3, H, W]
        pixel = pixel.to(device=self.device, dtype=self.dtype)
        pixel = pixel * 2.0 - 1.0
        with torch.no_grad():
            latents = self._encode_pixel_to_latents(pixel)
        return latents.to(device=self.device, dtype=self.dtype)

    def _pad_frames_to_block(self, frames_np: np.ndarray) -> np.ndarray:
        """
        Pad last frame to make T divisible by num_frame_per_block.
        """
        if frames_np.size == 0:
            return frames_np
        remainder = frames_np.shape[0] % self.num_frame_per_block
        if remainder == 0:
            return frames_np
        pad = self.num_frame_per_block - remainder
        pad_frames = np.repeat(frames_np[-1:], pad, axis=0)
        return np.concatenate([frames_np, pad_frames], axis=0)

    def _pad_latents_to_block(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pad last latent to make T_latent divisible by num_frame_per_block.
        """
        if latents.shape[1] % self.num_frame_per_block == 0:
            return latents
        pad = self.num_frame_per_block - (latents.shape[1] % self.num_frame_per_block)
        pad_latents = latents[:, -1:].repeat(1, pad, 1, 1, 1)
        return torch.cat([latents, pad_latents], dim=1)

    # ----------------------------------------------------------------------
    # Sliding-window tail management
    # ----------------------------------------------------------------------

    def _update_tail_from_video(
        self,
        session: SessionState,
        new_video_np: np.ndarray,
        chunk_latents: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Maintain a sliding window of up to 21 video frames (in [0,1]) and
        re-encode them to latents for the next call as start_latents.
        """
        if new_video_np.size == 0:
            return

        if chunk_latents is None:
            chunk_latents = self._frames_to_latents(new_video_np)

        # Combine with previous tail if present
        if session.video_tail is not None:
            frames_combined = np.concatenate([session.video_tail, new_video_np], axis=0)
            latents_combined = torch.cat([session.latents, chunk_latents], dim=1)
        else:
            frames_combined = new_video_np
            latents_combined = chunk_latents

        # Keep last up to overlap_frames frames
        overlap_frames = min(self.overlap_frames, frames_combined.shape[0])
        # adjust overlap_frames such that ((overlap_frames-1) // 4 +1) % 3 == 0
        while (
            ((overlap_frames - 1) // 4 + 1) % self.num_frame_per_block != 0
            or (overlap_frames - 1) % self.video_frames_per_latent != 0
        ):
            overlap_frames -= 1
        logging.debug("Adjusted overlap_frames: %d", overlap_frames)

        start_frame = frames_combined[-overlap_frames:-overlap_frames + 1]
        start_frame_latent = self._frames_to_latents(start_frame)

        start_latents = torch.cat(
            [start_frame_latent, latents_combined[:, -(overlap_frames - 1) // 4 :]],
            dim=1,
        )
        assert start_latents.shape[1] % self.num_frame_per_block == 0
       
        
        session.latents = start_latents
        session.video_tail = frames_combined[-overlap_frames:]

        return start_latents

    # ----------------------------------------------------------------------
    # User sketch handling
    # ----------------------------------------------------------------------

    def append_user_segment(self, session: SessionState, frames: List[str]) -> Dict[str, object]:
        """
        User-drawn sketch frames:
          1. Decode to [T, H, W, 3] in [0,1].
          2. Save as 'sketch' MP4.
          3. Update total_frames.
          4. Update sliding tail (video_tail + latents) for next run().
        """
        with self.pipeline_lock:
            if not frames:
                raise HTTPException(status_code=400, detail="no frames provided")

            frames_np = self._decode_base64_frames(frames, session.width, session.height)  # [T, H, W, 3] in [0,1]
            frames_np = self._pad_frames_to_block(frames_np)

            num_video_frames = frames_np.shape[0]
            session.total_frames += num_video_frames

            logging.info("[APPEND USER SEGMENT] appended video shape: %s", frames_np.shape)

            # Update tail (this includes the last canvas the user saw)
            tail_len = min(self.overlap_frames, frames_np.shape[0])
            truncated_frames_np = frames_np[-tail_len:]
            truncated_latents = self._frames_to_latents(truncated_frames_np)
            self._update_tail_from_video(session, truncated_frames_np, truncated_latents)

            segment_url = self._video_chunk_to_path(session, frames_np, "sketch")
            return self._segment_payload(session, segment_url, num_video_frames)

    # ----------------------------------------------------------------------
    # Model-driven generation
    # ----------------------------------------------------------------------

    def generate_model_segment(self, session: SessionState, requested_frames: int, seed: Optional[int]) -> Dict[str, object]:
        """
        Generate a new video segment:

          - Use session.latents (if any) as start_latents (context).
          - Convert requested video frames to latent frames.
          - Cap total latents this call by max_total_latents_per_call and model_limit.
          - Prepare noise with zeros for context, random for new.
          - Call pipeline.inference().
          - Return only the visible frames corresponding to the new latents.
          - Update the sliding tail from the decoded video.
        """
        with self.pipeline_lock:
            # 1. Context latents
            if session.latents is not None and session.latents.shape[1] > 0:
                start_latents = session.latents.to(device=self.device, dtype=self.dtype).contiguous()
                start_len = start_latents.shape[1]
            else:
                start_latents = None
                start_len = 0

            # 2. Desired total latent frames (context + new)
            new_latent_frames = self._video_to_latent_frames(session, requested_frames)
            new_latent_frames = min(new_latent_frames, self.max_total_latents_per_call, self.model_limit)
            logging.info("Adding %d latent frames", new_latent_frames)
            if new_latent_frames <= 0:
                raise HTTPException(status_code=400, detail="frames must be positive")

            total_chunk_latents = start_len + new_latent_frames

            # 3. Prepare noise
            noise = self._prepare_noise(total_chunk_latents, new_latent_frames, seed)

            # 4t. Model inference
            with torch.no_grad():
                logging.info("[MODEL GENERATION] Using prompt = '%s'", session.prompt)
                video, latents = self.pipeline.inference( # here video and latent is [start_len + new_len]
                    noise=noise,
                    text_prompts=[session.prompt],
                    start_latents=start_latents,
                    return_latents=True,
                    start_frame_index=0,
                )

            # 5. How many visible frames came from the new latents?
            added_video_frames = self._latent_increment_to_video_frames(
                start_latents=start_len,
                added_latents=new_latent_frames,
            )
            if added_video_frames <= 0:
                added_video_frames = video.shape[1]

            # Take only the last added_video_frames from the decoded video
            chunk_video = video[:, -added_video_frames:, :, :, :]  # [1, T_new, C, H, W]
            chunk_np = chunk_video[0].permute(0, 2, 3, 1).cpu().numpy()  # [T_new, H, W, 3] in [0,1]
            chunk_latents = latents[:, -new_latent_frames:, :, :, :]  # [1, T_new_latent, C, H, W]

            # 6. Update counts and sliding tail
            self._update_tail_from_video(session, chunk_np, chunk_latents)
            session.total_frames += added_video_frames
            session.latent_frames += new_latent_frames

            # 7. Save and respond
            segment_url = self._video_chunk_to_path(session, chunk_np, "model")
            return self._segment_payload(session, segment_url, added_video_frames)

    # ----------------------------------------------------------------------
    # Session clearing / reset
    # ----------------------------------------------------------------------

    def clear_session_state(self, session: SessionState) -> None:
        """
        Clear disk outputs and in-memory history for a session, but keep the
        session object itself alive.
        """
        with self.pipeline_lock:
            # if session.storage_dir.exists():
            #     for entry in session.storage_dir.iterdir():
                    # if entry.is_dir():
                    #     shutil.rmtree(entry, ignore_errors=True)
                    # elif entry.is_file():
                    #     entry.unlink(missing_ok=True)

            session.latents = None
            session.video_tail = None
            session.latent_frames = 0
            session.total_frames = 0
            session.segment_index = 0
            session.last_frame_path = None

    def reset_session(self, session_id: str) -> None:
        """
        Delete a session entirely (disk + memory).
        """
        with self.pipeline_lock:
            if session_id not in self.sessions:
                raise HTTPException(status_code=404, detail="session not found")
            state = self.sessions.pop(session_id)
            # if state.storage_dir.exists():
            #     shutil.rmtree(state.storage_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SessionCreateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    width: int = 832
    height: int = 480
    fps: int = 15


class RunRequest(BaseModel):
    frames: int = 16
    seed: Optional[int] = None
    overlap_frames: Optional[int] = None


class SketchRequest(BaseModel):
    frames: List[str]
    overlap_frames: Optional[int] = None


class OverlapUpdateRequest(BaseModel):
    overlap_frames: int


class PromptRefineRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app: FastAPI | None = None
engine: Optional[InteractiveEngine] = None
prompt_refiner: Optional[PromptRefiner] = None
args: Optional[argparse.Namespace] = None

DEFAULT_CONFIG_PATH = "configs/wan_causal_ode_finetune.yaml"
DEFAULT_CHECKPOINT_PATH = "../pretrained/VideoSketcher/VideoSketcher-models/VideoSketcher_AR/AR_1.3B.pt"
DEFAULT_DEVICE = "cuda"
DEFAULT_TORCH_DTYPE = "bfloat16"
DEFAULT_MODEL_FRAME_LIMIT = 21
DEFAULT_USER_FRAME_LIMIT = 20
DEFAULT_OVERLAP_FRAMES = 9
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_REFINE_BACKEND = "openai"
DEFAULT_REFINE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REFINE_DEVICE = "cuda"
DEFAULT_REFINE_TORCH_DTYPE = "auto"

def build_app(engine: InteractiveEngine) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    assets_dir = CURRENT_DIR / "assets"
    sessions_dir = CURRENT_DIR / "sessions"
    app.mount("/static", StaticFiles(directory=assets_dir), name="static")
    app.mount("/sessions", StaticFiles(directory=sessions_dir), name="sessions")

    @app.get("/")
    def index():
        return FileResponse(assets_dir / "index.html")

    @app.post("/api/session")
    def create_session(payload: SessionCreateRequest):
        state = engine.create_session(
            prompt=payload.prompt,
            width=payload.width,
            height=payload.height,
            fps=payload.fps,
        )
        return {
            "session_id": state.session_id,
            "model_frame_limit": engine._latent_to_video_frames(state.model_limit),
            "user_frame_limit": engine._latent_to_video_frames(state.user_limit),
            "frame_alignment": engine.num_frame_per_block,
            # "overlap_frames": engine.overlap_frames,
        }

    @app.get("/api/session/{session_id}")
    def session_info(session_id: str):
        state = engine.get_session(session_id)
        return {
            "session_id": state.session_id,
            "prompt": state.prompt,
            "total_frames": state.total_frames,
            "model_frame_limit": engine._latent_to_video_frames(state.model_limit),
            "user_frame_limit": engine._latent_to_video_frames(state.user_limit),
            "frame_alignment": engine.num_frame_per_block,
            "overlap_frames": engine.overlap_frames,
            "last_frame_url": f"{state.public_dir}/last_frame.png" if state.last_frame_path else None,
        }

    @app.post("/api/overlap")
    def update_overlap(payload: OverlapUpdateRequest):
        value = engine.set_overlap_frames(payload.overlap_frames)
        return {"overlap_frames": value}

    @app.post("/api/session/{session_id}/prompt")
    async def update_prompt(session_id: str, request: Request):
        state = engine.get_session(session_id)

        data = await request.json()
        prompt = data["prompt"].strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        old_prompt = state.prompt
        engine.update_session_prompt(state, prompt)
        logging.info("[UPDATE PROMPT] session=%s '%s' -> '%s'", session_id, old_prompt, state.prompt)

        return {"status": "ok"}

    @app.post("/api/prompt/refine")
    def refine_prompt(payload: PromptRefineRequest):
        refined = prompt_refiner.refine(payload.prompt.strip())
        return {"prompt": refined}


    @app.post("/api/session/{session_id}/run")
    def run_model(session_id: str, payload: RunRequest):
        state = engine.get_session(session_id)
        if payload.overlap_frames is not None:
            engine.set_overlap_frames(payload.overlap_frames)
        return engine.generate_model_segment(state, payload.frames, payload.seed)

    @app.post("/api/session/{session_id}/sketch")
    def submit_sketch(session_id: str, payload: SketchRequest):
        state: SessionState = engine.get_session(session_id)
        if payload.overlap_frames is not None:
            engine.set_overlap_frames(payload.overlap_frames)
        return engine.append_user_segment(state, payload.frames)

    @app.delete("/api/session/{session_id}")
    def delete_session(session_id: str):
        engine.reset_session(session_id)
        return {"status": "reset"}

    @app.post("/api/session/{session_id}/clear")
    def clear_session(session_id: str):
        state = engine.get_session(session_id)
        engine.clear_session_state(state)
        return {"status": "cleared"}

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CausVid interactive demo server")
    parser.add_argument(
        "--config_path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default=DEFAULT_TORCH_DTYPE,
    )
    parser.add_argument("--model_frame_limit", type=int, default=DEFAULT_MODEL_FRAME_LIMIT)
    parser.add_argument("--user_frame_limit", type=int, default=DEFAULT_USER_FRAME_LIMIT)
    parser.add_argument("--overlap_frames", type=int, default=DEFAULT_OVERLAP_FRAMES)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no_load_pipe", action="store_true", default=False)
    parser.add_argument(
        "--refine_backend",
        type=str,
        choices=["openai", "qwen"],
        default=DEFAULT_REFINE_BACKEND,
    )
    parser.add_argument(
        "--refine_model_name",
        type=str,
        default=DEFAULT_REFINE_MODEL_NAME,
    )
    parser.add_argument(
        "--refine_device",
        type=str,
        default=DEFAULT_REFINE_DEVICE,
    )
    parser.add_argument(
        "--refine_torch_dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default=DEFAULT_REFINE_TORCH_DTYPE,
    )
    return parser.parse_args()


def build_engine_and_app(namespace: argparse.Namespace) -> tuple[InteractiveEngine, PromptRefiner, FastAPI]:
    eng = InteractiveEngine(
        config_path=namespace.config_path,
        checkpoint_path=namespace.checkpoint_path,
        device=namespace.device,
        torch_dtype=_to_torch_dtype(namespace.torch_dtype),
        model_limit=namespace.model_frame_limit,
        user_limit=namespace.user_frame_limit,
        overlap_frames=namespace.overlap_frames,
        no_load_pipe=namespace.no_load_pipe,
    )
    refiner = PromptRefiner(
        backend=namespace.refine_backend,
        model_name=namespace.refine_model_name,
        device=namespace.refine_device,
        torch_dtype=namespace.refine_torch_dtype,
    )
    application = build_app(eng)
    return eng, refiner, application


def _default_args() -> argparse.Namespace:
    return argparse.Namespace(
        config_path=DEFAULT_CONFIG_PATH,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        device=DEFAULT_DEVICE,
        torch_dtype=DEFAULT_TORCH_DTYPE,
        model_frame_limit=DEFAULT_MODEL_FRAME_LIMIT,
        user_frame_limit=DEFAULT_USER_FRAME_LIMIT,
        overlap_frames=DEFAULT_OVERLAP_FRAMES,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        refine_backend=DEFAULT_REFINE_BACKEND,
        refine_model_name=DEFAULT_REFINE_MODEL_NAME,
        refine_device=DEFAULT_REFINE_DEVICE,
        refine_torch_dtype=DEFAULT_REFINE_TORCH_DTYPE,
    )


def _maybe_auto_init() -> None:
    global args, engine, prompt_refiner, app
    if app is not None and engine is not None:
        return
    default_args = _default_args()
    default_args.no_load_pipe = False
    engine, prompt_refiner, app = build_engine_and_app(default_args)
    args = default_args


def main() -> None:
    global args, engine, prompt_refiner, app
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parsed_args = parse_args()
    args = parsed_args
    rebuild = True
    if args is not None and engine is not None and app is not None:
        rebuild = (
            args.config_path != parsed_args.config_path
            or args.checkpoint_path != parsed_args.checkpoint_path
            or args.device != parsed_args.device
            or args.torch_dtype != parsed_args.torch_dtype
            or args.model_frame_limit != parsed_args.model_frame_limit
            or args.user_frame_limit != parsed_args.user_frame_limit
            or args.overlap_frames != parsed_args.overlap_frames
            or args.refine_backend != parsed_args.refine_backend
            or args.refine_model_name != parsed_args.refine_model_name
            or args.refine_device != parsed_args.refine_device
            or args.refine_torch_dtype != parsed_args.refine_torch_dtype
        )
    if rebuild:
        engine, prompt_refiner, app = build_engine_and_app(parsed_args)
    if parsed_args.refine_backend == "qwen":
        prompt_refiner.warmup()
    args = parsed_args
    uvicorn.run(app, host=parsed_args.host, port=parsed_args.port)


if __name__ != "__main__":
    _maybe_auto_init()


if __name__ == "__main__":
    main()
