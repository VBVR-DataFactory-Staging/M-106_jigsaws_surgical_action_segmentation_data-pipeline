"""Pipeline configuration for M-106 JIGSAWS surgical action segmentation.

Source layout in s3://med-vr-datasets/M-106/JIGSAWS/ (LeRobot v2.0 format,
mirrored from huggingface.co/datasets/Potestates/jigsaws-lerobot)::

    JIGSAWS/
      meta/
        info.json
        tasks.jsonl                # task_index -> action label
        episodes.jsonl             # per-episode task list + length
      data/chunk-000/
        episode_000000.parquet     # per-frame: annotation.language.language_instruction
        episode_000001.parquet
        ...
      videos/chunk-000/video.exterior_image_1_left/
        episode_000000.mp4         # robotic surgery RGB video (30 fps)
        episode_000001.mp4
        ...

Each episode = one task sample. The pipeline overlays a per-frame action-label
banner (color-coded by action) onto the surgical video so transitions between
gestures (G1..G15-like steps) are visually obvious.
"""
from pathlib import Path

from pydantic import Field

from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Settings for the JIGSAWS surgical action-segmentation pipeline."""

    domain: str = Field(default="jigsaws_surgical_action_segmentation")

    # Empty generator name -> flat layout under data/questions/<domain>_task/.
    generator: str = Field(default="")

    # Source layout in s3://med-vr-datasets/M-106/JIGSAWS/ (mirror of HF dataset).
    s3_bucket: str = Field(default="med-vr-datasets")
    s3_prefix: str = Field(default="M-106/JIGSAWS/")
    raw_dir: Path = Field(default=Path("raw"))

    # Video layout: render at native frame size, then resize to a stable
    # square so the website can display consistently.
    fps: int = Field(default=10)              # downsample from 30 fps source
    frame_size: int = Field(default=480)      # output square size
    banner_height: int = Field(default=60)    # action banner above video
    max_frames: int = Field(default=240)      # cap per-episode frames (24s @ 10fps)

    # Optional per-episode index offset and cap.
    start_index: int = Field(default=0)
    max_samples: int = Field(default=300, ge=1)
