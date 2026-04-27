"""Pipeline configuration for M-106 Merlin abdominal-CT findings dataset.

Merlin (Stanford AIMI) is a foundation-model-grade abdominal CT dataset:
    - CT volumes:    s3://med-vr-datasets/M-106/merlin_ct/merlinabdominalctdataset/merlin_data/<study_id>.nii.gz
    - Metadata CSV:  s3://.../merlinabdominalctdataset/metadata.csv
        columns: study id, Age, Gender, Race, contrast, manufacturer,
                 manufacturermodelname, kvp, slicethickness, xraytubecurrent, phase

We adapt M-100 INSPECT (sister Stanford CT pipeline). For each study we render:
    first_video        = clean axial sweep through abdominal organs
    last_video         = annotated sweep with HU-based organ-window highlight
                         + reveal banner showing study metadata
    ground_truth_video = same as last_video

NOTE: GitHub repo name remains
``M-106_jigsaws_surgical_action_segmentation_data-pipeline`` (the website maps
by repo prefix, so we do not rename), but the actual ``domain`` (used for the
on-disk task folder name) is ``merlin_abdominal_ct_findings``.
"""
from pathlib import Path

from pydantic import Field

from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Dataset + rendering settings for Merlin abdominal CT pipeline."""

    domain: str = Field(default="merlin_abdominal_ct_findings")

    # Empty generator name → flat layout under data/questions/<domain>_task/<task_id>/
    generator: str = Field(default="")

    # Source layout in s3://med-vr-datasets/M-106/
    s3_bucket: str = Field(default="med-vr-datasets")
    s3_prefix: str = Field(
        default="M-106/merlin_ct/merlinabdominalctdataset/"
    )
    raw_dir: Path = Field(default=Path("raw"))

    # Abdominal soft-tissue window centred on liver/parenchyma (~+50, ±200 HU).
    window_level: int = Field(default=50)
    window_width: int = Field(default=400)

    # Video: axial sweep through the central abdomen (organs).
    fps: int = Field(default=8)
    num_frames: int = Field(default=28)

    # Frame output size.
    frame_height: int = Field(default=512)
    ehr_panel_width: int = Field(default=360)

    # Organ highlight (warm tint, 35% opacity) on annotated video.
    pe_alpha: float = Field(default=0.35)

    # Default cap (overridden by --num-samples on the CLI).
    max_samples: int = Field(default=300, ge=1)
