"""Pipeline for M-106 Merlin abdominal-CT findings.

Adapted from M-100 INSPECT (sister Stanford CT pipeline). Per-study workflow:

  1. Stream one Merlin abdominal NIfTI volume from S3 (delete after load).
  2. Pick an axial sweep through the central abdomen (~organ-rich slices).
  3. Render a study/acquisition side panel from metadata.csv row.
  4. Compose CT-slice + side-panel frames (clean + organ-overlay variants).
  5. Write first / last / ground_truth mp4s.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from core.pipeline import BasePipeline, SampleProcessor, TaskSample

from .config import TaskConfig
from .transforms import (
    colorize_ct_slice,
    compose_frame,
    pe_heuristic_mask,
    pick_sweep_indices,
    render_ehr_panel,
    resize_square,
    write_mp4,
)
from src.download.downloader import fetch_ct_volume, run_download


PROMPT = (
    "This is an abdominal CT axial sweep from the Merlin dataset (Stanford "
    "AIMI). Identify the dominant abdominal organs visible in the central "
    "slices and any salient finding (e.g. mass, fluid, parenchymal "
    "abnormality). Use the side panel for patient + acquisition context "
    "(contrast phase, slice thickness, scanner)."
)


TMP_DIR = Path("_tmp")
CT_CACHE_DIR = Path("raw/_ct_cache")


class TaskPipeline(BasePipeline):
    """Merlin abdominal CT pipeline."""

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.task_config = config

    # ── 1) Download ──────────────────────────────────────────────────────
    def download(self) -> Iterator[dict]:
        cap = getattr(self.task_config, "max_samples", None)
        ns = getattr(self.task_config, "num_samples", None)
        limit = ns if ns is not None else cap
        n = 0
        for s in run_download(self.task_config):
            yield s
            n += 1
            if limit is not None and n >= limit:
                break

    # ── 2) Process one study ─────────────────────────────────────────────
    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        cfg = self.task_config
        study_id = str(raw_sample.get("study_id", "unknown"))
        global_idx = int(getattr(cfg, "start_index", 0)) + idx
        task_id = f"{cfg.domain}_{global_idx:05d}"

        volume = fetch_ct_volume(
            raw_sample["s3_bucket"],
            raw_sample["s3_key"],
            CT_CACHE_DIR,
        )
        if volume is None or volume.ndim != 3:
            return None

        # Merlin volumes are int16 HU, typically (Z, 512, 512).
        if volume.shape[0] < 8 or volume.shape[1] < 64 or volume.shape[2] < 64:
            return None

        sweep_idx = pick_sweep_indices(volume.shape[0], cfg.num_frames)
        if not sweep_idx:
            return None

        panel_clean = render_ehr_panel(
            raw=raw_sample,
            width=cfg.ehr_panel_width,
            height=cfg.frame_height,
            title="Study context",
            reveal=False,
        )
        panel_reveal = render_ehr_panel(
            raw=raw_sample,
            width=cfg.ehr_panel_width,
            height=cfg.frame_height,
            title="Study context",
            reveal=True,
        )

        clean_frames: List[np.ndarray] = []
        annotated_frames: List[np.ndarray] = []

        for s_i in sweep_idx:
            hu = volume[s_i].astype(np.int16)

            clean_rgb = colorize_ct_slice(hu, cfg.window_level, cfg.window_width, pe_mask=None)
            clean_rgb = resize_square(clean_rgb, cfg.frame_height)
            clean_frames.append(compose_frame(clean_rgb, panel_clean))

            mask = pe_heuristic_mask(hu)
            ann_rgb = colorize_ct_slice(
                hu, cfg.window_level, cfg.window_width,
                pe_mask=mask, alpha=cfg.pe_alpha,
            )
            ann_rgb = resize_square(ann_rgb, cfg.frame_height)
            annotated_frames.append(compose_frame(ann_rgb, panel_reveal))

        if not clean_frames or not annotated_frames:
            return None

        # Free the volume before writing videos (bound peak RAM).
        del volume

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = TMP_DIR / task_id
        tmp.mkdir(parents=True, exist_ok=True)

        fps = cfg.fps
        first_video = tmp / "first_video.mp4"
        last_video = tmp / "last_video.mp4"
        gt_video = tmp / "ground_truth.mp4"

        write_mp4(clean_frames, first_video, fps)
        write_mp4(annotated_frames, last_video, fps)
        write_mp4(annotated_frames, gt_video, fps)

        first_rgb = clean_frames[len(clean_frames) // 2]
        final_rgb = annotated_frames[-1]

        metadata = {
            "task_id": task_id,
            "source_dataset": "Merlin",
            "study_id": study_id,
            "age": raw_sample.get("age"),
            "gender": raw_sample.get("gender"),
            "race": raw_sample.get("race"),
            "contrast": raw_sample.get("contrast"),
            "manufacturer": raw_sample.get("manufacturer"),
            "model": raw_sample.get("model"),
            "kvp": raw_sample.get("kvp"),
            "slice_mm": raw_sample.get("slice_mm"),
            "tube_current": raw_sample.get("tube_current"),
            "phase": raw_sample.get("phase"),
            "num_slices": int(len(sweep_idx)),
            "fps": fps,
            "window_level": cfg.window_level,
            "window_width": cfg.window_width,
        }

        return SampleProcessor.build_sample(
            task_id=task_id,
            domain=cfg.domain,
            first_image=first_rgb,
            prompt=PROMPT,
            final_image=final_rgb,
            first_video=str(first_video),
            last_video=str(last_video),
            ground_truth_video=str(gt_video),
            metadata=metadata,
        )

    # ── 3) Run (with tmp cleanup) ────────────────────────────────────────
    def run(self) -> List[TaskSample]:
        try:
            samples = super().run()
        finally:
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR, ignore_errors=True)
            if CT_CACHE_DIR.exists():
                shutil.rmtree(CT_CACHE_DIR, ignore_errors=True)
        return samples
