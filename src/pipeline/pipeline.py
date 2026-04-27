"""TaskPipeline for M-106 JIGSAWS surgical action segmentation.

Input  : LeRobot-format JIGSAWS at s3://med-vr-datasets/M-106/JIGSAWS/.
         103 episodes, each = robotic-surgery RGB clip (30 fps) + per-frame
         action label in episode_NNNNNN.parquet under
         ``annotation.language.language_instruction``.

Output (per episode -> one TaskSample):

    first_video        : surgical clip with NO action banner ("plain view").
    last_video         : same clip with a colored per-frame action banner
                         on top — color encodes the gesture/action so
                         transitions between actions are visually obvious.
    ground_truth.mp4   : same as last_video.
    first_frame.png    : middle frame of the plain clip.
    final_frame.png    : middle frame of the annotated clip (banner visible).

We intentionally:
    * Use ``aws s3 sync`` (subprocess) to fetch raw bytes — never urllib.
    * Do NOT horizontally flip — robotic instruments are asymmetric (PSM1 vs
      PSM2 have distinct left/right roles).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import pyarrow.parquet as pq

from core.pipeline import BasePipeline, SampleProcessor, TaskSample

from .config import TaskConfig
from .transforms import (
    build_action_palette,
    compose_frame,
    read_video_frames,
    render_action_banner,
    resize_square,
    write_mp4,
)
from src.download.downloader import run_download


# Live-stream stdout so EC2 log tail sees progress without buffering delay.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


PROMPT = (
    "This is a robotic-surgery training clip from the JIGSAWS dataset (Suturing, "
    "Knot-Tying, or Needle-Passing). Watch the dual-PSM da Vinci instruments "
    "perform a sequence of fine-grained gestures (e.g. position needle tip, "
    "push needle through tissue, pull suture, transfer needle, tie knot). "
    "Identify the action being executed in each segment and the moments of "
    "transition between actions. The annotated overlay banners encode the "
    "ground-truth action label per frame — each distinct action gets a "
    "consistent color, and transitions are highlighted with a white border."
)


TMP_DIR = Path("_tmp")


def _read_actions(parquet_path: Path) -> List[str]:
    """Return one action label per frame (length = video frame count).

    Parquet column ``annotation.language.language_instruction`` has one row per
    source-video frame at 30 fps.
    """
    try:
        table = pq.read_table(
            str(parquet_path),
            columns=["frame_index", "annotation.language.language_instruction"],
        )
    except Exception as exc:
        print(f"[M-106][warn] parquet read failed {parquet_path}: {exc}",
              flush=True)
        return []
    df = table.to_pandas().sort_values("frame_index")
    return df["annotation.language.language_instruction"].astype(str).tolist()


class TaskPipeline(BasePipeline):
    """JIGSAWS surgical action segmentation pipeline."""

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

    # ── 2) Process one episode ───────────────────────────────────────────
    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        cfg = self.task_config
        ep_idx = int(raw_sample["episode_index"])
        global_idx = int(getattr(cfg, "start_index", 0)) + idx
        task_id = f"{cfg.domain}_{global_idx:05d}"

        parquet_path = Path(raw_sample["parquet_path"])
        video_path = Path(raw_sample["video_path"])
        if not parquet_path.exists() or not video_path.exists():
            print(f"[M-106] ep{ep_idx:03d}: missing parquet or video, skipping",
                  flush=True)
            return None

        # Per-frame action labels (one per source-video frame at 30 fps).
        actions = _read_actions(parquet_path)
        if not actions:
            print(f"[M-106] ep{ep_idx:03d}: no actions, skipping", flush=True)
            return None

        # Decode video at a stride so output is at cfg.fps from a 30 fps source.
        # JIGSAWS source = 30 fps. cfg.fps = 10 → stride 3.
        src_fps = 30
        stride = max(1, src_fps // max(1, cfg.fps))
        frames = read_video_frames(video_path, max_frames=cfg.max_frames,
                                   stride=stride)
        if not frames:
            print(f"[M-106] ep{ep_idx:03d}: no decoded frames, skipping",
                  flush=True)
            return None

        # Sub-sample action labels at the same stride so they align 1:1 with frames.
        sampled_actions: List[str] = []
        for i in range(len(frames)):
            src_idx = min(i * stride, len(actions) - 1)
            sampled_actions.append(actions[src_idx])

        # Build palette ordered by first-appearance for stable, distinct colors.
        unique_in_order: List[str] = []
        for a in sampled_actions:
            if a not in unique_in_order:
                unique_in_order.append(a)
        palette = build_action_palette(unique_in_order)

        # Assemble plain (clean) and annotated (banner) frame lists.
        size = cfg.frame_size
        banner_h = cfg.banner_height
        clean_frames: List[np.ndarray] = []
        annotated_frames: List[np.ndarray] = []
        last_action: Optional[str] = None
        n_total = len(frames)

        for i, (frame, action) in enumerate(zip(frames, sampled_actions)):
            sq = resize_square(frame, size)
            clean_frames.append(sq)

            color = palette.get(action, (160, 160, 160))
            progress = (i + 1) / max(1, n_total)
            transition = (last_action is not None and action != last_action)
            banner = render_action_banner(
                width=size, height=banner_h,
                action=action, color=color,
                progress=progress, transition=transition,
            )
            annotated_frames.append(compose_frame(banner, sq))
            last_action = action

        if not clean_frames or not annotated_frames:
            return None

        # Pad clean frames to match annotated height (banner above black strip).
        ann_h, ann_w = annotated_frames[0].shape[:2]
        pad_h = ann_h - clean_frames[0].shape[0]
        if pad_h > 0:
            black = np.zeros((pad_h, ann_w, 3), dtype=np.uint8)
            clean_padded = []
            for f in clean_frames:
                if f.shape[1] != ann_w:
                    from PIL import Image as _Im
                    pil = _Im.fromarray(f).resize((ann_w, f.shape[0]),
                                                  _Im.Resampling.LANCZOS)
                    f = np.array(pil)
                clean_padded.append(np.concatenate([black, f], axis=0))
            clean_frames = clean_padded

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = TMP_DIR / task_id
        tmp.mkdir(parents=True, exist_ok=True)

        first_video = tmp / "first_video.mp4"
        last_video = tmp / "last_video.mp4"
        gt_video = tmp / "ground_truth.mp4"
        write_mp4(clean_frames, first_video, cfg.fps)
        write_mp4(annotated_frames, last_video, cfg.fps)
        write_mp4(annotated_frames, gt_video, cfg.fps)

        mid = len(clean_frames) // 2
        first_rgb = clean_frames[mid]
        final_rgb = annotated_frames[mid]

        # Action transitions for metadata.
        transitions: List[dict] = []
        prev = None
        for i, a in enumerate(sampled_actions):
            if a != prev:
                transitions.append({"frame": i, "action": a})
                prev = a

        ep_task_list = raw_sample.get("episode_meta", {}).get("tasks") or []
        metadata = {
            "task_id": task_id,
            "source_dataset": "JIGSAWS (LeRobot mirror)",
            "episode_index": ep_idx,
            "num_frames": len(clean_frames),
            "fps": cfg.fps,
            "src_fps": src_fps,
            "stride": stride,
            "frame_size": cfg.frame_size,
            "banner_height": cfg.banner_height,
            "unique_actions": unique_in_order,
            "action_palette": {
                a: list(palette[a]) for a in unique_in_order
            },
            "transitions": transitions,
            "episode_task_list": ep_task_list,
        }

        if (idx + 1) % 5 == 0 or idx < 3:
            print(
                f"[M-106] sample {idx:05d} done — ep{ep_idx:03d} "
                f"frames={len(clean_frames)} actions={len(unique_in_order)} "
                f"transitions={len(transitions)}",
                flush=True,
            )

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
        print(f"[M-106] generated {len(samples)} JIGSAWS samples total",
              flush=True)
        return samples
