"""Raw downloader for M-106 JIGSAWS surgical action segmentation.

Mirrors s3://med-vr-datasets/M-106/JIGSAWS/ → local raw/ via ``aws s3 sync``
(NOT HTTP). Yields one raw-sample dict per episode for the pipeline.

Contract:
    * ``aws s3 sync`` (subprocess) — never urllib/requests/boto download_file.
    * Stream-friendly: idempotent, resumable, no partial corruption.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List, Optional

# Live-stream stdout so EC2 log tail sees progress without buffering delay.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _aws_sync(s3_uri: str, local_dir: Path,
              exclude: Optional[List[str]] = None,
              include: Optional[List[str]] = None) -> None:
    """Run ``aws s3 sync`` from S3 to a local dir."""
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", s3_uri, str(local_dir)]
    if exclude:
        for e in exclude:
            cmd += ["--exclude", e]
    if include:
        for i in include:
            cmd += ["--include", i]
    print(f"[M-106] aws s3 sync {s3_uri} -> {local_dir}", flush=True)
    res = subprocess.run(cmd, check=False, text=True)
    if res.returncode != 0:
        print(f"[M-106][warn] aws s3 sync exit={res.returncode} (continuing)",
              flush=True)


def _ensure_raw(config) -> Path:
    """Sync the JIGSAWS LeRobot dataset from S3 to ``config.raw_dir``.

    Layout we expect after sync::

        raw/
          meta/{info.json, tasks.jsonl, episodes.jsonl, ...}
          data/chunk-000/episode_NNNNNN.parquet
          videos/chunk-000/video.exterior_image_1_left/episode_NNNNNN.mp4
    """
    raw_dir = Path(config.raw_dir)
    s3_uri = f"s3://{config.s3_bucket}/{config.s3_prefix.rstrip('/')}/"
    # Skip wrist/exterior_2 cameras to halve disk + transfer. Keep cam 1.
    _aws_sync(
        s3_uri, raw_dir,
        exclude=[
            "*video.exterior_image_2_left*",
            "*video.wrist_image_left*",
        ],
    )
    return raw_dir


def _list_episodes(raw_dir: Path) -> List[int]:
    """Return sorted episode indices that have BOTH a parquet and a video."""
    data_dir = raw_dir / "data" / "chunk-000"
    vid_dir = raw_dir / "videos" / "chunk-000" / "video.exterior_image_1_left"
    eps: List[int] = []
    if not data_dir.exists() or not vid_dir.exists():
        return eps
    for p in sorted(data_dir.glob("episode_*.parquet")):
        try:
            idx = int(p.stem.split("_")[1])
        except (ValueError, IndexError):
            continue
        if (vid_dir / f"episode_{idx:06d}.mp4").exists():
            eps.append(idx)
    return eps


def run_download(config) -> Iterator[dict]:
    """Yield one raw-sample dict per JIGSAWS episode."""
    raw_dir = _ensure_raw(config)
    eps = _list_episodes(raw_dir)
    print(f"[M-106] discovered {len(eps)} JIGSAWS episodes in {raw_dir}",
          flush=True)
    if not eps:
        return

    # Optional per-episode task list from meta/episodes.jsonl.
    ep_meta: dict = {}
    meta_eps = raw_dir / "meta" / "episodes.jsonl"
    if meta_eps.exists():
        try:
            with meta_eps.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ep_meta[int(rec["episode_index"])] = rec
        except Exception as e:
            print(f"[M-106][warn] could not parse episodes.jsonl: {e}",
                  flush=True)

    for idx in eps:
        parquet_path = raw_dir / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        video_path = (
            raw_dir / "videos" / "chunk-000"
            / "video.exterior_image_1_left" / f"episode_{idx:06d}.mp4"
        )
        yield {
            "episode_index": idx,
            "parquet_path": str(parquet_path),
            "video_path": str(video_path),
            "episode_meta": ep_meta.get(idx, {}),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Legacy compatibility shims (Merlin-era; not used by the new pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_ct_volume(*args, **kwargs):
    """Deprecated. M-106 is now JIGSAWS; CT helpers removed."""
    raise NotImplementedError("M-106 is JIGSAWS; fetch_ct_volume removed.")


class TaskDownloader:
    def __init__(self, config):
        self.config = config

    def iter_samples(self, limit=None):
        n = 0
        for s in run_download(self.config):
            yield s
            n += 1
            if limit is not None and n >= limit:
                return


def create_downloader(config):
    return TaskDownloader(config)
