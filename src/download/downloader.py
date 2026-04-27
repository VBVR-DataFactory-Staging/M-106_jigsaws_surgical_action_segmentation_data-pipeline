"""Merlin M-106 downloader.

Merlin layout under
``s3://med-vr-datasets/M-106/merlin_ct/merlinabdominalctdataset/``::

    metadata.csv                   # study id, Age, Gender, Race, contrast,
                                   # manufacturer, manufacturermodelname,
                                   # kvp, slicethickness, xraytubecurrent, phase
    merlin_data/
        <study_id>.nii.gz          # ~2668 abdominal CT volumes (NIfTI, int16 HU)

We can't pull every CT (309 GB). Instead:

  1. Pull the small metadata.csv once into ``raw/``.
  2. Build a ``study_id → s3_key`` lookup over merlin_data/.
  3. Yield one raw sample per metadata row that has a matching CT, pulling the
     volume on demand and deleting it after use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import boto3
import pandas as pd


_META_NAME = "metadata.csv"


def _ensure_metadata(bucket: str, prefix: str, raw_dir: Path) -> pd.DataFrame:
    """Download metadata.csv if not cached; return DataFrame."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = raw_dir / _META_NAME
    if not local.exists() or local.stat().st_size == 0:
        key = f"{prefix}{_META_NAME}"
        print(f"  [download] s3://{bucket}/{key} -> {local}", flush=True)
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(local))
    df = pd.read_csv(local)
    # Normalise column name "study id" → "study_id"
    df = df.rename(columns={"study id": "study_id"})
    df["study_id"] = df["study_id"].astype(str)
    return df


def _build_image_index(bucket: str, prefix: str, raw_dir: Path) -> dict:
    """Return {study_id(str) -> s3_key(str)} for all merlin_data/*.nii.gz files.

    Cached to ``raw/ct_index.csv`` so we don't re-list 2668 keys every run.
    """
    cache = raw_dir / "ct_index.csv"
    if cache.exists() and cache.stat().st_size > 0:
        df = pd.read_csv(cache)
        return dict(zip(df["study_id"].astype(str), df["s3_key"].astype(str)))

    s3 = boto3.client("s3")
    image_to_key: dict = {}
    sub_prefix = f"{prefix}merlin_data/"
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=sub_prefix)
    for page in pages:
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            stem = key.rsplit("/", 1)[-1]
            if not stem.endswith(".nii.gz"):
                continue
            study_id = stem[: -len(".nii.gz")]
            image_to_key[study_id] = key

    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"study_id": i, "s3_key": k} for i, k in sorted(image_to_key.items())]
    pd.DataFrame(rows).to_csv(cache, index=False)
    print(f"  [index] built ct_index.csv with {len(image_to_key)} CT volumes", flush=True)
    return image_to_key


def _safe_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()


def run_download(config) -> Iterator[dict]:
    """Yield one raw sample per Merlin study (where a CT volume exists).

    Each raw sample dict::

        {
            "study_id":      str,
            "age":           str,
            "gender":        str,
            "race":          str,
            "contrast":      str,
            "manufacturer":  str,
            "model":         str,
            "kvp":           str,
            "slice_mm":      str,
            "tube_current":  str,
            "phase":         str,
            "s3_bucket":     str,
            "s3_key":        str,
        }
    """
    bucket = config.s3_bucket
    prefix = config.s3_prefix
    raw_dir = Path(config.raw_dir)

    meta = _ensure_metadata(bucket, prefix, raw_dir)
    image_index = _build_image_index(bucket, prefix, raw_dir)

    # Stable order = sort by study_id; first 300 will be deterministic.
    meta_sorted = meta.sort_values("study_id").reset_index(drop=True)

    for _, row in meta_sorted.iterrows():
        study_id = str(row["study_id"])
        s3_key = image_index.get(study_id)
        if not s3_key:
            continue
        yield {
            "study_id": study_id,
            "age": _safe_str(row.get("Age")),
            "gender": _safe_str(row.get("Gender")),
            "race": _safe_str(row.get("Race")),
            "contrast": _safe_str(row.get("contrast")),
            "manufacturer": _safe_str(row.get("manufacturer")),
            "model": _safe_str(row.get("manufacturermodelname")),
            "kvp": _safe_str(row.get("kvp")),
            "slice_mm": _safe_str(row.get("slicethickness")),
            "tube_current": _safe_str(row.get("xraytubecurrent")),
            "phase": _safe_str(row.get("phase")),
            "s3_bucket": bucket,
            "s3_key": s3_key,
        }


def create_downloader(config) -> "TaskDownloader":
    return TaskDownloader(config)


class TaskDownloader:
    """Streams Merlin raw samples from S3 one study at a time."""

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config.raw_dir)

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for sample in run_download(self.config):
            yield sample
            count += 1
            if limit is not None and count >= limit:
                break


def fetch_ct_volume(bucket: str, key: str, cache_dir: Path):
    """Download one CT NIfTI volume, load as numpy (Z, H, W) int16, then delete file.

    Returns ``None`` if download or decode fails.
    """
    import nibabel as nib  # lazy import — only needed when actually fetching CTs
    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / Path(key).name
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, key, str(local))
    except Exception as e:
        print(f"  [download] failed {key}: {e}", flush=True)
        return None

    arr = None
    try:
        img = nib.load(str(local))
        data = img.get_fdata(caching="unchanged")
        # NIfTI conv: data is (X, Y, Z). Convert to (Z, H, W) where Z=axial slice.
        if data.ndim == 3:
            arr = np.transpose(data, (2, 1, 0)).astype(np.int16)
        elif data.ndim == 4:
            arr = np.transpose(data[..., 0], (2, 1, 0)).astype(np.int16)
    except Exception as e:
        print(f"  [load] failed {local}: {e}", flush=True)
        arr = None
    finally:
        try:
            local.unlink(missing_ok=True)
        except Exception:
            pass
    return arr
