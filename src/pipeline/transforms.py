"""Transforms for M-106 Merlin abdominal-CT findings pipeline.

Pipeline = abdominal CT volume + study metadata → axial sweep video (+ side panel).

Rendering contract:

    frame = [ CT_axial_slice (square)  ||  Clinical / acquisition panel ]

    first_video        = axial sweep through central abdomen, NO highlight.
    last_video         = same sweep with warm overlay on parenchymal (organ)
                         HU range and a reveal banner showing study metadata.
    ground_truth_video = same as last_video.

Merlin ships abdominal CT volumes plus per-study acquisition metadata
(age/gender/race/contrast/manufacturer/phase). It does NOT ship per-voxel
labels or radiology reports in this bundle, so the "annotation" is a
visual organ-window highlight (informative for the model) and the side
panel surfaces patient + scanner context.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────────
#  CT windowing
# ──────────────────────────────────────────────────────────────────────────────

def window_slice(hu: np.ndarray, wl: int, ww: int) -> np.ndarray:
    """Apply HU window → uint8 grayscale."""
    vmin = wl - ww / 2.0
    vmax = wl + ww / 2.0
    img = np.clip((hu.astype(np.float32) - vmin) / (vmax - vmin) * 255.0, 0, 255)
    return img.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
#  Axial-sweep slice selection
# ──────────────────────────────────────────────────────────────────────────────

def pick_sweep_indices(num_slices: int, num_frames: int) -> List[int]:
    """Pick N evenly-spaced slices from the central ~60% of the volume.

    For abdominal CT this hits liver / spleen / kidneys / pancreas region.
    """
    if num_slices <= 0:
        return []
    lo = int(num_slices * 0.20)
    hi = max(lo + 1, int(num_slices * 0.80))
    hi = min(hi, num_slices)
    if hi - lo < num_frames:
        lo, hi = 0, num_slices
    idxs = np.linspace(lo, hi - 1, num=min(num_frames, hi - lo)).round().astype(int)
    seen = set()
    out: List[int] = []
    for i in idxs:
        ii = int(i)
        if ii in seen:
            continue
        seen.add(ii)
        out.append(ii)
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Organ-region heuristic mask (Merlin has no per-voxel labels)
# ──────────────────────────────────────────────────────────────────────────────

def pe_heuristic_mask(hu_slice: np.ndarray) -> np.ndarray:
    """Heuristic abdominal-organ highlight: parenchymal HU (~+30..+80) inside
    the central abdomen ROI. Returns a bool mask.

    Function name kept as ``pe_heuristic_mask`` for code-shape compatibility
    with the M-100 sister pipeline; semantics here are "organ parenchyma".
    """
    h, w = hu_slice.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = min(h, w) * 0.38
    roi = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius ** 2)
    # Soft-tissue parenchyma window (liver, spleen, kidney, pancreas).
    parenchyma = (hu_slice >= 30) & (hu_slice <= 80)
    return roi & parenchyma


def colorize_ct_slice(
    hu: np.ndarray,
    wl: int,
    ww: int,
    pe_mask: Optional[np.ndarray] = None,
    alpha: float = 0.35,
) -> np.ndarray:
    """Render CT slice as RGB; warm-tint parenchymal organ regions (optional)."""
    gray = window_slice(hu, wl, ww)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    # Faint cool tint on vasculature (>+90 HU contrast-enhanced vessels).
    vessel_mask = (hu >= 90) & (hu < 200)
    if vessel_mask.any():
        cool = np.array([60, 140, 230], dtype=np.float32)
        base = rgb[vessel_mask].astype(np.float32)
        rgb[vessel_mask] = (base * 0.85 + cool * 0.15).astype(np.uint8)

    if pe_mask is not None and pe_mask.any():
        warm = np.array([255, 140, 30], dtype=np.float32)  # amber/organ overlay
        base = rgb[pe_mask].astype(np.float32)
        rgb[pe_mask] = (base * (1 - alpha) + warm * alpha).astype(np.uint8)

    return rgb


# ──────────────────────────────────────────────────────────────────────────────
#  Side panel (study + acquisition metadata)
# ──────────────────────────────────────────────────────────────────────────────

_FONT_CACHE: dict = {}


def _load_font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _fmt_age(age: str) -> str:
    if not age:
        return "?"
    try:
        return f"{float(age):.0f} yr"
    except (TypeError, ValueError):
        return age


def _fmt_phase(phase: str) -> str:
    if not phase:
        return "—"
    return phase.replace("_", " ")


def _fmt_contrast(contrast: str) -> str:
    if not contrast:
        return "—"
    s = contrast.strip().lower()
    if s in ("true", "1", "yes", "y"):
        return "Contrast-enhanced"
    if s in ("false", "0", "no", "n"):
        return "Non-contrast"
    return contrast


def render_ehr_panel(
    raw: dict,
    width: int,
    height: int,
    title: str,
    reveal: bool = False,
) -> np.ndarray:
    """Render a clinical / acquisition side panel (dark bg, white text)."""
    img = Image.new("RGB", (width, height), (18, 20, 26))
    draw = ImageDraw.Draw(img)

    font_title = _load_font(20)
    font_label = _load_font(14)
    font_value = _load_font(14)
    font_small = _load_font(12)

    pad = 16
    y = pad
    draw.text((pad, y), title, font=font_title, fill=(220, 230, 240))
    y += 28
    draw.line((pad, y, width - pad, y), fill=(80, 90, 110), width=1)
    y += 10

    label_col = pad
    value_col = pad + 130

    rows: List[Tuple[str, str]] = [
        ("Study ID", str(raw.get("study_id", "?"))),
        ("Age", _fmt_age(raw.get("age", ""))),
        ("Gender", raw.get("gender", "") or "—"),
        ("Race", raw.get("race", "") or "—"),
        ("", ""),
        ("— Acquisition —", ""),
        ("Contrast", _fmt_contrast(raw.get("contrast", ""))),
        ("Phase", _fmt_phase(raw.get("phase", ""))),
        ("Manufacturer", raw.get("manufacturer", "") or "—"),
        ("Model", raw.get("model", "") or "—"),
        ("kVp", raw.get("kvp", "") or "—"),
        ("Slice (mm)", raw.get("slice_mm", "") or "—"),
        ("Tube (mA)", raw.get("tube_current", "") or "—"),
    ]
    for label, value in rows:
        if not label and not value:
            y += 6
            continue
        if label.startswith("—"):
            draw.text((label_col, y), label, font=font_label, fill=(140, 180, 220))
        else:
            draw.text((label_col, y), label, font=font_label, fill=(180, 190, 200))
            # Truncate long values
            v = value if len(value) <= 22 else value[:21] + "…"
            draw.text((value_col, y), v, font=font_value, fill=(240, 245, 250))
        y += 18

    # Footer
    y_foot = height - pad - 40
    draw.line((pad, y_foot, width - pad, y_foot), fill=(80, 90, 110), width=1)
    draw.text(
        (pad, y_foot + 4),
        "Merlin  (Stanford AIMI, 2024)",
        font=font_small,
        fill=(140, 150, 170),
    )
    draw.text(
        (pad, y_foot + 18),
        "Abdominal CT  •  axial sweep",
        font=font_small,
        fill=(140, 150, 170),
    )

    if reveal:
        # Reveal banner: amber/warm to signal organ-window highlight is on.
        banner = (200, 120, 30)
        text = "ORGAN OVERLAY ON  (parenchyma)"
        bh = 30
        draw.rectangle((0, height - bh, width, height), fill=banner)
        draw.text((pad, height - bh + 7), text, font=font_label, fill=(250, 250, 255))

    return np.array(img, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
#  Frame composition + video writing
# ──────────────────────────────────────────────────────────────────────────────

def compose_frame(ct_rgb: np.ndarray, panel_rgb: np.ndarray) -> np.ndarray:
    """Concatenate CT slice and side panel side by side (same height)."""
    h = ct_rgb.shape[0]
    if panel_rgb.shape[0] != h:
        pil = Image.fromarray(panel_rgb)
        pil = pil.resize((panel_rgb.shape[1], h), Image.Resampling.LANCZOS)
        panel_rgb = np.array(pil)
    out = np.concatenate([ct_rgb, panel_rgb], axis=1)
    H, W = out.shape[:2]
    if W % 2:
        out = out[:, :-1]
    if H % 2:
        out = out[:-1, :]
    return out


def write_mp4(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
    """Write RGB frames to an H.264 mp4 via ffmpeg piping."""
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2 = w - (w % 2)
    h2 = h - (h % 2)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            if f.shape[:2] != (h, w):
                pil = Image.fromarray(f)
                pil = pil.resize((w, h), Image.Resampling.LANCZOS)
                f = np.array(pil)
            p.stdin.write(f.astype(np.uint8).tobytes())
    finally:
        p.stdin.close()
        p.wait()


def resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB frame to size x size via PIL LANCZOS."""
    pil = Image.fromarray(rgb)
    pil = pil.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(pil)
