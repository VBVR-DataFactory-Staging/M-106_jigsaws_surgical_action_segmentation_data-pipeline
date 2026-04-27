"""Rendering primitives for M-106 JIGSAWS action-segmentation pipeline.

Each frame = [ action banner (color-coded) | surgical video frame (square) ]
stacked vertically. A consistent palette assigns a distinct color per action
label so transitions between gestures are visually obvious in the output mp4.
"""
from __future__ import annotations

import colorsys
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────────
#  Action -> color palette (deterministic, distinct, contrasting)
# ──────────────────────────────────────────────────────────────────────────────

# Hand-picked HSL hues evenly spaced on the wheel (avoids reds-only collisions).
_BASE_PALETTE: List[Tuple[int, int, int]] = [
    (255, 196,  37),   # G1  amber/yellow
    ( 47, 134, 255),   # G2  azure blue
    ( 56, 200, 110),   # G3  emerald green
    (255, 105, 180),   # G4  hot pink
    (170,  90, 220),   # G5  purple
    (255, 130,  60),   # G6  orange
    ( 80, 220, 220),   # G7  cyan
    (220,  50,  50),   # G8  red
    (130, 200,  60),   # G9  lime
    (240, 200, 110),   # G10 light gold
    ( 50, 160, 200),   # G11 teal
    (200,  90, 140),   # G12 magenta
    (160, 130,  90),   # G13 taupe
    (110, 200, 170),   # G14 mint
    (220,  60, 100),   # G15 rose-red
]


def build_action_palette(actions: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """Assign a deterministic color to each unique action label.

    First N labels go to the curated palette; if there are more, additional
    colors are derived deterministically from a hash of the label so the
    same label always renders the same color even across episodes.
    """
    palette: Dict[str, Tuple[int, int, int]] = {}
    for i, act in enumerate(actions):
        if act in palette:
            continue
        if i < len(_BASE_PALETTE):
            palette[act] = _BASE_PALETTE[i]
        else:
            h = int(hashlib.md5(act.encode("utf-8")).hexdigest()[:6], 16)
            hue = (h % 360) / 360.0
            r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.65)
            palette[act] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


# ──────────────────────────────────────────────────────────────────────────────
#  Fonts
# ──────────────────────────────────────────────────────────────────────────────

_FONT_CACHE: dict = {}


def _load_font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
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


# ──────────────────────────────────────────────────────────────────────────────
#  Frame composition
# ──────────────────────────────────────────────────────────────────────────────

def resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB frame to size x size (LANCZOS).

    NOTE: We do *not* preserve aspect by padding because all JIGSAWS clips
    share the same camera aspect and a consistent square is what the website
    expects. We also intentionally do NOT horizontally flip (asymmetric
    robotic instruments — left vs right tool roles must be preserved).
    """
    pil = Image.fromarray(rgb)
    pil = pil.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(pil)


def render_action_banner(
    width: int,
    height: int,
    action: str,
    color: Tuple[int, int, int],
    progress: float,
    transition: bool,
) -> np.ndarray:
    """Render a colored action banner.

    Layout::

        [ ████████████░░░░░░░ ]   <- progress bar (top edge, 4 px)
        ACTION TEXT (white, bold, centered)
    """
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)

    # Top progress bar (light overlay over banner color).
    bar_h = 4
    bar_w = max(1, int(width * max(0.0, min(1.0, progress))))
    draw.rectangle((0, 0, bar_w, bar_h), fill=(255, 255, 255))
    draw.rectangle((bar_w, 0, width, bar_h), fill=(0, 0, 0))

    # Pulse a thin border on transitions so the gesture-change moment pops.
    if transition:
        draw.rectangle((0, bar_h, width - 1, height - 1), outline=(255, 255, 255), width=3)

    # Centered action text. Truncate if needed.
    font = _load_font(max(14, int(height * 0.35)))
    text = action.upper()
    # Measure
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = font.getsize(text) if hasattr(font, "getsize") else (len(text) * 8, 14)
    while tw > width - 20 and len(text) > 8:
        text = text[: max(8, len(text) - 4)] + "…"
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw = len(text) * 8
    tx = (width - tw) // 2
    ty = bar_h + (height - bar_h - th) // 2
    # Black drop shadow + white fill for legibility on any color.
    draw.text((tx + 1, ty + 1), text, font=font, fill=(0, 0, 0))
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255))

    return np.array(img, dtype=np.uint8)


def compose_frame(banner: np.ndarray, video: np.ndarray) -> np.ndarray:
    """Stack banner above the video frame; pad widths if mismatched."""
    bh, bw = banner.shape[:2]
    vh, vw = video.shape[:2]
    if bw != vw:
        pil = Image.fromarray(banner)
        pil = pil.resize((vw, bh), Image.Resampling.LANCZOS)
        banner = np.array(pil)
    out = np.concatenate([banner, video], axis=0)
    H, W = out.shape[:2]
    if W % 2:
        out = out[:, :-1]
    if H % 2:
        out = out[:-1, :]
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Video I/O
# ──────────────────────────────────────────────────────────────────────────────

def read_video_frames(video_path: Path, max_frames: int, stride: int) -> List[np.ndarray]:
    """Decode an mp4 to a list of RGB numpy frames via ffmpeg pipe.

    Frames are taken with the given stride (1 = every frame). Up to
    ``max_frames`` frames are kept (caps long episodes).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return []
    # Probe video size with ffprobe.
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=,",
             str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        wh = out.stdout.strip().split(",")
        w, h = int(wh[0]), int(wh[1])
    except Exception:
        return []

    frame_size = w * h * 3
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frames: List[np.ndarray] = []
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if not buf or len(buf) < frame_size:
                break
            if idx % stride == 0:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
                frames.append(arr)
                if len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
    return frames


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
