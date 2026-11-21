"""Contact sheets: a grid of frames with time labels, optional caption per tile and optional coloured mask overlay."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import paths

PALETTE = [(255, 60, 60), (60, 200, 255), (255, 210, 40), (90, 230, 90), (230, 90, 230), (255, 140, 40)]


def _font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def format_time(t: float) -> str:
    m, s = divmod(t, 60.0)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m:d}:{s:05.2f}"


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a mask onto an RGB image. ``mask``: bool / 0-1 (one colour) or integer labels (0 = none, k = colour k)."""
    img = image.astype(np.float32).copy()
    m = np.asarray(mask)
    if m.shape[:2] != img.shape[:2]:
        m = np.array(Image.fromarray(m.astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.NEAREST))
    labels = m.astype(np.int64) if m.dtype != bool else m.astype(np.int64)
    for k in np.unique(labels):
        if k == 0:
            continue
        col = np.array(PALETTE[(int(k) - 1) % len(PALETTE)], np.float32)
        sel = labels == k
        img[sel] = (1 - alpha) * img[sel] + alpha * col
    return img.clip(0, 255).astype(np.uint8)


def make_sheet(images: Sequence[np.ndarray], labels: Sequence[str] | None = None,
               captions: Sequence[str | None] | None = None, masks: Sequence[np.ndarray | None] | None = None,
               cols: int = 8, tile_width: int = 320, title: str | None = None,
               out: str | os.PathLike | None = None, quality: int = 88) -> Image.Image:
    """Lay out ``images`` (RGB uint8 arrays) in a grid. ``labels`` go in the top-left corner of each tile (time),
    ``captions`` under it, ``masks`` are blended on top. Saves a JPEG when ``out`` is given."""
    n = len(images)
    if n == 0:
        raise ValueError("no images")
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    h0, w0 = images[0].shape[:2]
    tile_h = int(round(tile_width * h0 / w0))
    font = _font(max(11, tile_width // 22))
    cap_font = _font(max(10, tile_width // 26))
    has_caps = captions is not None and any(captions)
    cap_h = (cap_font.size + 6) if has_caps else 0
    title_h = (font.size + 14) if title else 0
    pad = 4
    W = cols * (tile_width + pad) + pad
    H = title_h + rows * (tile_h + cap_h + pad) + pad
    sheet = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((pad + 2, 6), title, fill=(235, 235, 235), font=font)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = pad + c * (tile_width + pad)
        y = title_h + pad + r * (tile_h + cap_h + pad)
        arr = img if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)
        if masks is not None and i < len(masks) and masks[i] is not None:
            arr = overlay_mask(arr, masks[i])
        tile = Image.fromarray(arr).resize((tile_width, tile_h), Image.BILINEAR)
        sheet.paste(tile, (x, y))
        if labels is not None and i < len(labels) and labels[i]:
            tb = draw.textbbox((0, 0), labels[i], font=font)
            draw.rectangle((x, y, x + tb[2] + 6, y + tb[3] + 4), fill=(0, 0, 0))
            draw.text((x + 3, y + 1), labels[i], fill=(255, 255, 255), font=font)
        if has_caps and i < len(captions) and captions[i]:
            draw.text((x + 2, y + tile_h + 2), str(captions[i])[:80], fill=(220, 220, 220), font=cap_font)
    if out is not None:
        out = paths.check_writable_target(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out, "JPEG", quality=quality)
    return sheet


def video_sheet(path: str | os.PathLike, n: int = 40, out: str | os.PathLike | None = None, cols: int = 8,
                tile_width: int = 320, start: float = 0.0, end: float | None = None,
                with_clock: bool = True) -> tuple[Image.Image, list]:
    """Contact sheet of ``n`` frames evenly spaced over a video (random access), labelled with their time and, when
    readable, the burned-in clock."""
    from . import video, timecode

    tab = video.frame_table(path)
    if tab.n_frames == 0:
        raise ValueError(f"no frames in {path}")
    t_end = float(tab.times[-1]) if end is None else min(end, float(tab.times[-1]))
    times = np.linspace(max(start, float(tab.times[0])), t_end, n)
    with video.VideoReader(path, size=None) as r:
        frames = [f for f in r.frames_at(list(times)) if f is not None]
    return frames_sheet(path, frames, out=out, cols=cols, tile_width=tile_width, with_clock=with_clock)


def frames_sheet(path, frames, out=None, cols: int = 8, tile_width: int = 320, with_clock: bool = True,
                 title: str | None = None):
    from . import names, timecode
    caps = None
    if with_clock:
        try:
            gs = timecode.load_glyphs()
            hint = names.date_hint(path)
            caps = []
            for f in frames:
                rd = timecode.read_clock(f.image, gs, hint)
                caps.append(rd.text if rd else "no clock")
        except FileNotFoundError:
            caps = None
    labels = [f"{format_time(f.t)}  #{f.index}" for f in frames]
    title = title or f"{Path(path).name}  ({len(frames)} frames)"
    img = make_sheet([f.image for f in frames], labels, caps, cols=cols, tile_width=tile_width, title=title, out=out)
    return img, frames
