"""Synthetic test videos made on the fly (no real footage in tests).

Every frame carries its own index as a row of black / white blocks at the top (``decode_index``), so tests can tell
exactly which source frame a decoded or cut frame is. Optionally a clock is drawn bottom right in a fixed-pitch
layout, like the camera tags' overlay.
"""

from __future__ import annotations

import datetime as dt
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

W, H = 320, 240
N_BITS = 12
BLOCK = W // N_BITS
CLOCK_LAYOUT = dict(x0=40, y0=212, pitch=12, cell_h=18, n_chars=20, ref_width=W, ref_height=H)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def clock_text(t: dt.datetime) -> str:
    ms = (t.microsecond // 100000) * 100
    return f"{t.day:02d}{MONTHS[t.month - 1]}{t.year % 100:02d} {t:%H:%M:%S}.{ms:03d}"


def draw_clock(img: np.ndarray, text: str) -> None:
    L = CLOCK_LAYOUT
    x0, y0, p, h = L["x0"], L["y0"], L["pitch"], L["cell_h"]
    img[y0:y0 + h, x0:x0 + p * len(text)] = 0
    for k, ch in enumerate(text):
        cv2.putText(img, ch, (x0 + k * p + 1, y0 + h - 4), cv2.FONT_HERSHEY_PLAIN, 0.95, (255, 255, 255), 1,
                    cv2.LINE_8)


def frame_image(i: int, n: int, clock: str | None = None) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = (xx * 255 // W + 3 * i) % 256
    img[..., 1] = (yy * 255 // H) // 2 + 40
    img[..., 2] = 90
    for b in range(N_BITS):
        img[0:24, b * BLOCK:(b + 1) * BLOCK] = 255 if (i >> b) & 1 else 0
    if clock:
        draw_clock(img, clock)
    return img


def decode_index(img: np.ndarray) -> int:
    v = 0
    for b in range(N_BITS):
        if img[4:20, b * BLOCK + 4:(b + 1) * BLOCK - 4].mean() > 128:
            v |= 1 << b
    return v


def make_video(path: Path, n: int = 90, fps: int = 30, gop: int = 30, clock_start: dt.datetime | None = None,
               clock_every: int = 1, audio: bool = False, audio_rate: int = 32000, audio_channels: int = 2,
               movflags: str = "faststart", codec: str = "libx264") -> Path:
    """Encode ``n`` frames at ``fps`` with a keyframe every ``gop`` frames (no B-frames, closed GOPs)."""
    opts = {"movflags": movflags} if movflags else {}
    with av.open(str(path), "w", options=opts) as c:
        v = c.add_stream(codec, rate=fps)
        v.width, v.height, v.pix_fmt = W, H, "yuv420p"
        v.options = {"crf": "10", "g": str(gop), "keyint_min": str(gop), "sc_threshold": "0", "bf": "0",
                     "preset": "veryfast"}
        a = None
        if audio:
            a = c.add_stream("aac", rate=audio_rate)
            a.layout = "stereo" if audio_channels == 2 else "mono"
        for i in range(n):
            clock = None
            if clock_start is not None and (clock_every and i % clock_every == 0 or clock_every == 1):
                clock = clock_text(clock_start + dt.timedelta(seconds=i / fps))
            fr = av.VideoFrame.from_ndarray(frame_image(i, n, clock), format="rgb24")
            fr.pts = i
            fr.time_base = Fraction(1, fps)
            for pkt in v.encode(fr):
                c.mux(pkt)
        for pkt in v.encode():
            c.mux(pkt)
        if a is not None:
            total = int(n / fps * audio_rate)
            tt = np.arange(total) / audio_rate
            sig = (0.5 * np.sin(2 * np.pi * 440 * tt)).astype(np.float32)
            chunk = 1024
            for k in range(0, total, chunk):
                block = sig[k:k + chunk]
                arr = np.tile(block, (audio_channels, 1))
                af = av.AudioFrame.from_ndarray(arr, format="fltp", layout=a.layout.name)
                af.sample_rate = audio_rate
                af.pts = k
                af.time_base = Fraction(1, audio_rate)
                for pkt in a.encode(af):
                    c.mux(pkt)
            for pkt in a.encode():
                c.mux(pkt)
    return path


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    """Point footage / cache / sheets at temporary folders."""
    d = {k: tmp_path / k for k in ("footage", "cache", "sheets")}
    for p in d.values():
        p.mkdir()
    monkeypatch.setenv("UWID_FOOTAGE_DIR", str(d["footage"]))
    monkeypatch.setenv("UWID_CACHE_DIR", str(d["cache"]))
    monkeypatch.setenv("UWID_SHEETS_DIR", str(d["sheets"]))
    from uwid import video
    video._TABLE_CACHE.clear()
    return d
