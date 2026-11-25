"""Labelled sources of WP10 (whale against fish boxes, seafloor / sand / dark): where the frames, labels and masks are.

Three sources, used leave-one-source-out (PLAN.md, 2026-11-22):
- ``dev``: 2025-12-20 CG clip (191 labelled frames, 31 mask frames), frame index = OpenCV decode order of the .m4v;
- ``d1``: 2025-02-03, tag CC-11-96, pieces 00004-00010 (840 labelled frames, 14 mask frames), .m4v decode order;
- ``d2``: 2025-01-02, tag CC-09-85 (420 labelled frames, 29 mask frames), shown-frame index of the 4K .mov, frames
  saved at 1920 x 1080 (all label pixels in that space).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "experiments" / "02_reference_labels" / "labels"
REF = Path("/home/sat3737/NOAA_whales/underwater_cache/reference")
VIDEO = Path("/home/sat3737/NOAA_whales/underwater_video")
CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache/boxclf")
SF_CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache/seafloor")
SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
SOURCES = ("dev", "d1", "d2")
SOURCE_NAME = {"dev": "2025-12-20 clip", "d1": "2025-02-03 (CC-11-96)", "d2": "2025-01-02 (CC-09-85)"}
PARTNER_STEP = 15   # frames: the second frame of a 1 s window, 0.5 s later


@dataclass
class Recording:
    source: str
    stem: str          # label folder name
    video: Path
    fps: float

    @property
    def ref(self) -> Path:
        return REF / self.stem

    @property
    def labels(self) -> Path:
        return LABELS / self.stem

    def frames_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.labels / "frames.csv", keep_default_na=False)

    def mask_frames(self) -> list[int]:
        return sorted(int(p.name[:6]) for p in (self.ref / "masks").glob("*_class.png"))

    def image_path(self, frame: int) -> Path | None:
        for sub in ("frames", "sample", "maskframes"):
            p = self.ref / sub / f"{frame:06d}.jpg"
            if p.exists():
                return p
        return None

    def partner_path(self, frame: int) -> Path | None:
        """Frame + 0.5 s: on the dev clip the next labelled frame, otherwise decoded by partners.py."""
        f2 = frame + PARTNER_STEP
        if self.source == "dev":
            return self.image_path(f2)
        p = CACHE / "partners" / self.stem / f"{f2:06d}.jpg"
        return p if p.exists() else None

    def fish_calls(self) -> dict[int, list[list[float]]]:
        p = self.labels / "fish_calls.json"
        if not p.exists():
            return {}
        out = {}
        for r in json.loads(p.read_text()):
            bx = []
            for b in r["boxes"]:
                x0, y0, x1, y1 = b
                bx.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
            out[int(r["frame"])] = bx
        return out

    def labelled_frames(self) -> list[int]:
        fr = set(int(f) for f in self.frames_csv().frame) | set(self.mask_frames())
        return sorted(fr)


def recordings(sources=SOURCES) -> list[Recording]:
    out = []
    if "dev" in sources:
        out.append(Recording("dev", "2025-12-20_CG", VIDEO / "2025-12-20 CG.m4v", 29.9742))
    if "d1" in sources:
        for p in sorted(LABELS.glob("CC-11-96-20250203-*")):
            out.append(Recording("d1", p.name, VIDEO / "2025-2-3 Mini-mother 11-96" / "Video" / f"{p.name}.m4v",
                                 float(json.loads((REF / p.name / "index.json").read_text())["fps"])))
    if "d2" in sources:
        for p in sorted(LABELS.glob("CC-09-85-*")):
            out.append(Recording("d2", p.name, VIDEO / "TestSubset 2025-1-2 CATS -85" / "Video" / f"{p.name}.mov", 30.0))
    return out


def piece(stem: str) -> str:
    return stem.split("-")[-1]


def size_band(px: float) -> str:
    for lim, name in ((150, "<150"), (300, "150-300"), (500, "300-500"), (800, "500-800")):
        if px < lim:
            return name
    return ">=800"


BANDS = ("<150", "150-300", "300-500", "500-800", ">=800")


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of boxes a (n, 4) and b (m, 4)."""
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-6, None)


def inside_frac(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Share of each box a's area that lies inside box b: (n, m)."""
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return ix * iy / np.clip(aa[:, None], 1e-6, None)
