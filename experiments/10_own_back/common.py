"""Shared paths and helpers of 10_own_back (WP14)."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
VID = Path("/home/sat3737/NOAA_whales/underwater_video")
CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache")
OUT = CACHE / "ownback"
SCAN = CACHE / "scan"
TRACKS = CACHE / "tracks"
SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
WP5B_HEAD = CACHE / "patchseg" / "clf_w1280_h0_all.pt"   # the tracker's patch classifier (trackstage.patch_pass)
DEPL = {"d1": "2025-2-3 Mini-mother 11-96", "d2": "TestSubset 2025-1-2 CATS -85"}
ON_WHALE = {"d1": ["00004", "00005", "00006", "00007", "00008", "00009", "00010", "00011", "00012"],
            "d2": ["00010", "00011", "00012", "00017", "00023", "00024"]}
BACK = 2
W, H = 1280, 720
GRID = (45, 80)


def recordings():
    """(source, piece, stem, video path, window table path)."""
    out = []
    for src, name in DEPL.items():
        for f in sorted(glob.glob(str(SCAN / name / "sensitive" / "windows" / "*.mov.csv"))):
            stem = Path(f).name[:-len(".mov.csv")]
            piece = stem.split("-")[-1]
            if piece in ON_WHALE[src]:
                out.append((src, piece, stem, VID / name / "Video" / f"{stem}.mov", Path(f)))
    return out


def windows(path) -> pd.DataFrame:
    return pd.read_csv(path)


def lens_out_runs(w: pd.DataFrame) -> list[tuple[float, float]]:
    """Runs of windows whose water state is not `underwater` (air, surfaced, crossing): (t_start, t_end)."""
    out = (w.water != "underwater").to_numpy()
    runs, s = [], None
    for i, v in enumerate(list(out) + [False]):
        if v and s is None:
            s = i
        if not v and s is not None:
            runs.append((float(w.t0.iloc[s]), float(w.t1.iloc[i - 1])))
            s = None
    return runs


def whale_objects(w: pd.DataFrame, gap: int = 1) -> list[tuple[int, int]]:
    """Runs of consecutive windows the scan calls whale (column `whale` True), allowing ``gap`` missing windows:
    (first row, last row)."""
    wh = w.whale.astype(str).str.lower().eq("true").to_numpy()
    runs, s, last = [], None, None
    for i, v in enumerate(wh):
        if v:
            if s is None:
                s = i
            elif i - last > gap + 1:
                runs.append((s, last)); s = i
            last = i
    if s is not None:
        runs.append((s, last))
    return runs


def unpack(a, w=W):
    return np.unpackbits(a, axis=-1)[..., :w].astype(bool)
