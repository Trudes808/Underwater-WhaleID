"""WP15 frame rule (tables.md S0): which tracked frames are fit to feed a single-image 3D model.

Runs in the repo's .venv (CPU only: tracker outputs + decoding). Used by the smoke selection (`select_smoke.py`) and
by the phase-2 picker (`pick_frames.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from uwid import trackstage, tracks
from uwid import video as uv

CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache")
TRACKS = CACHE / "tracks"
MESH = CACHE / "mesh3d"


@dataclass(frozen=True)
class Rule:
    step: int = 5                 # every 5th frame
    min_area: float = 0.03        # share of the frame
    edge_px: int = 6              # at store size
    back_px: int = 12             # gap to the back outline, store size
    one_piece: float = 0.90
    erode_px: int = 3


def run_dirs(deployment_dir: Path, recordings: tuple[str, ...] = ()) -> list[Path]:
    """Finished online runs (no `_pre_*` copies) of a deployment folder, optionally only some recordings."""
    out = []
    for m in sorted(deployment_dir.glob("*/*_online/meta.json")):
        d = m.parent
        if recordings and not any(r in d.parent.name for r in recordings):
            continue
        meta = json.loads(m.read_text())
        if meta.get("done") and (d / "masks.npz").exists() and (d / "tracks.parquet").exists():
            out.append(d)
    return out


def blocked_set(meta: dict) -> set[int]:
    s: set[int] = set()
    for a, b in meta.get("blocked", []):
        s.update(range(int(a), int(b) + 1))
    return s


def cheap_candidates(tr: tracks.Tracks, rule: Rule) -> pd.DataFrame:
    """Rows (i, frame, obj, number) that pass the table-only parts of the rule (area, edges, back box, blocked)."""
    W, H = tr.meta["store_size"]
    sx = W / trackstage.TRACK_W
    tab = tr.table
    tab = tab[tab.obj.isin(list(tr.number)) & (tab.i % rule.step == 0)].copy()
    blk = blocked_set(tr.meta)
    back_top = tr.meta.get("back_box", [0, 810, 1920, 1080])[1] * sx
    x0, y0, x1, y1 = (tab[c] * sx for c in ("x0", "y0", "x1", "y1"))
    ok = ((tab.area >= rule.min_area * W * H) & (x0 >= rule.edge_px) & (y0 >= rule.edge_px)
          & (x1 <= W - rule.edge_px) & (y1 <= min(H - rule.edge_px, back_top)) & ~tab.frame.isin(blk))
    tab = tab[ok]
    tab["number"] = tab.obj.map(tr.number)
    return tab[["i", "frame", "t", "obj", "number", "area"]].reset_index(drop=True)


def mask_checks(tr: tracks.Tracks, obj: int, i: int, rule: Rule) -> tuple[bool, dict, np.ndarray]:
    """Exact outline checks on one frame (store size). Returns (pass, values, mask)."""
    m = tr.mask(obj, i)
    if (i, obj) in tr.hidden or tr.is_small(obj, i):
        return False, dict(why="hidden or small"), m
    vals: dict = {}
    if tr.back is not None:
        b = tr.mask(tracks.BACK_ID, i)
        if b.any():
            k = 2 * rule.back_px + 1
            near = cv2.dilate(b.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            vals["back_touch_px"] = int((near & m).sum())
            if vals["back_touch_px"] > 0:
                return False, dict(vals, why="near back"), m
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n < 2:
        return False, dict(why="empty"), m
    areas = st[1:, cv2.CC_STAT_AREA]
    vals["one_piece"] = float(areas.max() / areas.sum())
    if vals["one_piece"] < rule.one_piece:
        return False, dict(vals, why="split"), m
    return True, vals, m


def sharpness(rgb: np.ndarray, m: np.ndarray, erode_px: int) -> float:
    """Variance of the Laplacian inside the eroded outline (frame and mask at the same size)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    k = 2 * erode_px + 1
    inner = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return float(lap[inner].var()) if inner.sum() > 50 else 0.0


def evaluate_run(d: Path, rule: Rule = Rule(), frames_out: dict | None = None) -> pd.DataFrame:
    """All rule values for every 5th frame of every whale number in one run. ``frames_out`` (optional dict) collects
    the decoded store-size frames of passing rows, keyed by frame index."""
    tr = trackstage.load_online(d)
    cand = cheap_candidates(tr, rule)
    rows = []
    if not len(cand):
        return pd.DataFrame()
    W, H = tr.meta["store_size"]
    want = sorted(cand.frame.unique())
    by_frame = {f: g for f, g in cand.groupby("frame")}
    for fr in uv.iter_frames_isolated(tr.meta["video"], indices=want, size=(W, H)):
        for r in by_frame.get(fr.index, pd.DataFrame()).itertuples():
            ok, vals, m = mask_checks(tr, int(r.obj), int(r.i), rule)
            row = dict(run=str(d), video=Path(tr.meta["video"]).name, i=int(r.i), frame=int(r.frame),
                       t=round(float(r.t), 3), obj=int(r.obj), number=int(r.number),
                       area_share=round(float(r.area) / (W * H), 4), passed=ok, **vals)
            if ok:
                row["sharp"] = sharpness(fr.image, m, rule.erode_px)
                if frames_out is not None:
                    frames_out[fr.index] = fr.image
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) and df.passed.any():
        med = df[df.passed].groupby("number").sharp.transform("median")
        df.loc[df.passed, "sharp_ok"] = df.loc[df.passed, "sharp"] >= med
        df["final"] = df.passed & df.sharp_ok.fillna(False).astype(bool)
    else:
        df["final"] = False
    return df


def rgba_input(video: str, frame: int, tr: tracks.Tracks, obj: int, i: int, size=(1920, 1080)) -> np.ndarray:
    """The model input: full-size frame + outline (nearest neighbour up) as alpha, RGBA uint8."""
    fr = next(iter(uv.iter_frames_isolated(video, indices=[frame], size=size)))
    m = tr.mask(obj, i).astype(np.uint8) * 255
    a = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    return np.dstack([fr.image, a])
