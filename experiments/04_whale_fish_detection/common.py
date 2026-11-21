"""Shared helpers for WP5a analysis: box filters (clock, whole frame, back), per-frame scores, persistence,
matching boxes to reference mask instances."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from uwid import detect  # noqa: E402
from uwid import eval_presence as ep  # noqa: E402

DEV_VIDEO = "/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
DEV_LABELS = REPO / "experiments/02_reference_labels/labels/2025-12-20_CG/frames.csv"
DEV_MASKS = Path("/home/sat3737/NOAA_whales/underwater_cache/reference/2025-12-20_CG/masks")
SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
W, H = 1920, 1080
PROMPTS = json.loads((HERE / "prompts.json").read_text())
WHALE_Q = PROMPTS["groups"]["whale"]
FISH_Q = PROMPTS["groups"]["fish"]
CLASS_IDS = dict(water=0, whale=1, back=2, fish=3, bubbles=4, surface=5, air=6, seafloor=7, unsure=255)


def fps_of(video: str) -> float:
    return detect.video_info(video)["fps"]


def box_area(df):
    return (df.x1 - df.x0).clip(lower=0) * (df.y1 - df.y0).clip(lower=0)


def tag_boxes(b: pd.DataFrame, whole_frame: float = 0.6, back_bottom: float = 0.75, back_touch_px: float = 12):
    """Add boolean columns: ``clock`` (box inside the burned-in clock), ``whole`` (covers > ``whole_frame`` of the
    frame), ``bottom`` (touches the bottom edge and its top is below ``back_bottom`` * H: the 'back zone' rule)."""
    b = b.copy()
    b["clock"] = detect.in_clock(b)
    b["whole"] = box_area(b) > whole_frame * W * H
    b["bottom"] = (b.y1 >= H - back_touch_px) & (b.y0 >= back_bottom * H)
    return b


def static_bottom(b: pd.DataFrame, fps: float, span_s: float = 2.0, iou_thr: float = 0.6, min_share: float = 0.6,
                  query_set=None) -> pd.Series:
    """'Does not move' rule: a box touching the bottom edge counts as static when, in at least ``min_share`` of the
    other processed frames within +-``span_s``, some box of the same query set lies on it with IoU >= ``iou_thr``.
    Returns a boolean Series aligned with ``b``."""
    out = pd.Series(False, index=b.index)
    touch = b.y1 >= H - 12
    if not touch.any():
        return out
    frames = np.sort(b.frame.unique())
    by_f = {f: g[["x0", "y0", "x1", "y1"]].to_numpy() for f, g in b[touch].groupby("frame")}
    half = span_s * fps
    for i, r in b[touch].iterrows():
        nb = frames[(np.abs(frames - r.frame) <= half) & (frames != r.frame)]
        if len(nb) == 0:
            continue
        hit = 0
        for f in nb:
            bb = by_f.get(f)
            if bb is None:
                continue
            if iou_one(np.array([r.x0, r.y0, r.x1, r.y1]), bb).max() >= iou_thr:
                hit += 1
        out[i] = hit / len(nb) >= min_share
    return out


def iou_one(a, bs):
    bs = np.asarray(bs, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[2], bs[:, 2]) - np.maximum(a[0], bs[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], bs[:, 3]) - np.maximum(a[1], bs[:, 1]), 0, None)
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def frame_scores(b: pd.DataFrame, frames: np.ndarray, queries, drop=("clock", "whole")) -> pd.DataFrame:
    """Max score per processed frame over boxes of ``queries`` not flagged by any column in ``drop``
    (0 when none). Columns frame, score."""
    sel = b[b["query"].isin(queries)]
    for c in drop:
        sel = sel[~sel[c]]
    s = sel.groupby("frame").score.max()
    return pd.DataFrame(dict(frame=frames, score=s.reindex(frames).fillna(0.0).to_numpy()))


def persistent_scores(b: pd.DataFrame, frames: np.ndarray, queries, drop=("clock", "whole"), iou_thr=0.3,
                      reach: int = 1) -> pd.DataFrame:
    """Per frame, the best box score that is backed by an overlapping box (IoU >= ``iou_thr``, any query of the set)
    in one of the ``reach`` previous or next processed frames: score = min(own, best neighbour). A one-frame flash
    gets 0."""
    sel = b[b["query"].isin(queries)]
    for c in drop:
        sel = sel[~sel[c]]
    frames = np.sort(np.asarray(frames))
    pos = {f: i for i, f in enumerate(frames)}
    by_f = {f: (g[["x0", "y0", "x1", "y1"]].to_numpy(float), g.score.to_numpy(float)) for f, g in sel.groupby("frame")}
    out = np.zeros(len(frames))
    for f, (bx, sc) in by_f.items():
        i = pos.get(f)
        if i is None:
            continue
        best = 0.0
        nbs = [frames[j] for j in range(max(0, i - reach), min(len(frames), i + reach + 1)) if j != i]
        for k in range(len(bx)):
            if sc[k] <= best:
                continue
            m = 0.0
            for nf in nbs:
                if nf not in by_f:
                    continue
                nbx, nsc = by_f[nf]
                ious = iou_one(bx[k], nbx)
                ok = ious >= iou_thr
                if ok.any():
                    m = max(m, nsc[ok].max())
            best = max(best, min(sc[k], m))
        out[i] = best
    return pd.DataFrame(dict(frame=frames, score=out))


def load_masks(frame: int, mask_dir: Path = DEV_MASKS):
    from PIL import Image
    cls = np.array(Image.open(mask_dir / f"{frame:06d}_class.png"))
    inst = np.array(Image.open(mask_dir / f"{frame:06d}_inst.png"))
    meta = json.loads((mask_dir / f"{frame:06d}_meta.json").read_text())
    return cls, inst, meta


def mask_frames(mask_dir: Path = DEV_MASKS) -> list[int]:
    return sorted(int(p.name[:6]) for p in mask_dir.glob("*_class.png"))


def instances(frame: int, mask_dir: Path = DEV_MASKS) -> list[dict]:
    """Reference instances of one mask frame: id, class, grade, note, bbox, area, mask, cut (touches a frame edge),
    behind_bubbles (>= 15 % of the ring around it is bubbles, or note says so), small (area < 1 % of the frame)."""
    import cv2
    cls, inst, meta = load_masks(frame, mask_dir)
    out = []
    bub = (cls == CLASS_IDS["bubbles"]).astype(np.uint8)
    for k, m in meta["instances"].items():
        mk = inst == int(k)
        if not mk.any():
            continue
        ys, xs = np.nonzero(mk)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        ring = cv2.dilate(mk.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~mk
        note = (m.get("note") or "").lower()
        out.append(dict(frame=frame, id=int(k), cls=m["class"], grade=m.get("grade", ""), note=note,
                        box=np.array([x0, y0, x1, y1], float), area=int(mk.sum()), mask=mk,
                        cut=bool(x0 <= 2 or y0 <= 2 or x1 >= W - 2 or y1 >= H - 2),
                        behind_bubbles=bool("bubble" in note or (ring.any() and bub[ring].mean() >= 0.15)),
                        small=bool(mk.sum() < 0.01 * W * H)))
    return out


def match_instances(insts: list[dict], boxes: pd.DataFrame, iou_thr: float = 0.3, cover_thr: float = 0.5):
    """Greedy match of boxes (highest score first) to instances: a box matches an instance when IoU(box, instance
    bbox) >= ``iou_thr``, or when the box holds >= ``cover_thr`` of the instance's pixels and the instance's pixels
    fill >= 25 % of the box. Returns (instance id -> box row index), set of unmatched box indices."""
    order = boxes.sort_values("score", ascending=False).index
    taken, used = {}, set()
    for bi in order:
        r = boxes.loc[bi]
        a = np.array([r.x0, r.y0, r.x1, r.y1])
        best, best_v = None, 0.0
        for ins in insts:
            if ins["id"] in taken:
                continue
            iou = iou_one(a, ins["box"][None])[0]
            x0, y0, x1, y1 = [int(round(v)) for v in a]
            inside = ins["mask"][max(y0, 0):y1, max(x0, 0):x1].sum()
            cover = inside / max(ins["area"], 1)
            fill = inside / max((x1 - x0) * (y1 - y0), 1)
            v = iou if iou >= iou_thr else (cover if (cover >= cover_thr and fill >= 0.25) else 0.0)
            if v > best_v:
                best, best_v = ins["id"], v
        if best is not None:
            taken[best] = bi
            used.add(bi)
    return taken, set(boxes.index) - used


def box_cause(cls: np.ndarray, r) -> str:
    """What a box that matched no whale sits on, by the reference class map: the largest class among back, bubbles,
    surface, air, fish, whale (part of a whale, unmatched), water."""
    x0, y0, x1, y1 = [int(round(v)) for v in (r.x0, r.y0, r.x1, r.y1)]
    sub = cls[max(y0, 0):y1, max(x0, 0):x1]
    if sub.size == 0:
        return "water"
    fr = {k: float((sub == v).mean()) for k, v in CLASS_IDS.items()}
    if fr["whale"] >= 0.25:
        return "whale_part"
    cand = {k: fr[k] for k in ("back", "bubbles", "surface", "air", "fish")}
    k = max(cand, key=cand.get)
    return k if cand[k] >= 0.25 else "water"


def to_md(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    """Markdown table without the tabulate dependency."""
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
