"""The tagged whale's own back as a template (WP14 / WP17): where the back usually is in a stretch of a recording.

The camera is fixed on the tagged whale, so its own back sits at nearly the same place through a recording, while
another animal is in view only for a while. The tracker's own back (the SAM 3 back object, object 0, prompted from
the bottom 25 % of the frame) can grow over a calf swimming along the back's edge (2025-02-03 00005 604-629 s,
confirmed by the user 2026-11-23); every start box on the calf is then refused as "on the back" and its tracks get
the role ``on_back``.

With the tracking flag ``own_back_template`` (off by default; :func:`uwid.trackstage.run_span`) the own back becomes
the SAM back object INSIDE a template: the patches that the WP5b patch classifier (the tracker's own patch pass)
calls ``back`` in at least ``TAU`` of the under-water on-whale frames of the 300 s block of the recording, grown by
``DILATE`` patches. The frames are every ``STEP``-th scanned window of the block (the scan's window table says which
windows are under water and on the whale); their class maps are computed once and cached per block. Parameters
frozen as reported in WP14 (chosen on the two test deployments 2025-02-03 and 2025-01-02, see
experiments/10_own_back/tables.md); nothing here is tuned.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BACK = 2
TAU = 0.5            # share of a block's frames in which a patch is called back
DILATE = 1           # patches added around the template (8-neighbourhood)
BLOCK_S = 300.0      # template block length (s of video)
STEP = 10            # every STEP-th scanned under-water on-whale window gives one frame
MIN_FRAMES = 5       # fewer frames than this in a block: no template (the SAM back object is used as before)
GRID = (45, 80)      # patch grid of the 1280 px patch classifier


def back_template(label_maps: np.ndarray, tau: float = TAU, dilate: int = DILATE, back_code: int = BACK) -> np.ndarray:
    """(n, h, w) patch labels -> (h, w) bool: patch called back in >= ``tau`` of the frames, grown by ``dilate``."""
    lab = np.asarray(label_maps)
    if lab.ndim != 3 or len(lab) == 0:
        raise ValueError("need (n, h, w) label maps")
    t = (lab == back_code).mean(0) >= tau
    for _ in range(max(0, dilate)):
        g = t.copy()
        g[1:] |= t[:-1]; g[:-1] |= t[1:]; g[:, 1:] |= t[:, :-1]; g[:, :-1] |= t[:, 1:]
        g[1:, 1:] |= t[:-1, :-1]; g[1:, :-1] |= t[:-1, 1:]; g[:-1, 1:] |= t[1:, :-1]; g[:-1, :-1] |= t[1:, 1:]
        t = g
    return t


def upsample(template: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(h, w) bool patch grid -> (H, W) bool, nearest neighbour; ``size`` = (H, W)."""
    H, W = size
    h, w = template.shape
    yi = np.minimum((np.arange(H) * h) // H, h - 1)
    xi = np.minimum((np.arange(W) * w) // W, w - 1)
    return template[yi][:, xi]


def own_back(sam_back: np.ndarray | None, template_px: np.ndarray | None) -> np.ndarray | None:
    """The own back of one frame: the SAM back object inside the template. No template = the SAM back object itself
    (the same array, untouched)."""
    if template_px is None or sam_back is None:
        return sam_back
    return sam_back & template_px


def share_inside(mask: np.ndarray, region: np.ndarray) -> float:
    n = int(mask.sum())
    return float((mask & region).sum() / n) if n else 0.0


def box_on_back(box, region: np.ndarray, frac: float = 0.6) -> bool:
    """As tracks.box_covered's back test: does a box (pixels of ``region``) lie >= ``frac`` on the region?"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    area = max((x1 - x0) * (y1 - y0), 1)
    return bool(region[y0:y1, x0:x1].sum() >= frac * area)


def block_of(t: float, block_s: float = BLOCK_S) -> int:
    return int(np.floor(t / block_s))


def template_frames(windows: pd.DataFrame, block: int, block_s: float = BLOCK_S, step: int = STEP) -> list[int]:
    """Frames for a block's template: frame_0 of every ``step``-th scanned window of the block that is under water
    with the tag on the whale (window table columns t0, water, tag_state, frame_0; without water / tag_state all
    windows count)."""
    w = windows
    m = (w.t0 >= block * block_s) & (w.t0 < (block + 1) * block_s) & w.frame_0.notna()
    if "water" in w:
        m &= w.water == "underwater"
    if "tag_state" in w:
        m &= w.tag_state == "whale"
    if "scanned" in w:
        m &= w.scanned.astype(str).str.lower().eq("true")
    return [int(x) for x in w[m].frame_0.iloc[::step]]


def label_maps(video: str | os.PathLike, frames: list[int], model=None, clf=None) -> tuple[np.ndarray, list[int]]:
    """WP5b patch classes (45 x 80) of the given shown frames, as the tracker's patch pass computes them: (labels,
    the frames that decoded)."""
    import cv2
    import torch
    from . import patchseg as ps
    from . import video as uv
    model = model or ps.load_backbone()
    clf = clf or ps.load_classifier(ps.CACHE / "clf_w1280_h0_all.pt")[0]
    labs, batch = {}, []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            f = ps.extract(model, [b[1] for b in batch], 1280)
            for j, (idx, _) in enumerate(batch):
                labs[idx] = ps.predict_probs(clf, f[j]).argmax(-1).cpu().numpy().astype(np.uint8)
        batch.clear()

    for fr in uv.iter_frames_at(video, sorted(set(frames))):
        img = fr.image
        if img.shape[1] != 1920:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        batch.append((fr.index, img))
        if len(batch) == 8:
            flush()
    flush()
    got = [f for f in sorted(set(frames)) if f in labs]
    return np.stack([labs[f] for f in got]) if got else np.zeros((0, *GRID), np.uint8), got


def block_template(video: str | os.PathLike, windows: pd.DataFrame | None, t: float, cache_dir: str | os.PathLike,
                   model=None, clf=None, maps_fn=None) -> tuple[np.ndarray | None, dict]:
    """The template of the 300 s block holding time ``t`` (s): (45 x 80 bool or None, info). Label maps are cached in
    ``cache_dir/block_<b>.npz``; ``maps_fn(video, frames) -> (labels, frames)`` replaces :func:`label_maps` (tests)."""
    b = block_of(t)
    info = dict(block=b, block_s=BLOCK_S, tau=TAU, dilate=DILATE, step=STEP, frames=0)
    if windows is None or not len(windows):
        info["why_none"] = "no window table"
        return None, info
    frames = template_frames(windows, b)
    cache = Path(cache_dir) / f"block_{b:04d}.npz"
    if cache.exists():
        z = np.load(cache)
        labels, got = z["labels"], [int(x) for x in z["frames"]]
    else:
        if len(frames) < MIN_FRAMES:
            info.update(frames=len(frames), why_none=f"fewer than {MIN_FRAMES} frames in the block")
            return None, info
        labels, got = (maps_fn or (lambda v, f: label_maps(v, f, model, clf)))(video, frames)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, labels=labels, frames=np.asarray(got, np.int64))
    info["frames"] = int(len(got))
    if len(got) < MIN_FRAMES:
        info["why_none"] = f"fewer than {MIN_FRAMES} frames decoded"
        return None, info
    tm = back_template(labels)
    info["area_share"] = round(float(tm.mean()), 4)
    return tm, info


def save_run_template(out: str | os.PathLike, template: np.ndarray, info: dict) -> Path:
    """Store the template a run used beside its masks (``own_back.npz``); :func:`uwid.tracks.read_run` picks it up."""
    p = Path(out) / "own_back.npz"
    np.savez_compressed(p, template=template.astype(bool), info=json.dumps(info))
    return p


def load_run_template(run_dir: str | os.PathLike) -> tuple[np.ndarray, dict] | None:
    p = Path(run_dir) / "own_back.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return z["template"].astype(bool), json.loads(str(z["info"]))
