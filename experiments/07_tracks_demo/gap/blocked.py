"""WP8b: a blocked-view detector that needs no hand marking, and the blocked frame ranges of the real cases.

    flock ... .venv/bin/python experiments/07_tracks_demo/gap/blocked.py patch     # GPU: patch classes, 2025-02-03 cases
    .venv/bin/python experiments/07_tracks_demo/gap/blocked.py rule                 # CPU: rule -> results/blocked_<case>.json

RULE (fixed 2026-11-21 before any tracker result of WP8b was looked at):

A frame is BLOCKED when
  (1) the lens is not under water:
      - with a sensor log (2025-02-03): the frame's time lies in [exit - 0.2 s, entry + 0.2 s] of a surfacing of
        `tagdata/surfacings.csv` (exit / entry = `pts_exit_pred` / `pts_entry_pred`, depth mapped through the
        burned-in clock);
      - without a log (2025-12-20 clip): >= 25 % of the frame's patches are classed `air` by the WP5b patch classifier;
  or (2) bubbles hide most of the view above the back: >= 50 % of the patches NOT classed `back` are classed `bubbles`.
Patch classes: WP5b's linear classifier on DINOv3 ViT-B/16 patches at 1280 px. On the 2025-12-20 clip the cached
predictions of the four fold classifiers are used (each frame scored by the classifier whose held-out time block
holds it; still all fitted on that clip, so not an independent test there), every 5th frame; on 2025-02-03 the
classifier fitted on all 31 dev mask frames, every 3rd frame of the `.m4v` (the colour domain it was trained on).
Frames between scored frames take the nearest scored frame's call. Cleaning: blocked runs shorter than 0.2 s are
dropped, then unblocked runs shorter than 0.3 s between two blocked runs are filled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import REAL, SURFACINGS, tracks  # noqa: E402

from uwid import video as uv  # noqa: E402

CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache/tracks/blocked")
PATCH = Path("/home/sat3737/NOAA_whales/underwater_cache/patchseg")
AIR, BACK, BUBBLES = 6, 2, 4
AIR_SHARE, BUBBLE_SHARE, LENS_PAD_S = 0.25, 0.50, 0.2
MIN_BLOCK_S, MIN_OPEN_S = 0.2, 0.3


def shares(probs: np.ndarray) -> dict:
    """probs (N, h, w, K) -> per-frame shares of air (all patches) and bubbles (patches not back)."""
    lab = probs.argmax(-1)
    n = lab.shape[1] * lab.shape[2]
    air = (lab == AIR).reshape(len(lab), -1).sum(1) / n
    notback = np.maximum((lab != BACK).reshape(len(lab), -1).sum(1), 1)
    bub = (lab == BUBBLES).reshape(len(lab), -1).sum(1) / notback
    back = (lab == BACK).reshape(len(lab), -1).sum(1) / n
    return dict(air=air, bubbles_of_view=bub, back=back)


def cmd_patch():
    import torch
    from uwid import patchseg as ps
    torch.set_num_threads(8)
    from uwid.detect import _torch_setup
    _torch_setup()
    model = ps.load_backbone()
    clf = ps.load_classifier(PATCH / "clf_w1280_h0_all.pt")[0]
    CACHE.mkdir(parents=True, exist_ok=True)
    for cid, (v, t0, t1, short, what, sid) in REAL.items():
        if v == tracks.DEV_VIDEO:
            continue
        out = CACHE / f"patch_{cid}.npz"
        if out.exists():
            continue
        m4v = v.replace(".mov", ".m4v")
        tab = uv.frame_table(m4v)
        want = list(range(tab.index_at(t0), tab.index_at(t1), 3))
        fr_i, sh = [], []
        buf = []
        for fr in uv.iter_frames_at(m4v, want):
            buf.append(fr)
            if len(buf) == 8:
                f = ps.extract(model, [b.image for b in buf], 1280)
                p = torch.stack([ps.predict_probs(clf, f[j]) for j in range(len(buf))]).cpu().numpy()
                fr_i += [b.index for b in buf]
                sh.append(p.astype(np.float16))
                buf = []
        if buf:
            f = ps.extract(model, [b.image for b in buf], 1280)
            p = torch.stack([ps.predict_probs(clf, f[j]) for j in range(len(buf))]).cpu().numpy()
            fr_i += [b.index for b in buf]
            sh.append(p.astype(np.float16))
        probs = np.concatenate(sh)
        np.savez_compressed(out, frames=np.array(fr_i), **shares(probs.astype(np.float32)))
        print(out.name, len(fr_i), flush=True)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, a = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and a is None:
            a = i
        elif not m and a is not None:
            out.append((a, i))
            a = None
    return out


def clean(mask: np.ndarray, fps: float) -> np.ndarray:
    m = mask.copy()
    for a, b in runs(m):
        if b - a < MIN_BLOCK_S * fps:
            m[a:b] = False
    r = runs(m)
    for (a0, b0), (a1, b1) in zip(r[:-1], r[1:]):
        if a1 - b0 < MIN_OPEN_S * fps:
            m[b0:a1] = True
    return m


def blocked_frames(cid: str) -> dict:
    v, t0, t1, short, what, sid = REAL[cid]
    tab = uv.frame_table(v)
    fps = 1 / tab.frame_step
    i0 = int(np.ceil(tab.index_at(t0) / tracks.STAGE1_STRIDE) * tracks.STAGE1_STRIDE)
    i1 = tab.index_at(t1)
    frames = np.arange(i0, i1)
    if v == tracks.DEV_VIDEO:
        z = np.load(PATCH / "pred_dev_w1280_h0.npz")
        sel = (z["frames"] >= i0 - 10) & (z["frames"] < i1 + 10)
        sf = z["frames"][sel]
        sh = shares(z["probs"][sel].astype(np.float32))
        lens_by_sample = sh["air"] >= AIR_SHARE
        lens_source = "patch classifier air share (no sensor log)"
    else:
        z = np.load(CACHE / f"patch_{cid}.npz")
        sf = z["frames"]
        sh = dict(air=z["air"], bubbles_of_view=z["bubbles_of_view"], back=z["back"])
        lens_by_sample = np.zeros(len(sf), bool)
        lens_source = "depth sensor (surfacings.csv)"
    bub_by_sample = sh["bubbles_of_view"] >= BUBBLE_SHARE
    near = np.abs(frames[:, None] - sf[None, :]).argmin(1)
    lens = lens_by_sample[near]
    bub = bub_by_sample[near]
    if v != tracks.DEV_VIDEO:
        s = pd.read_csv(SURFACINGS)
        times = tab.times[frames]
        for r in s[s.in_video.astype(bool)].itertuples():
            if f"-{int(r.piece):05d}" in Path(v).stem:
                lens |= (times >= r.pts_exit_pred - LENS_PAD_S) & (times <= r.pts_entry_pred + LENS_PAD_S)
    raw = lens | bub
    m = clean(raw, fps)
    ranges = [(int(frames[a]), int(frames[b - 1]) + 1) for a, b in runs(m)]
    return dict(case=cid, video=Path(v).name, frames=[int(i0), int(i1)], fps=fps, lens_source=lens_source,
                rule=dict(air_share=AIR_SHARE, bubble_share=BUBBLE_SHARE, lens_pad_s=LENS_PAD_S,
                          min_block_s=MIN_BLOCK_S, min_open_s=MIN_OPEN_S),
                n_lens=int(lens.sum()), n_bubbles=int(bub.sum()), n_blocked=int(m.sum()),
                ranges=ranges, ranges_s=[[round(float(tab.times[a]), 2), round(float(tab.times[b - 1]), 2)] for a, b in ranges],
                per_frame=dict(frame=frames.tolist(), lens=lens.astype(int).tolist(), bubbles=bub.astype(int).tolist(),
                               blocked=m.astype(int).tolist()))


def cmd_rule():
    (HERE / "results").mkdir(exist_ok=True)
    for cid in REAL:
        try:
            b = blocked_frames(cid)
        except FileNotFoundError as e:
            print(cid, "missing", e)
            continue
        (HERE / "results" / f"blocked_{cid}.json").write_text(json.dumps(b))
        print(cid, b["video"], "blocked", b["ranges_s"], f"lens {b['n_lens']} bubbles {b['n_bubbles']} frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["patch", "rule"])
    a = ap.parse_args()
    {"patch": cmd_patch, "rule": cmd_rule}[a.cmd]()
