"""The 14 test mask frames of 2025-02-03 (WP3): whale and fish instances found by the OWLv2 run at the dev operating
points, fish boxes that land on whales and whale boxes that land on fish. Nothing tuned here.

    .venv/bin/python experiments/04_whale_fish_detection/test_masks.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import FISH_Q, detect, instances, iou_one, mask_frames, match_instances, tag_boxes, to_md
from score_test import FISH_THR, QUERY, RUN, THR, VD

REF = Path("/home/sat3737/NOAA_whales/underwater_cache/reference")


def main():
    rows, cross = [], dict(fish_boxes_on_whale=0, fish_boxes=0, whale_boxes_on_fish=0, whale_boxes=0)
    for v in sorted(VD.glob("CC-11-96-20250203-*.m4v")):
        mdir = REF / v.stem / "masks"
        b, frames = detect.load_cached(v, RUN)
        if not mdir.exists() or len(frames) == 0:
            continue
        b = tag_boxes(b)
        for f in mask_frames(mdir):
            if f not in set(frames):
                continue
            ins = instances(f, mdir)
            wi = [i for i in ins if i["cls"] == "whale"]
            fi = [i for i in ins if i["cls"] == "fish"]
            wb = b[(b.frame == f) & (b["query"] == QUERY) & (b.score >= THR) & ~b.clock & ~b.whole]
            fb = b[(b.frame == f) & b["query"].isin(FISH_Q) & (b.score >= 0.3) & ~b.clock & ~b.whole]
            mw, _ = match_instances(wi, wb)
            for th in (0.3, FISH_THR):
                mf, _ = match_instances(fi, fb[fb.score >= th])
                for i in fi:
                    rows.append(dict(video=v.stem[-5:], frame=f, cls="fish", thr=th, size=int(max(i["box"][2] - i["box"][0], i["box"][3] - i["box"][1])),
                                     found=i["id"] in mf))
            for i in wi:
                rows.append(dict(video=v.stem[-5:], frame=f, cls="whale", thr=THR, grade=i["grade"], found=i["id"] in mw))
            # cross-calls: a box lying mostly (>= 50 % of its area) on the other class's pixels
            for _, r in fb[fb.score >= FISH_THR].iterrows():
                cross["fish_boxes"] += 1
                if any(_on(r, i) for i in wi):
                    cross["fish_boxes_on_whale"] += 1
            for _, r in wb.iterrows():
                cross["whale_boxes"] += 1
                # on a fish: fish pixels fill >= 25 % of the box and whale pixels < 10 %
                if fi and sum(_frac(r, i) for i in fi) >= 0.25 and sum(_frac(r, i) for i in wi) < 0.10:
                    cross["whale_boxes_on_fish"] += 1
    d = pd.DataFrame(rows)
    if d.empty:
        print("no test mask frames with detections yet")
        return
    t = d.groupby(["cls", "thr"]).agg(n=("found", "size"), found=("found", "sum"), frames=("frame", "nunique")).reset_index()
    print("### test mask frames: instances found (OWLv2; whale prompt 'whale' >= 0.25; fish = any fish prompt)\n")
    print(to_md(t))
    if (d.cls == "fish").any():
        fd = d[(d.cls == "fish") & (d.thr == FISH_THR)]
        print(f"\nfish sizes (px, longest side): median {fd['size'].median():.0f}; found at {FISH_THR}: "
              f"<= 60 px {int(fd[fd['size'] <= 60].found.sum())}/{int((fd['size'] <= 60).sum())}, "
              f"> 60 px {int(fd[fd['size'] > 60].found.sum())}/{int((fd['size'] > 60).sum())}")
    print("\ncross-calls on these frames:", cross)
    d.to_csv(Path(detect.paths.cache_dir("detect")) / "test_maskframes_instances.csv", index=False)


def sheet():
    import cv2
    from make_sheets import draw, grid
    from common import SHEETS
    tiles = []
    for v in sorted(VD.glob("CC-11-96-20250203-*.m4v")):
        mdir = REF / v.stem / "masks"
        b, frames = detect.load_cached(v, RUN)
        if not mdir.exists() or len(frames) == 0:
            continue
        b = tag_boxes(b)
        for f in mask_frames(mdir):
            jp = next(iter((REF / v.stem).glob(f"*/{f:06d}.jpg")), None)
            if jp is None:
                continue
            img = cv2.cvtColor(cv2.imread(str(jp)), cv2.COLOR_BGR2RGB)
            sel = b[(b.frame == f) & ~b.clock & ~b.whole & (((b["query"] == QUERY) & (b.score >= THR)) |
                                                            (b["query"].isin(FISH_Q) & (b.score >= FISH_THR)))]
            whale_mask = np.zeros(img.shape[:2], bool)
            for i in instances(f, mdir):
                if i["cls"] == "whale":
                    whale_mask |= i["mask"]
            tiles.append(draw(img, sel, f"{v.stem[-5:]} f{f} | OWLv2 whale>={THR} (red), fish>={FISH_THR} (green); "
                                         f"reference whale pixels magenta", width=800, mask=whale_mask))
    grid(tiles, 3, SHEETS / "detect_test_maskframes.jpg",
         "2025-02-03 test mask frames (WP3): OWLv2 boxes at the dev operating points")


def _frac(r, ins):
    x0, y0, x1, y1 = [int(round(v)) for v in (r.x0, r.y0, r.x1, r.y1)]
    return ins["mask"][max(y0, 0):y1, max(x0, 0):x1].sum() / max((x1 - x0) * (y1 - y0), 1)


def _on(r, ins, need=0.5, of="box"):
    x0, y0, x1, y1 = [int(round(v)) for v in (r.x0, r.y0, r.x1, r.y1)]
    inside = ins["mask"][max(y0, 0):y1, max(x0, 0):x1].sum()
    denom = max((x1 - x0) * (y1 - y0), 1) if of == "box" else max(ins["area"], 1)
    return inside / denom >= need


if __name__ == "__main__":
    import sys
    sheet() if "--sheet" in sys.argv else main()
