"""The tagged whale's back as a class without training: SAM 2.1 prompted by position (the back fills the bottom of the
frame), scored against the reference back masks.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04_whale_fish_detection/back_sam.py

Prompts (fixed before looking): A = box over the bottom 25 % of the frame; B = three points along the bottom edge
(x = 25 / 50 / 75 %, y = 97 %); C = A and B together. On frames where bubbles cover the back the reference mask
holds only the visible back pixels.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import CLASS_IDS, DEV_VIDEO, H, W, detect, load_masks, mask_frames, to_md


def main():
    import argparse
    from pathlib import Path
    import cv2
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="*", help="WP3 reference dirs (masks/ + frames/ or sample/); default: the dev clip")
    a = ap.parse_args()
    sam = detect.SamMasker()
    items = []   # (tag, frame, image, mask dir)
    if not a.ref:
        mf = mask_frames()
        imgs = dict(detect.iter_frames_cv(DEV_VIDEO, frames=mf))
        items = [("dev", f, imgs[f], None) for f in mf]
    else:
        for r in map(Path, a.ref):
            for f in mask_frames(r / "masks"):
                jp = next(iter(list(r.glob(f"*/{f:06d}.jpg"))), None)
                items.append((r.name[-5:], f, cv2.cvtColor(cv2.imread(str(jp)), cv2.COLOR_BGR2RGB), r / "masks"))
    box = [0, 0.75 * H, W, H]
    pts = [[0.25 * W, 0.97 * H], [0.5 * W, 0.97 * H], [0.75 * W, 0.97 * H]]
    rows = []
    for tag, f, img, mdir in items:
        cls, _, _ = load_masks(f) if mdir is None else load_masks(f, mdir)
        back = cls == CLASS_IDS["back"]
        out = {}
        for name, kw in [("A box", dict(input_boxes=[[box]])),
                         ("B points", dict(input_points=[[pts]], input_labels=[[[1, 1, 1]]])),
                         ("C box+points", dict(input_boxes=[[box]], input_points=[[pts]], input_labels=[[[1, 1, 1]]]))]:
            inp = sam.proc(images=img, return_tensors="pt", **kw).to(sam.dev)
            with torch.no_grad():
                o = sam.model(**inp, multimask_output=True)
            ms = sam.proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0]
            m = ms[o.iou_scores[0, 0].float().cpu().argmax()].numpy() > 0
            u = (m | back).sum()
            out[name] = float((m & back).sum() / u) if u else np.nan
            out[name + " area"] = float(m.mean())
        rows.append(dict(video=tag, frame=f, back_px=int(back.sum()), **out))
    d = pd.DataFrame(rows)
    out_dir = detect.detect_cache_dir(DEV_VIDEO, "back_sam")
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_dir / ("back_sam_dev.csv" if not a.ref else "back_sam_test.csv"), index=False)
    has = d.back_px > 1000
    print(f"mask frames with a back mask: {int(has.sum())} of {len(d)}")
    print(to_md(pd.DataFrame([dict(prompt=k, median_iou=d.loc[has, k].median(), iou_ge_05=int((d.loc[has, k] >= 0.5).sum()),
                                   iou_ge_07=int((d.loc[has, k] >= 0.7).sum()),
                                   median_area_no_back=d.loc[~has, k + " area"].median())
                              for k in ("A box", "B points", "C box+points")])))


if __name__ == "__main__":
    main()
