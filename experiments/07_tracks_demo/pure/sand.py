"""WP8c step 5: does the WP8b blocked-view rule see stirred-up sand? (GPU, small.)
No sand span in the labelling agent's blocked_spans.csv has a whale before and after, so no tracker comparison is
possible; this only checks what the patch classifier calls sand frames, 1 frame per second, sand spans of
00017 (2025-01-02 deployment, 4K .mov, downscaled to 1920 first) and the open-water seconds around them.

    flock ... .venv/bin/python experiments/07_tracks_demo/pure/sand.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent / "gap"))
import blocked as BL  # noqa: E402

from uwid import video as uv  # noqa: E402

VIDEO = "/home/sat3737/NOAA_whales/underwater_video/TestSubset 2025-1-2 CATS -85/Video/CC-09-85-HRA-20250102-141304-000-00017.mov"
SPANS = HERE.parents[1] / "02_reference_labels/labels/CC-09-85-HRA-20250102-141304-000-00017/blocked_spans.csv"
NAMES = ["water", "whale", "back", "fish", "bubbles", "surface", "air", "seafloor"]


def main():
    import torch
    from uwid import patchseg as ps
    from uwid.detect import _torch_setup
    _torch_setup()
    sp = pd.read_csv(SPANS)
    sand = sp[sp.kind == "sand"]
    times = []
    for r in sand.itertuples():
        times += [(float(t), "sand") for t in np.arange(r.start_s + 0.5, r.end_s, 2.0)]
    blocked_any = [(r.start_s, r.end_s) for r in sp.itertuples()]
    for t in np.arange(330, 860, 10.0):
        if not any(a - 3 <= t <= b + 3 for a, b in blocked_any):
            times.append((float(t), "open"))
    times.sort()
    model = ps.load_backbone()
    clf = ps.load_classifier(ps.CACHE / "clf_w1280_h0_all.pt")[0]
    rows = []
    with uv.VideoReader(VIDEO) as rd:
        for t, kind in times:
            fr = rd.frame_at(t)
            if fr is None:
                continue
            img = cv2.resize(fr.image, (1920, 1080), interpolation=cv2.INTER_AREA)
            with torch.no_grad():
                f = ps.extract(model, [img], 1280)
                lab = ps.predict_probs(clf, f[0]).argmax(-1).cpu().numpy()[None]
            sh = BL.shares(np.eye(8)[lab])
            cls = np.bincount(lab.ravel(), minlength=8) / lab.size
            rows.append(dict(t=t, kind=kind, air=round(float(sh["air"][0]), 3),
                             bubbles_of_view=round(float(sh["bubbles_of_view"][0]), 3),
                             blocked=bool(sh["air"][0] >= BL.AIR_SHARE or sh["bubbles_of_view"][0] >= BL.BUBBLE_SHARE),
                             brightness=round(float(img.mean()), 1), contrast=round(float(img.std()), 1),
                             **{f"cls_{n}": round(float(c), 3) for n, c in zip(NAMES, cls)}))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "results" / "sand_00017.csv", index=False)
    print(d.groupby("kind")[["blocked", "air", "bubbles_of_view"] + [f"cls_{n}" for n in NAMES]].mean().round(3).to_string())
    print(d.groupby("kind").size())


if __name__ == "__main__":
    main()
