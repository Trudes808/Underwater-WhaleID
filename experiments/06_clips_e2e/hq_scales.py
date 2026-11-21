"""Scales of the picture parts of the hq_whale tier, from the DEV clip only (2025-12-20 CG.m4v).

For the whale boxes >= 0.25 that `uwid scan` found on the dev clip, the 10th and 90th percentiles of brightness,
contrast against the surrounding water and sharpness inside the box. These become `HQ_SCALES` in
`src/uwid/pipeline.py` (part = 0 at the 10th, 1 at the 90th percentile). No truth for "high quality" exists yet:
this only puts the three measures on a common 0-1 footing.

    .venv/bin/python experiments/06_clips_e2e/hq_scales.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from uwid import paths, pipeline as pl  # noqa: E402

rec = pl.Recording(paths.footage_dir() / "2025-12-20 CG.m4v", "m4v")
boxes, frames = pl.load_boxes(paths.cache_dir("scan") / "2025-12-20_CG" / "cache", rec)
w = boxes[(boxes["query"] == "whale") & (boxes.score >= 0.25)].dropna(subset=["q_bright"])
out = {"n_boxes": int(len(w)), "n_frames": int(w.frame.nunique())}
for c in ("q_bright", "q_contrast", "q_sharp"):
    out[c] = {"p10": float(np.percentile(w[c], 10)), "p50": float(np.percentile(w[c], 50)),
              "p90": float(np.percentile(w[c], 90))}
print(json.dumps(out, indent=1))
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "hq_scales_dev.json").write_text(json.dumps(out, indent=1))
