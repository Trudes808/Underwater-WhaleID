"""Burned-in clock per recording of a deployment (WP9): share of frames read, monotonic, fit against video time.

    .venv/bin/python experiments/06_clips_e2e/clock_report.py "<deployment folder>" [--every 10 --burst 10]

Glyphs are chosen or learned by ``uwid.timecode.glyphs_for`` (a new layout is learned from the deployment's own
first recording and its file name). Per-frame readings go to ``<cache>/timecode/<stem>_clock.csv``; the summary to
``results/clock_<tag>.json`` (no footage, numbers only)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uwid import names, paths, pipeline, timecode

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--every", type=float, default=10.0)
ap.add_argument("--burst", type=int, default=10)
a = ap.parse_args()
dep = pipeline.discover(a.folder)[0]
out = {}
tag = None
for rec in dep.recordings:
    info = names.parse_tag_video_name(rec.path)
    tag = tag or (info["tag"] if info else dep.name)
    gs = timecode.glyphs_for(rec.path, log=print)
    res = timecode.read_video_clock(rec.path, glyphs=gs, every_s=a.every, burst=a.burst)
    s = res["summary"]
    rows = pd.DataFrame(res["rows"])
    rows.to_csv(paths.cache_dir("timecode") / f"{rec.stem}_clock.csv", index=False)
    ok = rows[rows.clock_s.notna()] if len(rows) else rows
    if len(ok):
        st = np.diff(ok.clock_s.to_numpy())
        dt = np.diff(ok.t.to_numpy())
        cons = dt < 0.05                      # consecutive frames inside a burst
        s["step_ms_consecutive"] = {q: float(np.percentile(st[cons] * 1000, q)) for q in (1, 50, 99)} if cons.any() else None
        s["min_digit_score_p1"] = float(np.percentile(ok.score, 1))
        s["ms_last_digit_counts"] = ok.text.str[-1].value_counts().to_dict()
    s.pop("path", None)
    out[rec.stem] = s
    fit = s.get("fit") or {}
    print(f"{rec.stem}: read {s['n_read']}/{s['n_frames']} ({s['share_read']:.1%}), monotonic {s.get('monotonic')}, "
          f"backward steps {s.get('n_backward_steps')}, slope {fit.get('slope')}, rms {fit.get('rms_residual_inliers_s')}, "
          f"outliers {fit.get('n_outliers')}, clock-name {s.get('clock_at_t0_minus_name_start_s')}", flush=True)
res_dir = Path(__file__).parent / "results"
res_dir.mkdir(exist_ok=True)
(res_dir / f"clock_{tag}.json").write_text(json.dumps(out, indent=1, default=str))
