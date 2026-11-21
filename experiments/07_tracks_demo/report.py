"""Collect results/<clip>_<method>.json into the number tables of tables.md (printed as Markdown; the by-eye notes
are written by hand in tables.md).

    .venv/bin/python experiments/07_tracks_demo/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    rows, ov_rows = [], []
    for p in sorted((HERE / "results").glob("*_*.json")):
        if p.name in ("kept_share.json",):
            continue
        r = json.loads(p.read_text())
        if "clip" not in r:
            continue
        o = r.get("overlap_summary") or {}
        tr = pd.DataFrame(r["tracks"])
        gaps = int(tr.gaps_s.fillna("").astype(bool).sum()) if len(tr) and "gaps_s" in tr else 0
        rows.append(dict(clip=r["clip"], video=r["video"], times=f"{r['t0']:.1f}-{r['t1']:.1f}", method=r["method"],
                         tracks_started=r["n_started"], whale_ids_shown=r["n_whale_tracks"],
                         tracks_with_gaps=gaps,
                         dropped=", ".join(f"{k}:{v}" for k, v in r["roles"].items() if v not in ("whale", "back")),
                         ref_whales=o.get("n_ref_whales", ""), matched=o.get("matched", ""),
                         median_iou=o.get("median_iou_all", ""), back_iou=", ".join(f"{x:.2f}" for x in o.get("back_iou", [])),
                         gpu_s_per_s=r["gpu_s_per_video_s"], peak_gb=r["gpu_peak_gb"], mb=r.get("mb", "")))
        for x in r.get("overlap") or []:
            x = dict(x)
            x.update(clip=r["clip"], method=r["method"])
            ov_rows.append(x)
    d = pd.DataFrame(rows)
    print(d.to_markdown(index=False))
    if ov_rows:
        ov = pd.DataFrame(ov_rows)
        cols = [c for c in ["clip", "method", "frame", "t", "what", "ref_id", "grade", "track", "iou", "note"] if c in ov]
        print()
        print(ov[cols].to_markdown(index=False))
        w = ov[ov.what == "whale"]
        print()
        print(w.groupby(["method", "grade"]).iou.agg(["count", "median", lambda s: int((s >= 0.5).sum())]).to_markdown())


if __name__ == "__main__":
    main()
