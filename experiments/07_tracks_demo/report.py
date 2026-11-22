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


def to_md(d: pd.DataFrame, index: bool = False) -> str:
    if index:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


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
    print(to_md(d))
    if ov_rows:
        ov = pd.DataFrame(ov_rows)
        cols = [c for c in ["clip", "method", "frame", "t", "what", "ref_id", "grade", "track", "iou", "note"] if c in ov]
        print()
        print(to_md(ov[cols]))
        w = ov[ov.what == "whale"]
        print()
        g = w.groupby(["method", "grade"]).iou
        print(to_md(pd.DataFrame(dict(n=g.size(), median_iou=g.median(), iou_ge_05=g.apply(lambda s: int((s >= 0.5).sum())))), index=True))


if __name__ == "__main__":
    main()
