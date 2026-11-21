"""Stage 1 -> candidate stretches: read WP5a's cached OWLv2 run (prompt "whale", every 15th frame = 2 per 1 s
window), mark the 1 s windows with a whale (both frames >= 0.25, the fixed operating point), join them into runs and
list runs of >= 8 s with the size of the largest whale box and the number of whale boxes per frame.

    .venv/bin/python experiments/07_tracks_demo/stretches.py               # all cached videos -> stretches.csv
    .venv/bin/python experiments/07_tracks_demo/stretches.py --show 00006  # print the runs of one piece

Also writes the kept share per recording (share of 1 s windows stage 1 keeps, with and without 1 s padding), used
for the cost estimate. Nothing here is tuned: the window rule is WP5a's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from uwid import tracks  # noqa: E402

VIDEOS = [tracks.DEV_VIDEO] + sorted(str(p) for p in Path(tracks.TEST_DIR).glob("CC-11-96-20250203-*-000-000*.m4v")
                                     if tracks.has_stage1(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None)
    ap.add_argument("--min-s", type=float, default=8.0)
    a = ap.parse_args()
    rows, share = [], []
    for v in VIDEOS:
        win = tracks.stage1_windows(v)
        runs = tracks.runs_of(win, min_s=a.min_s, gap_s=1)
        for r in runs:
            r["video"] = Path(v).stem
            rows.append(r)
        n = len(win)
        k = int(win.whale.sum())
        pad = win.whale | win.whale.shift(1, fill_value=False) | win.whale.shift(-1, fill_value=False)
        share.append(dict(video=Path(v).stem, windows=n, whale_windows=k, kept_share=round(k / n, 4),
                          kept_share_pad1=round(float(pad.mean()), 4), runs_ge_8s=len(runs)))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "stretches.csv", index=False)
    pd.DataFrame(share).to_csv(HERE / "results" / "kept_share.csv", index=False)
    print(pd.DataFrame(share).to_string())
    if a.show:
        sel = df[df.video.str.contains(a.show)]
        print(sel.to_string())


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    main()
