"""WP9: the ok-against-low boundary of the scan's tier, chosen on the development clip ONLY (2025-12-20 CG.m4v).

WP3b's AI by-eye grades of the largest other whale (hq / ok / low, 145 whale frames) against the scan's window
columns (sensitive run of the dev clip). Size of the largest whale is the labeller's main criterion, so the rule is
a box-size limit; a detector-score limit is added only if it helps without losing hq / ok frames (it does not).

    .venv/bin/python experiments/06_clips_e2e/tier_rule_dev.py      # prints the grid, writes results/tier_rule_dev.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uwid import paths

HERE = Path(__file__).resolve().parent
win = pd.read_csv(next((paths.cache_dir("scan") / "2025-12-20_CG" / "sensitive" / "windows").glob("*.csv")))
lab = pd.read_csv(HERE.parents[1] / "experiments/02_reference_labels/labels/2025-12-20_CG/frames.csv")
lab = lab[lab.quality.notna()].copy()
W = win.set_index("k")
k = np.floor(lab.time_s).astype(int).clip(upper=W.index.max())
for c in ("whale", "whale_px", "top_score"):
    lab[c] = k.map(W[c]).to_numpy()
good = lab.quality.isin(["hq", "ok"])
rows = []
for px in (300, 400, 450, 500, 550, 600, 650, 700):
    for sc in (0.25, 0.3, 0.4, 0.5):
        p = lab.whale.astype(bool) & (lab.whale_px.fillna(0) >= px) & (lab.top_score.fillna(0) >= sc)
        rows.append({"min_px": px, "min_score": sc, "hq_ok_kept_ok": int((p & good).sum()), "hq_ok": int(good.sum()),
                     "low_called_ok": int((p & ~good).sum()), "low": int((~good).sum())})
g = pd.DataFrame(rows)
print(g.to_string(index=False))
# choice: the largest size limit that loses no hq / ok frame, score limit only if it then removes low frames
best = g[(g.hq_ok_kept_ok == g.hq_ok)].sort_values(["low_called_ok", "min_px"], ascending=[True, False]).iloc[0]
print("chosen:", best.to_dict())
(HERE / "results" / "tier_rule_dev.json").write_text(json.dumps({"grid": rows, "chosen": best.to_dict(),
                                                                 "grades": lab.quality.value_counts().to_dict()},
                                                                indent=1, default=str))
