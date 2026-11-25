"""WP9: the `silhouette` flag's limits, chosen on the development clip ONLY (2025-12-20 CG.m4v).

WP3b's AI by-eye `silhouette` column (yes / no, frames with a graded whale) against the window columns of the
sensitive dev scan: whale box length `whale_px`, mean grey inside the box `bright`, mean grey of the ring around it
`ring`. The rule follows the definition (a large whale box, dark inside, brighter around); the grid picks the
limits. Result: the box measures barely separate silhouettes on this clip (AUC of each measure 0.43-0.67).

    .venv/bin/python experiments/06_clips_e2e/silhouette_rule_dev.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uwid import paths

HERE = Path(__file__).resolve().parent
win = pd.read_csv(next((paths.cache_dir("scan") / "2025-12-20_CG" / "sensitive" / "windows").glob("*.csv")))
lab = pd.read_csv(HERE.parents[1] / "experiments/02_reference_labels/labels/2025-12-20_CG/frames.csv")
lab = lab[lab.silhouette.isin(["yes", "no"])].copy()
W = win.set_index("k")
k = np.floor(lab.time_s).astype(int).clip(upper=W.index.max())
for c in ("whale", "whale_px", "bright", "ring"):
    lab[c] = k.map(W[c]).to_numpy()
lab["lift"] = lab.ring - lab.bright
y = lab.silhouette.eq("yes")
rows = []
for px in (576, 768, 960):
    for mb in (0.30, 0.35, 0.40):
        for ml in (0.0, 0.02, 0.05):
            p = lab.whale.astype(bool) & (lab.whale_px >= px) & (lab.bright <= mb) & (lab.lift >= ml)
            tp, fp = int((p & y).sum()), int((p & ~y).sum())
            rows.append({"min_px": px, "max_bright": mb, "min_lift": ml, "found": tp, "of_yes": int(y.sum()),
                         "false": fp, "of_no": int((~y).sum()), "f1": 2 * tp / (2 * tp + fp + int(y.sum()) - tp)})
g = pd.DataFrame(rows).sort_values("f1", ascending=False)
print(g.head(10).to_string(index=False))
best = g.iloc[0].to_dict()
print("chosen:", best)
(HERE / "results" / "silhouette_rule_dev.json").write_text(json.dumps({"grid": rows, "chosen": best}, indent=1))
