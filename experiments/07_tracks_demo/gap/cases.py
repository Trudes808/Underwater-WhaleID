"""WP8b: the stretches used for the gap experiments.

Synthetic gaps (identity truth = the uncut run of the same stretch): BASES. Real blocked views: REAL.
The four 2025-02-03 breaths are all surfacings (tagdata/surfacings.csv, pieces 00004-00010, in video) with stage-1
whale windows in at least 2 of the 4 s before the sensor exit and 2 of the 4 s after the sensor entry (found by
`find_breaths()`, 4 of 66 surfacings).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

from uwid import tracks  # noqa: E402

V = tracks.TEST_DIR + "/CC-11-96-20250203-"
DEV = tracks.DEV_VIDEO
SURFACINGS = Path("/home/sat3737/NOAA_whales/underwater_cache/tagdata/2025-2-3 Mini-mother 11-96/surfacings.csv")

# base stretches for the synthetic gaps: (video, t0, t1); uncut owl_sam3 run of exactly this stretch
BASES = {
    "B1": (DEV, 24.0, 48.0),                                   # competitive group (extends demo clip 01)
    "B3": (V + "114256-000-00007.mov", 1212.5, 1230.0),        # demo clip 03
    "B4": (V + "101236-000-00004.mov", 650.0, 668.0),          # demo clip 04
    "B5": (V + "111250-000-00006.mov", 456.0, 476.0),          # demo clip 05
    "B6": (V + "104242-000-00005.mov", 736.0, 754.0),          # demo clip 06
}
GAPS_S = (0.5, 1.0, 2.0, 4.0, 8.0)
POSITIONS = (1 / 3, 2 / 3)   # gap centre as a share of the stretch (clamped so 2 s before and 2.5 s after fit)
PRE_S, POST_S = 2.0, 2.5     # cut runs start PRE_S before the gap and end POST_S after it

# real blocked views: id -> (video, t0, t1, short name, what), sensor surfacing id for the 2025-02-03 breaths
REAL = {
    "01": (DEV, 61.0, 76.0, "dev_bubble_curtain", "dev clip bubble curtain (demo clip 02)", None),
    "02": (DEV, 5.0, 22.0, "dev_breath", "dev clip breath at 13.9-16.4 s (whales 5-11 s and from 18 s)", None),
    "03": (V + "104242-000-00005.mov", 365.5, 381.5, "breath_00005_372s", "breath, 00005, sensor exit 371.6 s", 6),
    "04": (V + "111250-000-00006.mov", 575.5, 590.0, "breath_00006_582s", "breath, 00006, sensor exit 581.7 s", 20),
    "05": (V + "111250-000-00006.mov", 1514.0, 1529.0, "breath_00006_1520s", "breath, 00006, sensor exit 1520.3 s", 26),
    "06": (V + "114256-000-00007.mov", 1459.5, 1474.5, "breath_00007_1466s", "breath, 00007, sensor exit 1465.8 s", 36),
}


def find_breaths():
    import numpy as np
    import pandas as pd
    s = pd.read_csv(SURFACINGS)
    s = s[s.in_video.astype(bool)]
    rows = []
    for p in range(4, 11):
        v = next(Path(tracks.TEST_DIR).glob(f"CC-11-96-20250203-*-000-{p:05d}.m4v"))
        w = tracks.stage1_windows(v).set_index("window")
        for r in s[s.piece == p].itertuples():
            a, b = r.pts_exit_pred, r.pts_entry_pred
            if not np.isfinite(a):
                continue
            pre, post = w.loc[int(a) - 4:int(a) - 1], w.loc[int(b) + 1:int(b) + 4]
            rows.append(dict(piece=p, surfacing_id=r.surfacing_id, exit=a, entry=b, pre_whale=int(pre.whale.sum()),
                             post_whale=int(post.whale.sum())))
    d = pd.DataFrame(rows)
    return d, d[(d.pre_whale >= 2) & (d.post_whale >= 2)]


if __name__ == "__main__":
    d, c = find_breaths()
    print(len(d), "surfacings in video;", len(c), "with whales before and after")
    print(c.to_string())
