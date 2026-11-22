"""By-eye check of piece 00012: frames around the tag's release (14:38:44 sensor time) and during its float, with
the scan's per-window decision. Writes review_sheets/underwater/scan_release_00012.jpg.

    .venv/bin/python experiments/06_clips_e2e/release_sheet.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from uwid import paths, sheets, video  # noqa: E402

DEP = "2025-2-3 Mini-mother 11-96"
src = next((paths.footage_dir() / DEP / "Video").glob("CC-*-00012.mov"))
w = pd.read_csv(paths.cache_dir("scan") / DEP / "sensitive" / "windows" / f"{src.stem}.mov.csv").set_index("k")
times = [1480, 1500, 1508, 1512, 1515, 1517, 1520, 1525, 1530, 1540, 1546, 1549, 1555, 1600, 1650, 1700, 1750, 1780]
with video.VideoReader(src) as r:
    frs = [r.frame_at(t + 0.5) for t in times]
labels = [f"{t} s {w.loc[t, 'clock']}" for t in times]
caps = [f"{w.loc[t, 'tag_state']}/{w.loc[t, 'water']}: {'KEPT ' + w.loc[t, 'why'] if w.loc[t, 'kept'] else w.loc[t, 'why']}"
        for t in times]
out = paths.sheets_dir() / "scan_release_00012.jpg"
sheets.make_sheet([f.image for f in frs], labels, caps, cols=3, tile_width=640,
                  title=f"{src.stem}: release at about 1516 s and float (sensitive)", out=out)
print(out)
