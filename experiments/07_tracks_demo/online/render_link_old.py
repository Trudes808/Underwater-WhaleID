"""WP11a demo_link videos: old stretches, strict purity (after the fact, as WP8c) + the frozen lone-link layer.

    .venv/bin/python experiments/07_tracks_demo/online/render_link_old.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pure"))
sys.path.insert(0, str(HERE))
import evaluate as EV  # noqa: E402
import links_old as LO  # noqa: E402
from runs import tracks  # noqa: E402

SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
for key, name in (("R06b", "demo_link_02_breath_00007_1466s"),
                  ("demo07", "demo_link_03_hazy_calf_fish")):
    tr, n0 = LO.linked(EV.run_key_dir(key))
    tr.meta["label"] = "SAM 3 tracker, strict purity + lone links"
    r = tracks.render_tracks(tr, SHEETS / f"{name}.mp4", SHEETS / f"{name}.jpg", crf=20)
    print(name, n0, "->", len(set(tr.number.values())), "numbers;", sum(1 for L in tr.links if L["why"] == "linked"),
          "links", r["mb"], "MB", flush=True)
