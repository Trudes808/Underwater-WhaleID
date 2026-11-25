"""WP8c videos: demo stretches re-drawn with the purity policy; where a guard ends a number the video says so.

    .venv/bin/python experiments/07_tracks_demo/pure/render.py [--level medium]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as EV  # noqa: E402
from runs import tracks  # noqa: E402

SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
VIDEOS = [("01", "demo01", "comp_group"), ("02", "demo05", "calf_with_fish"), ("03", "demo02", "bubble_curtain"),
          ("04", "R06a", "breath_00007_1466s"), ("05", "demo07", "hazy_calf_fish_swap")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="medium")
    ap.add_argument("ids", nargs="*")
    a = ap.parse_args()
    cfg = EV.policy(a.level)
    for nn, key, short in VIDEOS:
        if a.ids and nn not in a.ids:
            continue
        tr = EV.apply(EV.run_key_dir(key), cfg)
        tr.meta["label"] = f"OWLv2 + SAM 3 tracker, purity guards ({a.level})"
        r = tracks.render_tracks(tr, SHEETS / f"demo_pure_{nn}_{short}.mp4", SHEETS / f"demo_pure_{nn}_{short}.jpg", crf=19)
        print(nn, key, len(tr.number), "whale numbers,", len(tr.endings), "endings", r["mb"], "MB")


if __name__ == "__main__":
    main()
