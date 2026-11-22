"""The demo stretches (chosen from stage 1's runs, `stretches.py` -> results/stretches.csv) and a runner.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/07_tracks_demo/clips.py run 01 --method owl_sam3
    .venv/bin/python experiments/07_tracks_demo/clips.py list

Each stretch lies inside one stage-1 run (whale in both frames of every 1 s window, gaps <= 1 s bridged), except
clip 02, which spans the bubble curtain at 66-69 s that splits two runs (that is the point of the clip).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from uwid import tracks  # noqa: E402

V = tracks.TEST_DIR + "/CC-11-96-20250203-"
DEV = tracks.DEV_VIDEO

# id: (video, t0, t1, short name, kind, stage-1 run it sits in)
CLIPS = {
    "01": (DEV, 24.0, 36.0, "comp_group_crossing", "a: competitive group, several whales crossing", "18-66 s"),
    "02": (DEV, 61.0, 76.0, "bubble_curtain_return", "d: whales hidden by a bubble curtain and back", "18-66 + 69-87 s"),
    "03": (V + "114256-000-00007.mov", 1212.5, 1230.0, "single_whale_passing", "b: one large whale passing in good light", "1214-1230 s"),
    "04": (V + "101236-000-00004.mov", 650.0, 668.0, "very_near_whale_on_back", "b: very near whale head-down, touching the back", "650-724 s"),
    "05": (V + "111250-000-00006.mov", 456.0, 476.0, "calf_fish_behind_back", "c + d: calf close with fish, goes behind the back and returns", "409-485 s"),
    "06": (V + "104242-000-00005.mov", 736.0, 754.0, "far_hazy_whale_approaching", "e: far hazy whale coming closer", "735-760 s"),
    "07": (V + "111250-000-00006.mov", 822.0, 840.0, "hazy_calf_near_back", "e + d: hazy calf near the back, dives out of view, fish", "821-868 s"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "list"])
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--method", default="owl_sam3", choices=list(tracks.METHODS))
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="start another stretch only while less than this many seconds have passed (keeps one "
                         "GPU-locked command short); items are clip:method or clip (with --method)")
    a = ap.parse_args()
    if a.cmd == "list":
        for k, c in CLIPS.items():
            print(k, Path(c[0]).name, c[1], c[2], c[3], "|", c[4], "| stage-1 run", c[5])
        return
    import time
    t_start = time.time()
    for item in a.clips or list(CLIPS):
        k, _, m = item.partition(":")
        m = m or a.method
        v, t0, t1 = CLIPS[k][:3]
        if tracks.is_done(v, t0, t1, m) and not a.force:
            continue
        if a.budget_s is not None and time.time() - t_start > a.budget_s:
            print(f"budget reached before {item}")
            break
        tracks.run_tracks(v, t0, t1, method=m, max_frames=a.max_frames, force=a.force)


if __name__ == "__main__":
    main()
