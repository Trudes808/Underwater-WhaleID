"""Dense sheets for judging a run by eye: the outlined frame every --step seconds, 6 per row, 480 px wide.

    .venv/bin/python experiments/07_tracks_demo/judge.py 01 --method owl_sam3 --out <dir> [--step 0.5] [--t0 .. --t1 ..]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from uwid import tracks, video as uv  # noqa: E402
from clips import CLIPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--method", default="owl_sam3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--tile", type=int, default=480)
    ap.add_argument("--cols", type=int, default=6)
    a = ap.parse_args()
    v, t0, t1 = CLIPS[a.clip][:3]
    back_from = tracks.tracks_dir(v, t0, t1, "owl_sam3") if a.method == "text_sam3" else None
    tr = tracks.load_tracks(tracks.tracks_dir(v, t0, t1, a.method), back_from=back_from)
    f0, f1 = tr.meta["frames"]
    tab = uv.frame_table(v)
    every = max(1, int(round(a.step * tr.meta["fps"])))
    want = list(range(0, tr.n, every))
    sw, sh = tr.meta["store_size"]
    tiles = []
    for fr in uv.iter_frames_at(v, [f0 + i for i in want]):
        i = fr.index - f0
        if (a.t0 is not None and fr.t < a.t0) or (a.t1 is not None and fr.t > a.t1):
            continue
        img = cv2.resize(fr.image, (sw, sh), interpolation=cv2.INTER_AREA)
        out = tracks.draw_frame(img, tr, i, f"t = {fr.t:.2f} s  (i {i})", {})
        tiles.append(cv2.resize(out, (a.tile, a.tile * sh // sw), interpolation=cv2.INTER_AREA))
    C = a.cols
    while len(tiles) % C:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[j:j + C]) for j in range(0, len(tiles), C)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # split into pages of 6 rows
    for p in range(0, len(rows), 6):
        fn = out / f"judge_{a.clip}_{a.method}_{p // 6}.jpg"
        cv2.imwrite(str(fn), cv2.cvtColor(np.vstack(rows[p:p + 6]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(fn)


if __name__ == "__main__":
    main()
