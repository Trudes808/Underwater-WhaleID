"""CPU: decode the frames WP14 needs at 1280 x 720 into underwater_cache/ownback/frames/<stem>/<frame>.jpg.

- template frames: every 10th scanned underwater on-whale window of every on-whale recording (frame_0 of the window);
- post-breath objects: for every scan whale object that begins within 10 s after a lens-out run, the best frame of
  its first window and of the windows 1 s and 2 s later.
The list is written to ownback/frames/index.csv (stem, frame, kind). DONE marker when finished.

    nice -n 10 .venv/bin/python experiments/10_own_back/sample.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
from uwid import video  # noqa: E402


def post_breath_objects(w: pd.DataFrame) -> list[dict]:
    runs = C.lens_out_runs(w)
    out = []
    for a, b in C.whale_objects(w):
        t = float(w.t0.iloc[a])
        prev = [r for r in runs if r[1] <= t + 1e-6]
        if not prev or t - prev[-1][1] > 10.0:
            continue
        out.append(dict(row0=a, row1=b, t_start=t, t_end=float(w.t1.iloc[b]), after_s=t - prev[-1][1],
                        lens_out=prev[-1]))
    return out


def main():
    od = C.OUT / "frames"; od.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, piece, stem, vpath, wpath in C.recordings():
        w = C.windows(wpath)
        uw = w[(w.water == "underwater") & (w.tag_state == "whale") & w.frame_0.notna()]
        tmpl = [int(x) for x in uw.frame_0.iloc[::10]]
        rows += [dict(src=src, piece=piece, stem=stem, frame=f, kind="template") for f in tmpl]
        for o in post_breath_objects(w):
            for d in (0, 1, 2):
                r = o["row0"] + d
                if r < len(w) and pd.notna(w.best_frame.iloc[r]):
                    rows.append(dict(src=src, piece=piece, stem=stem, frame=int(w.best_frame.iloc[r]),
                                     kind=f"post_breath_{d}"))
    idx = pd.DataFrame(rows).drop_duplicates(["stem", "frame", "kind"])
    idx.to_csv(od / "index.csv", index=False)
    print(idx.groupby(["src", "kind"]).size(), flush=True)
    for (src, piece, stem), g in idx.groupby(["src", "piece", "stem"]):
        d = od / stem; d.mkdir(exist_ok=True)
        want = sorted(set(int(f) for f in g.frame) - {int(p.stem) for p in d.glob("*.jpg")})
        if not want:
            continue
        t = time.time()
        vpath = C.VID / C.DEPL[src] / "Video" / f"{stem}.mov"
        n = 0
        for fr in video.iter_frames_at(vpath, want, size=(C.W, C.H)):
            cv2.imwrite(str(d / f"{fr.index:06d}.jpg"), cv2.cvtColor(fr.image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            n += 1
        print(f"{stem}: {n}/{len(want)} frames in {time.time() - t:.0f} s", flush=True)
    (od / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
