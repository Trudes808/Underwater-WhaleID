"""Diagnostic view of one stored tracker run (for my own look, cache only): plain frame, the SAM back object
(obj_0, purple), the other tracked objects (red / orange) and the WP5b patch map's back class (yellow grid cells)
at a few times.

    .venv/bin/python experiments/10_own_back/look_run.py <run dir> <out.jpg> t1 t2 ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from uwid import video


def main():
    run, out, ts = Path(sys.argv[1]), Path(sys.argv[2]), [float(x) for x in sys.argv[3:]]
    m = json.loads((run / "meta.json").read_text())
    z = np.load(run / "masks.npz"); p = np.load(run / "patch.npz")
    tab = video.frame_table(m["video"])
    f0 = tab.index_at(m["t0"])
    idx = [tab.index_at(t) for t in ts]
    frames = {fr.index: fr.image for fr in video.iter_frames_at(m["video"], idx, size=(1280, 720))}
    tiles = []
    for t, k in zip(ts, idx):
        im = cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR)
        plain = im.copy()
        i = k - f0
        for name, col in (("obj_0", (200, 80, 160)), ("obj_1", (40, 60, 255)), ("obj_2", (0, 160, 255)), ("obj_3", (0, 255, 0))):
            if name in z.files and 0 <= i < len(z[name]):
                mk = np.unpackbits(z[name][i], axis=1)[:, :1280].astype(np.uint8)
                cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cs, -1, col, 3)
        j = int(np.abs(p["frames"] - k).argmin())
        lab = p["labels"][j]
        back = cv2.resize((lab == 2).astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(back, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (0, 255, 255), 2)
        cv2.putText(im, f"t {t:.1f}s  purple SAM back obj_0, red/orange tracks, yellow = patch 'back'", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(np.hstack([cv2.resize(plain, (640, 360)), cv2.resize(im, (640, 360))]))
    cv2.imwrite(str(out), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
