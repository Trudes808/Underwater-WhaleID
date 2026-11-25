"""Decode the second frame of each labelled frame's window (frame + 15 = 0.5 s later) and save it as JPEG.

CPU only (no torch). The dev clip needs none (its labelled frames are every 15th frame). 2025-02-03: OpenCV decode
order of the .m4v, as the labels; 2025-01-02: shown-frame index of the 4K .mov through uwid.video, saved at
1920 x 1080 like the labelled frames. Output: underwater_cache/boxclf/partners/<stem>/<frame>.jpg, DONE marker.

    nice -n 10 .venv/bin/python experiments/08_whale_fish_boxes/partners.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402

import data as D  # noqa: E402
from uwid import detect, video  # noqa: E402


def main():
    out_root = D.CACHE / "partners"
    for rec in D.recordings(("d1", "d2")):
        out = out_root / rec.stem
        out.mkdir(parents=True, exist_ok=True)
        want = sorted({f + D.PARTNER_STEP for f in rec.labelled_frames()})
        todo = [f for f in want if not (out / f"{f:06d}.jpg").exists()]
        if not todo:
            continue
        t = time.time(); n = 0
        if rec.source == "d1":
            it = ((i, im) for i, im in detect.iter_frames_cv(rec.video, frames=todo))
        else:
            it = ((fr.index, fr.image) for fr in video.iter_frames_at(rec.video, todo, size=(1920, 1080)))
        for i, im in it:
            cv2.imwrite(str(out / f"{i:06d}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
        print(f"{rec.stem}: {n}/{len(todo)} partner frames in {time.time() - t:.0f} s", flush=True)
    (out_root / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
