"""Decode the two frames of every 1 s window (0.25 s and 0.75 s after the window start, as the scan does) of the
recordings scored densely in 09_seafloor_sand, at 1280 x 720 (the patch classifier's input size), into a temporary
JPEG folder in the cache. CPU only (no torch). The GPU step (run_frames.py) reads them and the folders are deleted
after the features are cached (``--clean``).

Recordings: the six on-whale recordings of 2025-01-02 (00010, 00011, 00012, 00017, 00023, 00024; .mov, 4K) and one
piece of 2025-02-03 (00006, .mov) as a seafloor-free stretch.

    nice -n 10 .venv/bin/python experiments/09_seafloor_sand/decode.py
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import cv2  # noqa: E402

from uwid import video  # noqa: E402

VID = Path("/home/sat3737/NOAA_whales/underwater_video")
OUT = Path("/home/sat3737/NOAA_whales/underwater_cache/seafloor/frames_tmp")
D2 = VID / "TestSubset 2025-1-2 CATS -85" / "Video"
D1 = VID / "2025-2-3 Mini-mother 11-96" / "Video"
DENSE = [D2 / f"CC-09-85-HRA-20250102-{s}.mov" for s in
         ("141304-000-00017", "154322-000-00023", "122956-000-00010", "124244-000-00011", "130824-000-00012",
          "160604-000-00024")] + [D1 / "CC-11-96-20250203-111250-000-00006.mov"]


def main():
    if "--clean" in sys.argv:
        shutil.rmtree(OUT, ignore_errors=True); return
    for v in DENSE:
        out = OUT / v.stem
        if (out / "DONE").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        t = time.time(); rows = []
        for fr in video.iter_frames(v, fps=2.0, start=0.25, size=(1280, 720)):
            cv2.imwrite(str(out / f"{fr.index:06d}.jpg"), cv2.cvtColor(fr.image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append(f"{fr.index},{fr.t:.4f}")
        (out / "index.csv").write_text("frame,t\n" + "\n".join(rows) + "\n")
        (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"{v.stem}: {len(rows)} frames in {time.time() - t:.0f} s", flush=True)
    (OUT / "ALL.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
