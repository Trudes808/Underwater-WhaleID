"""Test-recording sample for WP3 part 2: 120 evenly spaced frames with a fixed random offset, plus frames at 2 per
second inside chosen windows (for dense event labelling). One decode pass in order (no seeking).

    PY=.venv/bin/python
    $PY experiments/02_reference_labels/sample_test.py VIDEO OUTDIR --n 120 --seed 4 [--windows auto --wseed 1004 | --windows 0,180 600,780]

Sample: spacing = frames_listed / n; offset = uniform(0, spacing) from numpy default_rng(seed); frame k = round(offset
+ k * spacing). Windows "auto": two non-overlapping 180 s windows with starts from default_rng(wseed).uniform(0,
duration - 180). Writes OUTDIR/sample/<frame>.jpg (full size), OUTDIR/win_<start>/<frame>.jpg (every 15th frame,
960 px wide) and OUTDIR/index.json (frames decoded, fps, sample frame list with times from the decoder).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--windows", nargs="*", default=[]); ap.add_argument("--wseed", type=int, default=None)
    ap.add_argument("--wstep", type=int, default=15)
    a = ap.parse_args()
    v = cv2.VideoCapture(a.video); listed = int(v.get(cv2.CAP_PROP_FRAME_COUNT)); fps = v.get(cv2.CAP_PROP_FPS)
    sp = listed / a.n; off = float(np.random.default_rng(a.seed).uniform(0, sp))
    sample = {int(round(off + k * sp)) for k in range(a.n)}
    dur = listed / fps
    wins = []
    if a.windows == ["auto"]:
        r = np.random.default_rng(a.wseed)
        while len(wins) < 2:
            s = float(r.uniform(0, dur - 180))
            if all(abs(s - w[0]) >= 180 for w in wins):
                wins.append((s, s + 180))
    else:
        wins = [tuple(map(float, w.split(","))) for w in a.windows]
    wframes = {}
    for s, e in wins:
        d = a.out / f"win_{int(s):04d}"; d.mkdir(parents=True, exist_ok=True)
        for i in range(int(round(s * fps)), int(round(e * fps)) + 1, a.wstep):
            wframes[i] = d
    (a.out / "sample").mkdir(parents=True, exist_ok=True)
    i = 0; times = {}
    last = max(list(sample) + list(wframes))
    while i <= last:
        need = i in sample or i in wframes
        if need:
            ok, f = v.read()
        else:
            ok = v.grab()
        if not ok:
            break
        if need:
            t = v.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if i in sample:
                cv2.imwrite(str(a.out / "sample" / f"{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92]); times[i] = t
            if i in wframes:
                cv2.imwrite(str(wframes[i] / f"{i:06d}.jpg"), cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
        i += 1
    # count the rest of the frames
    while v.grab():
        i += 1
    meta = {"video": Path(a.video).name, "frames_listed": listed, "frames_decoded": i, "fps": fps, "n": a.n, "seed": a.seed,
            "spacing_frames": sp, "offset_frames": off, "sample": {str(k): round(times[k], 3) for k in sorted(times)},
            "windows": [[round(s, 3), round(e, 3)] for s, e in wins], "wseed": a.wseed, "wstep": a.wstep}
    (a.out / "index.json").write_text(json.dumps(meta, indent=1)); print(json.dumps({k: meta[k] for k in ("frames_listed", "frames_decoded", "fps", "offset_frames", "windows")}))


if __name__ == "__main__":
    main()
