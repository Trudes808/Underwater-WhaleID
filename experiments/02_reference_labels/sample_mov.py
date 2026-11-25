"""Sampling and stepping for the tag's own .mov files (WP3b part B, second deployment). Same sample design as
`sample_test.py` (part 2 of the first round), but frames are read with `uwid.video` (PyAV), so frame index = position
among the SHOWN frames in presentation order, the index `uwid scan` uses. Never pass a file still being copied.

    PY=.venv/bin/python; S=experiments/02_reference_labels/sample_mov.py
    $PY $S sample VIDEO OUTDIR --n 120 --seed S --wseed W     # 120 frames + two 180 s windows at 2 frames/s, seek per target
    $PY $S strip  VIDEO a b --out s.jpg [--tw 480 --cols 4]   # every frame a..b (seek to the keyframe before a)
    $PY $S coarse VIDEO OUTDIR --step 10 [--t0 --t1]            # one 480 px frame every 10 s -> OUTDIR/coarse_10s/
    $PY $S grab   VIDEO OUTDIR f1 f2 ...                       # extra full-size frames by index into OUTDIR/extra/

Sample: spacing = n_frames / n; offset = uniform(0, spacing) from numpy default_rng(seed); frame k = round(offset +
k * spacing). Windows: two non-overlapping 180 s windows, starts uniform in [0, duration - 180) from
default_rng(wseed). Writes OUTDIR/sample/<frame>.jpg (full size), OUTDIR/win_<start>/<frame>.jpg (every 15th frame,
960 px), OUTDIR/index.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid.video import frame_table, iter_frames, iter_frames_at, probe  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from frametool import tile  # noqa: E402


def copy_settled(path: Path, wait: float = 60.0) -> bool:
    import time
    s0 = path.stat().st_size; m0 = path.stat().st_mtime
    if time.time() - m0 < wait:
        return False
    return path.stat().st_size == s0


def sample(a):
    video = Path(a.video)
    if not copy_settled(video):
        sys.exit(f"{video} changed in the last 60 s: still being copied")
    tab = frame_table(video); n = tab.n_frames; times = tab.times
    dur = float(times[-1] - times[0]); fps = (n - 1) / dur if dur > 0 else 0
    sp = n / a.n; off = float(np.random.default_rng(a.seed).uniform(0, sp))
    want = {int(round(off + k * sp)) for k in range(a.n)}
    want = {min(i, n - 1) for i in want}
    wins = []
    r = np.random.default_rng(a.wseed)
    if a.wseed is None:
        pass  # no event windows (reduced plan, 2026-11-22)
    elif dur > 400:
        while len(wins) < 2:
            s = float(r.uniform(0, dur - 180))
            if all(abs(s - w[0]) >= 180 for w in wins):
                wins.append((s, s + 180))
    else:
        wins = [(0.0, dur)]
    wframes = {}
    for s, e in wins:
        d = a.out / f"win_{int(s):04d}"; d.mkdir(parents=True, exist_ok=True)
        i0 = int(np.searchsorted(times, times[0] + s)); i1 = int(np.searchsorted(times, times[0] + e))
        for i in range(i0, min(i1 + 1, n), a.wstep):
            wframes[i] = d
    (a.out / "sample").mkdir(parents=True, exist_ok=True)
    got = {}
    # 4K sources: frames are saved at 1920 x 1080 (all label / mask coordinates are in that space); grab gives full size
    for fr in iter_frames_at(video, sorted(want), size=(1920, 1080), threads=a.threads):
        Image.fromarray(fr.image).save(a.out / "sample" / f"{fr.index:06d}.jpg", quality=92); got[fr.index] = fr.t
    for fr in iter_frames_at(video, sorted(wframes), size=(960, 540), threads=a.threads) if wframes else []:
        Image.fromarray(fr.image).save(wframes[fr.index] / f"{fr.index:06d}.jpg", quality=88)
    nd = len(got)
    missing = sorted(want - set(got))
    meta = {"video": video.name, "reader": "uwid.video.iter_frames (PyAV); index = shown frame in presentation order",
            "frames_listed": n, "frames_saved": nd, "saved_size": [1920, 1080], "fps": fps, "duration_s": dur, "first_t": float(times[0]),
            "n": a.n, "seed": a.seed, "spacing_frames": sp, "offset_frames": off,
            "sample": {str(k): round(got[k], 3) for k in sorted(got)}, "missing": missing,
            "windows": [[round(s, 3), round(e, 3)] for s, e in wins], "wseed": a.wseed, "wstep": a.wstep,
            "size_bytes": video.stat().st_size}
    (a.out / "index.json").write_text(json.dumps(meta, indent=1))
    (a.out / "SAMPLED").write_text("ok\n")
    print(json.dumps({k: meta[k] for k in ("frames_listed", "fps", "duration_s", "offset_frames", "windows", "missing")}))


def coarse(a):
    """One frame every --step seconds at 480 x 270 into OUTDIR/coarse_<step>s/ (for finding seafloor spans)."""
    video = Path(a.video); tab = frame_table(video); t = tab.times - tab.times[0]
    idx = sorted({int(np.searchsorted(t, x)) for x in np.arange(a.t0, min(a.t1, t[-1]), a.step)})
    od = Path(a.args[0]) / f"coarse_{a.step:g}s"; od.mkdir(parents=True, exist_ok=True)
    old = json.loads((od / "index.json").read_text()) if (od / "index.json").exists() else {}
    times = {int(k): v for k, v in old.get("times", {}).items()}
    for fr in iter_frames_at(video, idx, size=(480, 270), threads=a.threads):
        Image.fromarray(fr.image).save(od / f"{fr.index:06d}.jpg", quality=85); times[fr.index] = round(fr.t - tab.times[0], 3)
    (od / "index.json").write_text(json.dumps({"video": video.name, "step_s": a.step, "times": {str(k): times[k] for k in sorted(times)},
                                                "fps": (tab.n_frames - 1) / float(t[-1])}, indent=0))
    print(len(times), "frames")


def strip(a):
    video = Path(a.video); tab = frame_table(video); s, e = int(a.args[0]), int(a.args[1])
    t0 = float(tab.times[s]); t1 = float(tab.times[min(e, tab.n_frames - 1)])
    ims, labs = [], []
    for fr in iter_frames(video, start=max(0.0, t0 - 1e-3), end=t1 + 1e-3, fmt="rgb24", size=(1280, 720), threads=a.threads):
        if s <= fr.index <= e:
            ims.append(Image.fromarray(fr.image)); labs.append(f"f{fr.index} t={fr.t - tab.times[0]:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True); tile(ims, labs, a.cols, a.tw, fs=14).save(a.out, quality=88)


def grab(a):
    """Full-size frames (the source's own resolution) by index, for fish crops."""
    video = Path(a.video); od = a.out / "extra"; od.mkdir(parents=True, exist_ok=True)
    size = (a.gw, a.gw * 9 // 16) if a.gw else None
    for fr in iter_frames_at(video, [int(x) for x in a.args], size=size, threads=a.threads):
        Image.fromarray(fr.image).save(od / f"{fr.index:06d}.jpg", quality=92)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "strip", "grab", "coarse"]); ap.add_argument("video")
    ap.add_argument("args", nargs="*"); ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--seed", type=int); ap.add_argument("--wseed", type=int)
    ap.add_argument("--wstep", type=int, default=15); ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--step", type=float, default=10.0); ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--tw", type=int, default=480); ap.add_argument("--gw", type=int, default=0, help="grab: width (0 = source size)"); ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_intermixed_args()
    if a.cmd == "sample":
        a.out = Path(a.args[0]) if a.args else a.out; a.out.mkdir(parents=True, exist_ok=True); sample(a)
    elif a.cmd == "coarse":
        coarse(a)
    elif a.cmd == "strip":
        strip(a)
    else:
        a.out = Path(a.args[0]); a.args = a.args[1:]; grab(a)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
