"""Read the clock burned into CATS tag video frames (bottom right, e.g. '03Feb25 10:22:38.200').

Stop-gap reader for WP1 until uwid.timecode exists. The font is a fixed 8-px-pitch bitmap font: 20 cells starting at
x = 1757 (for 1920x1080), rows 1061-1074. Glyph templates for the digits are built once from consecutive frames of
one video (the tenths digit walks through 0..9 in one second) and stored in the cache, not in the repo.

    .venv/bin/python experiments/01_metadata/clock_ocr.py build <video>       # build digit templates
    .venv/bin/python experiments/01_metadata/clock_ocr.py read <video> [step_s]  # clock vs pts table -> cache
"""
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd

TEMPLATES = Path("/home/sat3737/NOAA_whales/underwater_cache/tagdata/clock_templates.npz")
X0, PITCH, Y0, Y1 = 1757, 8, 1061, 1075
DIGIT_CELLS = {"h": (8, 9), "m": (11, 12), "s": (14, 15), "d": (17,)}


def cells_of(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    sx, sy = w / 1920, h / 1080
    assert sx == 1 and sy == 1, "reader written for 1920x1080"
    return np.stack([gray[Y0:Y1, X0 + PITCH * k - 1:X0 + PITCH * k + 7] for k in range(20)]) > 128


def build(video: str, start: float = 100.0):
    c = av.open(video)
    s = c.streams.video[0]
    c.seek(int(start / s.time_base), stream=s)
    pats, seq = {}, []
    for fr in c.decode(s):
        if fr.time < start:
            continue
        b = cells_of(fr.to_ndarray(format="gray"))[17]
        k = b.tobytes()
        if k not in pats:
            pats[k] = b
        seq.append(k)
        if len(pats) == 10 and len(seq) > 40:
            break
    # assign digits by order of appearance: tenths digit increases by one at each change
    order = []
    for k in seq:
        if not order or order[-1] != k:
            order.append(k)
    # the first complete cycle gives 10 distinct consecutive patterns; read the first digit from a later full read
    tmpl = np.stack([pats[k] for k in order[:10]])
    first_digit = int(input("tenths digit shown in the first frame (look at a frame): ")) if sys.stdin.isatty() else None
    return tmpl, first_digit


def load_templates():
    z = np.load(TEMPLATES)
    return z["tmpl"].astype(bool)  # [10, 14, 8], index = digit


def read_gray(gray: np.ndarray, tmpl: np.ndarray):
    """Return seconds of day (float) and a confidence (max Hamming distance of the chosen glyphs, 0 = exact)."""
    cells = cells_of(gray)
    worst = 0
    digits = {}
    for key, idx in DIGIT_CELLS.items():
        v = 0
        for k in idx:
            dist = (tmpl != cells[k][None]).reshape(10, -1).sum(1)
            dgt = int(np.argmin(dist))
            worst = max(worst, int(dist[dgt]))
            v = v * 10 + dgt
        digits[key] = v
    sod = digits["h"] * 3600 + digits["m"] * 60 + digits["s"] + digits["d"] / 10
    return sod, worst


def read_video(video: str, step_s: float = 30.0, frames_per_probe: int = 12, tmpl=None) -> pd.DataFrame:
    """Probe the clock every step_s seconds of presentation time; at each probe read frames_per_probe frames."""
    tmpl = load_templates() if tmpl is None else tmpl
    c = av.open(video)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    dur = (c.duration or 0) / 1e6
    rows = []
    for target in np.arange(0.5, dur - 1.0, step_s):
        try:
            c.seek(int(target / s.time_base), stream=s, backward=True)
        except av.error.PermissionError:  # some .m4v refuse a seek near pts 0: decode from the start
            c.close()
            c = av.open(video)
            s = c.streams.video[0]
            s.thread_type = "AUTO"
        n = 0
        for fr in c.decode(s):
            if fr.time < target:
                continue
            sod, worst = read_gray(fr.to_ndarray(format="gray"), tmpl)
            rows.append({"pts": fr.time, "clock_sod": sod, "worst_hamming": worst})
            n += 1
            if n >= frames_per_probe:
                break
    c.close()
    return pd.DataFrame(rows)


def fit_clock(df: pd.DataFrame, max_ham: int = 6):
    """Fit clock = a + b * pts using the frames where the tenths digit changes (the change happens between frames).

    Returns a (clock at pts 0, s of day), b (clock seconds per pts second), rms residual (s), number of changes.
    The clock is quantised to 0.1 s: the first frame showing a new tenth is at most one frame interval after the
    change, so each change gives the change time to within one frame (1/30 s).
    """
    d = df[df.worst_hamming <= max_ham].sort_values("pts").reset_index(drop=True)
    ch = d.index[(d.clock_sod.diff() > 0.05) & (d.clock_sod.diff() < 0.15) & (d.pts.diff() < 0.05)]
    if len(ch) < 3:
        x, y = d.pts.to_numpy(), d.clock_sod.to_numpy()
    else:
        x = (d.pts[ch].to_numpy() + d.pts[ch - 1].to_numpy()) / 2   # change between the two frames
        y = d.clock_sod[ch].to_numpy()
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    return float(a), float(b), float(np.sqrt(np.mean(res ** 2))), int(len(x)), float(np.abs(res).max())


if __name__ == "__main__":
    cmd, video = sys.argv[1], sys.argv[2]
    if cmd == "build":
        tmpl, first = build(video)
        if first is None:
            raise SystemExit("run interactively, or build the templates in code (see tables.md)")
        tmpl = np.roll(tmpl, first, axis=0)
        np.savez(TEMPLATES, tmpl=tmpl)
    elif cmd == "read":
        step = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
        df = read_video(video, step)
        print(fit_clock(df))
