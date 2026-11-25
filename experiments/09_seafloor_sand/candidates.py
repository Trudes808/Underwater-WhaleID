"""WP12: find the real seafloor in the on-whale recordings of 2025-01-02 (CPU only).

Candidate moments come from the sensors and the picture, not from the old labels:
- within 60 s of every dark window (every frame's mean grey < 20; 00017 only),
- every stretch where the depth (scan window table, sensor) is within 20 % of the recording's maximum and flat
  (20 s rolling std < 0.3 m), with 30 s on each side,
- the sand spans of 00017 (inside the first item).
The accelerometer is not in the 1 Hz sensor table; contact from motion was not used.

    decode   frames at 1 s steps over the candidate ranges, 1280 x 720, into underwater_cache/seafloor/cand/<stem>/
             (t in tenths of a second in the file name) -> DONE marker
    triage   working sheets of 16 frames (every 2nd second; black frames skipped) in cand/triage/ for my own look
    user     the user's sheets from candidate_calls.csv: review_sheets/underwater/seafloor_candidates_<piece>_NN.jpg
             (3 per row, frame at 640 px with a contrast-stretched copy beside it, time and call) and
             seafloor_candidates_best_01.jpg (the clearest 24)

    nice -n 10 .venv/bin/python experiments/09_seafloor_sand/candidates.py decode
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08_whale_fish_boxes"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402

OUT = D.SF_CACHE / "cand"
SCANW = Path("/home/sat3737/NOAA_whales/underwater_cache/scan/TestSubset 2025-1-2 CATS -85/sensitive/windows")
PIECES = ("00010", "00011", "00012", "00017", "00023", "00024")


def stem_of(piece):
    return [r.stem for r in D.recordings(("d2",)) if r.stem.endswith(piece)][0]


def _merge(rs, gap=5):
    rs = sorted(rs)
    out = []
    for a, b, why in rs:
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b), out[-1][2] + "+" + why if why not in out[-1][2] else out[-1][2])
        else:
            out.append((a, b, why))
    return out


def ranges() -> dict:
    res = {}
    for piece in PIECES:
        stem = stem_of(piece)
        w = pd.read_csv(SCANW / f"{stem}.mov.csv")
        dur = float(w.t1.max())
        rs = []
        d = w.depth_m.astype(float)
        if d.notna().any():
            mx = d.max()
            flat = ((d.rolling(20, center=True).std() < 0.3) & (d > 0.8 * mx)).fillna(False).to_numpy()
            start = None
            for i, v in enumerate(list(flat) + [False]):
                if v and start is None:
                    start = i
                if not v and start is not None:
                    if i - start >= 10:
                        rs.append((max(0, w.t0.iloc[start] - 30), min(dur, w.t0.iloc[i - 1] + 30), "deep-flat"))
                    start = None
        z = D.SF_CACHE / "frames" / f"dense_{stem}.npz"
        if z.exists():
            zz = np.load(z)
            k = np.floor(zz["t"]).astype(int)
            dark = sorted({int(x) for x, g in zip(k, zz["grey_mean"]) if g < 20})
            for t in dark:
                rs.append((max(0, t - 60), min(dur, t + 61), "dark"))
        res[piece] = [(int(a), int(b), why) for a, b, why in _merge(rs)]
    return res


# second pass (after the first look found the bottom only in 00017): every stretch deeper than 70 % of the
# recording's maximum depth in the other long recordings, flat or not
EXTRA = {"00011": [(84, 215, "deep"), (602, 695, "deep"), (1155, 1237, "deep"), (1319, 1349, "deep")],
         "00023": [(897, 958, "deep"), (1167, 1238, "deep")]}


def decode(extra: bool = False):
    from uwid import video
    OUT.mkdir(parents=True, exist_ok=True)
    R = EXTRA if extra else ranges()
    print("ranges", R, flush=True)
    for piece, rs in R.items():
        stem = stem_of(piece)
        od = OUT / stem
        od.mkdir(exist_ok=True)
        path = D.VIDEO / "TestSubset 2025-1-2 CATS -85" / "Video" / f"{stem}.mov"
        for a, b, why in rs:
            t = time.time(); n = 0
            print(f"{piece} {a}-{b} s ({why}): decoding", flush=True)
            for fr in video.iter_frames(path, fps=1.0, start=float(a), end=float(b), size=(1280, 720)):
                cv2.imwrite(str(od / f"{int(round(fr.t * 10)):06d}.jpg"), cv2.cvtColor(fr.image, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                n += 1
            print(f"{piece} {a}-{b} s: {n} frames in {time.time() - t:.0f} s", flush=True)
    (OUT / ("DONE_extra" if extra else "DONE")).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def frames(stem):
    fs = sorted(glob.glob(str(OUT / stem / "*.jpg")))
    return [(int(Path(f).stem) / 10.0, f) for f in fs]


def stretch(img):
    """Contrast-stretched copy: CLAHE on the lightness channel only (stretching the colour channels separately blows
    up the near-empty red channel underwater into red speckle)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def triage(extra: bool = False):
    td = OUT / ("triage_extra" if extra else "triage"); td.mkdir(exist_ok=True)
    for old in td.glob("*.jpg"):
        old.unlink()
    for piece in (EXTRA if extra else PIECES):
        stem = stem_of(piece)
        inside = (lambda t: any(a <= t <= b for a, b, _ in EXTRA[piece])) if extra else (lambda t: True)
        fr = [(t, f) for t, f in frames(stem) if int(round(t)) % (4 if extra else 2) == 0 and inside(t)]
        keep = []
        for t, f in fr:
            im = cv2.imread(f)
            if cv2.cvtColor(cv2.resize(im, (160, 90)), cv2.COLOR_BGR2GRAY).mean() < 20:
                continue
            keep.append((t, im))
        for k in range(0, len(keep), 16):
            tiles = []
            for t, im in keep[k:k + 16]:
                s = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
                cv2.rectangle(s, (0, 0), (90, 24), (0, 0, 0), -1)
                cv2.putText(s, f"{t:.0f}s", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                tiles.append(s)
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
            cv2.imwrite(str(td / f"{piece}_{k // 16 + 1:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(piece, len(keep), "frames on", (len(keep) + 15) // 16, "sheets")


def user_sheets():
    calls = pd.read_csv(HERE / "candidate_calls.csv")
    col = {"seafloor": (60, 160, 255), "sediment": (40, 200, 200), "unsure": (200, 200, 200), "none": (160, 160, 160)}

    def tile(r):
        f = OUT / stem_of(f"{int(r.piece):05d}") / f"{int(round(r.t * 10)):06d}.jpg"
        im = cv2.imread(str(f))
        a = cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
        b = cv2.resize(stretch(im), (320, 180), interpolation=cv2.INTER_AREA)
        t = np.zeros((360 + 46, 640 + 320, 3), np.uint8)
        t[:360, :640] = a; t[:180, 640:] = b
        cv2.putText(t, "contrast-stretched copy", (650, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(t, f"2025-01-02 {int(r.piece):05d}  t {r.t:.0f} s  call: {r.call}", (6, 385),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col.get(r.call, (255, 255, 255)), 2, cv2.LINE_AA)
        if isinstance(r.note, str) and r.note:
            cv2.putText(t, r.note[:90], (650, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        return t

    def write(rows, name):
        tiles = [tile(r) for r in rows.itertuples()]
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        cv2.imwrite(str(D.SHEETS / name), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])

    for old in D.SHEETS.glob("seafloor_candidates_000*_*.jpg"):
        old.unlink()
    for old in D.SHEETS.glob("seafloor_candidates_best_*.jpg"):
        old.unlink()
    for piece, g in calls.groupby("piece"):
        g = g.sort_values("t")
        for k in range(0, len(g), 12):
            write(g.iloc[k:k + 12], f"seafloor_candidates_{int(piece):05d}_{k // 12 + 1:02d}.jpg")
    best = calls[calls.best == 1].sort_values(["piece", "t"]).head(24)
    if len(best):
        write(best, "seafloor_candidates_best_01.jpg")
    print(calls.groupby(["piece", "call"]).size())


if __name__ == "__main__":
    {"decode": decode, "triage": triage, "user": user_sheets, "ranges": lambda: print(ranges()),
     "decode_extra": lambda: decode(True), "triage_extra": lambda: triage(True)}[sys.argv[1]]()
