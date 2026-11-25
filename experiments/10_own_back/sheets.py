"""WP14 sheets for the user (decision only; plain frame beside the decision).

    ownback   ownback_candidates_NN.jpg: every scan whale object of 3 windows or more whose best box lies mostly (>= 60 %
              of its height, in at least half its windows) in the bottom 25 % of the frame, where the tracker's own-back
              object is prompted and can swallow it (both deployments). Right: the scan's box and the proposed own-back
              region (the recording's template); caption: would a start pass the proposed own-back test.
    bubbles   bubbles_candidates_01.jpg: the scan whale objects that begin within 10 s after a lens-out run, ranked
              by how much they look like the confirmed bubble case (bubbles_scan.csv), plus the two confirmed
              tracker numbers of 00024.

    .venv/bin/python experiments/10_own_back/sheets.py ownback|bubbles
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
import ownback as O  # noqa: E402
import runs as R  # noqa: E402
from uwid import video  # noqa: E402

TW, TH = 640, 360
MINE = C.OUT / "sheets_written.txt"


def safe_write(name, img):
    p = C.SHEETS / name
    mine = set(MINE.read_text().split()) if MINE.exists() else set()
    if p.exists() and name not in mine:
        raise SystemExit(f"{p} exists and was not written by WP14")
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    mine.add(name); MINE.write_text("\n".join(sorted(mine)) + "\n")
    print("wrote", p)


def band_objects():
    rows = []
    for src, piece, stem, vp, wp in C.recordings():
        w = C.windows(wp)
        for a, b in C.whale_objects(w):
            if b - a + 1 < 3:
                continue
            s = w.iloc[a:b + 1]
            s = s[s.whale.astype(str).str.lower().eq("true") & s.by0.notna()]
            inb = ((s.by1.clip(upper=1080) - s.by0.clip(lower=810)).clip(lower=0)) / (s.by1 - s.by0)
            if len(s) and (inb >= 0.6).mean() >= 0.5:
                mid = s.iloc[len(s) // 2]
                rows.append(dict(src=src, piece=piece, stem=stem, video=str(vp), t0=float(s.t0.min()), t1=float(s.t1.max()),
                                 n=len(s), frame=int(mid.best_frame), box=[mid.bx0, mid.by0, mid.bx1, mid.by1],
                                 tier=mid.tier, depth=mid.depth_m))
    return pd.DataFrame(rows)


def tile(img, left_text, right_draw, lines):
    a = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
    b = a.copy()
    right_draw(b)
    t = np.hstack([a, b])
    bar = np.zeros((48, t.shape[1], 3), np.uint8)
    for i, (s, c) in enumerate(lines):
        cv2.putText(bar, s, (6, 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return np.vstack([t, bar])


def ownback():
    df = band_objects()
    T = R.templates()
    tiles = []
    for r in df.itertuples():
        img = next(iter(video.iter_frames_at(r.video, [r.frame], size=(C.W, C.H)))).image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        tm = T[r.stem][0]
        tpx = O.upsample(tm, (TH, TW))
        sc = TW / 1920
        box = [v * sc for v in r.box]
        passes = not O.box_on_back(box, tpx)

        def draw(b, box=box, tpx=tpx):
            cs, _ = cv2.findContours(tpx.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(b, cs, -1, (200, 80, 160), 2)
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            cv2.rectangle(b, (x0, y0), (x1, y1), (40, 60, 255), 2)
        tiles.append(tile(img, "", draw, [
            (f"{'2025-02-03' if r.src == 'd1' else '2025-01-02'} {r.piece}  {r.t0:.0f}-{r.t1:.0f} s ({r.n} windows), tier {r.tier}, depth {r.depth:.0f} m", (255, 255, 255)),
            (f"red = scan whale box; purple = proposed own back (where the back usually is); start would {'PASS' if passes else 'be refused'}",
             (120, 220, 120) if passes else (80, 80, 255))]))
    for k in range(0, len(tiles), 8):
        tt = tiles[k:k + 8]
        while len(tt) % 2:
            tt.append(np.zeros_like(tt[0]))
        safe_write(f"ownback_candidates_{k // 8 + 1:02d}.jpg", np.vstack([np.hstack(tt[i:i + 2]) for i in range(0, len(tt), 2)]))
    df.drop(columns=["video"]).to_csv(C.OUT / "ownback_candidates.csv", index=False)


def bubbles():
    s = pd.read_csv(C.OUT / "bubbles_scan.csv")
    s["bubble_like"] = s.p_bubbles.fillna(0) + s.p_surface.fillna(0) + s.p_air.fillna(0)
    s["flag"] = (s.contrast > 0).astype(int) + (s.bubble_like >= 0.2).astype(int)
    s = s.sort_values(["flag", "contrast"], ascending=False)
    tiles = []
    for r in s.itertuples():
        img = cv2.imread(str(C.OUT / "frames" / r.stem / f"{int(r.frame):06d}.jpg"))
        sc = TW / 1920

        def draw(b, r=r):
            cv2.rectangle(b, (int(r.bx0 * sc), int(r.by0 * sc)), (int(r.bx1 * sc), int(r.by1 * sc)), (40, 60, 255), 2)
        tiles.append(tile(img, "", draw, [
            (f"{'2025-02-03' if r.src == 'd1' else '2025-01-02'} {int(r.recording):05d}  t {r.t_start:.0f} s, {r.after_lens_out_s:.0f} s after the lens was out; scan: whale ({r.tier})", (255, 255, 255)),
            (f"paler than around it: {'YES' if r.contrast > 0 else 'no'} ({r.contrast:+.0f}); patch map: bubbles / surface / air {r.bubble_like:.0%}, whale {r.p_whale:.0%}", (80, 160, 255) if r.flag else (200, 200, 200))]))
    for k in range(0, len(tiles), 12):
        tt = tiles[k:k + 12]
        while len(tt) % 2:
            tt.append(np.zeros_like(tt[0]))
        safe_write(f"bubbles_candidates_{k // 12 + 1:02d}.jpg", np.vstack([np.hstack(tt[i:i + 2]) for i in range(0, len(tt), 2)]))


if __name__ == "__main__":
    {"ownback": ownback, "bubbles": bubbles}[sys.argv[1]]()
