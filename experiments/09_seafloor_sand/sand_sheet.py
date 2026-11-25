"""WP12 step B sheet for the user: where the refitted sand label (variant B) fires outside the confirmed sediment
(2025-01-02 00024, fitted without 00024; 2025-02-03 labelled frames and 00006, never fitted on). Plain frame |
decision. Writes review_sheets/underwater/sand_calls_01.jpg (refuses to replace a sheet it did not write).

    .venv/bin/python experiments/09_seafloor_sand/sand_sheet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "08_whale_fish_boxes")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import data as D  # noqa: E402
import sand_redo as R  # noqa: E402
from uwid import video  # noqa: E402

TW, TH = 640, 360
MINE = D.SF_CACHE / "wp12" / "sheets_written.txt"


def main():
    w17, X17 = R.label_00017()
    sets = R.others()
    d2 = {n: (X[(~w.air).to_numpy()], np.zeros(int((~w.air).sum()), int)) for n, s, p, st, w, X, _ in sets if s == "d2"}
    base = (w17.y >= 0).to_numpy()
    picks = []
    for name, src, piece, stem, w, X, st in sets:
        if name.startswith("dev"):
            continue
        ex = [v for k, v in d2.items() if k != name]
        m = R.fit(np.concatenate([X17[base]] + [e[0] for e in ex]), np.concatenate([w17.y[base]] + [e[1] for e in ex]))
        p = R.prob(m, X)
        ok = ((~w.air) & (w.grey_max >= 20)).to_numpy() & (p >= 0.15)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        rng = np.random.default_rng(1)
        sel = rng.choice(idx, min(4, len(idx)), replace=False)
        for i in sel:
            picks.append((name, src, stem if stem else w.stem.iloc[i], int(w.frame.iloc[i]), int(w.k.iloc[i]), float(p[i]), int(ok.sum()), int(((~w.air) & (w.grey_max >= 20)).sum())))
    tiles = []
    for name, src, stem, frame, k, p, n_call, n in picks:
        rec = [r for r in D.recordings() if r.stem == stem]
        img = None
        if "labelled" in name and rec:
            img = cv2.imread(str(rec[0].image_path(frame)))
        else:
            vid = D.VIDEO / ("TestSubset 2025-1-2 CATS -85" if src == "d2" else "2025-2-3 Mini-mother 11-96") / "Video" / f"{stem}.mov"
            fr = next(iter(video.iter_frames_at(vid, [frame], size=(1280, 720))))
            img = cv2.cvtColor(fr.image, cv2.COLOR_RGB2BGR)
        a = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
        b = a.copy()
        cv2.rectangle(b, (3, 3), (TW - 4, TH - 4), (40, 200, 200), 6)
        t = np.hstack([a, b])
        bar = np.zeros((48, t.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{name}  window {k} s   ({n_call} of {n} windows of this set called sand)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"decision: SAND (view blocked by sediment), p {p:.2f}; nothing confirmed here", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 200, 200), 1, cv2.LINE_AA)
        tiles.append(np.vstack([t, bar]))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    name = "sand_calls_01.jpg"
    mine = set(MINE.read_text().split()) if MINE.exists() else set()
    p = D.SHEETS / name
    if p.exists() and name not in mine:
        raise SystemExit(f"{p} exists, not written by WP12")
    cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    mine.add(name); MINE.write_text("\n".join(sorted(mine)) + "\n")
    print("wrote", p, len(picks))


if __name__ == "__main__":
    main()
