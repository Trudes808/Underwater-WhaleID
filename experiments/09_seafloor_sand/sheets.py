"""Sheets of 09_seafloor_sand: seafloor_NN.jpg = frame next to the predicted class map (the head that did NOT see
the frame's recording / source), with the window's truth and call. Order: hits, misses, false alarms on 2025-01-02,
false alarms on 2025-02-03, false alarms on the 2025-12-20 clip, then sand and dark examples (00017, held-out block).

    .venv/bin/python experiments/09_seafloor_sand/sheets.py    (before decode.py --clean: dense frames are read)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "08_whale_fish_boxes"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402
import score as S  # noqa: E402
from uwid import patchseg as ps  # noqa: E402

TW, TH = 560, 315
LEG = ", ".join(f"{c}" for c in ps.CLASSES)


def frame_img(kind, stem, frame):
    if kind == "dense":
        p = S.OUT / "frames_tmp" / stem / f"{frame:06d}.jpg"
    else:
        rec = [r for r in D.recordings() if r.stem == stem][0]
        p = rec.image_path(frame)
    im = cv2.imread(str(p))
    return cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA)


def map_img(m):
    c = ps.colorize(m)[..., ::-1]
    return cv2.resize(np.ascontiguousarray(c), (TW, TH), interpolation=cv2.INTER_NEAREST)


def row(kind, stem, z, i, head, text):
    hidx = list(z["heads"]).index(head)
    a = frame_img(kind, stem, int(z["frame"][i])); b = map_img(z["maps"][i, hidx])
    r = np.hstack([a, b])
    bar = np.zeros((34, r.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([r, bar])


def main():
    rules = json.loads((S.RES / "rules.json").read_text())
    thr = json.loads((S.RES / "seafloor.json").read_text())["thresholds"]
    rng = np.random.default_rng(7)
    rows = []

    def pick(df, n):
        return df.iloc[rng.choice(len(df), min(n, len(df)), replace=False)] if len(df) else df

    def add(kind, stem, z, w, head, label, n):
        for r in pick(w, n).itertuples():
            i = r.emb_i[0]
            t = float(z["t"][i])
            txt = (f"{label}: {D.SOURCE_NAME.get(src_of(stem), stem)} {D.piece(stem) if src_of(stem) != 'dev' else ''} "
                   f"t {t:.1f} s  seafloor share {getattr(r, 'sea_' + head):.3f} (call >= {thr[head]})  head {head}")
            rows.append(row(kind, stem, z, i, head, txt))

    for piece, head in (("00017", "test17"), ("00023", "test23")):
        stem = S.D2_STEMS[piece]; z = S.load("dense", stem); w = S.d2_window_table(stem)
        w = w[(w.blocked != "air") & (w.seafloor_truth != "unsure")]
        call = w[f"sea_{head}"] >= thr[head]
        add("dense", stem, z, w[call & (w.seafloor_truth == "visible")], head, "HIT", 2)
        add("dense", stem, z, w[~call & (w.seafloor_truth == "visible")], head, "MISS", 2)
        add("dense", stem, z, w[call & (w.seafloor_truth == "none")], head, "FALSE", 2)
    for piece in ("00010", "00011", "00012", "00024"):
        stem = S.D2_STEMS[piece]; z = S.load("dense", stem); w = S.d2_window_table(stem)
        w = w[w.blocked != "air"]
        add("dense", stem, z, w[w["sea_test23"] >= thr["test23"]], "test23", "FALSE", 1)
    z6 = S.load("dense", S.D1_DENSE); w6 = S.windows_from(z6, S.HEADS)
    add("dense", S.D1_DENSE, z6, w6[w6.sea_test_d1 >= thr["test_d1"]], "test_d1", "FALSE", 3)
    for rec in D.recordings(("d1", "dev")):
        z = S.load("label", rec.stem)
        z = {**z, "t": np.asarray(z["t"], float)}
        w = S.windows_from({**z, "t": np.arange(len(z["frame"])) + 0.5} if rec.source == "d1" else z, S.HEADS)
        head = "test_d1" if rec.source == "d1" else "test_dev"
        add("label", rec.stem, z, w[w[f"sea_{head}"] >= thr[head]], head, "FALSE", 1 if rec.source == "d1" else 3)
    # a few correct negatives on the traps: froth / back-filled frames of 2025-01-02 not called
    stem = S.D2_STEMS["00024"]; z = S.load("dense", stem); w = S.d2_window_table(stem)
    add("dense", stem, z, w[(w.blocked != "air") & (w.sea_test23 < thr["test23"]) & (w.back_test23 > 0.4)], "test23",
        "NEGATIVE (back fills the view)", 2)
    for old in D.SHEETS.glob("seafloor_*.jpg"):
        old.unlink()
    per = 5
    for k in range(0, len(rows), per):
        g = rows[k:k + per]
        sheet = np.vstack(g)
        head = np.zeros((30, sheet.shape[1], 3), np.uint8)
        cv2.putText(head, "left: frame (1 of the window's 2)   right: class map; brown = seafloor, purple = back, "
                    "blue = water, red = whale, white = bubbles, cyan = surface", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(D.SHEETS / f"seafloor_{k // per + 1:02d}.jpg"), np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(len(rows), "rows,", (len(rows) + per - 1) // per, "sheets")


def src_of(stem):
    return "dev" if stem.startswith("2025-12") else ("d1" if stem.startswith("CC-11") else "d2")


if __name__ == "__main__":
    main()
