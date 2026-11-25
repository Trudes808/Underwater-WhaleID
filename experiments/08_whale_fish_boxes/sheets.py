"""Sheets for the user: boxclf_fixed_NN.jpg (the scan's rule got the box wrong, the classifier right) and
boxclf_wrong_NN.jpg (the classifier still wrong). Each tile: the frame around the box, the box drawn once in the colour
of the classifier's decision (red whale, yellow fish, grey other), truth and both decisions printed. The classifier's
answer is always from the fold that did NOT see the tile's source.

    .venv/bin/python experiments/08_whale_fish_boxes/sheets.py [feature set]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402

COL = {"whale": (40, 60, 255), "fish": (0, 220, 255), "other": (170, 170, 170)}  # BGR
BASE_MAP = {"whale": "whale", "small": "fish (small)", "fish": "fish", "none": "other"}
LOSO = {"dev": "test dev", "d1": "test d1", "d2": "test d2"}
TILE = (480, 330)


def tile(rec: D.Recording, r) -> np.ndarray:
    img = cv2.imread(str(r.img))
    H, W = img.shape[:2]
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    side = max(r.x1 - r.x0, r.y1 - r.y0) * 1.8 + 80
    w = max(side, 360); h = w * 0.62
    x0, y0 = int(max(0, min(W - w, cx - w / 2))), int(max(0, min(H - h, cy - h / 2)))
    x1, y1 = int(min(W, x0 + w)), int(min(H, y0 + h))
    crop = img[y0:y1, x0:x1].copy()
    s = TILE[0] / crop.shape[1]
    crop = cv2.resize(crop, (TILE[0], int(crop.shape[0] * s)))
    c = COL[r.pred]
    cv2.rectangle(crop, (int((r.x0 - x0) * s), int((r.y0 - y0) * s)), (int((r.x1 - x0) * s), int((r.y1 - y0) * s)), c, 2)
    out = np.zeros((TILE[1] + 62, TILE[0], 3), np.uint8)
    out[:min(TILE[1], crop.shape[0])] = crop[:TILE[1]]
    t1 = f"truth {r.truth}  |  scan rule: {BASE_MAP[r.base_a]}  |  classifier: {r.pred} {max(r.p_whale, r.p_fish, r.p_other):.2f}"
    t2 = (f"{r.source} {D.piece(r.stem) if r.source != 'dev' else 'clip'} f{int(r.frame)}  {r.px:.0f} px  "
          f"whale {r.s_whale:.2f} fish {r.s_fish:.2f}" + ("  (extra box)" if r.extra else ""))
    cv2.putText(out, t1, (6, TILE[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, t2, (6, TILE[1] + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def write(rows: pd.DataFrame, name: str, per=12, cols=3, max_sheets=5):
    recs = {r.stem: r for r in D.recordings()}
    recs.update({k: None for k in rows.stem if k not in recs})
    n = 0
    for k in range(0, min(len(rows), per * max_sheets), per):
        tiles = [tile(recs[r.stem], r) for r in rows.iloc[k:k + per].itertuples()]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        n += 1
        cv2.imwrite(str(D.SHEETS / f"boxclf_{name}_{n:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return n


def main(fs="tight+ctx"):
    obj = pd.read_parquet(D.CACHE / "objects.parquet")
    P = pd.read_parquet(D.CACHE / "predictions.parquet")
    P = P[(P.features == fs) & (P.fold == P.source.map(LOSO))]
    m = P.merge(obj[["stem", "frame", "obj", "x0", "y0", "x1", "y1", "s_whale", "s_fish"]], on=["stem", "frame", "obj"])
    m["extra"] = False
    m["img"] = [[r for r in D.recordings() if r.stem == st][0].image_path(int(f)) for st, f in zip(m.stem, m.frame)]
    pe = D.CACHE / "predictions_extra.parquet"
    if pe.exists():
        E = pd.read_parquet(pe)
        E = E[(E.features == fs) & (E.fold == E.source.map(LOSO))].copy()
        E["extra"] = True; E["obj"] = -1
        E["img"] = [D.CACHE / "extra" / f"{st}_{int(f):06d}.jpg" for st, f in zip(E.stem, E.frame)]
        m = pd.concat([m, E[[c for c in m.columns if c in E.columns]]], ignore_index=True)
    m["base3"] = m.base_a.map({"whale": "whale", "small": "fish", "fish": "fish", "none": "other"})
    # what matters most first: the 300-800 band and big whales, then the rest
    big = m.band.isin(["300-500", "500-800", ">=800"])
    swap = m.truth3.isin(["whale", "fish"]) & (m.base3.isin(["whale", "fish"]) | m.pred.isin(["whale", "fish"]))
    m["prio"] = np.where(big & swap, 0, np.where(big, 1, np.where(swap, 2, 3)))
    rng = np.random.default_rng(0)
    m["r"] = rng.random(len(m))
    fixed = m[(m.base3 != m.truth3) & (m.pred == m.truth3)].sort_values(["prio", "r"])
    wrong = m[(m.pred != m.truth3)].sort_values(["prio", "r"])
    # keep the sheets about whale against fish: drop 'other' boxes the baseline merely left unnamed below 300 px
    print("fixed", len(fixed), "wrong", len(wrong))
    for old in D.SHEETS.glob("boxclf_*.jpg"):
        old.unlink()
    a = write(fixed, "fixed"); b = write(wrong, "wrong")
    print(f"{a} fixed sheets, {b} wrong sheets")


if __name__ == "__main__":
    main(*sys.argv[1:])
