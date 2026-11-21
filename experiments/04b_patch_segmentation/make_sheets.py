"""Sheets for people: a frame next to its predicted class map, legend on top.

    .venv/bin/python experiments/04b_patch_segmentation/make_sheets.py test --stem <stem> [--frames 47,491,...] --out patchseg_test_00004_a
    .venv/bin/python experiments/04b_patch_segmentation/make_sheets.py dev --frames 720,900 --out patchseg_dev_cv

Class map = argmax of the stored class probabilities (dev: held-out fold classifier; test: all-31 classifier),
upsampled. ``--unknown T`` paints patches whose distance to the nearest training patch is > T magenta.
``--m2f`` adds a third panel: Mask2Former ADE seafloor classes (sand / rock / earth / floor / mountain) in brown.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd

from uwid import patchseg as ps
import common as C

W = 800
LEG = ps.CLASSES + ["unknown"]


def legend(width: int) -> np.ndarray:
    bar = np.zeros((34, width, 3), np.uint8)
    x = 8
    for i, n in enumerate(LEG):
        cv2.rectangle(bar, (x, 8), (x + 18, 26), tuple(int(v) for v in ps.COLORS[i][::-1]), -1)
        cv2.putText(bar, n, (x + 24, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        x += 34 + 11 * len(n)
    return bar


def panel(img_rgb, probs, dist=None, unknown=None, m2f=None, title=""):
    h = int(W * img_rgb.shape[0] / img_rgb.shape[1])
    im = cv2.resize(img_rgb, (W, h), interpolation=cv2.INTER_AREA)
    import torch
    p = ps.upsample_probs(torch.from_numpy(probs.astype(np.float32)), (h, W)).numpy()
    lab = p.argmax(-1)
    if dist is not None and unknown is not None:
        d = cv2.resize(dist.astype(np.float32), (W, h), interpolation=cv2.INTER_LINEAR)
        lab[d > unknown] = ps.UNKNOWN
    cmap = ps.colorize(lab)
    blend = (0.55 * im + 0.45 * cmap).astype(np.uint8)
    tiles = [im, blend, cmap]
    if m2f is not None:
        m = cv2.resize(m2f, (W, h), interpolation=cv2.INTER_NEAREST)
        sf = np.isin(m, [46, 34, 13, 3, 16])
        mm = (im * 0.5).astype(np.uint8); mm[sf] = ps.COLORS[ps.SEAFLOOR]
        tiles = [im, blend, mm]
    row = np.concatenate(tiles, 1)
    row = cv2.cvtColor(row, cv2.COLOR_RGB2BGR)
    cv2.rectangle(row, (0, 0), (row.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(row, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return row


def write_sheets(rows, out_stem, header, per=5):
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        width = chunk[0].shape[1]
        top = np.zeros((26, width, 3), np.uint8)
        cv2.putText(top, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        sheet = np.concatenate([top, legend(width)] + chunk, 0)  # legend is drawn in BGR already
        path = C.SHEETS / f"{out_stem}_{s // per + 1:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which"); ap.add_argument("--stem"); ap.add_argument("--frames", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--pred", default="w1280_h0")
    ap.add_argument("--unknown", type=float, default=None); ap.add_argument("--m2f", action="store_true")
    ap.add_argument("--per", type=int, default=5); ap.add_argument("--header", default="")
    a = ap.parse_args()
    if a.which == "dev":
        d = np.load(ps.CACHE / f"pred_dev_{a.pred}.npz")
        img_dir = C.REF / "2025-12-20_CG" / "frames"
        labels = pd.read_csv(C.REPO / "experiments/02_reference_labels/labels/2025-12-20_CG/frames.csv")
        header = a.header or "dev clip 2025-12-20 CG, classifier fitted on the OTHER time blocks (held-out block)"
        m2f = np.load(ps.CACHE / "m2f_dev.npz") if a.m2f else None
    else:
        d = np.load(ps.CACHE / f"pred_{a.stem}_{a.pred}.npz")
        img_dir = C.REF / a.stem / "sample"
        lp = C.REPO / "experiments/02_reference_labels/labels" / a.stem / "frames.csv"
        labels = pd.read_csv(lp) if lp.exists() else None
        header = a.header or f"TEST {a.stem} (never fitted on), classifier fitted on 31 dev frames"
        m2f = np.load(ps.CACHE / f"m2f_{a.stem}.npz") if a.m2f else None
    frames = list(d["frames"])
    want = [int(x) for x in a.frames.split(",")] if a.frames else sorted(int(p.stem) for p in img_dir.glob("*.jpg"))
    rows = []
    for f in want:
        if f not in frames or not (img_dir / f"{f:06d}.jpg").exists():
            continue
        i = frames.index(f)
        img = cv2.cvtColor(cv2.imread(str(img_dir / f"{f:06d}.jpg")), cv2.COLOR_BGR2RGB)
        t = f"f{f}"
        if labels is not None and (labels.frame == f).any():
            r = labels[labels.frame == f].iloc[0]
            t += (f"  labels: {r.state} view={r.view} whales={r.other_whales}+{r.faint_whales}faint back={r.back} "
                  f"seafloor={r.seafloor} fish={r.fish}")
        mm = None
        if m2f is not None and f in list(m2f["frames"]):
            mm = m2f["labels"][list(m2f["frames"]).index(f)]
        rows.append(panel(img, d["probs"][i], d["dist"][i], a.unknown, mm, t))
    write_sheets(rows, a.out, header, a.per)


if __name__ == "__main__":
    main()
