"""WP14 question 2: what separates the two false numbers right after a breath in 2025-01-02 00024 (bubbles, user
2026-11-23) from real whales right after a breath (WP11c encounters 00004, 00005, 00006 and 00007 1459-1475).

Part 1 (tracked objects): for each object, over its first second: patch classes under the outline (the run's own
WP5b patch map), grey level, colour saturation, texture (log variance of the Laplacian) inside the outline and the
contrast to a ring around it, and how the outline changes (IoU with the outline 0.5 s and 1 s later, area ratio,
centroid speed in outline sizes per second).
Part 2 (scan objects): every scan whale object (run of windows the scan calls whale) that begins within 10 s after a
lens-out run, both deployments: the same picture measures inside its best box (WP5b maps from maps.py), the box 1 s
and 2 s later, and the scan's own columns.

    .venv/bin/python experiments/10_own_back/bubbles.py      -> ownback/bubbles_tracks.csv, bubbles_scan.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
import sample as S  # noqa: E402
from uwid import video  # noqa: E402

D1 = C.TRACKS / C.DEPL["d1"]; D2 = C.TRACKS / C.DEPL["d2"]
# (run dir, object, truth, note); truth from the user (00024) and from WP11c's by-eye notes (breath_eye.csv)
CASES = [
    (D2 / "CC-09-85-HRA-20250102-160604-000-00024" / "0119.00-0142.00_online", "obj_1", "bubbles", "number 2: fluke-shaped pale patch (user: bubbles)"),
    (D2 / "CC-09-85-HRA-20250102-160604-000-00024" / "0119.00-0142.00_online", "obj_2", "bubbles", "number 3: pale shape top left (user: bubbles)"),
    (D1 / "CC-11-96-20250203-101236-000-00004" / "0622.05-0637.10_online", "obj_1", "whale", "whale at the left edge after the breath"),
    (D1 / "CC-11-96-20250203-104242-000-00005" / "1565.15-1579.50_online", "obj_1", "unsure", "faint pale shape bottom right (WP11c: uncertain)"),
    (D1 / "CC-11-96-20250203-104242-000-00005" / "1565.15-1579.50_online", "obj_2", "whale", "whale at the bottom right"),
    (D1 / "CC-11-96-20250203-111250-000-00006" / "0912.50-0927.00_online", "obj_1", "whale", "dark body at the left edge in the bubbles"),
    (D1 / "CC-11-96-20250203-114256-000-00007" / "1459.00-1475.00_online", "obj_2", "whale", "calf after the breath"),
]
NAMES = ["water", "whale", "back", "fish", "bubbles", "surface", "air"]


def picture_measures(img_bgr: np.ndarray, region: np.ndarray) -> dict:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(grey, cv2.CV_32F)
    ring = cv2.dilate(region.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~region
    if region.sum() < 20:
        return {}
    return {"grey": float(grey[region].mean()), "sat": float(hsv[..., 1][region].mean()),
            "texture": float(np.log10(lap[region].var() + 1e-6)),
            "contrast": float(grey[region].mean() - (grey[ring].mean() if ring.any() else grey[region].mean()))}


def class_shares(lab: np.ndarray, region: np.ndarray) -> dict:
    l = cv2.resize(lab, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    v = l[region]
    return {f"p_{n}": float((v == k).mean()) for k, n in enumerate(NAMES)}


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def tracked():
    rows = []
    for run, obj, truth, note in CASES:
        meta = json.loads((run / "meta.json").read_text())
        z = np.load(run / "masks.npz"); p = np.load(run / "patch.npz")
        m = z[obj]
        seen = np.nonzero(m.reshape(len(m), -1).any(1))[0]
        i0 = int(seen[0]); fps = float(meta["fps"])
        tab = video.frame_table(meta["video"]); f0 = tab.index_at(meta["t0"])
        want = [i for i in (i0, i0 + 15, i0 + 30) if i < len(m)]
        imgs = {fr.index: cv2.cvtColor(fr.image, cv2.COLOR_RGB2BGR)
                for fr in video.iter_frames_at(meta["video"], [f0 + i for i in want], size=(C.W, C.H))}
        masks = {i: C.unpack(m[i]) for i in want}
        pm = [picture_measures(imgs[f0 + i], masks[i]) for i in want if f0 + i in imgs and masks[i].sum() >= 20]
        cs = [class_shares(p["labels"][int(np.abs(p["frames"] - (f0 + i)).argmin())], masks[i]) for i in want if masks[i].sum() >= 20]
        areas = [masks[i].sum() for i in want]
        cy = [np.nonzero(masks[i])[0].mean() / C.H if masks[i].any() else np.nan for i in want]
        cx = [np.nonzero(masks[i])[1].mean() / C.W if masks[i].any() else np.nan for i in want]
        size = np.sqrt(max(areas[0], 1)) / C.W
        speed = (np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]) / size / ((want[-1] - want[0]) / fps)) if len(want) > 1 and size > 0 else np.nan
        r = dict(recording=run.parent.name.split("-")[-1], run=run.name, obj=obj, truth=truth, note=note,
                 t_start=round(meta["t0"] + i0 / fps, 2), dur_s=round(len(seen) / fps, 2),
                 area_share=float(areas[0] / (C.W * C.H)), centre_y=float(cy[0]),
                 iou_0_5s=iou(masks[want[0]], masks[want[1]]) if len(want) > 1 else np.nan,
                 iou_1s=iou(masks[want[0]], masks[want[2]]) if len(want) > 2 else np.nan,
                 area_ratio_1s=float(max(areas) / max(min(areas), 1)), speed_sizes_per_s=float(speed))
        for d in pm[:1]:
            r.update(d)
        if cs:
            r.update({k: float(np.mean([c[k] for c in cs])) for k in cs[0]})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "bubbles_tracks.csv", index=False)
    return df


def scan_objects():
    maps = {f.stem: np.load(f) for f in (C.OUT / "maps").glob("*.npz")}
    rows = []
    for src, piece, stem, vpath, wpath in C.recordings():
        w = C.windows(wpath)
        for o in S.post_breath_objects(w):
            boxes, frames = [], []
            for d in (0, 1, 2):
                r = o["row0"] + d
                if r < len(w) and pd.notna(w.best_frame.iloc[r]) and pd.notna(w.bx0.iloc[r]):
                    boxes.append([w.bx0.iloc[r], w.by0.iloc[r], w.bx1.iloc[r], w.by1.iloc[r]])
                    frames.append(int(w.best_frame.iloc[r]))
            if not frames:
                continue
            first = w.iloc[o["row0"]]
            row = dict(src=src, recording=piece, stem=stem, t_start=o["t_start"], t_end=o["t_end"],
                       dur_s=o["t_end"] - o["t_start"], after_lens_out_s=round(o["after_s"], 1),
                       frame=frames[0], tier=first.tier, whale_score=first.whale_score, whale_px=first.whale_px,
                       cover=first.cover, bubbles_box=first.bubbles, veto_other=first.get("veto_other_score", np.nan),
                       depth_m=first.depth_m, bx0=boxes[0][0], by0=boxes[0][1], bx1=boxes[0][2], by1=boxes[0][3])
            sc = 1280 / 1920
            def region(b):
                reg = np.zeros((C.H, C.W), bool)
                x0, y0, x1, y1 = [int(round(v * sc)) for v in b]
                reg[max(y0, 0):y1, max(x0, 0):x1] = True
                return reg
            img = cv2.imread(str(C.OUT / "frames" / stem / f"{frames[0]:06d}.jpg"))
            if img is not None:
                row.update(picture_measures(img, region(boxes[0])))
            if stem in maps:
                z = maps[stem]
                j = np.nonzero(z["frames"] == frames[0])[0]
                if len(j):
                    row.update(class_shares(z["labels"][j[0]], region(boxes[0])))
            for d, b in enumerate(boxes[1:], 1):
                ix = max(0, min(b[2], boxes[0][2]) - max(b[0], boxes[0][0])) * max(0, min(b[3], boxes[0][3]) - max(b[1], boxes[0][1]))
                ua = (b[2] - b[0]) * (b[3] - b[1]) + (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1]) - ix
                row[f"box_iou_{d}s"] = float(ix / max(ua, 1))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "bubbles_scan.csv", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    t = tracked()
    print(t.drop(columns=["note", "run"]).round(3).to_string())
    s = scan_objects()
    print(len(s), "scan whale objects within 10 s after a lens-out run")
    print(s.drop(columns=["stem"]).round(3).to_string())
