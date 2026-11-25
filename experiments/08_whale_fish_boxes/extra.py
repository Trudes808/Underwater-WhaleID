"""Extra test boxes for the band that matters (300-800 px), labelled by eye by the WP10 agent (AI by-eye labels).

Why: the reference labels hold only 4 fish of 300 px or more (all on 2025-01-02) and none on 2025-02-03, so the
question "does it call a large near fish a fish" rests on 4 boxes. These boxes come from the scan's own detector
cache (2 frames per 1 s window, the .mov recordings) of both test deployments: per frame one object whose "whale"
box scores >= 0.25 with a longest side of 300-800 px (the box the scan keeps as a whale), half of them chosen among
boxes whose best fish prompt scores >= 0.3. Frames within 90 frames of a reference-labelled frame are skipped. Used
only as TEST boxes (never for training a scored fold).

    pick    choose the boxes (seed 20261122), decode their frames (1920 px) into CACHE/extra/, write the labelling
            sheets (box drawn in white, a number, no scores) to the scratchpad-like cache folder CACHE/extra/sheets/
    feat    (GPU) crop embeddings for the chosen objects -> CACHE/extra/feat.npz
    The labels are typed into experiments/08_whale_fish_boxes/extra_labels.json by hand after looking.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402
from uwid import boxclf as B  # noqa: E402

SCAN = Path("/home/sat3737/NOAA_whales/underwater_cache/scan")
DEPL = {"d1": ("2025-2-3 Mini-mother 11-96", D.VIDEO / "2025-2-3 Mini-mother 11-96" / "Video"),
        "d2": ("TestSubset 2025-1-2 CATS -85", D.VIDEO / "TestSubset 2025-1-2 CATS -85" / "Video")}
N = {"d1": (24, 12), "d2": (24, 24)}      # (fishy, other) per deployment
X = D.CACHE / "extra"


def candidates(src: str) -> pd.DataFrame:
    name, vdir = DEPL[src]
    labelled = {r.stem: np.array(r.labelled_frames()) for r in D.recordings((src,))}
    out = []
    for d in sorted(glob.glob(str(SCAN / name / "cache" / "det" / "*.mov"))):
        stem = Path(d).stem
        df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(d + "/part_*.parquet"))], ignore_index=True)
        df = df[df.frame.notna()]
        lf = labelled.get(stem, np.array([]))
        for f, g in df.groupby("frame"):
            if len(lf) and np.abs(lf - f).min() < 90:
                continue
            u = B.unique_boxes(g)
            if not len(u):
                continue
            u = B.group_objects(u)
            u["px"] = np.maximum(u.x1 - u.x0, u.y1 - u.y0)
            area = (u.x1 - u.x0) * (u.y1 - u.y0) / (1920 * 1080)
            w = u[u.queries.apply(lambda q: "whale" in q) & (u.s_whale >= 0.25) & (u.px >= 300) & (u.px < 800) & (area <= 0.6)]
            if not len(w):
                continue
            t = w.sort_values("s_fish").iloc[-1]
            rep = u[(u.obj == t.obj) & u.rep].iloc[0]
            out.append(dict(source=src, stem=stem, frame=int(f), obj=int(t.obj), x0=rep.x0, y0=rep.y0, x1=rep.x1,
                            y1=rep.y1, fishy=bool(t.s_fish >= 0.3)))
    return pd.DataFrame(out)


def pick():
    X.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261122)
    sel = []
    for src in ("d1", "d2"):
        c = candidates(src)
        nf, no = N[src]
        a = c[c.fishy]; b = c[~c.fishy]
        sel.append(a.iloc[rng.choice(len(a), min(nf, len(a)), replace=False)])
        sel.append(b.iloc[rng.choice(len(b), min(no, len(b)), replace=False)])
        print(src, "candidates", len(c), "fishy", len(a))
    s = pd.concat(sel).sample(frac=1, random_state=1).reset_index(drop=True)
    s["id"] = np.arange(1, len(s) + 1)
    from uwid import video
    for (src, stem), g in s.groupby(["source", "stem"]):
        path = DEPL[src][1] / f"{stem}.mov"
        for fr in video.iter_frames_at(path, sorted(g.frame), size=(1920, 1080)):
            cv2.imwrite(str(X / f"{stem}_{fr.index:06d}.jpg"), cv2.cvtColor(fr.image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    s.to_csv(X / "chosen.csv", index=False)
    sheets(s)


def sheets(s: pd.DataFrame):
    sd = X / "sheets"; sd.mkdir(exist_ok=True)
    tiles = []
    for r in s.itertuples():
        img = cv2.imread(str(X / f"{r.stem}_{r.frame:06d}.jpg"))
        H, W = img.shape[:2]
        side = max(r.x1 - r.x0, r.y1 - r.y0) * 1.6 + 120
        w = max(side, 500); h = w * 0.62
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        x0 = int(max(0, min(W - w, cx - w / 2))); y0 = int(max(0, min(H - h, cy - h / 2)))
        crop = img[y0:int(min(H, y0 + h)), x0:int(min(W, x0 + w))].copy()
        sc = 640 / crop.shape[1]
        crop = cv2.resize(crop, (640, int(crop.shape[0] * sc)))
        cv2.rectangle(crop, (int((r.x0 - x0) * sc), int((r.y0 - y0) * sc)), (int((r.x1 - x0) * sc), int((r.y1 - y0) * sc)),
                      (255, 255, 255), 2)
        t = np.zeros((420, 640, 3), np.uint8); t[:min(400, crop.shape[0])] = crop[:400]
        cv2.putText(t, f"#{r.id}", (8, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(t)
    for k in range(0, len(tiles), 6):
        tt = tiles[k:k + 6]
        while len(tt) < 6:
            tt.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tt[i:i + 2]) for i in range(0, 6, 2)])
        cv2.imwrite(str(sd / f"label_{k // 6 + 1:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])


def feat():
    import torch
    from uwid import patchseg
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8.5e9 / torch.cuda.get_device_properties(0).total_memory))
    s = pd.read_csv(X / "chosen.csv")
    model = patchseg.load_backbone()
    T, C = [], []
    for r in s.itertuples():
        img = cv2.cvtColor(cv2.imread(str(X / f"{r.stem}_{r.frame:06d}.jpg")), cv2.COLOR_BGR2RGB)
        e = B.crop_embeddings(model, img, np.array([[r.x0, r.y0, r.x1, r.y1]]), (1.0, 2.0))
        T.append(e[1.0][0]); C.append(e[2.0][0])
    np.savez(X / "feat.npz", id=s.id.to_numpy(), tight=np.stack(T), ctx=np.stack(C))
    (X / "feat.DONE").write_text("ok")


def table() -> pd.DataFrame:
    """The labelled extra objects with the scores, the baseline and the truth (for train.py)."""
    from table import baseline
    s = pd.read_csv(X / "chosen.csv")
    lab = json.loads((HERE / "extra_labels.json").read_text())["labels"]
    rows = []
    for r in s.itertuples():
        name = DEPL[r.source][0]
        df = pd.concat([pd.read_parquet(p) for p in
                        sorted(glob.glob(str(SCAN / name / "cache" / "det" / f"{r.stem}.mov" / "part_*.parquet")))])
        g = df[df.frame == r.frame]
        u = B.group_objects(B.unique_boxes(g))
        grp = u[u.obj == r.obj]
        rep = grp[grp.rep].iloc[0]
        ba, bb = baseline(grp, 1920, 1080)
        px = max(r.x1 - r.x0, r.y1 - r.y0)
        rows.append(dict(id=r.id, source=r.source, stem=r.stem, piece=D.piece(r.stem), frame=r.frame, obj=r.obj,
                         x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1, W=1920, H=1080, px=px, band=D.size_band(px),
                         truth=lab.get(str(r.id), "ambiguous"), fishy=r.fishy, base_a=ba, base_b=bb,
                         **{c: rep[c] for c in ("s_whale", "s_fish", "s_fish_prompt", "s_school_of_fish",
                                                "s_small_fish", "s_air_bubbles")}))
    t = pd.DataFrame(rows)
    t["truth3"] = t.truth.replace({"back": "other", "nothing": "other"})
    return t


if __name__ == "__main__":
    {"pick": pick, "feat": feat}[sys.argv[1]]()
