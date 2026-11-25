"""WP14 question 1, measures (a) and (b) (CPU; needs maps.py and runs.py).

(a) on the labelled back pixels (reference masks of 2025-02-03, 2025-01-02 and the dev clip): share kept inside the
    recording's template, at several tau.
(b) 00005 604-629 s: the scan's whale boxes (window best boxes) and the tracker's objects there against the template.

    .venv/bin/python experiments/10_own_back/measure.py     -> results/measure.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[0] / "08_whale_fish_boxes")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import data as D  # noqa: E402
import ownback as O  # noqa: E402
from uwid import patchseg as ps  # noqa: E402

TAUS = (0.3, 0.5, 0.7)


def rec_templates(tau, dil=1):
    idx = pd.read_csv(C.OUT / "frames" / "index.csv")
    out = {}
    for f in (C.OUT / "maps").glob("*.npz"):
        z = np.load(f)
        tf = set(idx[(idx.stem == f.stem) & (idx.kind == "template")].frame)
        sel = np.array([int(x) in tf for x in z["frames"]])
        out[f.stem.split("-")[-1] + ("_d1" if f.stem.startswith("CC-11") else "_d2")] = O.back_template(z["labels"][sel], tau, dil)
    return out


def dev_template(tau, dil=1):
    """Dev clip: no scan sample; the WP5b head's maps on the 31 dev mask frames (IN-SAMPLE: the head was fitted on
    these frames)."""
    clf, _ = ps.load_classifier(C.WP5B_HEAD, "cpu")
    d = np.load(C.CACHE / "patchseg" / "labels_dev_w1280.npz")
    labs = [ps.predict_probs(clf, torch.from_numpy(f)).argmax(-1).numpy() for f in d["feats"]]
    return O.back_template(np.stack(labs), tau, dil), d["frames"]


def part_a():
    res = {}
    for tau in TAUS:
        T = rec_templates(tau)
        dt, _ = dev_template(tau)
        rows = {}
        for rec in D.recordings():
            fr = rec.mask_frames()
            if not fr:
                continue
            if rec.source == "dev":
                tmpl = dt
            else:
                key = D.piece(rec.stem) + "_" + rec.source
                if key not in T:
                    continue
                tmpl = T[key]
            kept = tot = 0
            for f in fr:
                cm = cv2.imread(str(rec.ref / "masks" / f"{f:06d}_class.png"), cv2.IMREAD_UNCHANGED)
                back = cm == C.BACK
                tpx = O.upsample(tmpl, cm.shape)
                kept += int((back & tpx).sum()); tot += int(back.sum())
            g = rec.source
            r = rows.setdefault(g, [0, 0, 0]); r[0] += kept; r[1] += tot; r[2] += len(fr)
        res[f"tau {tau}"] = {g: {"back_px": v[1], "kept": v[0], "share": v[0] / max(v[1], 1), "mask_frames": v[2]} for g, v in rows.items()}
    return res


def part_b(tau=0.5):
    T = rec_templates(tau)
    tmpl = T["00005_d1"]
    stem = "CC-11-96-20250203-104242-000-00005"
    w = pd.read_csv(C.SCAN / C.DEPL["d1"] / "sensitive" / "windows" / f"{stem}.mov.csv")
    s = w[(w.t0 >= 604) & (w.t0 < 629) & w.whale.astype(str).str.lower().eq("true")]
    tpx = O.upsample(tmpl, (1080, 1920))
    rows = []
    for r in s.itertuples():
        box = [r.bx0, r.by0, r.bx1, r.by1]
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        inside = float(tpx[max(y0, 0):y1, max(x0, 0):x1].mean())
        rows.append(dict(t0=r.t0, inside_template=inside, passes=not O.box_on_back(box, tpx)))
    df = pd.DataFrame(rows)
    tr = pd.read_parquet(C.OUT / "runs_tracks.parquet")
    trk = tr[(tr.stem == stem) & (tr.run == "0607.00-0627.00_online")]
    return {"windows_with_scan_whale": len(df), "box_share_inside_template_median": float(df.inside_template.median()) if len(df) else None,
            "boxes_passing_back_test": int(df.passes.sum()) if len(df) else 0,
            "tracks": trk[["obj", "dur_s", "in_sam_back", "in_own_back", "in_template"]].round(3).to_dict("records"),
            "template_area_share": float(tmpl.mean())}


def main():
    out = {"a_back_kept": part_a(), "b_00005_604_629": part_b()}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "measure.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
