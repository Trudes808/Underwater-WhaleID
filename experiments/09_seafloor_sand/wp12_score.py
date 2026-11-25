"""WP12 scoring (CPU): pixels on held-out episodes, the combined back + seafloor class, whale safety, windows.
Reads wp12/mask_preds.npz (wp12.py heads) and frames_v2/*.npz (wp12.py frames2). Writes results/wp12_*.json.

Evidence design (fixed before looking at results, orchestrator 2026-11-23): seafloor exists in ONE recording
(2025-01-02 00017); fold k (k = 1..4, the episodes E1 292-327, E2 405-422, E3 465-487, E4 510-524 + 527-533 s) is a
head that saw no mask and no window within 5 s of episode k; its window threshold is chosen on 00017 windows outside
episode k +- 5 s only (positives: the other episodes; negatives: `none` spans of 00017). False calls everywhere
else; the hard negatives (the withdrawn `visible` spans) and the weak positives (00023 286-301, 1180-1190) apart.

    .venv/bin/python experiments/09_seafloor_sand/wp12_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "08_whale_fish_boxes"), str(HERE.parent / "04b_patch_segmentation")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402
import wp12 as W  # noqa: E402
from uwid import patchseg as ps  # noqa: E402

RES = HERE / "results"
FV2 = D.SF_CACHE / "frames_v2"
GRID = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
SEA, BACK, WATER, WHALE = ps.SEAFLOOR, ps.BACK, ps.WATER, ps.WHALE
K = len(ps.CLASSES)
D2 = {D.piece(r.stem): r.stem for r in D.recordings(("d2",))}
D1_DENSE = "CC-11-96-20250203-111250-000-00006"


# ------------------------------------------------------------------------------------------------ pixels

def class_mask(name, frame, fs):
    return fs[name][1].class_mask(int(frame))


def confusion(pred_maps, frames, name, fs):
    from common import border_band
    c = np.zeros((K, K), np.int64); cb = np.zeros((K, K), np.int64)
    for m, f in zip(pred_maps, frames):
        cm = class_mask(name, f, fs)
        pred = cv2.resize(m, (cm.shape[1], cm.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        lab = cm != 255
        c += np.bincount(cm[lab].astype(np.int64) * K + pred[lab], minlength=K * K).reshape(K, K)
        lab2 = lab & ~border_band(cm, 12)
        cb += np.bincount(cm[lab2].astype(np.int64) * K + pred[lab2], minlength=K * K).reshape(K, K)
    return c, cb


def share(c, a, b):
    n = int(c[a].sum())
    return {"n": n, "k": int(c[a, b]), "share": float(c[a, b] / max(1, n))}


def iou(c, k):
    return float(c[k, k] / max(1, c[k].sum() + c[:, k].sum() - c[k, k]))


def merge_bg(c):
    """Confusion with back and seafloor merged into one class (index BACK)."""
    m = c.copy()
    m[BACK] += m[SEA]; m[SEA] = 0
    m[:, BACK] += m[:, SEA]; m[:, SEA] = 0
    return m


def pixel_part(z, fs):
    out = {}
    fr = z["frames|wp12"]
    tt = fr / W.FPS
    tot = np.zeros((K, K), np.int64); totb = np.zeros((K, K), np.int64)
    for k in (1, 2, 3, 4):
        sel = np.array([W.episode_of(t) == k for t in tt])
        c, cb = confusion(z[f"v2_fold{k}|wp12"][sel], fr[sel], "wp12", fs)
        tot += c; totb += cb
        out[f"E{k}"] = summarize(c, cb, int(sel.sum()))
    out["all held-out"] = summarize(tot, totb, len(fr))
    return out


def summarize(c, cb, n):
    mc, mcb = merge_bg(c), merge_bg(cb)
    return {"frames": n,
            "seafloor_px": int(c[SEA].sum()), "seafloor_found": share(c, SEA, SEA), "seafloor_iou": iou(c, SEA),
            "seafloor_iou_band12": iou(cb, SEA), "seafloor_called_back": share(c, SEA, BACK),
            "seafloor_called_water": share(c, SEA, WATER), "back_called_seafloor": share(c, BACK, SEA),
            "water_called_seafloor": share(c, WATER, SEA), "back_iou": iou(c, BACK),
            "combined": {"bg_px": int(mc[BACK].sum()), "bg_found": share(mc, BACK, BACK), "bg_iou": iou(mc, BACK),
                         "bg_iou_band12": iou(mcb, BACK), "bg_called_water": share(mc, BACK, WATER),
                         "water_called_bg": share(mc, WATER, BACK), "bg_called_whale": share(mc, BACK, WHALE),
                         "whale_called_bg": share(mc, WHALE, BACK)}}


def whale_safety(z, fs):
    """Whale pixels called seafloor / back, new head against old head on the same frames."""
    out = {}
    pairs = [("2025-02-03 (heads not fitted on it)", [k for k in fs if k.startswith("CC-11")], "v2_test_d1", "old_test_d1"),
             ("dev clip (heads not fitted on it)", ["dev"], "v2_test_dev", "old_test_dev"),
             ("2025-01-02 (IN-SAMPLE: both heads saw these masks)", [k for k in fs if k.startswith("CC-09") or k == "wp12"],
              "v2_all", "old_all")]
    for label, names, new, old in pairs:
        r = {}
        for h in (new, old):
            c = np.zeros((K, K), np.int64)
            for n in names:
                cc, _ = confusion(z[f"{h}|{n}"], z[f"frames|{n}"], n, fs)
                c += cc
            r[h] = {"whale_px": int(c[WHALE].sum()), "whale_called_seafloor": share(c, WHALE, SEA),
                    "whale_called_back": share(c, WHALE, BACK), "whale_found": share(c, WHALE, WHALE)}
        out[label] = r
    return out


# ------------------------------------------------------------------------------------------------ windows

def load(kind, stem):
    p = FV2 / f"{kind}_{stem}.npz"
    return dict(np.load(p)) if p.exists() else None


def windows(z, unit_t=None):
    hs = list(z["heads"])
    t = z["t"] if unit_t is None else unit_t
    k = np.floor(t).astype(int)
    rows = []
    for w in np.unique(k):
        ii = np.nonzero(k == w)[0]
        r = {"k": int(w), "n": len(ii), "grey_max": float(z["grey_mean"][ii].max()), "i0": int(ii[0])}
        for h in hs:
            maps = z["maps"][ii, hs.index(h)]
            r[f"sea_{h}"] = float(np.mean([(m == SEA).mean() for m in maps]))
        rows.append(r)
    return pd.DataFrame(rows)


def spans(stem):
    p = D.LABELS / stem / "seafloor_spans.csv"
    return pd.read_csv(p, keep_default_na=False) if p.exists() else pd.DataFrame()


def truth_of(sp, t):
    m = sp[(sp.start_s <= t) & (t < sp.end_s)]
    return m.iloc[0].seafloor if len(m) else "none"


def blocked(stem, t):
    p = D.LABELS / stem / "blocked_spans.csv"
    if not p.exists():
        return ""
    b = pd.read_csv(p)
    m = b[(b.start_s <= t) & (t < b.end_s)]
    return m.iloc[0].kind if len(m) else ""


def best_threshold(score, y):
    best = (-1, GRID[0])
    for t in GRID:
        p = score >= t
        br = 0.5 * (p[y == 1].mean() + (~p[y == 0]).mean())
        if br > best[0]:
            best = (br, t)
    return best[1]


def window_part():
    res = {}
    stem = D2[ "00017"]
    w17 = windows(load("dense", stem)); sp = spans(stem)
    w17["truth"] = [truth_of(sp, k + 0.5) for k in w17.k]
    w17["ep"] = [W.episode_of(k + 0.5) for k in w17.k]
    w17["near"] = [W.episode_of(k + 0.5, W.BUFFER) for k in w17.k]
    w17["air"] = [blocked(stem, k + 0.5) == "air" for k in w17.k]
    w17["dark"] = w17.grey_max < 20
    w17 = w17[~w17.air]
    thr = {}
    for f in (1, 2, 3, 4):
        h = f"v2_fold{f}"
        tr = w17[(w17.near != f) & (w17.truth.isin(["visible", "none"])) & ((w17.ep > 0) | (w17.truth == "none"))]
        thr[h] = best_threshold(tr[f"sea_{h}"].to_numpy(), (tr.truth == "visible").to_numpy().astype(int))
    tr = w17[w17.truth.isin(["visible", "none"])]
    for h in ("v2_test_d1", "v2_test_dev", "v2_all"):
        thr[h] = best_threshold(tr[f"sea_{h}"].to_numpy(), (tr.truth == "visible").to_numpy().astype(int))
    res["thresholds"] = thr
    # held-out episodes
    for f in (1, 2, 3, 4):
        h = f"v2_fold{f}"
        e = w17[w17.ep == f]
        res[f"00017 E{f} held out|{h}"] = {"windows": len(e), "found": int((e[f"sea_{h}"] >= thr[h]).sum())}
    # false calls inside 00017 per fold (windows not near the held-out episode and not in any episode)
    groups = {"none 0-292 / 644-673": (w17.truth == "none") & (w17.k < 842),
              "HARD NEG: withdrawn 'visible' 842-1390": (w17.truth == "none") & (w17.k >= 842),
              "sediment": w17.truth == "sediment",
              "unsure, black (grey < 20)": (w17.truth == "unsure") & w17.dark,
              "unsure, not black": (w17.truth == "unsure") & ~w17.dark}
    for g, m in groups.items():
        r = {}
        for f in (1, 2, 3, 4):
            h = f"v2_fold{f}"
            x = w17[m & (w17.near != f)]
            r[h] = [int((x[f"sea_{h}"] >= thr[h]).sum()), len(x)]
        res[f"00017 {g}"] = r
    # other recordings of 2025-01-02
    for piece in ("00010", "00011", "00012", "00023", "00024"):
        z = load("dense", D2[piece])
        if z is None:
            continue
        w = windows(z)
        w["air"] = [blocked(D2[piece], k + 0.5) == "air" for k in w.k]
        w = w[~w.air]
        sets = {piece: np.ones(len(w), bool)}
        if piece == "00023":
            weak = ((w.k >= 286) & (w.k < 301)) | ((w.k >= 1180) & (w.k < 1190))
            sets = {"00023 HARD NEG: withdrawn 'visible' 0-306 without 286-301": ((w.k < 306) & ~weak).to_numpy(),
                    "00023 WEAK POSITIVES 286-301, 1180-1190 (not fitted on)": weak.to_numpy(),
                    "00023 rest": ((w.k >= 306) & ~weak).to_numpy()}
        for name, m in sets.items():
            x = w[m]
            res[name] = {f"v2_fold{f}": [int((x[f"sea_v2_fold{f}"] >= thr[f"v2_fold{f}"]).sum()), len(x)] for f in (1, 2, 3, 4)}
    # 2025-02-03 and the dev clip (heads not fitted on them)
    z6 = load("dense", D1_DENSE)
    sw = pd.read_csv(f"/home/sat3737/NOAA_whales/underwater_cache/scan/2025-2-3 Mini-mother 11-96/sensitive/windows/{D1_DENSE}.mov.csv")
    uw = set(sw[(sw.water == "underwater") & (sw.tag_state == "whale")].k)
    w6 = windows(z6); w6 = w6[w6.k.isin(uw)]
    res["2025-02-03 00006 dense|v2_test_d1"] = [int((w6.sea_v2_test_d1 >= thr["v2_test_d1"]).sum()), len(w6)]
    res["2025-02-03 00006 dense|v2_all (saw these masks' deployment)"] = [int((w6.sea_v2_all >= thr["v2_all"]).sum()), len(w6)]
    for src, h in (("d1", "v2_test_d1"), ("dev", "v2_test_dev")):
        n_call, n = 0, 0
        for rec in D.recordings((src,)):
            z = load("label", rec.stem)
            fc = rec.frames_csv().set_index("frame")
            keep = np.array([fc.loc[int(f), "state"] == "underwater" for f in z["frame"]])
            zz = {k: (v[keep] if k != "heads" else v) for k, v in z.items()}
            ut = np.arange(keep.sum()) + 0.5 if src == "d1" else None
            w = windows(zz, ut)
            n_call += int((w[f"sea_{h}"] >= thr[h]).sum()); n += len(w)
        res[f"{'2025-02-03 labelled' if src == 'd1' else 'dev clip labelled'}|{h}"] = [n_call, n]
    w17.to_parquet(W.W12 / "windows_00017.parquet", index=False)
    return res


def main():
    RES.mkdir(exist_ok=True)
    fs = W.folders()
    z = dict(np.load(W.W12 / "mask_preds.npz"))
    pix = pixel_part(z, fs)
    whale = whale_safety(z, fs)
    (RES / "wp12_pixels.json").write_text(json.dumps({"pixels": pix, "whale_safety": whale}, indent=1))
    print(json.dumps(pix["all held-out"], indent=1)); print(json.dumps(whale, indent=1))
    if (FV2 / "DONE").exists():
        win = window_part()
        (RES / "wp12_windows.json").write_text(json.dumps(win, indent=1))
        print(json.dumps(win, indent=1))


if __name__ == "__main__":
    main()
