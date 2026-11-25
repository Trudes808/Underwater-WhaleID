"""Scoring of 09_seafloor_sand (CPU): seafloor per window and per pixel, false seafloor, the back confusion, sand
(within recording 00017), dark (brightness rule). Reads gpu.py's outputs; writes results/*.json, the window tables
in the cache, the sand model and the rules.

Window = the 1 s grid slot [k, k + 1) s; the dense recordings carry two frames per window (0.25 / 0.75 s, as the
scan), the dev clip's labelled frames are two per window, the 2025-02-03 labelled frames one per window.

    .venv/bin/python experiments/09_seafloor_sand/score.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "08_whale_fish_boxes"))
sys.path.insert(0, str(HERE.parent / "04b_patch_segmentation"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import data as D  # noqa: E402
from uwid import patchseg as ps  # noqa: E402

warnings.filterwarnings("ignore")
RES = HERE / "results"
OUT = D.SF_CACHE
HEADS = ("test17", "test23", "test_d1", "test_dev", "all")
SEA = ps.SEAFLOOR
GRID = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
DEFAULT_SHARE = 0.05
D2_STEMS = {D.piece(r.stem): r.stem for r in D.recordings(("d2",))}
D1_DENSE = "CC-11-96-20250203-111250-000-00006"


def load(kind, stem):
    p = OUT / "frames" / f"{kind}_{stem}.npz"
    return dict(np.load(p)) if p.exists() else None


def spans(stem, name):
    p = D.LABELS / stem / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def span_at(sp: pd.DataFrame, t: float, col: str):
    if not len(sp):
        return None
    m = sp[(sp.start_s <= t) & (t < sp.end_s)]
    return m.iloc[0][col] if len(m) else None


def windows_from(z: dict, head_names, unit="dense") -> pd.DataFrame:
    """Per window: mean share of each class per head, frames, grey levels, embedding mean."""
    hidx = {h: i for i, h in enumerate(list(z["heads"]))}
    k = np.floor(z["t"]).astype(int)
    rows = []
    for w in np.unique(k):
        ii = np.nonzero(k == w)[0]
        r = {"k": int(w), "n": len(ii), "frames": [int(z["frame"][i]) for i in ii],
             "grey_max": float(z["grey_mean"][ii].max()), "grey_mean": float(z["grey_mean"][ii].mean()),
             "p95_max": float(z["grey_p95"][ii].max())}
        for h in head_names:
            maps = z["maps"][ii, hidx[h]]
            sh = np.stack([np.bincount(m.ravel(), minlength=len(ps.CLASSES)) / m.size for m in maps]).mean(0)
            r[f"sea_{h}"] = float(sh[SEA]); r[f"back_{h}"] = float(sh[ps.BACK])
            seam = maps == SEA
            r[f"sea_low_{h}"] = float(seam[:, seam.shape[1] // 2:].sum() / max(1, seam.sum()))
        rows.append(r)
    df = pd.DataFrame(rows)
    df["emb_i"] = [list(np.nonzero(k == w)[0]) for w in df.k]
    return df


def best_threshold(score, y):
    best = (-1, None)
    for t in GRID:
        p = score >= t
        br = 0.5 * ((p[y == 1]).mean() + (~p[y == 0]).mean())
        if br > best[0]:
            best = (br, t)
    return best[1]


def d2_window_table(stem: str) -> pd.DataFrame | None:
    z = load("dense", stem)
    if z is None:
        return None
    w = windows_from(z, HEADS)
    sf, bl = spans(stem, "seafloor_spans.csv"), spans(stem, "blocked_spans.csv")
    w["seafloor_truth"] = [span_at(sf, k + 0.5, "seafloor") or "none" for k in w.k]
    w["blocked"] = [span_at(bl, k + 0.5, "kind") or "" for k in w.k]
    w["stem"] = stem; w["piece"] = D.piece(stem)
    return w


def seafloor_part(tabs: dict, other: dict) -> dict:
    res = {}
    # thresholds chosen on training recordings' windows only
    def train_windows(head):
        if head == "test17":
            parts = [tabs["00023"]]
        elif head == "test23":
            parts = [tabs["00017"]]
        else:
            parts = [tabs["00017"], tabs["00023"]]
        t = pd.concat(parts)
        t = t[(t.seafloor_truth != "unsure") & (t.blocked != "air")]
        return t[f"sea_{head}"].to_numpy(), (t.seafloor_truth == "visible").to_numpy().astype(int)
    thr = {h: best_threshold(*train_windows(h)) for h in HEADS}
    res["thresholds"] = thr
    for piece, head in (("00017", "test17"), ("00023", "test23")):
        t = tabs[piece]
        t = t[t.blocked != "air"]
        for name, th in (("chosen", thr[head]), ("fixed", DEFAULT_SHARE)):
            ev = t[t.seafloor_truth != "unsure"]
            call = ev[f"sea_{head}"] >= th
            vis = ev.seafloor_truth == "visible"
            res[f"{piece}|{head}|{name}"] = {
                "threshold": th, "visible_windows": int(vis.sum()), "found": int((call & vis).sum()),
                "none_windows": int((~vis).sum()), "false": int((call & ~vis).sum()),
                "unsure_windows_called": int((t[t.seafloor_truth == "unsure"][f"sea_{head}"] >= th).sum()),
                "unsure_windows": int((t.seafloor_truth == "unsure").sum())}
    # the other on-whale recordings of 2025-01-02 (no seafloor anywhere)
    for piece in ("00010", "00011", "00012", "00024"):
        if piece not in tabs:
            continue
        t = tabs[piece]; t = t[t.blocked != "air"]
        for head in ("test17", "test23"):
            res[f"{piece}|{head}"] = {"windows": len(t), "called": int((t[f"sea_{head}"] >= thr[head]).sum()),
                                      "called_fixed": int((t[f"sea_{head}"] >= DEFAULT_SHARE).sum())}
    for key, (t, head) in other.items():
        for name, th in (("chosen", thr[head]), ("fixed", DEFAULT_SHARE)):
            call = t[f"sea_{head}"] >= th
            res[f"{key}|{head}|{name}"] = {"threshold": th, "windows": len(t), "called": int(call.sum()),
                                           "called_low_half": int((call & (t[f"sea_low_{head}"] >= 0.5)).sum())}
    return res


def pixel_part() -> dict:
    from common import border_band  # 04b_patch_segmentation
    z = np.load(OUT / "heads" / "heldout_preds.npz")
    keys = sorted({k.rsplit("|", 1)[0] for k in z.files})
    K = len(ps.CLASSES)
    res = {}
    for key in keys:
        head, stem = key.split("|")
        frames, maps = z[f"{key}|frames"], z[f"{key}|map"]
        conf = np.zeros((K, K), np.int64); confb = np.zeros((K, K), np.int64)
        for f, m in zip(frames, maps):
            cm = cv2.imread(str(D.REF / (stem if stem != "dev" else "2025-12-20_CG") / "masks" / f"{int(f):06d}_class.png"),
                            cv2.IMREAD_UNCHANGED)
            if stem == "dev":
                cm = cm.copy(); cm[cm == ps.FISH] = 255
            pred = cv2.resize(m, (cm.shape[1], cm.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            lab = cm != 255
            conf += np.bincount(cm[lab].astype(np.int64) * K + pred[lab], minlength=K * K).reshape(K, K)
            lab2 = lab & ~border_band(cm, 12)
            confb += np.bincount(cm[lab2].astype(np.int64) * K + pred[lab2], minlength=K * K).reshape(K, K)
        group = "d1" if stem.startswith("CC-11") else ("dev" if stem == "dev" else D.piece(stem))
        r = res.setdefault(f"{head}|{group}", {"frames": 0, "conf": np.zeros((K, K), np.int64),
                                               "conf_band": np.zeros((K, K), np.int64)})
        r["frames"] += len(frames); r["conf"] += conf; r["conf_band"] += confb
    out = {}
    for key, r in res.items():
        c, cb = r["conf"], r["conf_band"]
        def share(cc, a, b):
            return float(cc[a, b] / max(1, cc[a].sum()))
        iou = lambda cc, k: float(cc[k, k] / max(1, cc[k].sum() + cc[:, k].sum() - cc[k, k]))  # noqa: E731
        out[key] = {"frames": r["frames"], "seafloor_px": int(c[SEA].sum()), "back_px": int(c[ps.BACK].sum()),
                    "seafloor_recall": share(c, SEA, SEA), "seafloor_iou": iou(c, SEA),
                    "seafloor_iou_band": iou(cb, SEA), "back_iou": iou(c, ps.BACK),
                    "back_called_seafloor": share(c, ps.BACK, SEA), "seafloor_called_back": share(c, SEA, ps.BACK),
                    "seafloor_called_water": share(c, SEA, ps.WATER),
                    "water_called_seafloor": share(c, ps.WATER, SEA), "bubbles_called_seafloor": share(c, ps.BUBBLES, SEA),
                    "whale_called_seafloor": share(c, ps.WHALE, SEA),
                    "pred_seafloor_px": int(c[:, SEA].sum()),
                    "pred_seafloor_on": {ps.CLASSES[k]: int(c[k, SEA]) for k in range(K) if c[k, SEA]}}
    return out


def _xy(tab: pd.DataFrame, z: dict, positive: bool = True):
    """Window embeddings (mean of the window's frame embeddings) and sand labels; air / dark windows and windows
    the dark rule calls dark are left out (dark is decided first)."""
    t = tab
    if "blocked" in t:
        t = t[(t.blocked != "air") & (t.blocked != "dark")]
    t = t[t.grey_max >= ps.WINDOW_RULES["dark_grey"]]
    e = z["emb"].astype(np.float32)
    X = np.stack([e[i].mean(0) for i in t.emb_i]) if len(t) else np.zeros((0, e.shape[1]), np.float32)
    y = (t.blocked == "sand").to_numpy().astype(int) if "blocked" in t else np.zeros(len(t), int)
    return t, X, y


def _fit(Xa, ya, C=0.001):
    s = StandardScaler().fit(Xa)
    return s, LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(s.transform(Xa), ya)


def sand_part(tab17: pd.DataFrame, z17: dict, d2_others: dict, unseen: dict, extra_neg: dict):
    """Sand = stirred-up sediment hiding the view (blocked_spans kind sand, recording 00017 only).

    Positives exist in 00017 only, so: recall = leave-one-100-s-block-out within 00017 (5 s buffer), the training
    negatives being the other 00017 blocks plus every other on-whale recording of 2025-01-02; false alarms on another
    2025-01-02 recording = that recording left out of training; false alarms on 2025-02-03 and the 2025-12-20 clip =
    a model fitted on 2025-01-02 only. C fixed at 0.001 (the value an inner block search chose in every fold of a
    first run with 00017 negatives only)."""
    t17, X17, y17 = _xy(tab17, z17)
    others = {k: _xy(*v) for k, v in d2_others.items()}
    blocks = (t17.k // 100).to_numpy()
    p = np.zeros(len(t17))
    Xo_all = np.concatenate([v[1] for v in others.values()]); yo_all = np.zeros(len(Xo_all), int)
    for b in np.unique(blocks):
        te = blocks == b
        tr = (np.abs(t17.k.to_numpy()[:, None] - t17.k.to_numpy()[te][None]).min(1) > 5) & ~te
        s, m = _fit(np.concatenate([X17[tr], Xo_all]), np.concatenate([y17[tr], yo_all]))
        p[te] = m.predict_proba(s.transform(X17[te]))[:, 1]
    call = p >= 0.5
    res = {"windows": int(len(t17)), "sand_windows": int(y17.sum()), "found": int((call & (y17 == 1)).sum()),
           "false": int((call & (y17 == 0)).sum()), "blocks_with_sand": [int(b) for b in np.unique(blocks[y17 == 1])]}
    for k in others:
        Xtr = np.concatenate([X17] + [v[1] for kk, v in others.items() if kk != k])
        ytr = np.concatenate([y17] + [np.zeros(len(v[1]), int) for kk, v in others.items() if kk != k])
        s, m = _fit(Xtr, ytr)
        q = m.predict_proba(s.transform(others[k][1]))[:, 1] if len(others[k][1]) else np.zeros(0)
        res[f"false|{k} (left out)"] = {"windows": int(len(q)), "called": int((q >= 0.5).sum())}
    s, m = _fit(np.concatenate([X17, Xo_all]), np.concatenate([y17, yo_all]))
    d2model = ps.SandModel(s.mean_, s.scale_, m.coef_[0], float(m.intercept_[0]), {})
    for k, v in unseen.items():
        t, X, _ = _xy(*v)
        q = d2model.proba(X) if len(X) else np.zeros(0)
        res[f"false|{k} (fitted on 2025-01-02 only)"] = {"windows": int(len(q)), "called": int((q >= 0.5).sum())}
    # delivered model: 2025-01-02 plus the seafloor-free windows of the other two sources as more negatives (its
    # false-alarm rate on those sources is therefore not measured)
    Xn = [_xy(*v)[1] for v in extra_neg.values()]
    s, m = _fit(np.concatenate([X17, Xo_all] + Xn), np.concatenate([y17, yo_all] + [np.zeros(len(x), int) for x in Xn]))
    model = ps.SandModel(s.mean_, s.scale_, m.coef_[0], float(m.intercept_[0]),
                         {"trained_on": "2025-01-02 on-whale dense windows (00017 sand spans = positives; all other "
                                        "windows of 00010-00024 negatives) + 2025-02-03 00006 dense and labelled windows "
                                        "+ the 2025-12-20 clip's labelled windows as negatives",
                          "n": int(len(X17) + len(Xo_all) + sum(len(x) for x in Xn)), "n_sand": int(y17.sum()), "C": 0.001})
    return res, model, t17.assign(sand_p=p)


def dark_part(tabs: dict, others: dict) -> dict:
    res = {}
    t = tabs["00017"]; t = t[t.blocked != "air"]
    y = (t.blocked == "dark").to_numpy()
    for thr in (10, 15, 20, 25, 30, 40):
        call = (t.grey_max < thr).to_numpy()
        res[f"00017|grey<{thr}"] = {"dark_windows": int(y.sum()), "found": int((call & y).sum()),
                                    "other_windows": int((~y).sum()), "false": int((call & ~y).sum()),
                                    "false_in_sand": int((call & (t.blocked == "sand").to_numpy()).sum())}
    for key, tab in list(others.items()) + [(f"d2 {p}", tabs[p]) for p in ("00010", "00011", "00012", "00023", "00024") if p in tabs]:
        tt = tab[tab.blocked != "air"] if "blocked" in tab else tab
        res[f"false|{key}"] = {"windows": len(tt), **{f"grey<{thr}": int((tt.grey_max < thr).sum()) for thr in (20, 30)},
                               "min_grey": float(tt.grey_max.min()) if len(tt) else None}
    return res


def main():
    RES.mkdir(exist_ok=True)
    tabs = {}
    for piece, stem in D2_STEMS.items():
        w = d2_window_table(stem)
        if w is not None:
            tabs[piece] = w
    # seafloor-free sources
    others = {}
    z6 = load("dense", D1_DENSE)
    scan_w = pd.read_csv(f"/home/sat3737/NOAA_whales/underwater_cache/scan/2025-2-3 Mini-mother 11-96/sensitive/windows/{D1_DENSE}.mov.csv")
    uw = set(scan_w[(scan_w.water == "underwater") & (scan_w.tag_state == "whale")].k)
    w6 = windows_from(z6, HEADS); w6 = w6[w6.k.isin(uw)].reset_index(drop=True)
    others["d1 00006 dense"] = (w6, z6)
    zl = {}
    for rec in D.recordings(("d1", "dev")):
        z = load("label", rec.stem)
        fc = rec.frames_csv().set_index("frame")
        keep = np.array([fc.loc[int(f), "state"] == "underwater" for f in z["frame"]])
        z = {k: (v[keep] if k != "heads" else v) for k, v in z.items()}
        if rec.source == "d1":
            z["t"] = np.arange(len(z["frame"])) + 0.5    # one labelled frame = one window
        zl.setdefault(rec.source, []).append(z)
    for src, zs in zl.items():
        z = {k: (np.concatenate([q[k] for q in zs]) if k != "heads" else zs[0][k]) for k in zs[0]}
        if src == "d1":
            z["t"] = np.arange(len(z["frame"])) + 0.5
        others[f"{src} labelled"] = (windows_from(z, HEADS), z)
    sea_other = {"d1 00006 dense": (others["d1 00006 dense"][0], "test_d1"),
                 "d1 labelled": (others["d1 labelled"][0], "test_d1"),
                 "dev labelled": (others["dev labelled"][0], "test_dev")}
    sea = seafloor_part(tabs, sea_other)
    pix = pixel_part()
    d2o = {f"d2 {p}": (tabs[p], load("dense", D2_STEMS[p])) for p in ("00010", "00011", "00012", "00023", "00024") if p in tabs}
    sand, sand_model, t17 = sand_part(tabs["00017"], load("dense", D2_STEMS["00017"]), d2o, others, others)
    sand_model.save(ps.SAND_CLF)
    dark = dark_part(tabs, {k: v[0] for k, v in others.items()})
    (RES / "seafloor.json").write_text(json.dumps(sea, indent=1))
    (RES / "pixels.json").write_text(json.dumps(pix, indent=1))
    (RES / "sand.json").write_text(json.dumps(sand, indent=1))
    (RES / "dark.json").write_text(json.dumps(dark, indent=1))
    rules = {"seafloor_share": sea["thresholds"]["all"], "dark_grey": 20.0, "sand_p": 0.5}
    (RES / "rules.json").write_text(json.dumps(rules, indent=1))
    keep_cols = ["stem", "piece", "k", "n", "seafloor_truth", "blocked", "grey_max"] + [f"sea_{h}" for h in HEADS] + \
                [f"back_{h}" for h in HEADS] + [f"sea_low_{h}" for h in HEADS]
    allw = pd.concat([tabs[p][keep_cols] for p in tabs], ignore_index=True)
    allw.to_parquet(OUT / "d2_windows.parquet", index=False)
    t17[["k", "blocked", "sand_p"]].to_parquet(OUT / "sand_00017_heldout.parquet", index=False)
    for k, (w, _) in others.items():
        w.drop(columns=["emb_i", "frames"]).to_parquet(OUT / f"windows_{k.replace(' ', '_')}.parquet", index=False)
    print(json.dumps({"seafloor": sea, "sand": sand, "dark": dark}, indent=1)[:6000])
    print(json.dumps(pix, indent=1)[:4000])


if __name__ == "__main__":
    main()
