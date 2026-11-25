"""Steps 2-4 of 08_whale_fish_boxes: baseline vs a linear box classifier, leave one source out.

Reads CACHE/objects.parquet (table.py), CACHE/feat/<stem>.npz and CACHE/owl/<stem>.parquet (extract.py).
Classes: whale / fish / other (back or nothing); ambiguous objects are left out of training and scoring.
Model: StandardScaler + multinomial logistic regression (class weights balanced); C chosen by an inner
leave-one-recording-out search on the TRAINING sources only (mean per-class recall).

Folds: LOSO (test dev: train d1 + d2; test d1: train dev + d2; test d2: train dev + d1) and the two pairwise folds
d1 -> d2, d2 -> d1 (no dev). Feature sets: geom+scores (no image), tight (+geom+scores), tight+ctx (+geom+scores),
tight+ctx+motion (step 4: the box 0.5 s later).

    .venv/bin/python experiments/08_whale_fish_boxes/train.py   -> results/*.json, CACHE/predictions.parquet,
                                                                  CACHE/boxclf_v1.npz (all sources, delivered)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import data as D  # noqa: E402
from uwid import boxclf as B  # noqa: E402

warnings.filterwarnings("ignore")
RES = HERE / "results"
CLS3 = ("whale", "fish", "other")
CS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
FEATURE_SETS = {
    "geom+scores": ("geom", "scores"),
    "tight": ("tight", "geom", "scores"),
    "tight+ctx": ("tight", "ctx", "geom", "scores"),
    "tight+ctx+motion": ("tight", "ctx", "geom", "scores", "motion"),
    "geom+scores+motion": ("geom", "scores", "motion"),
}
FOLDS = {"test dev": (("d1", "d2"), "dev"), "test d1": (("dev", "d2"), "d1"), "test d2": (("dev", "d1"), "d2"),
         "d1 -> d2": (("d1",), "d2"), "d2 -> d1": (("d2",), "d1")}


def _key(stem, f, x0, y0, x1, y1):
    return (stem, int(f), round(float(x0), 2), round(float(y0), 2), round(float(x1), 2), round(float(y1), 2))


def motion_features(obj: pd.DataFrame) -> np.ndarray:
    """Per object: max IoU with any box (whale / fish score >= 0.2) 0.5 s later, log size ratio of that box,
    cosine similarity of the same square's DINOv3 class token and mean patch token 0.5 s later, has-partner flag."""
    out = np.zeros((len(obj), 5))
    owl = {}
    for rec in D.recordings():
        o = pd.read_parquet(D.CACHE / "owl" / f"{rec.stem}.parquet")
        owl[rec.stem] = {int(f): g for f, g in o.groupby("frame")}
    for i, r in enumerate(obj.itertuples()):
        g = owl[r.stem].get(int(r.frame) + D.PARTNER_STEP)
        if g is None or not r.has_partner:
            continue
        u = B.unique_boxes(g.drop(columns=["kind"]))
        out[i, 4] = 1.0
        if len(u):
            xy = u[["x0", "y0", "x1", "y1"]].to_numpy(float)
            iou = D.iou_xyxy(np.array([[r.x0, r.y0, r.x1, r.y1]]), xy)[0]
            j = int(np.argmax(iou))
            out[i, 0] = iou[j]
            pj = max(xy[j, 2] - xy[j, 0], xy[j, 3] - xy[j, 1]) * 1920.0 / r.W
            out[i, 1] = np.log(pj / r.px) if iou[j] > 0 else 0.0
    return out


def load_table() -> tuple[pd.DataFrame, dict]:
    obj = pd.read_parquet(D.CACHE / "objects.parquet")
    feats = {}
    for rec in D.recordings():
        z = np.load(D.CACHE / "feat" / f"{rec.stem}.npz")
        if not len(z["frame"]):
            continue
        for i in range(len(z["frame"])):
            feats[_key(rec.stem, z["frame"][i], z["x0"][i], z["y0"][i], z["x1"][i], z["y1"][i])] = \
                (z["tight"][i], z["ctx"][i], z["partner_tight"][i], z["has_partner"][i])
    idx = [feats.get(_key(r.stem, r.frame, r.x0, r.y0, r.x1, r.y1)) for r in obj.itertuples()]
    miss = sum(x is None for x in idx)
    if miss:
        print(f"WARNING: {miss} objects without features (dropped)")
    obj = obj[[x is not None for x in idx]].reset_index(drop=True)
    idx = [x for x in idx if x is not None]
    tight = np.stack([x[0] for x in idx]).astype(np.float32)
    ctx = np.stack([x[1] for x in idx]).astype(np.float32)
    ptight = np.stack([x[2] for x in idx]).astype(np.float32)
    obj["has_partner"] = [bool(x[3]) for x in idx]
    xy = obj[["x0", "y0", "x1", "y1"]].to_numpy(float)
    geom = np.concatenate([B.geometry(xy[i:i + 1], int(w), int(h)) for i, (w, h) in enumerate(zip(obj.W, obj.H))])
    u = obj.rename(columns={})
    scores = B.score_features(u)
    mot = motion_features(obj)

    def cos(a, b):
        return (a * b).sum(1) / np.clip(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-6, None)
    sim_cls = np.where(obj.has_partner, cos(tight[:, :768], ptight[:, :768]), 0.0)
    sim_mean = np.where(obj.has_partner, cos(tight[:, 768:], ptight[:, 768:]), 0.0)
    motion = np.concatenate([mot[:, :2], sim_cls[:, None], sim_mean[:, None], mot[:, 4:5]], 1)
    obj["m_iou"], obj["m_ratio"], obj["m_sim_cls"], obj["m_sim_mean"] = motion[:, 0], motion[:, 1], motion[:, 2], motion[:, 3]
    parts = {"tight": tight, "ctx": ctx, "geom": geom, "scores": scores, "motion": motion}
    return obj, parts


def load_extra():
    """The extra 300-800 px test boxes (extra.py, AI by-eye labels by the WP10 agent) with their features."""
    fp = D.CACHE / "extra" / "feat.npz"
    if not fp.exists() or not (HERE / "extra_labels.json").exists():
        return None, None
    import extra
    t = extra.table()
    z = np.load(fp)
    pos = {int(i): k for k, i in enumerate(z["id"])}
    ii = [pos[int(i)] for i in t.id]
    xy = t[["x0", "y0", "x1", "y1"]].to_numpy(float)
    parts = {"tight": z["tight"][ii].astype(np.float32), "ctx": z["ctx"][ii].astype(np.float32),
             "geom": B.geometry(xy, 1920, 1080), "scores": B.score_features(t)}
    return t, parts


def X_of(parts, blocks, rows):
    return np.concatenate([parts[b][rows] for b in blocks], 1)


def balanced_recall(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def fit(X, y, groups, C=None):
    """Scaler + LR; C by inner leave-one-group-out (groups = recordings) when not given."""
    if C is None:
        ug = np.unique(groups)
        best = (-1, CS[0])
        if len(ug) >= 2:
            for c in CS:
                sc = []
                for g in ug:
                    tr, te = groups != g, groups == g
                    if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                        continue
                    s = StandardScaler().fit(X[tr])
                    m = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(s.transform(X[tr]), y[tr])
                    pr = m.predict(s.transform(X[te]))
                    sc.append((balanced_recall(y[te], pr), te.sum()))
                if sc:
                    v = np.average([a for a, _ in sc], weights=[n for _, n in sc])
                    if v > best[0]:
                        best = (v, c)
        C = best[1]
    s = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(s.transform(X), y)
    return s, m, C


def main():
    RES.mkdir(exist_ok=True)
    obj, parts = load_table()
    lab = obj.truth3.isin(CLS3).to_numpy()
    y_all = obj.truth3.map({c: i for i, c in enumerate(CLS3)}).fillna(-1).astype(int).to_numpy()
    preds = []
    summary = {}
    ext, ext_parts = load_extra()
    ext_preds = []
    for fs_name, blocks in FEATURE_SETS.items():
        for fold, (train_src, test_src) in FOLDS.items():
            tr = lab & obj.source.isin(train_src).to_numpy()
            te = lab & (obj.source == test_src).to_numpy()
            # recordings as inner groups; the dev clip is one group
            groups = obj.stem.to_numpy()[tr]
            s, m, C = fit(X_of(parts, blocks, tr), y_all[tr], groups)
            P = m.predict_proba(s.transform(X_of(parts, blocks, te)))
            full = np.zeros((te.sum(), 3)); full[:, m.classes_] = P
            d = obj.loc[te, ["source", "stem", "frame", "obj", "band", "px", "truth", "truth3", "base_a", "base_b"]].copy()
            d["fold"], d["features"], d["C"] = fold, fs_name, C
            d["p_whale"], d["p_fish"], d["p_other"] = full[:, 0], full[:, 1], full[:, 2]
            d["pred"] = [CLS3[k] for k in full.argmax(1)]
            preds.append(d)
            if ext is not None and "motion" not in blocks:
                sel = ((ext.source == test_src) & ext.truth3.isin(CLS3)).to_numpy()
                if sel.any():
                    Pe = m.predict_proba(s.transform(X_of(ext_parts, blocks, sel)))
                    fe = np.zeros((sel.sum(), 3)); fe[:, m.classes_] = Pe
                    e = ext.loc[sel, ["id", "source", "stem", "frame", "band", "px", "truth", "truth3", "base_a",
                                      "base_b", "fishy", "s_whale", "s_fish", "x0", "y0", "x1", "y1"]].copy()
                    e["fold"], e["features"] = fold, fs_name
                    e["p_whale"], e["p_fish"], e["p_other"] = fe[:, 0], fe[:, 1], fe[:, 2]
                    e["pred"] = [CLS3[k] for k in fe.argmax(1)]
                    ext_preds.append(e)
            summary[f"{fs_name}|{fold}"] = {"C": C, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                                            "train_counts": obj.loc[tr, "truth3"].value_counts().to_dict(),
                                            "balanced_recall": balanced_recall(y_all[te], full.argmax(1))}
            print(fs_name, fold, summary[f"{fs_name}|{fold}"], flush=True)
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(D.CACHE / "predictions.parquet", index=False)
    if ext_preds:
        pd.concat(ext_preds, ignore_index=True).to_parquet(D.CACHE / "predictions_extra.parquet", index=False)
    (RES / "train_summary.json").write_text(json.dumps(summary, indent=1))
    # delivered model: all sources, tight+ctx (+geom+scores) unless main() is told otherwise
    blocks = FEATURE_SETS[DELIVERED]
    Xf, yf, gf = X_of(parts, blocks, lab), y_all[lab], obj.stem.to_numpy()[lab]
    counts = obj.loc[lab, "truth3"].value_counts().to_dict()
    if ext is not None:
        sel = ext.truth3.isin(CLS3).to_numpy()
        Xf = np.concatenate([Xf, X_of(ext_parts, blocks, sel)])
        yf = np.concatenate([yf, ext.truth3[sel].map({c: i for i, c in enumerate(CLS3)}).to_numpy()])
        gf = np.concatenate([gf, ("extra_" + ext.stem[sel]).to_numpy()])
        counts = {"reference": counts, "extra": ext.truth3[sel].value_counts().to_dict()}
    s, m, C = fit(Xf, yf, gf)
    meta = {"trained_on": "dev + d1 + d2 (all labelled underwater boxes, ambiguous left out) + the 81 extra boxes",
            "n": int(len(yf)), "counts": counts, "C": C,
            "features": blocks, "date": pd.Timestamp.now().isoformat(timespec="minutes")}
    m.classes_ = np.asarray(m.classes_)
    B.LinearBoxModel.from_sklearn(s, m, B.FeatureSpec(tuple(blocks)), meta).save(B.DEFAULT_WEIGHTS)
    print("saved", B.DEFAULT_WEIGHTS, meta)


DELIVERED = "tight+ctx"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DELIVERED = sys.argv[1]
    main()
