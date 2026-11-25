"""WP12 step B (CPU only): a `sand` label per 1 s window ("view blocked by stirred-up sediment"), refitted on the
user-confirmed sediment of 2025-01-02 00017, and a sensor table for the bottom-contact question.

Truth (00017): the user confirmed the 22 sediment frames of `sand_candidates_best_01.jpg` (2026-11-23: "Yes those
are the sediment frames of dragging along seafloor."). Episodes = the WP12 `sediment` spans that hold confirmed
frames: S1 327-336, S2 422-443 + 449-459, S3 524-527 + 533-536 + 540-557 (plus the 522 s frame), S4 610-622 s.
Not-sediment: `visible` (user-confirmed seafloor) and `none` spans of 00017 (incl. the old false cases 411 s, 660 s);
`unsure` (black / murk) and windows dark by the grey rule are left out; the AI-only sediment spans 736-765 and
874-898 s (no confirmed frame) are left out of fitting and scored apart; 519-524 s (bottom with sediment rising) is
left out.

Features: the cached frame embeddings (gpu.py frames: mean DINOv3 patch feature of the frame and its four
quarters, 3,840 numbers), window = mean of its two frames. Logistic regression (C 0.001, class weights balanced).
Design fixed before results, as step A: leave one episode out inside 00017 (5 s buffer); the fold's threshold is
chosen by an inner leave-one-episode-out on its training episodes. Variant A fits on 00017 only; variant B adds every
window of the other on-whale recordings of 2025-01-02 as not-sediment, and its false calls on such a recording come
from a fit that left that recording out. 2025-02-03 and the dev clip are never in a fit.

    nice -n 10 .venv/bin/python experiments/09_seafloor_sand/sand_redo.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "08_whale_fish_boxes")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import data as D  # noqa: E402
from uwid import patchseg as ps  # noqa: E402

warnings.filterwarnings("ignore")
FR = D.SF_CACHE / "frames"
SCANW = Path("/home/sat3737/NOAA_whales/underwater_cache/scan")
DEPN = {"d1": "2025-2-3 Mini-mother 11-96", "d2": "TestSubset 2025-1-2 CATS -85"}
D2 = {D.piece(r.stem): r.stem for r in D.recordings(("d2",))}
EPIS = {1: [(327, 336)], 2: [(422, 443), (449, 459)], 3: [(522, 527), (533, 536), (540, 557)], 4: [(610, 622)]}
AI_ONLY = [(736, 765), (874, 898)]
EXCLUDE = [(519, 522)]
BUF = 5.0
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def ep_of(t, buf=0.0):
    for k, rs in EPIS.items():
        if any(a - buf <= t < b + buf for a, b in rs):
            return k
    return 0


def in_any(t, rs):
    return any(a <= t < b for a, b in rs)


def windows(stem, kind="dense", src="d2"):
    z = np.load(FR / f"{kind}_{stem}.npz")
    t = z["t"].astype(float)
    if kind == "label" and src == "d1":
        t = np.arange(len(t)) + 0.5
    k = np.floor(t).astype(int)
    rows, X = [], []
    for w in np.unique(k):
        ii = np.nonzero(k == w)[0]
        rows.append(dict(k=int(w), grey_max=float(z["grey_mean"][ii].max()), frame=int(z["frame"][ii[0]])))
        X.append(z["emb"][ii].astype(np.float32).mean(0))
    return pd.DataFrame(rows), np.stack(X)


def scan_table(src, stem):
    p = SCANW / DEPN[src] / "sensitive" / "windows" / f"{stem}.mov.csv"
    return pd.read_csv(p) if p.exists() else None


def label_00017():
    stem = D2["00017"]
    w, X = windows(stem)
    sp = pd.read_csv(D.LABELS / stem / "seafloor_spans.csv", keep_default_na=False)
    bl = pd.read_csv(D.LABELS / stem / "blocked_spans.csv")
    t = w.k + 0.5
    span = [sp[(sp.start_s <= x) & (x < sp.end_s)].seafloor.iloc[0] if ((sp.start_s <= x) & (x < sp.end_s)).any() else "none" for x in t]
    air = [((bl.kind == "air") & (bl.start_s <= x) & (x < bl.end_s)).any() for x in t]
    oldsand = [((bl.kind == "sand") & (bl.start_s <= x) & (x < bl.end_s)).any() for x in t]
    w["span"] = span; w["air"] = air; w["old_sand_label"] = oldsand
    w["ep"] = [ep_of(x) for x in t]
    w["near"] = [ep_of(x, BUF) for x in t]
    w["ai_only"] = [in_any(x, AI_ONLY) for x in t]
    w["excl"] = [in_any(x, EXCLUDE) for x in t]
    w["dark"] = w.grey_max < ps.WINDOW_RULES["dark_grey"]
    w["y"] = np.where(w.ep > 0, 1, np.where(w.span.isin(["none", "visible"]), 0, -1))
    use = (~w.air) & (~w.dark) & (~w.ai_only) & (~w.excl)
    w.loc[~use, "y"] = -1
    return w, X


def fit(X, y, C=0.001):
    s = StandardScaler().fit(X)
    return s, LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(s.transform(X), y)


def prob(model, X):
    s, m = model
    return m.predict_proba(s.transform(X))[:, 1]


def best_thr(p, y):
    best = (-1, 0.5)
    for t in GRID:
        c = p >= t
        br = 0.5 * (c[y == 1].mean() + (~c[y == 0]).mean())
        if br > best[0]:
            best = (br, float(t))
    return best[1]


def fold_models(w17, X17, extra: list[tuple[np.ndarray, np.ndarray]]):
    """{fold: (model, threshold)} for folds 1-4 and 0 (all episodes)."""
    out = {}
    for f in (0, 1, 2, 3, 4):
        tr = (w17.y >= 0) & (w17.near != f) if f else (w17.y >= 0)
        tr = tr.to_numpy()
        # inner leave-one-episode-out on the training episodes for the threshold
        inner_p, inner_y = [], []
        for g in [e for e in EPIS if e != f]:
            itr = tr & (w17.near != g).to_numpy()
            ite = tr & (w17.ep == g).to_numpy()
            neg_te = tr & (w17.y == 0).to_numpy() & (np.arange(len(w17)) % 4 == g % 4)   # a quarter of the negatives
            itr = itr & ~neg_te
            Xi = np.concatenate([X17[itr]] + [e[0] for e in extra]); yi = np.concatenate([w17.y[itr]] + [e[1] for e in extra])
            mi = fit(Xi, yi)
            sel = ite | neg_te
            inner_p.append(prob(mi, X17[sel])); inner_y.append(w17.y[sel].to_numpy())
        thr = best_thr(np.concatenate(inner_p), np.concatenate(inner_y))
        Xa = np.concatenate([X17[tr]] + [e[0] for e in extra]); ya = np.concatenate([w17.y[tr]] + [e[1] for e in extra])
        out[f] = (fit(Xa, ya), thr)
    return out


def others():
    """Every other window set: (name, src, df, X, groups dict name -> mask)."""
    sets = []
    for piece in ("00010", "00011", "00012", "00023", "00024"):
        stem = D2[piece]
        w, X = windows(stem)
        st = scan_table("d2", stem)
        bl = pd.read_csv(D.LABELS / stem / "blocked_spans.csv")
        t = w.k + 0.5
        air = np.array([((bl.kind == "air") & (bl.start_s <= x) & (x < bl.end_s)).any() for x in t])
        w = w.assign(air=air)
        sets.append((f"2025-01-02 {piece}", "d2", piece, stem, w, X, st))
    stem6 = "CC-11-96-20250203-111250-000-00006"
    w, X = windows(stem6)
    st = scan_table("d1", stem6)
    uw = set(st[(st.water == "underwater") & (st.tag_state == "whale")].k)
    w = w.assign(air=~w.k.isin(uw))
    sets.append(("2025-02-03 00006 dense", "d1", "00006", stem6, w, X, st))
    for src, name in (("d1", "2025-02-03 labelled frames"), ("dev", "dev clip labelled frames")):
        ws, Xs = [], []
        for rec in D.recordings((src,)):
            z = np.load(FR / f"label_{rec.stem}.npz")
            fc = rec.frames_csv().set_index("frame")
            keep = np.array([fc.loc[int(f), "state"] == "underwater" for f in z["frame"]])
            ww, XX = windows(rec.stem, "label", src)
            if src == "d1":
                ww = ww[keep]; XX = XX[keep]
            else:
                kk = {int(np.floor(t)) for t, k in zip(z["t"], keep) if k}
                m = ww.k.isin(kk).to_numpy(); ww = ww[m]; XX = XX[m]
            ww = ww.assign(stem=rec.stem, air=False)
            ws.append(ww); Xs.append(XX)
        sets.append((name, src, "", None, pd.concat(ws, ignore_index=True), np.concatenate(Xs), None))
    return sets


def hard_negative_masks(name, piece, w, st):
    """Hard negative groups inside a window set."""
    out = {}
    base = (~w.air) & (w.grey_max >= ps.WINDOW_RULES["dark_grey"])
    out["all underwater"] = base
    if st is not None:
        d = w.k.map(st.set_index("k").depth_m)
        out["deep (> 30 m)"] = base & (d > 30)
        lo = (st.water != "underwater").to_numpy()
        ends = [float(st.t1.iloc[i]) for i in range(len(st) - 1) if lo[i] and not lo[i + 1]]
        after = np.array([any(0 <= k - e < 10 for e in ends) for k in w.k])
        out["within 10 s after a lens-out (bubbles)"] = base & after
    if piece == "00023":
        out["back with haze, 0-306 s (old 'visible')"] = base & (w.k < 306) & ~((w.k >= 286) & (w.k < 301))
    return out


def sensors():
    """Per-window sensor measures in 00017 against the recording's own labels (scan sensor time per window)."""
    from uwid import tagdata as T
    s = T.load(DEPN["d2"])
    st = scan_table("d2", D2["00017"])
    w17, _ = label_00017()
    t = s.t.to_numpy()
    ax, ay, az = s.ax.to_numpy(), s.ay.to_numpy(), s.az.to_numpy()
    dep = s.depth.to_numpy(); light = s.light1.to_numpy()
    rows = []
    for r in st.itertuples():
        if not np.isfinite(r.sensor_t0):
            continue
        i0, i1 = np.searchsorted(t, [r.sensor_t0, r.sensor_t0 + 1.0])
        j0, j1 = np.searchsorted(t, [r.sensor_t0 - 10, r.sensor_t0 + 11])
        if i1 <= i0:
            continue
        g = np.array([ax[i0:i1].mean(), ay[i0:i1].mean(), az[i0:i1].mean()])
        dyn = np.sqrt((ax[i0:i1] - g[0]) ** 2 + (ay[i0:i1] - g[1]) ** 2 + (az[i0:i1] - g[2]) ** 2).mean()
        rows.append(dict(k=int(r.k), depth=float(np.nanmean(dep[i0:i1])), depth_sd21=float(np.nanstd(dep[j0:j1])),
                         roll=float(np.degrees(np.arctan2(g[1], g[2]))), pitch=float(np.degrees(np.arctan2(-g[0], np.hypot(g[1], g[2])))),
                         odba=float(dyn), light=float(np.nanmean(light[i0:i1]))))
    sw = pd.DataFrame(rows).merge(w17[["k", "span", "ep", "dark", "air"]], on="k")
    mx = sw.depth.max()
    sw["depth_below_max"] = mx - sw.depth
    sw["absroll_from_median"] = (sw.roll - sw.roll.median()).abs()

    def grp(r):
        if r.air:
            return "breath (air)"
        if r.ep > 0:
            return "sediment (user-confirmed episodes)"
        if r.span == "visible":
            return "seafloor visible (user-confirmed)"
        if r.span == "sediment":
            return "sediment (AI only)"
        if r.span == "unsure":
            return "black / murk (unsure)" if r.dark else "unsure, not black"
        if r.k >= 842:
            return "resting, back with haze (842-1390)"
        return "none, swimming (0-292, 644-673)"
    sw["group"] = sw.apply(grp, axis=1)
    tab = sw.groupby("group").agg(windows=("k", "size"), depth_m=("depth", "median"),
                                  below_max_m=("depth_below_max", "median"), depth_sd_21s=("depth_sd21", "median"),
                                  roll_deg=("roll", "median"), abs_roll_dev=("absroll_from_median", "median"),
                                  pitch_deg=("pitch", "median"), odba=("odba", "median"), light=("light", "median")).round(2)
    return tab, sw, float(mx)


def main():
    res = {}
    w17, X17 = label_00017()
    sets = others()
    old = ps.SandModel.load(ps.SAND_CLF)
    d2extra = {name: (X[(~w.air).to_numpy()], np.zeros(int((~w.air).sum()), int)) for name, src, piece, stem, w, X, st in sets if src == "d2"}
    res["counts_00017"] = {"sediment windows (confirmed episodes)": int((w17.y == 1).sum()),
                           "not-sediment windows": int((w17.y == 0).sum()),
                           "per episode": {f"S{k}": int(((w17.ep == k) & (w17.y == 1)).sum()) for k in EPIS}}
    for variant in ("A", "B"):
        extra = list(d2extra.values()) if variant == "B" else []
        M = fold_models(w17, X17, extra)
        r = {"thresholds": {f"fold{f}" if f else "all": M[f][1] for f in M}}
        for f in (1, 2, 3, 4):
            p = prob(M[f][0], X17); thr = M[f][1]
            e = (w17.ep == f) & (w17.y == 1)
            r[f"S{f} held out: found"] = [int((p[e] >= thr).sum()), int(e.sum())]
            fs = (w17.near != f) & (w17.ep == 0)
            for gname, gm in {"none + visible (not sediment)": fs & w17.span.isin(["none", "visible"]) & ~w17.dark & ~w17.air & ~w17.excl,
                              "  of which seafloor visible": fs & (w17.span == "visible") & ~w17.dark & ~w17.excl,
                              "AI-only sediment 736-765, 874-898 (not fitted)": fs & w17.ai_only & ~w17.dark,
                              "unsure, not black": fs & (w17.span == "unsure") & ~w17.dark & ~w17.air}.items():
                r.setdefault(f"00017 {gname}", {})[f"fold{f}"] = [int((p[gm] >= thr).sum()), int(gm.sum())]
        # other recordings
        for name, src, piece, stem, w, X, st in sets:
            if variant == "B" and src == "d2":
                ex = [v for k, v in d2extra.items() if k != name]
                Mo = {0: (fit(np.concatenate([X17[(w17.y >= 0).to_numpy()]] + [e[0] for e in ex]),
                              np.concatenate([w17.y[w17.y >= 0]] + [e[1] for e in ex])), M[0][1])}
            else:
                Mo = M
            p = prob(Mo[0][0], X); thr = Mo[0][1]
            po = old.proba(X)
            for gname, gm in hard_negative_masks(name, piece, w, st).items():
                gm = gm.to_numpy() if hasattr(gm, "to_numpy") else gm
                r[f"{name} | {gname}"] = {"new": [int((p[gm] >= thr).sum()), int(gm.sum())],
                                          "old sand model": [int((po[gm] >= 0.5).sum()), int(gm.sum())]}
            if variant == "A":
                w2 = w.assign(p_new=p, p_old=po)
                w2.to_parquet(D.SF_CACHE / f"sand_redo_{name.replace(' ', '_')}.parquet", index=False)
        res[f"variant {variant}"] = r
        if variant == "A":
            pa = prob(M[0][0], X17)
            w17.assign(p_new=pa, p_old=old.proba(X17)).to_parquet(D.SF_CACHE / "sand_redo_00017.parquet", index=False)
        # delivered weights (not switched): fold 0 of each variant
        s, m = M[0][0]
        ps.SandModel(s.mean_, s.scale_, m.coef_[0], float(m.intercept_[0]),
                     {"wp12b": True, "variant": variant, "threshold": M[0][1],
                      "trained_on": "00017 confirmed sediment vs none / visible" + (" + other 2025-01-02 recordings as negatives" if variant == "B" else "")}
                     ).save(D.SF_CACHE / f"sand_v2_{variant}.npz")
    # old sand label against the new truth in 00017
    m = w17.y >= 0
    res["old sand label (blocked_spans) vs new truth, 00017"] = {
        "sediment windows labelled sand": [int((w17.old_sand_label & (w17.y == 1)).sum()), int((w17.y == 1).sum())],
        "not-sediment windows labelled sand": [int((w17.old_sand_label & (w17.y == 0)).sum()), int((w17.y == 0).sum())]}
    tab, sw, mx = sensors()
    res["sensors_max_depth_m"] = mx
    res["sensors"] = tab.reset_index().to_dict("records")
    sw.to_parquet(D.SF_CACHE / "sand_redo_sensors_00017.parquet", index=False)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sand_redo.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(tab.to_string())


if __name__ == "__main__":
    main()
