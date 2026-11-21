"""WP4 main run: frame scores per method, window step, scores on the fit / check / cross-deployment videos.

Protocol (PLAN.md): fit on pieces 00004-00006 of 2025-02-03 (sensor truth); check on 00007-00008 (same deployment,
unseen pieces); test across deployments on the 2025-12-20 clip (AI by-eye labels) and on the public Zenodo 4902325
deployment (depth truth, built by zenodo_truth.py). Nothing is fitted on the check or test videos.

Inside the fit pieces: classifier C chosen by leave-one-piece-out (LOPO); the window step (bias, pooling length,
hysteresis width, chain penalty) tuned on the LOPO out-of-fold scores; final classifiers refitted on all three.

Inputs: feature files from extract.py in underwater_cache/waterstate/feats/. Outputs: results.json and per-video
window predictions in underwater_cache/waterstate/pred/, the tables in tables_auto.md (pasted into tables.md).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import eval_state as es  # noqa: E402
from uwid import waterstate as ws  # noqa: E402

HERE = Path(__file__).parent
REPO = HERE.parents[1]
C = Path("/home/sat3737/NOAA_whales/underwater_cache/waterstate")
RUN_TAG = os.environ.get("WS_RUN_TAG", "")
FEATS = C / "feats"
PRED = C / "pred"
DEP = "2025-2-3 Mini-mother 11-96"
VDIR = Path("/home/sat3737/NOAA_whales/underwater_video") / DEP / "Video"
FIT = ["CC-11-96-20250203-101236-000-00004", "CC-11-96-20250203-104242-000-00005",
       "CC-11-96-20250203-111250-000-00006"]
CHECK = ["CC-11-96-20250203-114256-000-00007", "CC-11-96-20250203-121302-000-00008",
         "CC-11-96-20250203-124310-000-00009"]
DEV = "2025-12-20 CG"
ZEN = ["CATS_58-20200312-181746-359-00011", "CATS_58-20200312-183732-336-00013"]
ZDIR = Path("/home/sat3737/NOAA_whales/underwater_cache/public/zenodo_4902325")
ZTRUTH = C / "zenodo_truth"

METHODS = ["sat", "color", "clip0", "clipLR", "dino"]
STEPS = ["raw", "pool", "hyst", "chain"]
CLIP_LOGIT = 10.0
# Window-step choice among near-equal settings on the fit pieces. v1 (first run) took the arg-max; v2 takes the least
# smoothing within TIE_TOL of the best. v2 was decided after v1's dev-clip result (the chain with penalty 30 erased
# a 1.8 s air spell there), so v2's dev-clip numbers are not a clean test; the Zenodo deployment was first scored
# with v2.
TIE_RULE = os.environ.get("WS_TIE_RULE", "least_smoothing")
TIE_TOL = 0.02


# ------------------------------------------------------------------ data

def video_duration(path: Path) -> float:
    from uwid import video as uv
    return float(uv.frame_table(path).duration)


def load(name: str) -> dict:
    f = np.load(FEATS / f"{name}.npz")
    d = {k: f[k] for k in f.files}
    d["name"] = name
    if name in FIT + CHECK:
        path = VDIR / f"{name}.mov"
        dur = video_duration(path)
        tl = es.sensor_timeline(DEP, path.name, duration=dur)
        d.update(duration=dur, tl=tl, wtruth=es.windows_from_timeline(tl), ftruth=es.frame_truth(tl, d["t"]),
                 events=tl.events, truth_src="sensor")
    elif name == DEV:
        lab = REPO / "experiments/02_reference_labels/labels/2025-12-20_CG"
        fr, ev = es.load_label_dir(lab)
        dur = 2857 / 29.974222168934716
        evdf = es.label_events(ev)
        tl = es.timeline_from_events(evdf, dur, ev["clip_start_state"])
        d.update(duration=dur, tl=tl, wtruth=es.label_windows(fr, ev, dur), ftruth=es.frame_truth(tl, d["t"]),
                 events=evdf, truth_src="AI by-eye")
    else:  # zenodo: segments / events written by zenodo_truth.py
        seg = pd.read_csv(ZTRUTH / f"{name}_segments.csv")
        evs = pd.read_csv(ZTRUTH / f"{name}_events.csv")
        meta = json.loads((ZTRUTH / f"{name}_meta.json").read_text())
        tl = es.Timeline(seg, evs, meta["duration"], "depth")
        d.update(duration=meta["duration"], tl=tl, wtruth=es.windows_from_timeline(tl),
                 ftruth=es.frame_truth(tl, d["t"]), events=evs, truth_src="depth")
    return d


# ------------------------------------------------------------------ frame scores

def clip_zero_shot(E: np.ndarray) -> np.ndarray:
    T = np.load(C / "clip_text.npz")
    sim = E.astype(np.float32) @ T["emb"].T * float(T["logit_scale"])
    a = sim[:, T["is_air"] == 1]
    u = sim[:, T["is_air"] == 0]
    lse = lambda x: x.max(1) + np.log(np.exp(x - x.max(1, keepdims=True)).sum(1))  # noqa: E731
    return lse(a) - lse(u)


def X_of(d: dict, method: str) -> np.ndarray:
    if method == "sat":
        return d["color"][:, [0]]
    if method == "color":
        return d["color"]
    if method == "clipLR":
        return d["clip"].astype(np.float32)
    if method == "dino":
        return d["dino"].astype(np.float32)
    raise ValueError(method)


def fit_lr(Xs, ys, Cval):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=Cval, class_weight="balanced", max_iter=3000))
    m.fit(np.concatenate(Xs), np.concatenate(ys))
    return m


def train_rows(d):
    y = d["ftruth"]
    keep = np.isin(y, ["air", "underwater"])
    return keep, (y[keep] == "air").astype(int)


def score(model, X):
    return np.clip(model.decision_function(X), -CLIP_LOGIT, CLIP_LOGIT)


# ------------------------------------------------------------------ window step grid

def step_grid():
    biases = [-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6]
    g = [{"kind": "raw", "bias": b} for b in biases]
    g += [{"kind": "pool", "k": k, "bias": b} for k in (3, 5, 9) for b in biases]
    g += [{"kind": "hyst", "k": k, "bias": b, "h": h} for k in (3, 5, 9) for b in biases for h in (0.5, 1, 2, 4)]
    g += [{"kind": "chain", "k": k, "bias": b, "penalty": p} for k in (1, 3, 5) for b in biases
          for p in (1, 3, 10, 30, 100)]
    return g


def apply_step(d, s, step):
    air = ws.binarize(s, step)
    p = 1 / (1 + np.exp(-(s - step.get("bias", 0.0))))
    wp, ev = ws.windows_from_binary(d["t"], air, p, d["duration"])
    return air, wp, ev


def evaluate(ds, scores, step, tol=0.5):
    """Pooled window confusion and events over several videos."""
    T, P, TA, PA, Fr_t, Fr_p = [], [], [], [], [], []
    ev_res = []
    dur = 0.0
    for d, s in zip(ds, scores):
        air, wp, ev = apply_step(d, s, step)
        wt = d["wtruth"]
        n = min(len(wt), len(wp))
        T.append(wt.truth.values[:n]); P.append(wp.state.values[:n])
        amb = wt.ambiguous.values[:n]
        TA.append(wt.truth.values[:n][~amb]); PA.append(wp.state.values[:n][~amb])
        ev_res.append(es.match_events(d["events"], ev, tol=tol))
        dur += d["duration"]
        Fr_t.append(d["ftruth"]); Fr_p.append(np.where(air, "air", "underwater"))
    tab, per = es.confusion(np.concatenate(T), np.concatenate(P))
    tabA, perA = es.confusion(np.concatenate(TA), np.concatenate(PA))
    agg = {}
    for k in ("exit", "entry", "all"):
        n_t = sum(e[k]["n_truth"] for e in ev_res)
        f = sum(e[k]["found"] for e in ev_res)
        n_p = sum(e[k]["n_pred"] for e in ev_res)
        agg[k] = {"n_truth": n_t, "found": f, "n_pred": n_p, "false": n_p - f,
                  "recall": f / n_t if n_t else np.nan, "false_per_h": (n_p - f) / (dur / 3600)}
    errs = [p - t for e in ev_res for _, t, p in e["pairs"]]
    agg["all"]["abs_err_median"] = float(np.median(np.abs(errs))) if errs else np.nan
    agg["all"]["err_median"] = float(np.median(errs)) if errs else np.nan
    for k in ("exit", "entry"):
        ek = [p - t for e in ev_res for kk, t, p in e["pairs"] if kk == k]
        agg[k]["err_median"] = float(np.median(ek)) if ek else np.nan
    ft = np.concatenate(Fr_t); fp = np.concatenate(Fr_p)
    ftab, fper = es.confusion(ft, fp, states=("underwater", "air"))
    cross = ft == "crossing"
    return {"tab": tab, "per": per, "tabA": tabA, "perA": perA, "ev": agg, "dur": dur, "ftab": ftab, "fper": fper,
            "cross_frames": int(cross.sum()), "cross_called_air": int((fp[cross] == "air").sum())}


def objective(r):
    rec = r["per"].set_index("state").recall
    prec = r["per"].set_index("state").precision.fillna(0)
    e = r["ev"]["all"]
    f1 = 2 * e["found"] / max(e["n_truth"] + e["n_pred"], 1)
    return float(np.nanmean(rec.values) + np.nanmean(prec.values) + 2 * f1)


# ------------------------------------------------------------------ main

def main():
    t0 = time.time()
    names = FIT + CHECK + [DEV] + [z for z in ZEN if (FEATS / f"{z}.npz").exists() and (ZTRUTH / f"{z}_meta.json").exists()]
    data = {n: load(n) for n in names if (FEATS / f"{n}.npz").exists()}
    print("loaded", list(data), f"{time.time() - t0:.0f}s", flush=True)
    fit = [data[n] for n in FIT]
    tests = {"check (00007-00009)": [data[n] for n in CHECK if n in data],
             "dev clip 2025-12-20 (cross-deployment)": [data[DEV]]}
    zl = [data[z] for z in ZEN if z in data]
    if zl:
        tests["Zenodo 4902325 (cross-deployment)"] = zl
    for k in list(tests):
        if not tests[k]:
            del tests[k]
    results = {"fit_counts": {}, "methods": {}}
    grid = step_grid()
    PRED.mkdir(parents=True, exist_ok=True)
    final_models = {}
    for m in METHODS:
        print("method", m, flush=True)
        # out-of-fold scores on the fit pieces
        if m == "clip0":
            oof = [clip_zero_shot(d["clip"]) for d in fit]
            bestC = None
        else:
            best = None
            for Cv in ([1.0] if m == "sat" else [0.01, 0.1, 1.0]):
                oofC = []
                for i, d in enumerate(fit):
                    tr = [x for j, x in enumerate(fit) if j != i]
                    Xs, ys = [], []
                    for x in tr:
                        keep, y = train_rows(x)
                        Xs.append(X_of(x, m)[keep]); ys.append(y)
                    mod = fit_lr(Xs, ys, Cv)
                    oofC.append(score(mod, X_of(d, m)))
                from sklearn.metrics import average_precision_score
                ap = []
                for d, s in zip(fit, oofC):
                    keep, y = train_rows(d)
                    ap.append(average_precision_score(y, s[keep]))
                print(f"  C={Cv} LOPO AP(air) per piece {np.round(ap, 3)}", flush=True)
                if best is None or np.mean(ap) > best[0]:
                    best = (np.mean(ap), Cv, oofC)
            _, bestC, oof = best
        # tune the window step on the out-of-fold scores, separately per step kind
        tuned = {}
        for kind in STEPS:
            cand = [g for g in grid if g["kind"] == kind]
            scored = [(objective(evaluate(fit, oof, g)), i) for i, g in enumerate(cand)]
            if TIE_RULE == "least_smoothing":
                # v2: among settings within TIE_TOL of the best objective, take the least smoothing
                # (smallest penalty / hysteresis width, then pooling length, then |bias|)
                best_obj = max(scored)[0]
                near = [cand[i] for o, i in scored if o >= best_obj - TIE_TOL]
                near.sort(key=lambda g: (g.get("penalty", 0), g.get("h", 0), g.get("k", 1), abs(g.get("bias", 0))))
                tuned[kind] = near[0]
            else:
                tuned[kind] = cand[max(scored)[1]]
        # final model on all three fit pieces
        if m == "clip0":
            final_fn = lambda d: clip_zero_shot(d["clip"])  # noqa: E731
        else:
            Xs, ys = [], []
            for x in fit:
                keep, y = train_rows(x)
                Xs.append(X_of(x, m)[keep]); ys.append(y)
            mod = fit_lr(Xs, ys, bestC)
            final_models[m] = mod
            final_fn = (lambda mod_: (lambda d: score(mod_, X_of(d, m))))(mod)
        res = {"C": bestC, "steps": tuned, "sets": {}}
        sets = {"fit LOPO (00004-00006)": (fit, oof)}
        for name, lst in tests.items():
            sets[name] = (lst, [final_fn(d) for d in lst])
        for sname, (lst, scs) in sets.items():
            res["sets"][sname] = {}
            for kind in STEPS:
                r = evaluate(lst, scs, tuned[kind])
                r1 = evaluate(lst, scs, tuned[kind], tol=1.0)
                res["sets"][sname][kind] = {
                    "per": r["per"].to_dict("records"), "perA": r["perA"].to_dict("records"),
                    "tab": r["tab"].to_dict(), "ev05": r["ev"], "ev10": r1["ev"], "hours": r["dur"] / 3600,
                    "fper": r["fper"].to_dict("records"), "ftab": r["ftab"].to_dict(),
                    "cross_frames": r["cross_frames"], "cross_called_air": r["cross_called_air"]}
            # zero-shot at its natural threshold (bias 0, per frame) for clip0
            if m == "clip0":
                r = evaluate(lst, scs, {"kind": "raw", "bias": 0.0})
                res["sets"][sname]["raw_bias0"] = {"per": r["per"].to_dict("records"),
                                                   "fper": r["fper"].to_dict("records"), "ev05": r["ev"],
                                                   "hours": r["dur"] / 3600}
            # store per-video predictions for the sheets
            for d, s in zip(lst, scs):
                out = {"t": d["t"], "score": s}
                for kind in STEPS:
                    air, wp, ev = apply_step(d, s, tuned[kind])
                    out[f"air_{kind}"] = air
                np.savez(PRED / f"{m}__{d['name']}.npz", **out)
        results["methods"][m] = res
        print(json.dumps({k: v for k, v in tuned.items()}), flush=True)
    # the fitted linear models + tuned window steps (small), for the shipped predictor (uwid.waterstate)
    params = {}
    for m, mod in final_models.items():
        sc, lr = mod[0], mod[1]
        params[m] = {"feature": {"sat": "color", "color": "color", "clipLR": "clip", "dino": "dino"}[m],
                     "columns": [0] if m == "sat" else None,
                     "mean": [round(float(v), 6) for v in sc.mean_], "scale": [round(float(v), 6) for v in sc.scale_],
                     "coef": [round(float(v), 6) for v in lr.coef_[0]], "intercept": float(lr.intercept_[0]),
                     "clip": CLIP_LOGIT, "C": results["methods"][m]["C"],
                     "step": results["methods"][m]["steps"]["chain"], "steps": results["methods"][m]["steps"]}
    import pickle
    with open(C / f"final_models{RUN_TAG}.pkl", "wb") as fh:
        pickle.dump(final_models, fh)
    (C / f"linear_params{RUN_TAG}.json").write_text(json.dumps(params, indent=1))
    (C / f"results{RUN_TAG}.json").write_text(json.dumps(results, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
