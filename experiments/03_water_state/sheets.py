"""Review sheets for WP4 failures (for people), from run.py's predictions.

Writes to /home/sat3737/NOAA_whales/review_sheets/underwater/:
- waterstate_false_air.jpg      windows that are under water (truth) but called air or crossing by some method
- waterstate_missed_air.jpg     windows in air (truth) called underwater or crossing
- waterstate_crossings_<set>.jpg frame strips around exits / entries with each method's per-frame call
- waterstate_drop.jpg           the drone flight end and the drop in 00004 with each method's call
- waterstate_scores_<video>.png  frame scores of each method against the truth over time (zooms)

Zenodo videos are untrusted downloads: run this script with ``python -I``.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402
from uwid import sheets as us  # noqa: E402
from uwid import video as uv  # noqa: E402
from uwid import waterstate as ws  # noqa: E402

OUT = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
SHOW = ["color", "clip0", "clipLR", "dino"]
SHORT = {"sat": "S", "color": "C", "clip0": "Z", "clipLR": "L", "dino": "D"}
COL = {"color": "#2a78d6", "clip0": "#eb6834", "clipLR": "#1baf7a", "dino": "#6250d6", "sat": "#e87ba4"}
KIND = "chain"


def vpath(name):
    if name in run.FIT + run.CHECK:
        return run.VDIR / f"{name}.mov"
    if name == run.DEV:
        return Path("/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v")
    return run.ZDIR / f"{name}.mov"


def short(name):
    if name.startswith("CC-11-96"):
        return name[-5:]
    if name.startswith("CATS"):
        return "Z" + name[-5:]
    return "dev"


def preds(d, kind=KIND):
    out = {}
    for m in SHOW + ["sat"]:
        f = run.PRED / f"{m}__{d['name']}.npz"
        if not f.exists():
            continue
        p = np.load(f)
        air = p[f"air_{kind}"]
        wp, ev = ws.windows_from_binary(p["t"], air, 1 / (1 + np.exp(-p["score"])), d["duration"])
        out[m] = {"air": air, "score": p["score"], "w": wp, "ev": ev}
    return out


def window_errors(data):
    rows = []
    for d in data:
        P = preds(d)
        wt = d["wtruth"]
        for m, p in P.items():
            n = min(len(wt), len(p["w"]))
            for k in range(n):
                tr, pr = wt.truth.iat[k], p["w"].state.iat[k]
                if tr in ("underwater", "air", "crossing") and tr != pr:
                    rows.append((d["name"], k, tr, m, pr))
    return pd.DataFrame(rows, columns=["video", "k", "truth", "method", "pred"])


def tile_sheet(err: pd.DataFrame, truth: str, out: Path, title: str, n=40, seed=0):
    e = err[(err.truth == truth) & err.method.isin(SHOW)]
    if not len(e):
        return 0
    g = e.groupby(["video", "k"]).apply(lambda x: " ".join(f"{SHORT[m]}:{p[:3]}" for m, p in zip(x.method, x.pred)),
                                        include_groups=False)
    # up to n_per windows per method (windows several methods got wrong first), so one method's many errors do
    # not fill the sheet
    n_per = max(4, n // len(SHOW))
    keys = []
    for m in SHOW:
        km = list(e[e.method == m].set_index(["video", "k"]).index.unique())
        random.Random(seed).shuffle(km)
        km.sort(key=lambda kk: -len(g[kk].split()))
        keys += [kk for kk in km if kk not in keys][:n_per]
    keys = sorted(set(keys))[:n]
    imgs, labels, caps = [], [], []
    for vid, grp in pd.Series(keys).groupby([kk[0] for kk in keys]):
        with uv.VideoReader(vpath(vid), size=(384, 216)) as r:
            for (v, k) in grp:
                f = r.frame_at(k + 0.5)
                if f is None:
                    continue
                imgs.append(f.image)
                labels.append(f"{short(v)} {k}s")
                caps.append(g[(v, k)])
    us.make_sheet(imgs, labels, caps, cols=8, tile_width=240, title=title, out=out)
    return len(imgs)


def crossing_strip(data, out: Path, n_events=8, seed=1):
    ev = []
    for d in data:
        for r in d["events"].itertuples():
            if 1.5 < r.t < d["duration"] - 1.5:
                ev.append((d, r.t, r.kind))
    random.Random(seed).shuffle(ev)
    ev = sorted(ev[:n_events], key=lambda x: (x[0]["name"], x[1]))
    imgs, labels, caps = [], [], []
    for d, te, kind in ev:
        P = preds(d)
        ts = np.round(np.arange(te - 1.2, te + 1.21, 0.4), 2)
        with uv.VideoReader(vpath(d["name"]), size=(320, 180)) as r:
            frs = r.frames_at(list(ts))
        for f, dt in zip(frs, ts - te):
            if f is None:
                continue
            i = int(np.argmin(np.abs(d["t"] - f.t)))
            imgs.append(f.image)
            labels.append(f"{short(d['name'])} {kind} {dt:+.1f}")
            caps.append(" ".join(f"{SHORT[m]}:{'A' if P[m]['air'][i] else 'U'}" for m in SHOW if m in P))
    us.make_sheet(imgs, labels, caps, cols=7, tile_width=230,
                  title="frames around exits / entries (truth time = 0); calls after the window step (A air, U under)",
                  out=out)


def drop_sheet(d, out: Path, t0=86.0, t1=97.0):
    P = preds(d)
    ts = list(np.arange(t0, t1, 0.5))
    with uv.VideoReader(vpath(d["name"]), size=(320, 180)) as r:
        frs = r.frames_at(ts)
    imgs, labels, caps = [], [], []
    truth = d["ftruth"]
    for f in frs:
        i = int(np.argmin(np.abs(d["t"] - f.t)))
        imgs.append(f.image)
        labels.append(f"{f.t:.1f}s truth {truth[i][:5]}")
        caps.append(" ".join(f"{SHORT[m]}:{'A' if P[m]['air'][i] else 'U'}({P[m]['score'][i]:+.1f})" for m in SHOW
                             if m in P))
    us.make_sheet(imgs, labels, caps, cols=6, tile_width=300,
                  title="00004: end of the drone flight and the drop (impact ~91.6 s); method calls (score = logit air)",
                  out=out)


def score_plot(d, out: Path, spans):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = preds(d)
    fig, axes = plt.subplots(len(spans), 1, figsize=(13, 2.6 * len(spans)), squeeze=False)
    for ax, (a, b) in zip(axes[:, 0], spans):
        for r in d["tl"].segments.itertuples():
            if r.state == "air" and r.t1 > a and r.t0 < b:
                ax.axvspan(max(r.t0, a), min(r.t1, b), color="#d9d8d3", lw=0)
        for m in SHOW:
            if m not in P:
                continue
            sel = (d["t"] >= a) & (d["t"] <= b)
            ax.plot(d["t"][sel], P[m]["score"][sel], lw=1.2, color=COL[m], label=m)
        ax.axhline(0, color="#52514e", lw=0.6)
        for r in d["events"].itertuples():
            if a <= r.t <= b:
                ax.axvline(r.t, color="#0b0b0b", lw=0.6, ls=":")
        ax.set_xlim(a, b)
        ax.set_ylim(-10.5, 10.5)
        ax.set_ylabel("logit air")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0, 0].legend(ncol=4, loc="upper right", frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel(f"video time (s), {d['name']}; grey = truth air, dotted = truth events")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = run.CHECK + [run.DEV] + [z for z in run.ZEN if (run.ZTRUTH / f"{z}_meta.json").exists()]
    data = [run.load(n) for n in names if (run.FEATS / f"{n}.npz").exists()]
    err = window_errors(data)
    err.to_csv(run.C / "window_errors.csv", index=False)
    print(err.groupby(["truth", "method", "pred"]).size().to_string())
    for tr, fn, title in (("underwater", "waterstate_false_air.jpg",
                           "truth UNDERWATER windows called otherwise (C colour, Z CLIP zero-shot, L CLIP probe, D DINOv3)"),
                          ("air", "waterstate_missed_air.jpg", "truth AIR windows called otherwise"),
                          ("crossing", "waterstate_crossing_errors.jpg", "truth CROSSING windows called otherwise")):
        print(fn, tile_sheet(err, tr, OUT / fn, title))
    crossing_strip([d for d in data if d["name"] in run.CHECK], OUT / "waterstate_crossings_check.jpg")
    crossing_strip([d for d in data if d["name"] == run.DEV], OUT / "waterstate_crossings_dev.jpg", n_events=4)
    zd = [d for d in data if d["name"] in run.ZEN]
    if zd:
        crossing_strip(zd, OUT / "waterstate_crossings_zenodo.jpg", n_events=8)
    fit4 = run.load(run.FIT[0])
    drop_sheet(fit4, OUT / "waterstate_drop.jpg")
    score_plot(fit4, OUT / "waterstate_scores_00004.png", [(0, 120), (430, 490), (600, 640)])
    dev = [d for d in data if d["name"] == run.DEV][0]
    score_plot(dev, OUT / "waterstate_scores_dev.png", [(0, 48), (47, 96)])
    for d in zd:
        score_plot(d, OUT / f"waterstate_scores_{short(d['name'])}.png", [(0, 120), (280, 400), (480, 592)])


if __name__ == "__main__":
    main()
