"""WP8b step 1: how long a pure discontinuity does the tracker survive? Identity truth = the uncut run.

For each base stretch (cases.BASES, uncut OWLv2 + SAM 3 tracker run), each gap g in GAPS_S and each position, a cut
run starts PRE_S before the gap (fresh tracker, same chain), does not show the g seconds of the gap to the tracker
(it meets a jump), and ends POST_S after the gap.

Outcome per base whale A that the uncut run shows both in the last 0.5 s before the gap and 0.5-2.5 s after it:
- before the gap, A's number in the cut run = the cut track with the highest mean IoU with A over the last 0.5 s
  (>= 0.3; else A is "not tracked before the gap" and left out);
- after the gap, per frame the cut track with the highest IoU with A (>= 0.3, else none); the most common answer
  over the window decides: none -> LOST; A's own number -> KEPT; the number another whale B had before the gap ->
  SWAPPED (took another whale's number); any other track (started after the gap, or untracked before) -> NEW number.
Size: A's longest side in the last 0.5 s before the gap (median), large >= 0.3 of the frame width.
Whales in view: base whales visible on the last frame before the gap.

    .venv/bin/python experiments/07_tracks_demo/gap/synthetic.py plan     # list the cut runs
    .venv/bin/python experiments/07_tracks_demo/gap/synthetic.py eval     # -> results/gap_outcomes.csv, gap_curve.csv, demo_gap_curve.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import BASES, GAPS_S, POSITIONS, POST_S, PRE_S, tracks  # noqa: E402

from uwid import video as uv  # noqa: E402

LARGE = 0.3
SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")


def base_dir(b: str) -> Path:
    v, t0, t1 = BASES[b]
    return tracks.tracks_dir(v, t0, t1, "owl_sam3")


def plan() -> list[dict]:
    """The cut runs: base, g, pos, gap frames [ga, gb), run frames [s, e), output name."""
    out = []
    for b, (v, t0, t1) in BASES.items():
        tab = uv.frame_table(v)
        fps = 1 / tab.frame_step
        f0 = int(np.ceil(tab.index_at(t0) / 15) * 15)
        f1 = tab.index_at(t1)
        seen = set()
        for g in GAPS_S:
            ng = int(round(g * fps))
            for pos in POSITIONS:
                ga = int(round(f0 + pos * (f1 - f0) - ng / 2))
                lo = f0 + int(np.ceil(PRE_S * fps / 15)) * 15
                hi = f1 - int(np.ceil(POST_S * fps)) - ng
                if hi < lo:
                    continue
                ga = int(np.clip(ga, lo, hi))
                if (g, ga) in seen:
                    continue
                seen.add((g, ga))
                s = max(f0, int(np.floor((ga - PRE_S * fps) / 15)) * 15)
                e = ga + ng + int(np.ceil(POST_S * fps))
                out.append(dict(base=b, video=v, g=g, pos=round(pos, 2), ga=ga, gb=ga + ng, start=s, end=e,
                                name=f"gap_{b}_g{g:g}_f{ga}"))
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def evaluate_run(p: dict) -> list[dict]:
    base = tracks.load_tracks(base_dir(p["base"]))
    cut_dir = tracks.tracks_dir(p["video"], 0, 0, "owl_sam3").with_name(p["name"])
    cut = tracks.load_tracks(cut_dir)
    bf0, cf0 = base.meta["frames"][0], cut.meta["frames"][0]
    fps = base.meta["fps"]
    pre = list(range(p["ga"] - int(round(0.5 * fps)), p["ga"]))
    post = list(range(p["gb"] + int(round(0.5 * fps)), p["gb"] + int(round(2.5 * fps))))
    post = [f for f in post if f - bf0 < base.n and f - cf0 < cut.n]
    bw = [k for k, r in base.roles.items() if r in ("whale", "small")]
    cw = [k for k, r in cut.roles.items() if r in ("whale", "small", "duplicate", "on_back")]
    vis = lambda tr, k, f, f0: 0 <= f - f0 < tr.n and bool(tr.packed[k][f - f0].any())  # noqa: E731
    in_view = sum(vis(base, k, p["ga"] - 1, bf0) for k in bw)
    num = {}
    sizes = {}
    for a in bw:
        fa = [f for f in pre if vis(base, a, f, bf0)]
        if len(fa) < len(pre) / 2:
            continue
        sc = {c: np.mean([_iou(base.mask(a, f - bf0), cut.mask(c, f - cf0)) if vis(cut, c, f, cf0) else 0.0 for f in fa])
              for c in cw}
        best = max(sc, key=sc.get) if sc else None
        num[a] = best if best is not None and sc[best] >= 0.3 else None
        ms = []
        for f in fa:
            m = base.mask(a, f - bf0)
            ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
            ms.append(max(xs[-1] - xs[0] + 1, ys[-1] - ys[0] + 1) / m.shape[1])
        sizes[a] = float(np.median(ms))
    rows = []
    for a in num:
        fa = [f for f in post if vis(base, a, f, bf0)]
        if len(fa) < len(post) / 2:
            continue                       # A not in view after the gap in the uncut run
        if num[a] is None:
            outcome = "not tracked before"
        else:
            picks = []
            for f in fa:
                m = base.mask(a, f - bf0)
                sc = {c: _iou(m, cut.mask(c, f - cf0)) for c in cw if vis(cut, c, f, cf0)}
                best = max(sc, key=sc.get) if sc else None
                picks.append(best if best is not None and sc[best] >= 0.3 else None)
            vals, cnt = np.unique(np.array([-1 if x is None else x for x in picks]), return_counts=True)
            mode = int(vals[cnt.argmax()])
            others = {num[b] for b in num if b != a and num[b] is not None}
            if mode == -1:
                outcome = "lost"
            elif mode == num[a]:
                outcome = "kept"
            elif mode in others:
                outcome = "swapped"
            else:
                outcome = "new number"
        rows.append(dict(base=p["base"], g=p["g"], ga=p["ga"], name=p["name"], whale=a, size=round(sizes[a], 3),
                         large=sizes[a] >= LARGE, in_view=in_view, outcome=outcome))
    return rows


def cmd_eval():
    rows, missing = [], 0
    for p in plan():
        d = tracks.tracks_dir(p["video"], 0, 0, "owl_sam3").with_name(p["name"])
        if not (d / "meta.json").exists():
            missing += 1
            continue
        rows += evaluate_run(p)
    d = pd.DataFrame(rows)
    res = HERE / "results"
    d.to_csv(res / "gap_outcomes.csv", index=False)
    print(f"{len(d)} whale cases, {missing} cut runs missing")
    ev = d[d.outcome != "not tracked before"]
    d["view"] = np.where(d.in_view <= 1, "1", np.where(d.in_view <= 3, "2-3", "4+"))
    ev = d[d.outcome != "not tracked before"].copy()
    tabs = []
    for split, col in [("all", None), ("size", "large"), ("whales in view", "view")]:
        groups = [("all", ev)] if col is None else [(str(k), g) for k, g in ev.groupby(col)]
        for k, g in groups:
            for gap, gg in g.groupby("g"):
                c = gg.outcome.value_counts()
                tabs.append(dict(split=split, group=k, g=gap, n=len(gg), kept=int(c.get("kept", 0)),
                                 new=int(c.get("new number", 0)), lost=int(c.get("lost", 0)),
                                 swapped=int(c.get("swapped", 0))))
    t = pd.DataFrame(tabs)
    for c in ("kept", "new", "lost", "swapped"):
        t[c + "_share"] = (t[c] / t.n).round(2)
    t.to_csv(res / "gap_curve.csv", index=False)
    print(t.to_string())
    print("not tracked before the gap (left out):", int((d.outcome == "not tracked before").sum()))
    plot(t)


def plot(t: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    colors = {"kept": "#2a9d4b", "new": "#e0a400", "lost": "#9a9a9a", "swapped": "#d62728"}
    for ax, (split, title) in zip(axes, [("all", "all whales"), ("size", "by size (large >= 0.3 of width)"),
                                        ("whales in view", "by whales in view")]):
        sub = t[t.split == split]
        styles = ["-", "--", ":"]
        for j, (grp, g) in enumerate(sub.groupby("group")):
            g = g.sort_values("g")
            for c in ("kept", "new", "lost", "swapped"):
                lab = f"{c}" if split == "all" else f"{c}, {('large' if grp == 'True' else 'small') if split == 'size' else grp}"
                ax.plot(g.g, g[c + "_share"], styles[j % 3], marker="o", color=colors[c], label=lab, lw=2 if c == "kept" else 1.4)
        ns = sub.groupby("group").apply(lambda g: "/".join(str(x) for x in g.sort_values("g").n))
        ax.set_title(title + "\nn per gap: " + "; ".join(f"{('large' if k == 'True' else 'small') if split == 'size' else k}: {v}" for k, v in ns.items()), fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(GAPS_S))
        ax.set_xticklabels([f"{g:g}" for g in GAPS_S])
        ax.set_xlabel("gap not shown to the tracker (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2 if split != "all" else 1)
    axes[0].set_ylabel("share of whales after the gap")
    fig.suptitle("Gap tolerance of OWLv2 boxes + SAM 3 tracker (identity truth = the uncut run; 5 stretches, 2 gap positions)", fontsize=11)
    fig.tight_layout()
    fig.savefig(SHEETS / "demo_gap_curve.png", dpi=120)
    print("wrote", SHEETS / "demo_gap_curve.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "eval"])
    a = ap.parse_args()
    if a.cmd == "plan":
        P = plan()
        print(pd.DataFrame(P).drop(columns="video").to_string())
        print(len(P), "cut runs,", round(sum((p["end"] - p["start"] - (p["gb"] - p["ga"])) for p in P) / 30, 1), "s shown")
    else:
        cmd_eval()
