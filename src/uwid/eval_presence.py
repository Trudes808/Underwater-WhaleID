"""Window truth from frame labels, and presence scorers (whale / fish / seafloor / back).

Definitions follow ``PLAN.md`` ("How each question is measured"):

- A window is a fixed 1 s slot of the video: window ``k`` holds the frames with ``k <= frame / fps < k + 1``
  (``win_s`` changes the length). Frame index = position in decode order, as in the WP3 label files.
- A window's truth comes from the labelled frames inside it. Its state is ``underwater`` or ``air`` when all its
  labelled frames agree, otherwise ``crossing``. A whale (fish, seafloor) is present when any labelled frame in it
  shows one clearly (``other_whales > 0``, ``fish != none``, ``seafloor == visible``).
- Whale windows are split three ways: ``whale`` (a clear whale on some labelled frame), ``faint_only`` (no clear
  whale, but a faint shape on some labelled frame) and ``whale_free`` (no clear and no faint whale on any labelled
  frame). False alarms are counted on ``whale_free`` windows; ``faint_only`` windows are neither hit nor false alarm
  and are reported on their own.

Typical use (a detector gives one score per processed frame; higher = more whale-like)::

    labels = load_frame_labels("experiments/02_reference_labels/labels/2025-12-20_CG/frames.csv")
    truth = window_truth(labels, fps=29.9742)
    win = window_scores(frame_scores, fps=29.9742, rule="kth", k=2)        # frame_scores: DataFrame frame, score
    tab = truth.merge(win, on="window", how="left")
    curve = roc(tab.loc[tab.whale | tab.whale_free, "score"], tab.loc[tab.whale | tab.whale_free, "whale"])
    print(operating_point(curve, max_far=0.10))

Only pandas / numpy; no model code here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_frame_labels", "window_of", "window_truth", "window_scores", "frame_truth", "roc", "auc",
           "operating_point", "rate_at", "summarize"]


def load_frame_labels(path: str | Path) -> pd.DataFrame:
    """Read a WP3 ``frames.csv`` and add boolean columns: ``whale`` (other_whales > 0), ``faint`` (faint_whales > 0),
    ``whale_free`` (neither), ``fish_present``, ``seafloor_present`` (``visible``), ``seafloor_unsure``,
    ``back_present`` (back column not ``none``), ``bubbles`` (view == bubbles)."""
    d = pd.read_csv(path)
    for col, default in (("other_whales", 0), ("faint_whales", 0), ("fish", "none"), ("seafloor", "none"),
                         ("back", "none"), ("view", "clear"), ("state", "underwater")):
        if col not in d.columns:
            d[col] = default
    d["other_whales"] = pd.to_numeric(d["other_whales"], errors="coerce").fillna(0).astype(int)
    d["faint_whales"] = pd.to_numeric(d["faint_whales"], errors="coerce").fillna(0).astype(int)
    d["whale"] = d["other_whales"] > 0
    d["faint"] = d["faint_whales"] > 0
    d["whale_free"] = ~d["whale"] & ~d["faint"]
    d["fish_present"] = d["fish"].fillna("none").astype(str).str.lower().ne("none")
    sf = d["seafloor"].fillna("none").astype(str).str.lower()
    d["seafloor_present"] = sf.eq("visible")
    d["seafloor_unsure"] = sf.eq("unsure")
    d["back_present"] = d["back"].fillna("none").astype(str).str.lower().ne("none")
    d["bubbles"] = d["view"].fillna("clear").astype(str).str.lower().eq("bubbles")
    return d


def window_of(frame, fps: float, win_s: float = 1.0):
    """Window index of frame index(es) ``frame``."""
    return np.floor(np.asarray(frame, dtype=np.float64) / (fps * win_s) + 1e-9).astype(np.int64)


def window_truth(labels: pd.DataFrame, fps: float, win_s: float = 1.0) -> pd.DataFrame:
    """One row per window that contains at least one labelled frame.

    Columns: window, t0, t1, n_labelled, frames (list), state, whale, faint_only, whale_free, max_whales,
    fish, seafloor, seafloor_unsure, back, bubbles (any labelled frame with view == bubbles)."""
    d = labels.copy()
    d["window"] = window_of(d["frame"], fps, win_s)
    rows = []
    for w, g in d.groupby("window", sort=True):
        states = set(g["state"].astype(str))
        state = states.pop() if len(states) == 1 else "crossing"
        whale = bool(g["whale"].any())
        faint = bool(g["faint"].any())
        rows.append(dict(window=int(w), t0=w * win_s, t1=(w + 1) * win_s, n_labelled=len(g),
                         frames=list(map(int, g["frame"])), state=state, whale=whale,
                         faint_only=(not whale) and faint, whale_free=(not whale) and (not faint),
                         max_whales=int(g["other_whales"].max()), fish=bool(g["fish_present"].any()),
                         seafloor=bool(g["seafloor_present"].any()), seafloor_unsure=bool(g["seafloor_unsure"].any()),
                         back=bool(g["back_present"].any()), bubbles=bool(g["bubbles"].any())))
    return pd.DataFrame(rows)


def frame_truth(labels: pd.DataFrame) -> pd.DataFrame:
    """The labelled frames as scoring units, with the same column names as :func:`window_truth`."""
    d = labels.drop(columns=[c for c in ("fish", "seafloor", "back") if c in labels.columns]).copy()
    d["faint_only"] = ~d["whale"] & d["faint"]
    d["max_whales"] = d["other_whales"]
    d["fish"], d["seafloor"], d["back"] = d["fish_present"], d["seafloor_present"], d["back_present"]
    return d


def window_scores(frame_scores: pd.DataFrame, fps: float, win_s: float = 1.0, rule: str = "kth", k: int = 2,
                  neighbours: int = 0, score_col: str = "score") -> pd.DataFrame:
    """Turn per-frame scores into one score per window.

    ``frame_scores`` has columns ``frame`` and ``score_col`` (frames with no detection should carry 0, not be
    missing, so that the k-th rule sees them). The frames pooled for window ``w`` are those in windows
    ``w - neighbours .. w + neighbours``. Rules:

    - ``max``: the highest frame score (a one-frame flash counts);
    - ``kth``: the k-th highest frame score, i.e. at least ``k`` frames must reach the threshold (persistence);
    - ``mean``: the mean frame score;
    - ``frac``: like ``kth`` with k = ceil(frac * n frames pooled); pass ``k`` as a float in (0, 1].

    Returns columns window, score, n_frames."""
    f = frame_scores[["frame", score_col]].rename(columns={score_col: "score"}).copy()
    f["window"] = window_of(f["frame"], fps, win_s)
    by_w = {int(w): g["score"].to_numpy() for w, g in f.groupby("window")}
    out = []
    for w in sorted(by_w):
        pool = np.concatenate([by_w[j] for j in range(w - neighbours, w + neighbours + 1) if j in by_w])
        pool = np.sort(pool)[::-1]
        if rule == "max":
            s = pool[0]
        elif rule == "kth":
            s = pool[min(int(k), len(pool)) - 1]
        elif rule == "frac":
            kk = max(1, int(np.ceil(float(k) * len(pool))))
            s = pool[kk - 1]
        elif rule == "mean":
            s = float(pool.mean())
        else:
            raise ValueError(f"unknown rule {rule!r}")
        out.append(dict(window=w, score=float(s), n_frames=len(pool)))
    return pd.DataFrame(out)


def roc(scores, positive) -> pd.DataFrame:
    """Hit rate against false-alarm rate at every distinct threshold (a unit fires when score >= threshold).

    ``scores``: array of scores (NaN = never fires); ``positive``: booleans. Rows run from the strictest threshold
    (nothing fires) to the loosest. Columns threshold, tp, fp, n_pos, n_neg, recall, far."""
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    y = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    thr = np.unique(s[np.isfinite(s)])[::-1]
    rows = [dict(threshold=np.inf, tp=0, fp=0)]
    for t in thr:
        fire = s >= t
        rows.append(dict(threshold=float(t), tp=int((fire & y).sum()), fp=int((fire & ~y).sum())))
    df = pd.DataFrame(rows)
    df["n_pos"], df["n_neg"] = n_pos, n_neg
    df["recall"] = df.tp / max(n_pos, 1)
    df["far"] = df.fp / max(n_neg, 1)
    return df


def auc(curve: pd.DataFrame) -> float:
    """Area under the recall / false-alarm curve (trapezoid; the curve is closed at (1, 1))."""
    x = np.r_[curve["far"].to_numpy(), 1.0]
    y = np.r_[curve["recall"].to_numpy(), 1.0]
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def operating_point(curve: pd.DataFrame, max_far: float | None = None, min_recall: float | None = None) -> dict:
    """The loosest threshold whose false-alarm rate is <= ``max_far`` (best recall there), or the strictest one
    whose recall is >= ``min_recall``. Returns the curve row as a dict."""
    c = curve[np.isfinite(curve.threshold)]
    if max_far is not None:
        ok = c[c.far <= max_far + 1e-12]
        if ok.empty:
            return dict(curve.iloc[0])
        best = ok[ok.recall == ok.recall.max()]
        return dict(best.iloc[-1] if min_recall is None else best.iloc[0])
    if min_recall is not None:
        ok = c[c.recall >= min_recall - 1e-12]
        return dict(ok.iloc[0]) if not ok.empty else dict(c.iloc[-1])
    raise ValueError("give max_far or min_recall")


def rate_at(scores, threshold: float) -> tuple[int, int]:
    """(units that fire, units) at ``threshold``."""
    s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    return int((s >= threshold).sum()), int(len(s))


def summarize(table: pd.DataFrame, threshold: float, score_col: str = "score", positive: str = "whale",
              negative: str = "whale_free", groups: dict[str, pd.Series] | None = None) -> dict:
    """Counts at one threshold: hits on ``positive`` units, false alarms on ``negative`` units, firing rate on the
    rest, and hits inside each boolean mask in ``groups`` (for example clear / hazy / bubbles subsets)."""
    s = table[score_col].fillna(-np.inf)
    fire = s >= threshold
    pos, neg = table[positive].astype(bool), table[negative].astype(bool)
    other = ~pos & ~neg
    out = dict(threshold=float(threshold), hits=int((fire & pos).sum()), n_pos=int(pos.sum()),
               false_alarms=int((fire & neg).sum()), n_neg=int(neg.sum()),
               fire_other=int((fire & other).sum()), n_other=int(other.sum()))
    out["recall"] = out["hits"] / max(out["n_pos"], 1)
    out["far"] = out["false_alarms"] / max(out["n_neg"], 1)
    for name, m in (groups or {}).items():
        m = m.reindex(table.index).fillna(False).astype(bool)
        out[f"hits_{name}"] = int((fire & m).sum())
        out[f"n_{name}"] = int(m.sum())
    return out
