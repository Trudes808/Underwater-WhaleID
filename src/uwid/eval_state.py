"""Truth and scoring for the water state (underwater / crossing / air) per 1 s window, and surfacing events.

Unit of decision (user, 2026-11-20): a 1 s window on video time; window k covers [k, k+1) seconds of
presentation time.

Truth sources
-------------
* Sensor truth (:func:`sensor_timeline`): the tag's depth sensor and its tag-state timeline
  (``underwater_cache/tagdata/<deployment>/states.csv``, ``surfacings.csv``, ``drops.csv`` from WP1).
  On the whale the lens is in air from ``start_t + LENS_OUT_LAG_S`` to ``end_t - LENS_IN_LEAD_S`` of each
  surfacing (measured by eye on 18 events, +-0.3 s); deck (``air``) and drone flight (``flight``) are ``air``.
  After the drop that attached the tag, the lens stays in air until the depth first passes the surfacing
  threshold, then the same lead as at a surfacing applies (the tag lands on a whale at the surface).
  ``floating`` (tag free in the sea) is its own state ``floating``: the lens bobs at the waterline, neither
  side is a fair truth, so those windows are excluded from the confusion tables and reported apart.
  Events are the changes between ``air`` and ``underwater``.
* By-eye truth (:func:`label_windows`): frame labels (``frames.csv`` of experiments/02_reference_labels) and
  events (``events.json``). As in PLAN.md: a window is ``underwater`` or ``air`` when all its labelled frames
  agree, otherwise ``crossing``; a window holding an event or a ``crossing`` frame is ``crossing`` too.

Window truth (:func:`windows_from_timeline`): ``crossing`` when the window contains an exit or an entry;
``underwater`` / ``air`` when the lens is in that state for the whole window; ``excluded`` when it overlaps
``floating`` / unknown time or runs past the end of the video. Windows with an event within ``AMBIG_S`` of their
edges are flagged ``ambiguous`` (the event could fall in the neighbouring window given the truth's timing error).

Scorers: :func:`confusion` (window or frame confusion table, recall and precision per state) and
:func:`match_events` (each exit / entry found within a tolerance, false events per hour, timing error).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("underwater", "crossing", "air")
LENS_OUT_LAG_S = 0.3    # lens out of the water this long after a surfacing's start_t (WP1, 18 events by eye)
LENS_IN_LEAD_S = 0.4    # lens under water again this long before end_t
AMBIG_S = 0.3           # timing uncertainty of the sensor truth vs the picture


@dataclass
class Timeline:
    """Piecewise-constant state on VIDEO time: segments [t0, t1) with a state, and the events between them."""
    segments: pd.DataFrame   # columns t0, t1, state ('air', 'underwater', 'floating', 'unknown')
    events: pd.DataFrame     # columns t, kind ('exit' = underwater -> air, 'entry' = air -> underwater)
    duration: float
    source: str

    def state_at(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, float)
        out = np.full(t.shape, "unknown", dtype=object)
        for r in self.segments.itertuples():
            out[(t >= r.t0) & (t < r.t1)] = r.state
        return out


def _events_from_segments(seg: pd.DataFrame) -> pd.DataFrame:
    ev = []
    for a, b in zip(seg.itertuples(), seg.iloc[1:].itertuples()):
        if a.state == "underwater" and b.state == "air":
            ev.append((b.t0, "exit"))
        elif a.state == "air" and b.state == "underwater":
            ev.append((b.t0, "entry"))
    return pd.DataFrame(ev, columns=["t", "kind"])


def _merge(seg: list[tuple[float, float, str]]) -> pd.DataFrame:
    seg = sorted((a, b, s) for a, b, s in seg if b > a)
    out: list[list] = []
    for a, b, s in seg:
        if out and out[-1][2] == s and abs(out[-1][1] - a) < 1e-9:
            out[-1][1] = b
        else:
            out.append([a, b, s])
    return pd.DataFrame(out, columns=["t0", "t1", "state"])


def sensor_timeline(deployment: str, video_name: str, duration: float | None = None,
                    out_lag: float = LENS_OUT_LAG_S, in_lead: float = LENS_IN_LEAD_S) -> Timeline:
    """Lens state on the video's presentation time, from the tag's sensors (see module docstring)."""
    from . import tagdata
    cd = tagdata.cache_dir(deployment)
    vc = tagdata.video_clock(deployment, video_name)
    states = pd.read_csv(cd / "states.csv")
    surf = pd.read_csv(cd / "surfacings.csv")
    drops = pd.read_csv(cd / "drops.csv") if (cd / "drops.csv").exists() else pd.DataFrame()
    # sensor-time segments
    seg: list[tuple[float, float, str]] = []
    for r in states.itertuples():
        s = {"air": "air", "flight": "air", "floating": "floating", "whale": "underwater"}.get(r.state, "unknown")
        seg.append((float(r.start_t), float(r.end_t) + 0.02, s))   # states.csv rows are on a 20 ms grid
    seg_df = _merge(seg)
    # the attaching drop: the tag lands on a whale at the surface. states.csv keeps 'flight' until depth > 1 m;
    # here the lens is in air until the depth first passes the surfacing threshold (minus the usual lead), then
    # under water.
    rows = seg_df.values.tolist()
    if len(drops):
        dep = tagdata.load_depth_updates(deployment)
        thr = float(surf.air_reading_m.median()) + 0.10
        for d in drops[drops.attached.astype(bool)].itertuples():
            after = dep[(dep.t > d.impact_t) & (dep.depth > thr)]
            t_in = (float(after.t.iloc[0]) if len(after) else float(d.impact_t)) - in_lead
            new = []
            for t0, t1, s in rows:
                if s == "air" and t0 < t_in < t1 and t1 <= float(d.impact_t) + 10:
                    new += [[t0, t_in, "air"], [t_in, t1, "underwater"]]
                else:
                    new.append([t0, t1, s])
            rows = new
    # surfacings: lens in air from start_t + lag to end_t - lead, painted over 'underwater'
    for r in surf.itertuples():
        a, b = float(r.start_t) + out_lag, float(r.end_t) - in_lead
        if b <= a:
            continue
        new = []
        for t0, t1, s in rows:
            if t1 <= a or t0 >= b or s != "underwater":
                new.append([t0, t1, s])
                continue
            if t0 < a:
                new.append([t0, a, s])
            new.append([max(t0, a), min(t1, b), "air"])
            if t1 > b:
                new.append([b, t1, s])
        rows = new
    seg_df = _merge([tuple(r) for r in rows])
    # to video time
    v0 = vc.to_video(seg_df.t0.values)
    v1 = vc.to_video(seg_df.t1.values)
    vs = pd.DataFrame({"t0": v0, "t1": v1, "state": seg_df.state.values})
    if duration is not None:
        vs = vs[(vs.t1 > 0) & (vs.t0 < duration)].copy()
        vs["t0"] = vs.t0.clip(lower=0)
        vs["t1"] = vs.t1.clip(upper=duration)
    vs = vs.reset_index(drop=True)
    ev = _events_from_segments(vs)
    if duration is not None:
        ev = ev[(ev.t > 0) & (ev.t < duration)].reset_index(drop=True)
    return Timeline(vs, ev, float(duration) if duration else float(vs.t1.max()), f"sensor ({vc.source})")


def timeline_from_events(events: pd.DataFrame, duration: float, start_state: str) -> Timeline:
    """Timeline from a list of exits / entries (by-eye events) and the state at t = 0."""
    ev = events.sort_values("t").reset_index(drop=True)
    seg, t, s = [], 0.0, start_state
    for r in ev.itertuples():
        seg.append((t, r.t, s))
        s = "air" if r.kind == "exit" else "underwater"
        t = r.t
    seg.append((t, duration, s))
    return Timeline(_merge(seg), ev[["t", "kind"]], duration, "events")


def windows_from_timeline(tl: Timeline, win: float = 1.0, ambig_s: float = AMBIG_S) -> pd.DataFrame:
    """One row per window k (covering [k win, (k+1) win)): truth state, ambiguous flag, events inside."""
    n = int(np.floor(tl.duration / win + 1e-9))
    rows = []
    ev_t = tl.events.t.values
    for k in range(n):
        a, b = k * win, (k + 1) * win
        inside = tl.events[(tl.events.t >= a) & (tl.events.t < b)]
        ov = tl.segments[(tl.segments.t1 > a) & (tl.segments.t0 < b)]
        states = set(ov.state)
        covered = ov.t1.clip(upper=b).sub(ov.t0.clip(lower=a)).sum()
        if states & {"floating", "unknown"} or covered < win - 1e-6:
            st = "floating" if "floating" in states else "excluded"
        elif len(inside):
            st = "crossing"
        elif states == {"air"}:
            st = "air"
        elif states == {"underwater"}:
            st = "underwater"
        else:
            st = "crossing"
        near = np.any((np.abs(ev_t - a) < ambig_s) | (np.abs(ev_t - b) < ambig_s)) if len(ev_t) else False
        rows.append((k, a, b, st, bool(near), ";".join(inside.kind)))
    return pd.DataFrame(rows, columns=["k", "t0", "t1", "truth", "ambiguous", "events"])


def frame_truth(tl: Timeline, t: np.ndarray, near_s: float = AMBIG_S) -> np.ndarray:
    """State per frame time: 'air' / 'underwater', 'crossing' within near_s of an event, else the segment state."""
    st = tl.state_at(t)
    if len(tl.events):
        d = np.min(np.abs(np.asarray(t)[:, None] - tl.events.t.values[None, :]), axis=1)
        st = np.where(d < near_s, "crossing", st)
    return st


# ---------------------------------------------------------------- by-eye labels

def load_label_dir(d: str | Path) -> tuple[pd.DataFrame, dict | None]:
    d = Path(d)
    fr = pd.read_csv(d / "frames.csv")
    ev = json.loads((d / "events.json").read_text()) if (d / "events.json").exists() else None
    return fr, ev


def label_windows(frames: pd.DataFrame, events: dict | None, duration: float, win: float = 1.0) -> pd.DataFrame:
    """Window truth from by-eye frame labels (PLAN.md rule) plus by-eye events."""
    n = int(np.floor(duration / win + 1e-9))
    ev = pd.DataFrame(events["events"]) if events else pd.DataFrame(columns=["time_s", "type"])
    rows = []
    for k in range(n):
        a, b = k * win, (k + 1) * win
        f = frames[(frames.time_s >= a) & (frames.time_s < b)]
        st_set = set(f.state) - {"unusable"}
        inside = ev[(ev.time_s >= a) & (ev.time_s < b)] if len(ev) else ev
        if not st_set:
            st = "excluded"
        elif len(inside) or "crossing" in st_set or len(st_set) > 1:
            st = "crossing"
        else:
            st = st_set.pop()
        near = bool(len(ev) and np.any((np.abs(ev.time_s.values - a) < 0.1) | (np.abs(ev.time_s.values - b) < 0.1)))
        rows.append((k, a, b, st, near, ";".join(inside["type"]) if len(inside) else ""))
    return pd.DataFrame(rows, columns=["k", "t0", "t1", "truth", "ambiguous", "events"])


def label_events(events: dict) -> pd.DataFrame:
    return pd.DataFrame({"t": [e["time_s"] for e in events["events"]], "kind": [e["type"] for e in events["events"]]})


# ---------------------------------------------------------------- scorers

def confusion(truth, pred, states=STATES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Confusion table (rows truth, columns prediction) and per-state recall / precision with counts."""
    truth = np.asarray(truth, object)
    pred = np.asarray(pred, object)
    keep = np.isin(truth, states)
    truth, pred = truth[keep], pred[keep]
    tab = pd.DataFrame(0, index=list(states), columns=list(states))
    for s in states:
        for p in states:
            tab.loc[s, p] = int(np.sum((truth == s) & (pred == p)))
    rows = []
    for s in states:
        n_t = int(tab.loc[s].sum())
        n_p = int(tab[s].sum())
        tp = int(tab.loc[s, s])
        rows.append((s, n_t, n_p, tp, tp / n_t if n_t else np.nan, tp / n_p if n_p else np.nan))
    per = pd.DataFrame(rows, columns=["state", "n_truth", "n_pred", "correct", "recall", "precision"])
    return tab, per


def match_events(truth: pd.DataFrame, pred: pd.DataFrame, tol: float = 0.5, duration_s: float | None = None) -> dict:
    """One-to-one matching of predicted to true events of the same kind within ``tol`` seconds (closest first).

    Returns per kind and overall: n_truth, found, recall, n_pred, false (unmatched predictions), false per hour
    (when duration_s is given), timing error (pred - truth) median, median absolute, and the matched pairs.
    """
    out = {}
    pairs_all = []
    for kind in ("exit", "entry"):
        tt = truth[truth.kind == kind].t.values
        pp = pred[pred.kind == kind].t.values if len(pred) else np.array([])
        cand = sorted((abs(p - t), i, j) for i, t in enumerate(tt) for j, p in enumerate(pp) if abs(p - t) <= tol)
        ut, up, pairs = set(), set(), []
        for dd, i, j in cand:
            if i in ut or j in up:
                continue
            ut.add(i); up.add(j)
            pairs.append((float(tt[i]), float(pp[j])))
        err = np.array([p - t for t, p in pairs])
        r = {"n_truth": len(tt), "found": len(pairs), "recall": len(pairs) / len(tt) if len(tt) else np.nan,
             "n_pred": len(pp), "false": len(pp) - len(pairs),
             "err_median": float(np.median(err)) if len(err) else np.nan,
             "abs_err_median": float(np.median(np.abs(err))) if len(err) else np.nan}
        if duration_s:
            r["false_per_h"] = r["false"] / (duration_s / 3600.0)
        out[kind] = r
        pairs_all += [(kind, t, p) for t, p in pairs]
    n_t = sum(out[k]["n_truth"] for k in ("exit", "entry"))
    f = sum(out[k]["found"] for k in ("exit", "entry"))
    n_p = sum(out[k]["n_pred"] for k in ("exit", "entry"))
    err = np.array([p - t for _, t, p in pairs_all])
    out["all"] = {"n_truth": n_t, "found": f, "recall": f / n_t if n_t else np.nan, "n_pred": n_p, "false": n_p - f,
                  "abs_err_median": float(np.median(np.abs(err))) if len(err) else np.nan,
                  "err_median": float(np.median(err)) if len(err) else np.nan}
    if duration_s:
        out["all"]["false_per_h"] = (n_p - f) / (duration_s / 3600.0)
    out["pairs"] = pairs_all
    return out
