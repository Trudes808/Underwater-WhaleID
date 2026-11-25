"""WP11c held-out test of the lone-whale link across breaths + one look at groups. Every tracker rule is FROZEN.

THE RULE, written down 2026-11-23 before any tracker run or by-eye look at these encounters. Inputs are the scan's
window tables and detector cache (`underwater_cache/scan/<deployment>/sensitive/`) and the depth sensor only, for
both test deployments (2025-02-03 and CC-09-85 2025-01-02; the dev clip is out).

Breath encounters
- Lens out by the frozen blocked rule: depth lens-out (`trackstage.lens_out_spans`, with LENS_PAD_S) AND the picture
  label of the 1 s window not `underwater` (`trackstage.picture_out`), on a 0.1 s grid, runs < 0.2 s dropped, gaps
  < 0.3 s filled (the blocked rule's own cleaning). A breath = one such run of 1-12 s.
- Whale before: a window of tier `hq` or `ok` starting within the 5 s before the breath.
- Whale after: a window with `active` and `whale` starting within the 12 s after the breath.
- (Dropped 2026-11-23 13:40 by the coordinator, before any tracker run: "lone = at most one scan whale decision
  object in every window of the 10 s before / after". It found 0 of 11 otherwise eligible breaths, including the
  known lone case 00007 1465 s: the scan puts two whale boxes on one near whale. The count is kept as the column
  `multi_windows` (windows within 10 s with >= 2 whale objects); lone or not is now decided by eye on plain frames
  before the tracker output is looked at.)
- Run span: 10 s before the breath to 15 s after it, capped at 40 s.
- Left out (used for setting or checking rules): 00007 1459-1475 (and the WP8b breath 1459.5-1474.5), 00024 75-164,
  WP11a breath spans 00011 499-506 and 00017 176-186, the 8 WP11a held-out stretches (`results/selection.csv`), the
  WP11a whole encounter 00008 25-103, the WP8 demo clips and WP8b stretches on 2025-02-03 (00004 650-668, 00005
  365.5-381.5 and 736-754, 00006 456-476, 575.5-590, 822-840 and 1514-1529, 00007 1212.5-1230), the WP8c sand spans of
  00017 (`02_reference_labels/.../blocked_spans.csv`). An encounter whose run span overlaps any of them is skipped.
- Take up to 8, at most 2 per recording, earliest first: first the earliest eligible breath of every recording (in
  recording order, 2025-01-02 first), then the second of every recording, until 8.

Groups
- A run of >= 3 consecutive windows (3 s) in which >= 2 scan whale decision objects share a window; stretch = 20 s
  from 5 s before the run's first window (clamped to the recording), not overlapping the left-out list or a selected
  breath encounter. Take the 2 longest runs from different recordings. None found = a result.

Budget (50 GPU min since 13:40, was 30; estimate 10 GPU s per video s + 60 s per span for the patch pass and model load): groups first,
then breath encounters in selection order until the estimate reaches 1800 s; when the 8 do not fit, the cap per
recording drops to 1 before any encounter is cut, then the list is cut from the end.

    .venv/bin/python experiments/07_tracks_demo/online/select_breath.py   # -> results/selection_breath.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from uwid import trackstage as T  # noqa: E402

DEPS = ["TestSubset 2025-1-2 CATS -85", "2025-2-3 Mini-mother 11-96"]
GPU_S_PER_S, SPAN_OVERHEAD_S, BUDGET_S = 10.0, 60.0, 3000.0


def left_out() -> dict[str, list[tuple[float, float]]]:
    lo = {"00007": [(1459, 1475), (1459.5, 1474.5), (1212.5, 1230)], "00024": [(75, 164)],
          "00011": [(499, 506)], "00017": [(176, 186)], "00008": [(25, 103)],
          "00004": [(650, 668)], "00005": [(365.5, 381.5), (736, 754)],
          "00006": [(456, 476), (575.5, 590), (822, 840), (1514, 1529)]}
    sel = pd.read_csv(HERE / "results" / "selection.csv")
    for r in sel.itertuples():
        lo.setdefault(r.recording[-9:-4], []).append((float(r.t0), float(r.t1)))
    sp = HERE.parents[1] / "02_reference_labels/labels/CC-09-85-HRA-20250102-141304-000-00017/blocked_spans.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        lo["00017"] += [(float(r.start_s), float(r.end_s)) for r in s[s.kind == "sand"].itertuples()]
    return lo


def overlaps(rec: str, a: float, b: float, lo: dict) -> str:
    for x, y in lo.get(rec, []):
        if a < y and b > x:
            return f"overlaps left-out {rec} {x:g}-{y:g}"
    return ""


def window_counts(dep: str, name: str, w: pd.DataFrame) -> np.ndarray:
    """Scan whale decision objects per window (max over its two detector frames)."""
    f = np.concatenate([w.frame_0.dropna().to_numpy(), w.frame_1.dropna().to_numpy()])
    if not len(f):
        return np.zeros(len(w), int)
    s = T.scan_starts(dep, name, int(f.min()), int(f.max()) + 1)
    per = s.groupby("frame").size().to_dict() if len(s) else {}
    out = []
    for r in w.itertuples():
        c = [per.get(int(v), 0) for v in (r.frame_0, r.frame_1) if np.isfinite(v)]
        out.append(max(c) if c else 0)
    return np.array(out)


def breaths(dep: str, name: str, w: pd.DataFrame) -> list[tuple[float, float]]:
    lens = T.lens_out_spans(dep, name)
    if not lens:
        return []
    t = np.arange(0, float(w.t1.max()), 0.1)
    m = np.zeros(len(t), bool)
    for a, b in lens:
        m |= (t >= a - T.LENS_PAD_S) & (t <= b + T.LENS_PAD_S)
    m &= T.picture_out(dep, name, t)
    m = T.clean_blocked(m, 10.0)
    return [(round(t[a], 2), round(t[b - 1] + 0.1, 2)) for a, b in T.runs_of(m)]


def main():
    lo = left_out()
    cand, skips, groups = [], [], []
    for dep in DEPS:
        sd = T.scan_dir(dep, "sensitive")
        for p in sorted((sd / "windows").glob("*.csv")):
            name = p.name[:-4]
            rec = name[-9:-4]
            w = T.windows_table(dep, name)
            src = next((v["source"] for v in T.scan_config(dep, "sensitive")[1]["videos"]
                        if Path(v["source"]).name == name), None)
            n = window_counts(dep, name, w)
            t0s = w.t0.to_numpy()
            for a, b in breaths(dep, name, w):
                d = b - a
                why = ""
                if not 1.0 <= d <= 12.0:
                    why = f"lens out {d:.1f} s (not 1-12 s)"
                else:
                    pre = w[(t0s >= a - 5) & (t0s < a)]
                    post = w[(t0s > b - 1) & (t0s <= b + 12)]
                    near = (t0s >= a - 10) & (t0s < b + 10)
                    if not pre.tier.isin(["hq", "ok"]).any():
                        why = "no hq / ok whale window in the 5 s before"
                    elif not (post.active.astype(bool) & post.whale.astype(bool)).any():
                        why = "no active whale window in the 12 s after"
                r0, r1 = round(max(0.0, a - 10), 2), round(b + 15, 2)
                if r1 - r0 > 40:
                    r1 = round(r0 + 40, 2)
                why = why or overlaps(rec, r0, r1, lo)
                row = dict(set="breath", deployment=dep, recording=name, source=src, rec=rec, breath_t0=a, breath_t1=b,
                           breath_s=round(d, 2), t0=r0, t1=r1,
                           multi_windows=int((n[near] > 1).sum()) if 1.0 <= d <= 12.0 else None)
                (skips if why else cand).append(dict(row, skip=why))
            # groups: >= 3 consecutive windows with >= 2 whale objects
            g = n >= 2
            for a_, b_ in T.runs_of(g):
                if b_ - a_ >= 3:
                    s0 = max(0.0, float(t0s[a_]) - 5)
                    groups.append(dict(set="group", deployment=dep, recording=name, source=src, rec=rec,
                                       run_t0=float(t0s[a_]), run_s=int(b_ - a_), t0=s0, t1=s0 + 20,
                                       skip=overlaps(rec, s0, s0 + 20, lo)))
    cand = pd.DataFrame(cand)
    order = {r: i for i, r in enumerate(dict.fromkeys(cand.recording))} if len(cand) else {}

    def pick(cap):
        if not len(cand):
            return cand
        c = cand.sort_values(["breath_t0"]).copy()
        c["k"] = c.groupby("recording").cumcount()
        c = c[c.k < cap]
        c["o"] = c.recording.map(order)
        return c.sort_values(["k", "o"]).head(8)

    # groups: 2 longest from different recordings, not overlapping a selected breath
    G = pd.DataFrame(groups)
    chosen_g = []
    if len(G):
        for r in G.sort_values(["run_s", "run_t0"], ascending=[False, True]).itertuples():
            if r.skip or len(chosen_g) == 2 or any(x["recording"] == r.recording for x in chosen_g):
                continue
            chosen_g.append(r._asdict())
    cost = lambda rows: sum((x["t1"] - x["t0"]) * GPU_S_PER_S + SPAN_OVERHEAD_S for x in rows)  # noqa: E731
    sel = pick(2)
    if cost(chosen_g) + cost(sel.to_dict("records")) > BUDGET_S:
        sel = pick(1)
    rows = sel.to_dict("records")
    while rows and cost(chosen_g) + cost(rows) > BUDGET_S:
        rows = rows[:-1]
    # groups must not overlap a selected breath encounter
    chosen_g = [x for x in chosen_g if not any(x["recording"] == r["recording"] and x["t0"] < r["t1"] and
                                              x["t1"] > r["t0"] for r in rows)]
    out = pd.DataFrame(rows + chosen_g)
    out["est_gpu_s"] = [(r.t1 - r.t0) * GPU_S_PER_S + SPAN_OVERHEAD_S for r in out.itertuples()]
    keep = ["set", "deployment", "recording", "source", "t0", "t1", "breath_t0", "breath_t1", "breath_s", "multi_windows",
            "run_t0",
            "run_s", "est_gpu_s"]
    out = out[[c for c in keep if c in out]]
    out.to_csv(HERE / "results" / "selection_breath.csv", index=False)
    sk = pd.concat([pd.DataFrame(skips),
                    cand[~cand.set_index(["recording", "breath_t0"]).index.isin(
                        sel.set_index(["recording", "breath_t0"]).index if len(sel) else [])].assign(
                        skip="eligible, not taken (per-recording cap / 8 / budget)") if len(cand) else pd.DataFrame(),
                    G[G.skip != ""] if len(G) else pd.DataFrame(),
                    G[(G.skip == "") & ~G.run_t0.isin([x["run_t0"] for x in chosen_g])].assign(
                        skip="group not taken (2 longest from different recordings / overlaps a breath)") if len(G) else
                    pd.DataFrame()], ignore_index=True)
    sk.to_csv(HERE / "results" / "selection_breath_skipped.csv", index=False)
    print(out.to_string())
    print(f"estimated {out.est_gpu_s.sum():.0f} GPU s; skipped {len(sk)}")
    print(sk.skip.str.split(" \\(").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
