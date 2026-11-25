"""WP15 phase 2: pick frames for the 3D trial from WP13's tracker output (repo .venv, CPU).

    .venv/bin/python experiments/10_mesh3d/pick_frames.py [--per-number 4] [--cap 240] [--min-life-s 3]

Input: the span folders listed in `underwater_cache/tracks/wp13_spans.csv` (nothing else) and the shown numbers of
`wp13_numbers.csv` (after `trackstage.load_online` + `trackstage.link`, as WP13 numbered them), numbers lasting
>= --min-life-s only. Two numbers are excluded by name whatever the rule says (reviewer: wrong or unclear):
2025-02-03 00005 span 375-389 number 2 and 00005 span 365-367 number 1.

Per number, every 5th frame where it has an outline is judged by the frame rule of tables.md S0 (unchanged); the
first failing clause, in rule order, is recorded: hidden duplicate, drawn small, blocked, area < 3 %, at the frame
edge, below the fixed back-box top, within 12 px of the back outline, split outline, outside the scan's hq spans,
sharpness below the number's median (median over the number's frames that passed everything before it).
Then up to --per-number frames per number, spread over outline shape (orientation folded to 0-180 deg, elongation,
area share, fill), >= 1 s apart, sharpest first. If the total exceeds --cap, the per-number limit drops to 3, then 2.
Writes underwater_cache/mesh3d/trial/{judged.csv, picks.csv, pick_counts.json}. The pick list is fixed once written:
the script refuses to overwrite picks.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import framerule as fr  # noqa: E402
from uwid import trackstage  # noqa: E402
from uwid import video as uv  # noqa: E402

OUT = fr.MESH / "trial"
EXCLUDE = [("CC-11-96-20250203-104242-000-00005", "0375.00-0389.00", 2),
           ("CC-11-96-20250203-104242-000-00005", "0365.00-0367.00", 1)]
DEP_SHORT = {"2025-2-3 Mini-mother 11-96": "d0203", "TestSubset 2025-1-2 CATS -85": "cats85"}


def shape_features(m: np.ndarray) -> dict:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    if w < h:
        w, h, ang = h, w, ang + 90
    return dict(orient=float(ang % 180), elong=float(w / max(h, 1)), fill=float(cv2.contourArea(c) / max(w * h, 1)))


def spread(g: pd.DataFrame, k: int, min_gap: float) -> pd.DataFrame:
    """Farthest-point sampling in (orientation as a circle, log elongation, area share, fill), sharpest first."""
    th = np.deg2rad(g.orient.to_numpy() * 2)
    X = np.c_[np.cos(th), np.sin(th), np.log(g.elong.to_numpy()), g.area_share.to_numpy() * 10, g.fill.to_numpy() * 2]
    order = [int(np.argmax(g.sharp.to_numpy()))]
    t = g.t.to_numpy()
    while len(order) < k:
        d = np.min(np.linalg.norm(X[:, None] - X[order][None], axis=2), axis=1)
        d[np.min(np.abs(t[:, None] - t[order][None]), axis=1) < min_gap] = -1
        j = int(np.argmax(d))
        if d[j] <= 0:
            break
        order.append(j)
    return g.iloc[order]


def judge_span(dep: str, folder: Path, nums: set[int], hq: pd.DataFrame, rule: fr.Rule,
               skip_edge: bool = False) -> pd.DataFrame:
    """``skip_edge``: arm B (whales cut by the frame) judges every clause except 'at the frame edge'."""
    tr = trackstage.load_online(folder)
    trackstage.link(tr, dep, "scan")
    W, H = tr.meta["store_size"]
    sx = W / trackstage.TRACK_W
    back_top = tr.meta.get("back_box", [0, 810, 1920, 1080])[1] * sx
    blk = fr.blocked_set(tr.meta)
    video = Path(tr.meta["video"]).name
    spans = hq[hq.recording == video]
    tab = tr.table[(tr.table.area > 0) & (tr.table.i % rule.step == 0)]
    rows = []
    for obj, num in tr.number.items():
        if num not in nums:
            continue
        for r in tab[tab.obj == obj].itertuples():
            i = int(r.i)
            row = dict(deployment=dep, folder=str(folder), video=video, span=folder.name.replace("_online", ""),
                       number=int(num), obj=int(obj), i=i, frame=int(r.frame), t=round(float(r.t), 3),
                       area_share=round(float(r.area) / (W * H), 4), why="")
            x0, y0, x1, y1 = r.x0 * sx, r.y0 * sx, r.x1 * sx, r.y1 * sx
            if not tr.visible(obj, i):
                row["why"] = "hidden duplicate"
            elif tr.is_small(obj, i):
                row["why"] = "drawn small"
            elif int(r.frame) in blk:
                row["why"] = "blocked"
            elif r.area < rule.min_area * W * H:
                row["why"] = "area < 3 %"
            elif not skip_edge and (min(x0, y0) < rule.edge_px or x1 > W - rule.edge_px or y1 > H - rule.edge_px):
                row["why"] = "at frame edge"
            elif y1 > back_top:
                row["why"] = "below back-box top"
            else:
                ok, vals, m = fr.mask_checks(tr, obj, i, rule)
                if not ok:
                    row["why"] = {"near back": "near back outline", "split": "split outline"}.get(vals.get("why"),
                                                                                                vals.get("why"))
                elif not ((spans.start <= r.t) & (spans.end >= r.t)).any():
                    row["why"] = "outside hq span"
                else:
                    row.update(shape_features(m))
            rows.append(row)
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    want = df[df.why == ""]
    if len(want):
        sharp = {}
        for f in uv.iter_frames_isolated(tr.meta["video"], indices=sorted(want.frame.unique()), size=(W, H)):
            for r in want[want.frame == f.index].itertuples():
                sharp[(r.obj, r.i)] = fr.sharpness(f.image, tr.mask(r.obj, r.i), rule.erode_px)
        df["sharp"] = [sharp.get((o, i), np.nan) for o, i in zip(df.obj, df.i)]
        cand = df.why == ""
        df.loc[cand & df.sharp.isna(), "why"] = "not decoded"
        cand = df.why == ""
        med = df[cand].groupby("number").sharp.transform("median")
        low = df[cand].sharp < med
        df.loc[low[low].index, "why"] = "sharpness below median"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-number", type=int, default=4)
    ap.add_argument("--cap", type=int, default=240)
    ap.add_argument("--min-life-s", type=float, default=3.0)
    ap.add_argument("--min-gap-s", type=float, default=1.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "picks.csv").exists():
        sys.exit("picks.csv exists: the pick list is fixed, not overwritten")
    print("pick_frames: start", flush=True)
    S = pd.read_csv(fr.TRACKS / "wp13_spans.csv")
    N = pd.read_csv(fr.TRACKS / "wp13_numbers.csv")
    N["rec"] = N.recording.str.replace(".mov", "", regex=False)
    N["span_key"] = N.span.str.replace("_online", "", regex=False)
    excl = [(r.rec, r.span_key, r.number, r.life_s) for r in N.itertuples()
            if (r.rec, r.span_key, r.number) in EXCLUDE]
    keep = N[(N.life_s >= a.min_life_s) & ~N.apply(lambda r: (r.rec, r.span_key, r.number) in EXCLUDE, axis=1)]
    print("excluded by name (life_s):", excl, "-> also under the life rule:", all(e[3] < a.min_life_s for e in excl),
          flush=True)
    hq = {dep: trackstage.hq_spans(dep) for dep in S.deployment.unique()}
    judged = []
    for folder, g in keep.groupby("folder"):
        if folder not in set(S.folder):
            continue
        dep = g.deployment.iloc[0]
        df = judge_span(dep, Path(folder), set(g.number.astype(int)), hq[dep], fr.Rule())
        print(dep[:8], Path(folder).parent.name[-5:], Path(folder).name, "numbers", len(g), "judged", len(df),
              "pass", int((df.why == "").sum()) if len(df) else 0, flush=True)
        if len(df):
            judged.append(df)
    J = pd.concat(judged, ignore_index=True)
    J.to_csv(OUT / "judged.csv", index=False)
    P = J[J.why == ""].copy()
    k = a.per_number
    while True:
        picks = []
        for (folder, num), g in P.groupby(["folder", "number"]):
            sel = spread(g.reset_index(drop=True), k, a.min_gap_s)
            for j, r in enumerate(sel.itertuples()):
                rec = r.video.split("-")[-1].split(".")[0]
                picks.append(dict(id=f"{DEP_SHORT[r.deployment]}_{rec}_{r.span[:7]}_n{num}_{j}", run=folder,
                                  obj=r.obj, i=r.i, frame=r.frame, t=r.t, video=r.video, number=num,
                                  animal=f"{DEP_SHORT[r.deployment]}_{rec}_{r.span}_n{num}", deployment=r.deployment,
                                  kind="", sharp=r.sharp, area_share=r.area_share, orient=r.orient,
                                  elong=r.elong, fill=r.fill))
        if len(picks) <= a.cap or k <= 2:
            break
        print(f"{len(picks)} picks at {k} per number > cap {a.cap}: lowering", flush=True)
        k -= 1
    # counts for the record
    numbers = keep[keep.folder.isin(set(S.folder))][["deployment", "folder", "number"]].drop_duplicates()
    pc = P.groupby(["folder", "number"]).size()
    hist = {}
    for dep, g in numbers.groupby("deployment"):
        c = [int(pc.get((f, n), 0)) for f, n in zip(g.folder, g.number)]
        hist[dep] = {"numbers": len(c), **{str(b): sum(min(x, 4) == b for x in c) for b in range(5)}}
    pk = pd.DataFrame(picks)
    pick_hist = {dep: pk[pk.deployment == dep].groupby("animal").size().value_counts().sort_index().to_dict()
                 for dep in pk.deployment.unique()}
    reasons = {dep: J[J.deployment == dep].why.replace("", "passed").value_counts().to_dict()
               for dep in J.deployment.unique()}
    counts = dict(per_number_limit=k, picks=len(pk), numbers_ge_life=hist, passing_frames_bucket_note="4 = 4 or more",
                  picks_per_number=pick_hist, frames_judged_by_first_failing_clause=reasons,
                  excluded_by_name=[list(e) for e in excl], rule="tables.md S0, every 5th frame")
    (OUT / "pick_counts.json").write_text(json.dumps(counts, indent=1, default=int))
    pk.to_csv(OUT / "picks.csv", index=False)
    print(json.dumps(counts, indent=1, default=int))
    print("pick_frames: done", len(pk), "picks", flush=True)


if __name__ == "__main__":
    main()
