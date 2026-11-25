"""WP7 measures from the `uwid scan` outputs (nothing is fitted or tuned here).

    .venv/bin/python experiments/06_clips_e2e/evaluate.py            # all run labels found under the scan folder
    .venv/bin/python experiments/06_clips_e2e/evaluate.py --dep d2   # the second deployment (tag CC-09-85)

The second deployment is scored against WP3b's labels (AI by-eye: `labels/<video stem>/frames.csv`, 20-60 frames
per recording, `seafloor_spans.csv`, `blocked_spans.csv`), only when every recording has its
`underwater_cache/reference/<stem>/DONE`. Writes `results/eval_d2.json` and prints its tables.

Truth: WP3's AI by-eye frame labels (120 frames per 30-min piece for 2025-02-03 00004-00010; 191 frames of the dev
clip). Each labelled frame is placed on the `.mov` timeline: the label files index the `.m4v` conversions, whose
frames are the `.mov` frames in the same order (WP0); this script checks that on 3 frames per piece by reading the
burned-in clock in both forms. A labelled frame "falls inside a kept stretch" when its `.mov` time does.

Writes `experiments/06_clips_e2e/results/eval.json` and prints the tables used in `tables.md`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from uwid import eval_presence as ep, paths, timecode, video  # noqa: E402

DEP = "2025-2-3 Mini-mother 11-96"
VD = paths.footage_dir() / DEP / "Video"
LABELS = REPO / "experiments/02_reference_labels/labels"
SCAN = paths.cache_dir("scan")
LABELLED = ["00004", "00005", "00006", "00007", "00008", "00009", "00010"]
DEV = "2025-12-20_CG"
OUT = HERE / "results"
from uwid.pipeline import REASONS  # noqa: E402


def stem_of(piece):
    return next(VD.glob(f"CC-11-96-20250203-*-{piece}.mov")).stem


def check_index_mapping(cache: Path) -> dict:
    """Clock text of label frames read in the .m4v and in the .mov at the same index."""
    f = cache / "index_mapping_check.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {}
    for p in LABELLED:
        st = stem_of(p)
        lab = pd.read_csv(LABELS / st / "frames.csv")
        idx = [int(lab.frame.iloc[i]) for i in (5, len(lab) // 2, len(lab) - 3)]
        rows = []
        for form in ("m4v", "mov"):
            path = VD / f"{st}.{form}"
            tab = video.frame_table(path)
            rds = timecode.clock_at(path, [float(tab.times[i]) for i in idx])
            rows.append([r.text if r else None for r in rds])
        out[p] = {"frames": idx, "m4v": rows[0], "mov": rows[1], "same": rows[0] == rows[1]}
    f.write_text(json.dumps(out, indent=1))
    return out


def kept_intervals(label_dir: Path) -> dict:
    iv = json.loads((label_dir / "intervals.json").read_text())
    return {Path(v["source"]).stem: v for v in iv["videos"]}


def frames_vs_keep(labels: pd.DataFrame, times: np.ndarray, keep: list, windows: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["t_mov"] = times[lab.frame.clip(upper=len(times) - 1)]
    lab["kept"] = False
    for k in keep:
        lab.loc[(lab.t_mov >= k["start"]) & (lab.t_mov < k["end"]), "kept"] = True
    win = windows.set_index("k")
    kk = np.floor(lab.t_mov).astype(int).clip(upper=win.index.max())
    lab["why"] = kk.map(win["why"]).to_numpy()
    lab["whale_score"] = kk.map(win["whale_score"]).to_numpy()
    lab["tag_state"] = kk.map(win["tag_state"]).to_numpy()
    lab["water"] = kk.map(win["water"]).to_numpy()
    return lab


def evaluate_label(label: str) -> dict:
    d = SCAN / DEP / label
    if not (d / "intervals.json").exists():
        return {}
    ivs = kept_intervals(d)
    res = {"label": label, "pieces": {}, "deployment": {}}
    allf = []
    for p in LABELLED:
        st = stem_of(p)
        if st not in ivs:
            continue
        labels = ep.load_frame_labels(LABELS / st / "frames.csv")
        tab = video.frame_table(VD / f"{st}.mov")
        win = pd.read_csv(d / "windows" / f"{st}.mov.csv")
        lab = frames_vs_keep(labels, tab.times, ivs[st]["keep"], win).assign(piece=p)
        lab = label_window_columns(lab, win)
        allf.append(lab)
    lab = pd.concat(allf, ignore_index=True)
    lab["faint_only"] = ~lab.whale & lab.faint
    res["activity_tiers"] = activity_and_tiers(lab)

    def share(m):
        return {"kept": int(lab[m].kept.sum()), "n": int(m.sum())}

    res["frames"] = {"clear_whale": share(lab.whale), "whale_free": share(lab.whale_free),
                     "faint_only": share(lab.faint_only),
                     "whale_free_underwater": share(lab.whale_free & (lab.state == "underwater")),
                     "fish_frames": share(lab.fish_present)}
    res["frames_per_piece"] = {p: {"clear_whale": [int(g[g.whale].kept.sum()), int(g.whale.sum())],
                                   "whale_free": [int(g[g.whale_free].kept.sum()), int(g.whale_free.sum())]}
                               for p, g in lab.groupby("piece")}
    # WP3b size / quality labels of the largest other whale (AI by-eye, 0.1 steps of the frame width)
    if "whale_size" in lab.columns:
        wl = lab[lab.whale].copy()
        wl["size_bin"] = pd.cut(pd.to_numeric(wl.whale_size, errors="coerce"), [0, 0.15, 0.25, 1.01],
                                labels=["0.1", "0.2", ">=0.3"])
        res["clear_whale_by_size"] = {str(k): [int(g.kept.sum()), int(len(g))] for k, g in wl.groupby("size_bin",
                                                                                                    observed=False)}
        res["clear_whale_by_quality"] = {str(k): [int(g.kept.sum()), int(len(g))] for k, g in wl.groupby("quality")}
    lost = lab[lab.whale & ~lab.kept]
    cols = ["piece", "frame", "t_mov", "why", "whale_score", "other_whales", "note"] + \
        [c for c in ("whale_size", "quality") if c in lost.columns]
    res["lost_whale_frames"] = lost[cols] \
        .assign(t_mov=lambda x: x.t_mov.round(2), whale_score=lambda x: x.whale_score.round(3)).to_dict("records")
    res["lost_by_reason"] = lost.why.value_counts().to_dict()
    res["kept_whale_free_by_why"] = lab[lab.whale_free & lab.kept].why.value_counts().to_dict()
    # minutes, 00004-00010 and the whole deployment
    summ = json.loads((d / "summary.json").read_text())
    for scope, pieces in (("labelled 00004-00010", LABELLED), ("all pieces", None)):
        recs = [r for r in summ["recordings"] if pieces is None or r["recording"][-5:] in pieces]
        tot = sum(r["seconds"] for r in recs)
        res["deployment"][scope] = {
            "recordings": len(recs), "minutes": tot / 60, "kept_min": sum(r["kept_s"] for r in recs) / 60,
            "n_clips": sum(r["n_clips"] for r in recs),
            "dropped_min": {k: sum(r["dropped_s"].get(k, 0) for r in recs) / 60 for k in REASONS},
            "windows": sum(r["windows"] for r in recs), "windows_scanned": sum(r["windows_scanned"] for r in recs),
            "hq_min": sum(r.get("hq_windows", 0) for r in recs) / 60,
            "hq_stretch_min": sum(r.get("hq_stretch_s", 0) for r in recs) / 60,
            "tier_window_min": {t: sum(r.get("tier_window_s", {}).get(t, 0) for r in recs) / 60 for t in TIER_ORDER[:5]},
            "clip_min_by_tier": {t: sum(r.get("clip_s_by_tier", {}).get(t, 0) for r in recs) / 60
                                 for t in TIER_ORDER[:5]},
            "n_clips_by_tier": {t: sum(r.get("n_clips_by_tier", {}).get(t, 0) for r in recs) for t in TIER_ORDER[:5]},
            "whale_windows": sum(r["whale_windows"] for r in recs)}
    res["per_recording"] = {r["recording"][-5:]: {"min": r["seconds"] / 60, "kept_min": r["kept_s"] / 60,
                                                   "clips": r["n_clips"], "hq_min": r.get("hq_windows", 0) / 60,
                                                   "hq_clips": r.get("n_hq_clips", 0),
                                                   "dropped_min": {k: round(v / 60, 2) for k, v in r["dropped_s"].items()}}
                            for r in summ["recordings"]}
    res["offset_check"] = summ.get("offset_check")
    res["timings"] = summ.get("timings_this_run")
    return res


GRADES = ["hq", "ok", "low", "whale-free"]
TIER_ORDER = ["hq", "ok", "low", "small", "fish", "none"]


def label_window_columns(lab: pd.DataFrame, win: pd.DataFrame) -> pd.DataFrame:
    """Add the window's tier, activity, water and detector columns to labelled frames (by ``t_mov``)."""
    W = win.set_index("k")
    kk = np.floor(lab.t_mov).astype(int).clip(upper=W.index.max())
    lab = lab.copy()
    for c in ("tier", "active", "whale", "small", "fish", "hq_whale", "silhouette", "dark", "sand", "whale_px", "fish_score", "small_score",
              "whale_score", "water", "water_picture", "bubbles", "tag_state", "why", "kept"):
        if c in W.columns:
            lab["w_" + c] = kk.map(W[c]).to_numpy()
    lab["tier"] = lab.get("w_tier", pd.Series("none", index=lab.index)).fillna("none")
    q = lab["quality"].astype(object).fillna("").astype(str) if "quality" in lab.columns else pd.Series("", index=lab.index)
    lab["grade"] = np.where(lab.whale & q.isin(["hq", "ok", "low"]), q,
                            np.where(lab.whale, "low", np.where(lab.whale_free, "whale-free", "faint only")))
    return lab


def activity_and_tiers(lab: pd.DataFrame) -> dict:
    """The user's measures since 2026-11-22: activity kept (costly error: activity dropped), empty kept (cheap),
    labelled grade against the tool's tier, and the advanced_ok error (a low or whale-free frame in an hq window)."""
    # a whale seen from the drone through the surface (frames labelled air) is not activity the scan can keep
    # (orchestrator's review of WP9: 5 of the 15 "lost" CC-09-85 frames were the drone's approach)
    act = ((lab.whale | lab.faint) & lab.state.ne("air")) | lab.fish_present
    empty = lab.whale_free & ~lab.fish_present & (lab.state == "underwater")
    out = {"activity_frames": [int(lab[act].kept.sum()), int(act.sum())],
           "activity_lost": lab[act & ~lab.kept][["piece", "frame", "t_mov", "why", "grade", "fish"]
                                                 ].assign(t_mov=lambda x: x.t_mov.round(1)).to_dict("records"),
           "empty_underwater_frames_kept": [int(lab[empty].kept.sum()), int(empty.sum())],
           "fish_frames_kept": [int(lab[lab.fish_present].kept.sum()), int(lab.fish_present.sum())]}
    tab = pd.crosstab(lab.grade, lab.tier).reindex(index=GRADES + ["faint only"], columns=TIER_ORDER, fill_value=0)
    out["grade_vs_tier"] = tab.astype(int).to_dict("index")
    good = lab.grade.isin(["hq", "ok"])
    out["hq_ok_frames_in_tier_ok_or_better"] = [int((good & lab.tier.isin(["hq", "ok"])).sum()), int(good.sum())]
    okp = lab.tier.isin(["hq", "ok"])
    out["tool_ok_or_better_frames_graded_low_or_free"] = [
        int((okp & lab.grade.isin(["low", "whale-free", "faint only"])).sum()), int(okp.sum())]
    hq = lab.tier == "hq"
    out["advanced_ok_frames_graded_low_or_free"] = [int((hq & ~lab.grade.isin(["hq", "ok"])).sum()), int(hq.sum())]
    out["hq_frames_in_tier_hq"] = [int(((lab.grade == "hq") & hq).sum()), int((lab.grade == "hq").sum())]
    for c in ("dark", "sand"):
        if "w_" + c in lab.columns and "view" in lab.columns:
            flag = lab["w_" + c].astype(str).str.lower().eq("true")
            out[f"{c}_label_vs_flag"] = pd.crosstab(lab.view.astype(str).eq(c).map({True: f"view {c}", False: "other"}),
                                                    flag.map({True: "flag", False: "no flag"})).astype(int).to_dict("index")
    if "silhouette" in lab.columns and "w_silhouette" in lab.columns:
        m = lab.silhouette.astype(str).isin(["yes", "no"])
        flag = lab.loc[m, "w_silhouette"].fillna(False).astype(bool).map({True: "flag", False: "no flag"})
        out["silhouette_label_vs_flag"] = pd.crosstab(lab.loc[m, "silhouette"].astype(str), flag).astype(int).to_dict(
            "index")
    return out


def evaluate_dev(label: str) -> dict:
    d = SCAN / DEV / label
    if not (d / "intervals.json").exists():
        return {}
    iv = json.loads((d / "intervals.json").read_text())["videos"][0]
    path = Path(iv["source"])
    tab = video.frame_table(path)
    win = pd.read_csv(d / "windows" / f"{path.stem.replace(' ', '_')}.{path.suffix[1:]}.csv")
    labels = ep.load_frame_labels(LABELS / DEV / "frames.csv")
    lab = frames_vs_keep(labels, tab.times, iv["keep"], win)
    out = {"label": label, "minutes": tab.duration / 60, "kept_min": sum(k["end"] - k["start"] for k in iv["keep"]) / 60,
           "keep": [(round(k["start"], 2), round(k["end"], 2)) for k in iv["keep"]],
           "drop": [(round(x["start"], 2), round(x["end"], 2), x["reason"]) for x in iv["drop"]],
           "clear_whale_frames": [int(lab[lab.whale].kept.sum()), int(lab.whale.sum())],
           "whale_free_frames": [int(lab[lab.whale_free].kept.sum()), int(lab.whale_free.sum())],
           "whale_free_by_state": lab[lab.whale_free].groupby("state").kept.agg(["sum", "count"]).astype(int)
           .to_dict("index")}
    # picture water state per window against the labels' state (frames)
    lab["water"] = lab.water.fillna("unknown")
    out["water_vs_label_state"] = pd.crosstab(lab.state, lab.water).to_dict("index")
    out["windows_whale"] = int(win.whale.sum())
    out["windows_scanned"] = int(win.scanned.sum())
    out["windows"] = int(len(win))
    return out


def m4v_crosscheck() -> dict:
    """WP5a's cached .m4v window decisions (strict rule) against this run's .mov decisions, on the windows of the
    same labelled frames that the scan scanned."""
    out = {}
    for label in ("strict+nonear+px0",):   # the WP5a settings: no size limit, whole-frame boxes dropped
        d = SCAN / DEP / label
        if not (d / "windows").exists():
            continue
        agree = n = only_m4v = only_mov = 0
        for p in LABELLED:
            st = stem_of(p)
            f = paths.cache_dir("detect") / st / "owl_owl_all_s15" / "test_windows.csv"
            if not f.exists():
                continue
            wp5 = pd.read_csv(f)
            tab = video.frame_table(VD / f"{st}.mov")
            win = pd.read_csv(d / "windows" / f"{st}.mov.csv").set_index("k")
            for r in wp5.itertuples():
                frame = json.loads(r.frames)[0]
                k = int(np.floor(tab.times[min(frame, len(tab.times) - 1)]))
                if k not in win.index or not bool(win.loc[k, "scanned"]):
                    continue
                a = (r.score if np.isfinite(r.score) else 0) >= 0.25
                b = bool(win.loc[k, "whale"])
                n += 1
                agree += a == b
                only_m4v += a and not b
                only_mov += b and not a
        if n:
            out[label] = {"labelled_windows_compared": n, "agree": agree, "fires_m4v_only": only_m4v,
                          "fires_mov_only": only_mov}
    return out


D2 = "TestSubset 2025-1-2 CATS -85"


def d2_labels_ready() -> tuple[bool, list]:
    vd = paths.footage_dir() / D2 / "Video"
    stems = sorted(p.stem for p in vd.glob("*.mov"))
    missing = [s for s in stems if not (paths.cache_dir("reference") / s / "DONE").exists()
               or not (LABELS / s / "frames.csv").exists()]
    return not missing, stems


def evaluate_d2(label: str = "sensitive") -> dict:
    """Second deployment against WP3b's AI by-eye labels: tag / water state per frame, activity kept, grade against
    tier, and what the scan does over seafloor and blocked (sand / dark) spans."""
    ok, stems = d2_labels_ready()
    if not ok:
        return {"not_ready": True}
    d = SCAN / D2 / label
    ivs = kept_intervals(d)
    allf, spans_sf, spans_bl, wins = [], [], [], {}
    for st in stems:
        labels = ep.load_frame_labels(LABELS / st / "frames.csv")
        win = pd.read_csv(d / "windows" / f"{st}.mov.csv")
        wins[st] = win
        tab = video.frame_table(paths.footage_dir() / D2 / "Video" / f"{st}.mov")
        lab = frames_vs_keep(labels, tab.times, ivs[st]["keep"], win).assign(piece=st[-5:])
        allf.append(label_window_columns(lab, win))
        for name, lst in (("seafloor_spans.csv", spans_sf), ("blocked_spans.csv", spans_bl)):
            f = LABELS / st / name
            if f.exists():
                lst.append(pd.read_csv(f))
    lab = pd.concat(allf, ignore_index=True)
    res = {"label": label, "n_frames": int(len(lab))}
    # tag state / water: label state air on the boat recordings, underwater / air / crossing elsewhere
    res["state_vs_scan"] = pd.crosstab(lab.state, lab.w_water.fillna("unknown")).astype(int).to_dict("index")
    res["state_vs_tag_state"] = pd.crosstab(lab.state, lab.w_tag_state.fillna("unknown")).astype(int).to_dict("index")
    res["activity_tiers"] = activity_and_tiers(lab)
    res["whale_frames"] = {"clear_kept": [int(lab[lab.whale].kept.sum()), int(lab.whale.sum())],
                           "by_grade": {g: [int(x.kept.sum()), int(len(x))] for g, x in lab[lab.whale].groupby("grade")}}
    res["whale_free_underwater_kept_by_why"] = lab[lab.whale_free & ~lab.fish_present & (lab.state == "underwater")
                                                   & lab.kept].why.value_counts().to_dict()
    # labelled views
    for v in ("sand", "dark", "bubbles"):
        m = lab.view.astype(str).eq(v)
        if m.any():
            res[f"view_{v}"] = {"frames": int(m.sum()), "kept": int(lab[m].kept.sum()),
                                "tier": lab[m].tier.value_counts().to_dict(),
                                "water_picture": lab[m].w_water_picture.value_counts().to_dict()}
    # spans: per second window inside each span kind
    def span_stats(spans: pd.DataFrame, kind_col: str) -> dict:
        out = {}
        for (kind), g in spans.groupby(kind_col):
            rows = []
            for r in g.itertuples():
                w = wins[r.recording]
                rows.append(w[(w.t0 >= r.start_s) & (w.t1 <= r.end_s)])
            w = pd.concat(rows) if rows else pd.DataFrame()
            if w.empty:
                continue
            sc = w[w.scanned]
            out[str(kind)] = {"windows": int(len(w)), "scanned": int(len(sc)),
                              "whale_fires": int(sc.whale.sum()), "small_fires": int(sc.small.sum()),
                              "fish_fires": int(sc.fish.sum()) if "fish" in sc else None,
                              "kept": int(w.kept.sum()), "hq_whale": int(w.hq_whale.sum()),
                              "tier": w.tier.value_counts().to_dict() if "tier" in w else None,
                              "water_picture": w.water_picture.value_counts().to_dict(),
                              "bubbles_ge_0.3_share": int((sc.bubbles.fillna(0) > 0).sum())}
        return out
    if spans_sf:
        # the labeller's seafloor spans are a whale back with haze above it (user, 2026-11-23): void
        res["seafloor_spans"] = "labels withdrawn 2026-11-23"
    if spans_bl:
        res["blocked_spans"] = span_stats(pd.concat(spans_bl), "kind")
    summ = json.loads((d / "summary.json").read_text())
    res["minutes"] = {"total": summ["seconds"] / 60, "kept": summ["kept_s"] / 60,
                      "dropped": {k: v / 60 for k, v in summ["dropped_s"].items()},
                      "tier_window_min": {k: v / 60 for k, v in summ.get("tier_window_s", {}).items()},
                      "clip_min_by_tier": {k: v / 60 for k, v in summ.get("clip_s_by_tier", {}).items()},
                      "n_clips_by_tier": summ.get("n_clips_by_tier")}
    res["lost_activity"] = res["activity_tiers"]["activity_lost"]
    return res


def main_d2():
    ok, stems = d2_labels_ready()
    if not ok:
        print("labels not finished; run again when every underwater_cache/reference/<stem>/DONE exists")
        return
    OUT.mkdir(exist_ok=True)
    res = {lb: evaluate_d2(lb) for lb in ("sensitive", "strict", "sensitive+noclf+guard9", "sensitive+guard9",
                                           "sensitive+noclf") if (SCAN / D2 / lb / "intervals.json").exists()}
    (OUT / "eval_d2.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:12000])


def main():
    OUT.mkdir(exist_ok=True)
    cache = paths.cache_dir("scan") / DEP
    res = {"index_mapping": check_index_mapping(cache)}
    labels = sorted(p.name for p in (SCAN / DEP).iterdir() if (p / "intervals.json").exists())
    res["runs"] = {lb: evaluate_label(lb) for lb in labels}
    if (SCAN / DEV).exists():
        res["dev"] = {lb.name: evaluate_dev(lb.name) for lb in (SCAN / DEV).iterdir()
                      if (lb / "intervals.json").exists()}
    res["m4v_crosscheck"] = m4v_crosscheck()
    (OUT / "eval.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "tables_auto.md").write_text(markdown(res))
    for lb, r in res["runs"].items():
        if not r:
            continue
        fr = r["frames"]
        dl = r["deployment"]["labelled 00004-00010"]
        da = r["deployment"]["all pieces"]
        print(f"\n[{lb}] clear-whale frames kept {fr['clear_whale']['kept']}/{fr['clear_whale']['n']}, whale-free kept "
              f"{fr['whale_free']['kept']}/{fr['whale_free']['n']}, faint-only kept {fr['faint_only']['kept']}/"
              f"{fr['faint_only']['n']}")
        print(f"  lost by reason {r['lost_by_reason']}; kept whale-free by why {r['kept_whale_free_by_why']}")
        print(f"  clear whale kept by size {r.get('clear_whale_by_size')} by quality {r.get('clear_whale_by_quality')}")
        for name, dd in (("00004-00010", dl), ("all", da)):
            print(f"  {name}: kept {dd['kept_min']:.1f} of {dd['minutes']:.1f} min in {dd['n_clips']} stretches; "
                  + ", ".join(f"{k} {v:.1f}" for k, v in dd["dropped_min"].items()))
    for lb, r in res.get("dev", {}).items():
        print(f"\n[dev {lb}] kept {r['kept_min']:.2f}/{r['minutes']:.2f} min; whale frames {r['clear_whale_frames']}, "
              f"whale-free {r['whale_free_frames']}; drops {r['drop']}")
    print("\nindex mapping:", {p: v["same"] for p, v in res["index_mapping"].items()})
    print("m4v crosscheck:", res["m4v_crosscheck"])




def markdown(res: dict) -> str:
    """Tables for tables.md from eval.json."""
    L = []
    runs = res["runs"]
    L.append("| run | clear-whale frames kept | whale-free frames kept | faint-only kept | fish frames kept | "
             "kept min (00004-00010) | kept min (all 9) | hq_whale windows min (all 9) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lb, r in runs.items():
        if not r:
            continue
        f = r["frames"]
        dl, da = r["deployment"]["labelled 00004-00010"], r["deployment"]["all pieces"]
        L.append(f"| {lb} | {f['clear_whale']['kept']}/{f['clear_whale']['n']} | {f['whale_free']['kept']}/"
                 f"{f['whale_free']['n']} | {f['faint_only']['kept']}/{f['faint_only']['n']} | "
                 f"{f['fish_frames']['kept']}/{f['fish_frames']['n']} | {dl['kept_min']:.1f} of {dl['minutes']:.1f} | "
                 f"{da['kept_min']:.1f} of {da['minutes']:.1f} | {da.get('hq_min', float('nan')):.1f} |")
    L.append("")
    L.append("Minutes dropped by reason (all 9 recordings):")
    L.append("")
    L.append("| run | " + " | ".join(REASONS) + " |")
    L.append("|---|" + "---|" * len(REASONS))
    for lb, r in runs.items():
        if r:
            d = r["deployment"]["all pieces"]["dropped_min"]
            L.append(f"| {lb} | " + " | ".join(f"{d.get(k, 0):.1f}" for k in REASONS) + " |")
    L.append("")
    for lb in ("sensitive", "strict"):
        r = runs.get(lb)
        if not r:
            continue
        L.append(f"Per recording, {lb}:")
        L.append("")
        L.append("| piece | min | kept min | stretches | hq_whale min (windows) | " + " | ".join(REASONS) + " |")
        L.append("|---|---|---|---|---|" + "---|" * len(REASONS))
        for p, x in r["per_recording"].items():
            L.append(f"| {p} | {x['min']:.1f} | {x['kept_min']:.1f} | {x['clips']} | {x.get('hq_min', 0):.1f} | "
                     + " | ".join(f"{x['dropped_min'].get(k, 0):.1f}" for k in REASONS) + " |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    if "--dep" in sys.argv and sys.argv[sys.argv.index("--dep") + 1] == "d2":
        main_d2()
    else:
        main()
