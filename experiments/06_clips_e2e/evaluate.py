"""WP7 measures from the `uwid scan` outputs (nothing is fitted or tuned here).

    .venv/bin/python experiments/06_clips_e2e/evaluate.py            # all run labels found under the scan folder

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
        allf.append(lab)
    lab = pd.concat(allf, ignore_index=True)
    lab["faint_only"] = ~lab.whale & lab.faint

    def share(m):
        return {"kept": int(lab[m].kept.sum()), "n": int(m.sum())}

    res["frames"] = {"clear_whale": share(lab.whale), "whale_free": share(lab.whale_free),
                     "faint_only": share(lab.faint_only),
                     "whale_free_underwater": share(lab.whale_free & (lab.state == "underwater")),
                     "fish_frames": share(lab.fish_present)}
    res["frames_per_piece"] = {p: {"clear_whale": [int(g[g.whale].kept.sum()), int(g.whale.sum())],
                                   "whale_free": [int(g[g.whale_free].kept.sum()), int(g.whale_free.sum())]}
                               for p, g in lab.groupby("piece")}
    lost = lab[lab.whale & ~lab.kept]
    res["lost_whale_frames"] = lost[["piece", "frame", "t_mov", "why", "whale_score", "other_whales", "note"]] \
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
            "dropped_min": {k: sum(r["dropped_s"][k] for r in recs) / 60 for k in REASONS},
            "windows": sum(r["windows"] for r in recs), "windows_scanned": sum(r["windows_scanned"] for r in recs),
            "whale_windows": sum(r["whale_windows"] for r in recs)}
    res["per_recording"] = {r["recording"][-5:]: {"min": r["seconds"] / 60, "kept_min": r["kept_s"] / 60,
                                                   "clips": r["n_clips"],
                                                   "dropped_min": {k: round(v / 60, 2) for k, v in r["dropped_s"].items()}}
                            for r in summ["recordings"]}
    res["offset_check"] = summ.get("offset_check")
    res["timings"] = summ.get("timings_this_run")
    return res


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
    for label in ("strict",):
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
        for name, dd in (("00004-00010", dl), ("all", da)):
            print(f"  {name}: kept {dd['kept_min']:.1f} of {dd['minutes']:.1f} min in {dd['n_clips']} stretches; "
                  + ", ".join(f"{k} {v:.1f}" for k, v in dd["dropped_min"].items()))
    for lb, r in res.get("dev", {}).items():
        print(f"\n[dev {lb}] kept {r['kept_min']:.2f}/{r['minutes']:.2f} min; whale frames {r['clear_whale_frames']}, "
              f"whale-free {r['whale_free_frames']}; drops {r['drop']}")
    print("\nindex mapping:", {p: v["same"] for p, v in res["index_mapping"].items()})
    print("m4v crosscheck:", res["m4v_crosscheck"])


if __name__ == "__main__":
    main()
