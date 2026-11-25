"""WP11b measures from the scan tables (no fitting): clips, tiers, the veto's effect, the activity guard's effect.

    .venv/bin/python experiments/06_clips_e2e/wp11b_measures.py      # writes results/wp11b_measures.json

Labels compared: sensitive+noclf+guard9 (~ before: no veto, old-style merging), sensitive+guard9 (veto only),
sensitive+noclf (guard only), sensitive (both; the default), strict. Jacks / breath boxes: WP10's AI by-eye extra
and breath labels (those boxes were in the classifier's training set: not a clean test)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uwid import paths

HERE = Path(__file__).resolve().parent
SCAN = paths.cache_dir("scan")
DEPS = {"d1": "2025-2-3 Mini-mother 11-96", "d2": "TestSubset 2025-1-2 CATS -85"}
LABELS = ["sensitive+noclf+guard9", "sensitive+guard9", "sensitive+noclf", "sensitive", "strict"]
BOXCLF = Path("/home/sat3737/NOAA_whales/underwater_cache/boxclf")


def windows(dep, label):
    d = SCAN / dep / label / "windows"
    return pd.concat([pd.read_csv(f).assign(key=f.name[:-4]) for f in sorted(d.glob("*.csv"))], ignore_index=True)


def truthy(c):
    return c.astype(str).str.lower().eq("true")


def labelled_boxes(kind):
    lab = json.loads((HERE.parents[0] / "08_whale_fish_boxes" / f"{kind}_labels.json").read_text())["labels"]
    ch = pd.read_csv(BOXCLF / kind / "chosen.csv")
    ch["label"] = ch.id.astype(str).map(lab)
    return ch


def measure(dep, label):
    d = SCAN / dep / label
    if not (d / "intervals.json").exists():
        return None
    iv = json.loads((d / "intervals.json").read_text())
    w = windows(dep, label)
    clips = [(Path(v["source"]).stem, k) for v in iv["videos"] for k in v["keep"]]
    lens = np.array([k["end"] - k["start"] for _, k in clips]) if clips else np.zeros(0)
    quiet = sum(k.get("quiet_s", 0) for _, k in clips)
    wh = w[w.whale.astype(bool)]
    hq = w[w.hq_whale.astype(bool)]
    out = {"kept_min": round(float(lens.sum()) / 60, 1), "n_clips": int(len(lens)),
           "median_clip_s": round(float(np.median(lens)), 1) if len(lens) else None,
           "longest_clip_s": round(float(lens.max()), 1) if len(lens) else None,
           "clips_under_10s": int((lens < 10).sum()), "quiet_s_in_clips": int(quiet),
           "clips_by_tier": pd.Series([k["tier"] for _, k in clips]).value_counts().to_dict(),
           "advanced_ok_clips": int(sum(bool(k.get("advanced_ok")) for _, k in clips)),
           "hq_span_s": round(sum(b - a for _, k in clips for a, b in k.get("hq_spans", [])), 1),
           "continuing_marks": int(sum(("continues_in" in k) for _, k in clips)),
           "window_tiers": w.tier.value_counts().to_dict(), "whale_windows": int(len(wh)),
           "hq_windows": int(len(hq)), "hq_windows_under_1m": int((hq.depth_m < 1).sum()),
           "active_windows": int(w.active.astype(bool).sum())}
    if "vetoed_other_only" in w:
        v = w[w.vetoed_other_only.astype(bool)]
        out["vetoed_other_only_windows"] = int(len(v))
        out["veto_fish_windows"] = int((w.veto_fish_score.fillna(0) >= 0.25).sum())
        out["veto_other_windows"] = int((w.veto_other_score.fillna(0) >= 0.25).sum())
    for c in ("dark", "sand"):
        if c in w:
            m = truthy(w[c])
            out[f"{c}_windows"] = int(m.sum())
            out[f"{c}_windows_kept"] = int((m & w.kept.astype(bool)).sum())
            out[f"{c}_windows_hq"] = int((m & w.hq_whale.astype(bool)).sum())
    # WP10's by-eye boxes: tier of the window that holds the box's frame
    for kind, want in (("extra", "fish"), ("breath", "other")):
        try:
            ch = labelled_boxes(kind)
        except FileNotFoundError:
            continue
        ch = ch[(ch.label == want)]
        tiers = []
        for r in ch.itertuples():
            ww = w[w.key == f"{r.stem}.mov"]
            m = ww[(ww.frame_0 == r.frame) | (ww.frame_1 == r.frame)]
            if len(m):
                tiers.append(m.tier.iloc[0])
        if tiers:
            out[f"{kind}_{want}_boxes_window_tier"] = pd.Series(tiers).value_counts().to_dict()
    s = json.loads((d / "summary.json").read_text())
    out["timings"] = s.get("timings_this_run")
    return out


def made_inactive(dep, with_clf, without_clf):
    """Windows the veto itself made inactive: active without the classifier, not active with it (the flag
    ``vetoed_other_only`` also counts windows the shallow rule had already emptied), and how many of them are still
    inside a kept clip."""
    a, b = windows(dep, without_clf), windows(dep, with_clf)
    m = a[["key", "k", "active"]].merge(b[["key", "k", "active", "kept", "why", "clock", "t0"]], on=["key", "k"],
                                         suffixes=("_noclf", ""))
    x = m[m.active_noclf.astype(bool) & ~m.active.astype(bool)]
    flag = b[b.vetoed_other_only.astype(bool)] if "vetoed_other_only" in b else b.iloc[:0]
    return {"made_inactive_windows": int(len(x)), "still_inside_a_clip": int(x.kept.astype(bool).sum()),
            "no_longer_kept": int((~x.kept.astype(bool)).sum()), "flag_vetoed_other_only": int(len(flag)),
            "no_longer_kept_list": x[~x.kept.astype(bool)][["key", "k", "clock"]].to_dict("records")}


res = {k: {lb: measure(dep, lb) for lb in LABELS} for k, dep in DEPS.items()}
for k, dep in DEPS.items():
    res[k]["veto_effect_default"] = made_inactive(dep, "sensitive", "sensitive+noclf")
    res[k]["veto_effect_guard9"] = made_inactive(dep, "sensitive+guard9", "sensitive+noclf+guard9")
(HERE / "results" / "wp11b_measures.json").write_text(json.dumps(res, indent=1, default=str))
for k, r in res.items():
    for lb in ("veto_effect_default", "veto_effect_guard9"):
        print(k, lb, {x: v for x, v in r[lb].items() if x != "no_longer_kept_list"})
    for lb, m in r.items():
        if m and lb in LABELS:
            print(k, lb, {x: m[x] for x in ("kept_min", "n_clips", "median_clip_s", "longest_clip_s", "clips_under_10s",
                                            "quiet_s_in_clips", "hq_windows", "hq_windows_under_1m", "whale_windows")},
                  m.get("vetoed_other_only_windows"), m.get("extra_fish_boxes_window_tier"),
                  m.get("breath_other_boxes_window_tier"), m.get("dark_windows"), m.get("sand_windows"),
                  m.get("dark_windows_hq"), m.get("sand_windows_hq"), m["window_tiers"])
