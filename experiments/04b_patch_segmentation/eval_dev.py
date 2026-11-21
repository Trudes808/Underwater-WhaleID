"""Dev clip, held-out time blocks: single frame against the 1 s window, for masks and for whale presence.

    .venv/bin/python experiments/04b_patch_segmentation/eval_dev.py [--pred w1280_h0]

Input: underwater_cache/patchseg/pred_dev_<pred>.npz (every frame scored by the fold classifier of its own held-out
block). Writes results/dev_windows_<pred>.json and prints the tables.
- Masks (31 frames): the mask frame's own class map, against the average of the class maps at -15..+15 frames
  (7 frames at stride 5, about 1 s centred on the mask frame).
- Presence: frame score = largest whale region (patches with p(whale) > 0.5, 8-connected). Units: labelled frames
  (single frame) and 1 s grid windows (``uwid.eval_presence``), rules: max / 2nd highest frame (persistence) /
  largest whale region on the window-averaged map.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from uwid import eval_presence as ep
from uwid import patchseg as ps
import common as C

OUT = C.HERE / "results"


def presence_tables(frames, wins, probs, labels, fps, thr=0.5, fixed=None, keep=None):
    """Returns dict of rule -> {auc, op10 (operating point at false-alarm <= 10 %), at_fixed}."""
    size = np.array([ps.whale_region_size(probs[i].astype(np.float32), thr) for i in range(len(frames))])
    fs = pd.DataFrame({"frame": frames, "score": size})
    res = {}
    # single labelled frames
    ft = ep.frame_truth(labels).merge(fs, on="frame", how="left")
    units = {"frame": ft}
    wt = ep.window_truth(labels, fps)
    for rule, kw in [("max", {}), ("kth2", {"rule": "kth", "k": 2}), ("kth3", {"rule": "kth", "k": 3})]:
        ws = ep.window_scores(fs, fps, **({"rule": "max"} if rule == "max" else kw))
        units[f"win_{rule}"] = wt.merge(ws, on="window", how="left")
    # averaged class map per window
    rows = []
    for k in np.unique(wins):
        sel = wins == k
        rows.append({"window": int(k), "score": ps.whale_region_size(probs[sel].astype(np.float32).mean(0), thr)})
    units["win_meanmap"] = wt.merge(pd.DataFrame(rows), on="window", how="left")
    if keep is not None:
        keep.append(units)
    for name, tab in units.items():
        t = tab[tab.whale | tab.whale_free]
        curve = ep.roc(t.score, t.whale)
        op = ep.operating_point(curve, max_far=0.10)
        r = {"auc": ep.auc(curve), "n_whale": int(t.whale.sum()), "n_free": int(t.whale_free.sum()),
             "n_faint_only": int(tab.faint_only.sum()), "op_far10": {k: float(v) for k, v in op.items()}}
        # a fixed grid of region sizes (patches of 24 x 24 px at 1280 px input), not tuned on anything
        r["sweep"] = {int(th): {k: ep.summarize(tab, th)[k] for k in ("hits", "n_pos", "false_alarms", "n_neg")}
                      for th in (1, 4, 8, 16, 32, 64)}
        if fixed is not None and name in fixed:
            r["at_fixed"] = ep.summarize(tab, fixed[name])
        res[name] = r
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", default="w1280_h0")
    a = ap.parse_args()
    d = np.load(ps.CACHE / f"pred_dev_{a.pred}.npz")
    frames, wins, probs = list(d["frames"]), d["window"], d["probs"]
    _, _, mframes = C.load_dev(a.pred.split("_h")[0])
    single, win7 = C.Scorer(), C.Scorer()
    for f in mframes:
        cm, inst, meta = C.read_mask(C.DEV, f)
        p0 = torch.from_numpy(probs[frames.index(f)].astype(np.float32))
        single.add(C.pred_codes(p0), cm, inst, meta)
        idx = [frames.index(g) for g in range(f - 15, f + 16, 5) if g in frames]
        pw = torch.from_numpy(probs[idx].astype(np.float32).mean(0))
        win7.add(C.pred_codes(pw), cm, inst, meta)
    res = {"masks_single": single.summary(), "masks_window7": win7.summary()}
    labels = ep.load_frame_labels(C.REPO / "experiments/02_reference_labels/labels/2025-12-20_CG/frames.csv")
    res["presence"] = presence_tables(np.array(frames), wins, probs, labels, C.DEV_FPS)
    OUT.mkdir(exist_ok=True)
    (OUT / f"dev_windows_{a.pred}.json").write_text(json.dumps(res, indent=1, default=float))
    for k in ["masks_single", "masks_window7"]:
        s = res[k]
        print(k, {c: C.fmt(v, 3) for c, v in s["iou_band"].items()},
              {g: round(v["pixel_recall"], 3) for g, v in s["whale_recall_by_grade"].items()}, s["confusions"])
    for k, v in res["presence"].items():
        print(k, round(v["auc"], 3), v["n_whale"], v["n_free"], v["n_faint_only"],
              {x: v["op_far10"][x] for x in ["threshold", "tp", "fp", "recall", "far"]})


if __name__ == "__main__":
    main()
