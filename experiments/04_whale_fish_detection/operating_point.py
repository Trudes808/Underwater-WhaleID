"""Choose the operating point on the development video, then list what it finds, misses and fires on.

    .venv/bin/python experiments/04_whale_fish_detection/operating_point.py --run owl_owl_all_s5

Settings compared (all chosen on the dev video only): prompt in {whale, humpback whale}, frames per window in
{2, 3, 6}, rule in {window max, all-but-one frames (k = n-1, min 2), max of persistent boxes}. The threshold is the
strictest one with no false alarm on the dev whale-free windows that keeps the best recall ("FA0"), and the loosest
with at most 10 % false alarms ("FA10"). Writes ``op_dev.json`` and per-window tables to the run's cache folder.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import (DEV_LABELS, DEV_VIDEO, detect, ep, fps_of, frame_scores, persistent_scores, tag_boxes, to_md)

DROP = ("clock", "whole")


def window_table(b, frames, fps, labels, queries, stride, rule):
    fr = np.asarray(frames)
    fr = fr[fr % stride == 0]
    if rule == "persistent max":
        ws = ep.window_scores(persistent_scores(b, fr, queries, DROP), fps, rule="max")
    elif rule == "max":
        ws = ep.window_scores(frame_scores(b, fr, queries, DROP), fps, rule="max")
    elif rule == "all-but-one":
        n = int(round(30 / stride))
        ws = ep.window_scores(frame_scores(b, fr, queries, DROP), fps, rule="kth", k=max(2, n - 1))
    else:
        raise ValueError(rule)
    return ep.window_truth(labels, fps).merge(ws, on="window", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="owl_owl_all_s5")
    ap.add_argument("--video", default=DEV_VIDEO)
    args = ap.parse_args()
    labels = ep.load_frame_labels(DEV_LABELS)
    b, frames = detect.load_cached(args.video, args.run)
    b = tag_boxes(b)
    fps = fps_of(args.video)
    rows = []
    for q in ["whale", "humpback whale"]:
        lab = labels.merge(frame_scores(b, frames, [q], DROP), on="frame", how="left")
        m = lab.whale | lab.whale_free
        c = ep.roc(lab.score[m], lab.whale[m])
        o0, o10 = ep.operating_point(c, max_far=0.0), ep.operating_point(c, max_far=0.10)
        rows.append(dict(prompt=q, frames_per_window=1, rule="single frame", n_pos=o0["n_pos"], n_neg=o0["n_neg"],
                         auc=ep.auc(c), thr_FA0=o0["threshold"], recall_FA0=o0["recall"],
                         thr_FA10=o10["threshold"], recall_FA10=o10["recall"], fp_FA10=o10["fp"]))
        for stride in (15, 10, 5):
            for rule in ("max", "all-but-one", "persistent max"):
                t = window_table(b, frames, fps, labels, [q], stride, rule)
                mw = t.whale | t.whale_free
                c = ep.roc(t.score[mw], t.whale[mw])
                o0, o10 = ep.operating_point(c, max_far=0.0), ep.operating_point(c, max_far=0.10)
                rows.append(dict(prompt=q, frames_per_window=int(round(30 / stride)), rule=rule, n_pos=o0["n_pos"],
                                 n_neg=o0["n_neg"], auc=ep.auc(c), thr_FA0=o0["threshold"],
                                 recall_FA0=o0["recall"], thr_FA10=o10["threshold"], recall_FA10=o10["recall"],
                                 fp_FA10=o10["fp"]))
    d = pd.DataFrame(rows)
    out_dir = detect.detect_cache_dir(args.video, args.run)
    d.to_csv(out_dir / "op_compare_dev.csv", index=False)
    print("### settings (dev video; window units = 1 s windows; single frame = the 191 labelled frames)\n")
    print(to_md(d))


if __name__ == "__main__":
    main()
