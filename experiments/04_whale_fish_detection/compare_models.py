"""Whale presence of several runs on the SAME dev frames (the frames every run processed), single frame vs window.

    .venv/bin/python experiments/04_whale_fish_detection/compare_models.py \
        --runs owl_owl_all_s5:whale llmdet_llmdet_A_s10:whale llmdet_llmdet_B_s10:"humpback whale" sam3_sam3_all_s15:whale
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import DEV_LABELS, DEV_VIDEO, detect, ep, fps_of, frame_scores, tag_boxes, to_md

DROP = ("clock", "whole")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run:query")
    a = ap.parse_args()
    labels = ep.load_frame_labels(DEV_LABELS)
    fps = fps_of(DEV_VIDEO)
    loaded = {}
    common = None
    for rq in a.runs:
        run, q = rq.split(":", 1)
        b, fr = detect.load_cached(DEV_VIDEO, run)
        loaded[rq] = (tag_boxes(b), fr, q)
        common = set(fr) if common is None else common & set(fr)
    common = np.array(sorted(common))
    truth = ep.window_truth(labels, fps)
    mw = truth.whale | truth.whale_free
    lab = labels[labels.frame.isin(common)]
    rows = []
    for rq, (b, _, q) in loaded.items():
        fs = frame_scores(b, common, [q], DROP)
        l2 = lab.merge(fs, on="frame", how="left")
        m = l2.whale | l2.whale_free
        c = ep.roc(l2.score[m], l2.whale[m])
        o = ep.operating_point(c, max_far=0.0)
        rows.append(dict(run=rq, units=f"labelled frames ({int(m.sum())})", rule="single frame", auc=ep.auc(c),
                         recall_FA0=f"{o['tp']}/{o['n_pos']}", thr_FA0=o["threshold"]))
        n = int(np.median(pd.Series(ep.window_of(common, fps)).value_counts()))
        for rule, kw in (("max", dict(rule="max")), (f"all {n} frames", dict(rule="kth", k=n)),
                         (f"{max(n - 1, 1)} of {n}", dict(rule="kth", k=max(n - 1, 1)))):
            t = truth.merge(ep.window_scores(fs, fps, **kw), on="window", how="left")
            c = ep.roc(t.score[mw], t.whale[mw])
            o = ep.operating_point(c, max_far=0.0)
            rows.append(dict(run=rq, units=f"windows ({int(mw.sum())}, {n} frames each)", rule=rule, auc=ep.auc(c),
                             recall_FA0=f"{o['tp']}/{o['n_pos']}", thr_FA0=o["threshold"]))
    print(f"common frames: {len(common)}")
    print(to_md(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()
