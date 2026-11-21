"""Whale presence on the development video: single frame against the 1 s window rules, per run / prompt / filter.

    .venv/bin/python experiments/04_whale_fish_detection/analyze_presence.py --run owl_owl_all_s5 [--video ...]

Writes ``underwater_cache/detect/<video>/<run>/presence_*.csv`` and prints markdown tables. Only the development
video is used to choose prompts, filters, rules and thresholds.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import (DEV_LABELS, DEV_VIDEO, WHALE_Q, detect, ep, fps_of, frame_scores, persistent_scores, to_md,
                    static_bottom, tag_boxes)

FILTERS = {
    "clock": ("clock",),
    "clock+whole": ("clock", "whole"),
    "clock+whole+bottom": ("clock", "whole", "bottom"),
    "clock+whole+static": ("clock", "whole", "static"),
}


def prepare(video, run):
    b, frames = detect.load_cached(video, run)
    fps = fps_of(video)
    b = tag_boxes(b)
    b["static"] = False
    wq = b["query"].isin(WHALE_Q + ["humpback whale. ", "whale tail"]) | b["query"].str.contains("whale")
    b.loc[wq, "static"] = static_bottom(b[wq], fps).values
    return b, frames, fps


def curve_stats(scores, positive, tag):
    c = ep.roc(scores, positive)
    op10 = ep.operating_point(c, max_far=0.10)
    op0 = ep.operating_point(c, max_far=0.0)
    op90 = ep.operating_point(c, min_recall=0.90)
    return dict(units=tag, n_pos=int(np.sum(positive)), n_neg=int(np.sum(~np.asarray(positive))),
                auc=round(ep.auc(c), 3),
                recall_far10=round(op10["recall"], 3), thr_far10=round(op10["threshold"], 3),
                fp_far10=int(op10["fp"]),
                recall_far0=round(op0["recall"], 3), thr_far0=round(op0["threshold"], 3),
                far_at_recall90=round(op90["far"], 3), thr_recall90=round(op90["threshold"], 3))


def evaluate(b, frames, fps, labels, queries, drop, stride_sub=None):
    """Rows of stats for the single-frame and window scorings of one query set / filter."""
    fr = np.asarray(frames)
    out = []
    # single frame: the labelled frames only (independent of the frames-per-window setting)
    lab = labels.merge(frame_scores(b, fr, queries, drop), on="frame", how="left")
    m = lab.whale | lab.whale_free
    out.append(dict(rule="single frame", **curve_stats(lab.score[m], lab.whale[m], "labelled frames")))
    if stride_sub:
        fr = fr[fr % stride_sub == 0]
    fs = frame_scores(b, fr, queries, drop)
    # persistent: a box counts only if an overlapping box is in the previous or next processed frame (which may lie
    # in the neighbouring window); the box itself must be inside the window
    ps = persistent_scores(b, fr, queries, drop)
    truth = ep.window_truth(labels, fps)
    mw = truth.whale | truth.whale_free
    for rule, kw, src in [("window max", dict(rule="max"), fs), ("window 2nd-best", dict(rule="kth", k=2), fs),
                          ("window 3rd-best", dict(rule="kth", k=3), fs),
                          ("window median", dict(rule="frac", k=0.5), fs),
                          ("window max of persistent", dict(rule="max"), ps),
                          ("window 2nd-best of persistent", dict(rule="kth", k=2), ps)]:
        ws = ep.window_scores(src, fps, **kw)
        t = truth.merge(ws, on="window", how="left")
        out.append(dict(rule=rule, **curve_stats(t.score[mw], t.whale[mw], "1 s windows")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEV_VIDEO)
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", default=str(DEV_LABELS))
    args = ap.parse_args()
    labels = ep.load_frame_labels(args.labels)
    b, frames, fps = prepare(args.video, args.run)
    queries_avail = sorted(set(b["query"]))
    wq = [q for q in queries_avail if "whale" in q and q not in ("whale skin", "whale back")]
    out_dir = detect.detect_cache_dir(args.video, args.run)
    rows = []
    # 1) prompt wording (fixed filter clock+whole, all rules)
    for q in wq + ["any whale prompt"]:
        qs = wq if q == "any whale prompt" else [q]
        for r in evaluate(b, frames, fps, labels, qs, FILTERS["clock+whole"]):
            rows.append(dict(table="prompt", query=q, filter="clock+whole", per_window=6, **r))
    # 2) filters, with the main prompt
    main_q = ["whale"] if "whale" in wq else wq[:1]
    for fname, drop in FILTERS.items():
        for r in evaluate(b, frames, fps, labels, main_q, drop):
            rows.append(dict(table="filter", query=main_q[0], filter=fname, per_window=6, **r))
    # 3) frames per window
    stride = json.loads((out_dir / "meta.json").read_text()).get("stride") or 5
    for sub, n in [(stride, int(round(30 / stride))), (10, 3), (15, 2), (30, 1)]:
        if sub % stride:
            continue
        for r in evaluate(b, frames, fps, labels, main_q, FILTERS["clock+whole"], stride_sub=sub):
            rows.append(dict(table="frames_per_window", query=main_q[0], filter="clock+whole", per_window=n, **r))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "presence_dev.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    for t, g in df.groupby("table", sort=False):
        print(f"\n### {args.run}: {t}\n")
        print(to_md(g.drop(columns=["table"])))


if __name__ == "__main__":
    main()
