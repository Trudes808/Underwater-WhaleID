"""Development video at the chosen operating point: windows vs single frames at the same threshold, misses and
false alarms with their causes, the back returned as a whale, and the fish prompts (no fish on this clip: every
fish box is a false alarm).

    .venv/bin/python experiments/04_whale_fish_detection/report_dev.py --run owl_owl_all_s5 --query whale --thr 0.25

The operating point (query, 2 frames per window, both frames >= thr) was chosen on this video by
``operating_point.py``. Writes ``report_dev.json`` and CSVs into the run's cache folder.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import (DEV_LABELS, DEV_VIDEO, FISH_Q, H, box_cause, detect, ep, fps_of, frame_scores, load_masks,
                    mask_frames, static_bottom, tag_boxes, to_md)

DROP = ("clock", "whole")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="owl_owl_all_s5")
    ap.add_argument("--video", default=DEV_VIDEO)
    ap.add_argument("--query", default="whale")
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=15, help="frames used per window: every stride-th frame")
    ap.add_argument("--fish-thr", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    a = ap.parse_args()
    labels = ep.load_frame_labels(DEV_LABELS)
    fps = fps_of(a.video)
    b, frames = detect.load_cached(a.video, a.run)
    b = tag_boxes(b)
    out_dir = detect.detect_cache_dir(a.video, a.run)
    rep = dict(run=a.run, query=a.query, thr=a.thr, stride=a.stride)
    q = [a.query]

    # --- windows vs single frames at the same threshold
    fr = frames[frames % a.stride == 0]
    fs = frame_scores(b, fr, q, DROP)
    truth = ep.window_truth(labels, fps)
    both = truth.merge(ep.window_scores(fs, fps, rule="kth", k=2), on="window", how="left")
    anyf = truth.merge(ep.window_scores(fs, fps, rule="max"), on="window", how="left")
    lab = labels.merge(frame_scores(b, frames, q, DROP), on="frame", how="left")
    lab["faint_only"] = ~lab.whale & lab.faint
    groups_w = {"bubbles": both.bubbles & both.whale, "clear view": ~both.bubbles & both.whale,
                "1 whale": (both.max_whales == 1) & both.whale, ">=2 whales": (both.max_whales >= 2) & both.whale,
                "underwater negatives": both.whale_free & (both.state == "underwater")}
    rep["window_both"] = ep.summarize(both, a.thr, groups=groups_w)
    rep["window_any"] = ep.summarize(anyf, a.thr, groups={k: v.reindex(anyf.index) for k, v in groups_w.items()})
    groups_f = {"bubbles": lab.bubbles & lab.whale, "clear view": ~lab.bubbles & lab.whale,
                "1 whale": (lab.other_whales == 1), ">=2 whales": (lab.other_whales >= 2),
                "underwater negatives": lab.whale_free & (lab.state == "underwater")}
    rep["single_frame"] = ep.summarize(lab, a.thr, groups=groups_f)
    for k in ("window_both", "window_any", "single_frame"):
        r = rep[k]
        print(f"{k:13s}: hits {r['hits']}/{r['n_pos']}  false alarms {r['false_alarms']}/{r['n_neg']}  "
              f"faint-only fired {r['fire_other']}/{r['n_other']}  | bubbles {r['hits_bubbles']}/{r['n_bubbles']} "
              f"clear {r['hits_clear view']}/{r['n_clear view']}  1 whale {r['hits_1 whale']}/{r['n_1 whale']}  "
              f">=2 {r['hits_>=2 whales']}/{r['n_>=2 whales']}")
    both.to_csv(out_dir / "op_windows_dev.csv", index=False)
    lab.to_csv(out_dir / "op_frames_dev.csv", index=False)
    rep["missed_windows"] = both[both.whale & (both.score.fillna(0) < a.thr)][["window", "max_whales", "bubbles", "score"]].to_dict("records")
    rep["false_alarm_frames"] = lab[lab.whale_free & (lab.score >= a.thr)][["frame", "state", "view", "score"]].to_dict("records")
    print("missed windows:", rep["missed_windows"])
    print("single-frame false alarms:", rep["false_alarm_frames"])

    # --- what the boxes >= thr on whale-free labelled frames sit on (all processed frames of whale-free windows)
    wf_windows = set(both.window[both.whale_free])
    sel = b[b["query"].isin(q) & (b.score >= a.thr)].copy()
    sel["window"] = ep.window_of(sel.frame, fps)
    fa = sel[sel.window.isin(wf_windows)]
    rep["boxes_in_whalefree_windows"] = dict(total=int(len(fa)), clock=int(fa.clock.sum()), whole=int(fa.whole.sum()),
                                             bottom_zone=int((fa.bottom & ~fa.whole).sum()),
                                             other=int((~fa.clock & ~fa.whole & ~fa.bottom).sum()),
                                             frames=sorted(map(int, fa.frame.unique())))
    print("boxes >= thr in whale-free windows (all processed frames):", rep["boxes_in_whalefree_windows"])

    # --- the back returned as a whale: bottom-zone boxes over the whole video
    wq = b[b["query"].isin(q) & ~b.clock & ~b.whole]
    st = static_bottom(wq, fps)
    bz = wq[wq.bottom & (wq.score >= a.thr)]
    rep["bottom_zone_boxes"] = dict(n=int(len(bz)), frames=int(bz.frame.nunique()), of_frames=int(len(frames)),
                                    static=int(st[bz.index].sum()),
                                    low_whale_boxes_touching_bottom=int(((wq.y1 >= H - 12) & (wq.score >= a.thr)).sum()))
    # on mask frames: boxes >= thr whose pixels are >= 50 % back
    on_back, mf_boxes = [], 0
    causes = []
    for f in mask_frames():
        cls, _, _ = load_masks(f)
        for _, r in wq[(wq.frame == f) & (wq.score >= a.thr)].iterrows():
            mf_boxes += 1
            c = box_cause(cls, r)
            causes.append(dict(frame=f, score=float(r.score), cause=c, bottom=bool(r.bottom)))
    cz = pd.DataFrame(causes)
    rep["mask_frame_boxes_by_cause"] = cz.groupby("cause").size().to_dict() if len(cz) else {}
    rep["mask_frame_back_boxes"] = cz[cz.cause == "back"].to_dict("records") if len(cz) else []
    print("bottom-zone boxes >= thr:", rep["bottom_zone_boxes"])
    print("mask-frame boxes >= thr by what they sit on:", rep["mask_frame_boxes_by_cause"])

    # --- fish prompts: every box is a false alarm on this clip
    dur_min = (frames.max() + 1) / fps / 60
    fish_rows = []
    fb = b[b["query"].isin(FISH_Q) & ~b.clock & ~b.whole]
    for qq in FISH_Q + ["any fish prompt"]:
        sub = fb if qq == "any fish prompt" else fb[fb["query"] == qq]
        for t in a.fish_thr:
            s = sub[sub.score >= t]
            s15 = s[s.frame % a.stride == 0]
            w = ep.window_of(s15.frame, fps)
            fish_rows.append(dict(prompt=qq, thr=t, boxes=len(s), frames_with_box=int(s.frame.nunique()),
                                  frames_processed=int(len(frames)),
                                  windows_with_box_2pw=int(len(set(w))),
                                  windows_both_frames=int(pd.Series(w).value_counts().ge(2).sum()),
                                  windows_per_min_both=round(pd.Series(w).value_counts().ge(2).sum() / dur_min, 2)))
    fdf = pd.DataFrame(fish_rows)
    print("\n### fish prompts on the dev clip (no fish in it: every box is a false alarm)\n")
    print(to_md(fdf))
    # what fish boxes sit on (mask frames), and whales called fish
    fc = []
    for f in mask_frames():
        cls, _, _ = load_masks(f)
        for _, r in fb[(fb.frame == f) & (fb.score >= 0.2)].iterrows():
            fc.append(dict(prompt=r["query"], cause=box_cause(cls, r), score=float(r.score)))
    fc = pd.DataFrame(fc)
    if len(fc):
        print("\nfish boxes >= 0.2 on the 31 mask frames, by what they sit on:\n")
        print(to_md(fc.groupby(["prompt", "cause"]).size().rename("boxes").reset_index()))
    rep["fish"] = fish_rows
    rep["fish_causes_maskframes"] = fc.groupby(["prompt", "cause"]).size().rename("boxes").reset_index().to_dict("records") if len(fc) else []
    (out_dir / "report_dev.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
