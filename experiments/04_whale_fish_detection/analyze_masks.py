"""Mask frames of the development video: instances found by grade / situation, what the unmatched boxes sit on,
SAM 2.1 mask overlap for the found ones, and the back as a class.

    # 1) SAM masks for the boxes (GPU, cached):
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04_whale_fish_detection/analyze_masks.py --run owl_owl_all_s5 --sam
    # 2) tables (CPU):
    .venv/bin/python experiments/04_whale_fish_detection/analyze_masks.py --run owl_owl_all_s5 --query whale --thr 0.42
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import (CLASS_IDS, DEV_VIDEO, box_cause, detect, instances, iou_one, load_masks, mask_frames,
                    match_instances, tag_boxes, to_md)

SAM_MIN = 0.12      # boxes at or above this score get a SAM mask (covers every threshold we consider)
BACK_Q = ["whale skin", "whale back"]


def sam_cache(run):
    return detect.detect_cache_dir(DEV_VIDEO, run) / "sam_maskframes.npz"


def run_sam(run, queries):
    b, _ = detect.load_cached(DEV_VIDEO, run)
    b = tag_boxes(b)
    mf = mask_frames()
    sam = detect.SamMasker()
    imgs = dict(detect.iter_frames_cv(DEV_VIDEO, frames=mf))
    store = {}
    for f in mf:
        sel = b[(b.frame == f) & b["query"].isin(queries) & (b.score >= SAM_MIN) & ~b.clock]
        if sel.empty:
            continue
        ms = sam.masks(imgs[f], sel[["x0", "y0", "x1", "y1"]].to_numpy())
        for idx, m in zip(sel.index, ms):
            store[f"{f}_{idx}"] = np.packbits(m)
    np.savez_compressed(sam_cache(run), **store)
    print("SAM masks:", len(store))


def unpack(store, key):
    return np.unpackbits(store[key])[: 1920 * 1080].reshape(1080, 1920).astype(bool)


def mask_iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--query", default="whale", help="whale query (or 'any' for every whale prompt)")
    ap.add_argument("--thr", type=float, default=0.42)
    ap.add_argument("--sam", action="store_true")
    args = ap.parse_args()
    b, _ = detect.load_cached(DEV_VIDEO, args.run)
    qs_all = sorted(set(b["query"]))
    whale_qs = [q for q in qs_all if "whale" in q and q not in BACK_Q]
    if args.sam:
        run_sam(args.run, qs_all)
        return
    qs = whale_qs if args.query == "any" else [args.query]
    b = tag_boxes(b)
    store = np.load(sam_cache(args.run)) if sam_cache(args.run).exists() else None
    rows, boxes_rows = [], []
    for f in mask_frames():
        cls, inst, meta = load_masks(f)
        ins = [i for i in instances(f) if i["cls"] == "whale"]
        sel = b[(b.frame == f) & b["query"].isin(qs) & (b.score >= args.thr) & ~b.clock & ~b.whole]
        matched, unmatched = match_instances(ins, sel)
        for i in ins:
            bi = matched.get(i["id"])
            miou = np.nan
            if bi is not None and store is not None and f"{f}_{bi}" in store:
                miou = mask_iou(unpack(store, f"{f}_{bi}"), i["mask"])
            rows.append(dict(frame=f, id=i["id"], grade=i["grade"], cut=i["cut"], bubbles=i["behind_bubbles"],
                             small=i["small"], area=i["area"], found=bi is not None,
                             box_iou=float(iou_one(sel.loc[bi, ["x0", "y0", "x1", "y1"]].to_numpy(float),
                                                   i["box"][None])[0]) if bi is not None else np.nan,
                             mask_iou=miou))
        for bi in unmatched:
            r = sel.loc[bi]
            boxes_rows.append(dict(frame=f, score=r.score, cause=box_cause(cls, r),
                                   whale_free_frame=len(ins) == 0))
        # boxes dropped by the whole-frame rule, for the count
        for _, r in b[(b.frame == f) & b["query"].isin(qs) & (b.score >= args.thr) & b.whole].iterrows():
            boxes_rows.append(dict(frame=f, score=r.score, cause="whole frame", whale_free_frame=len(ins) == 0))
    d = pd.DataFrame(rows)
    u = pd.DataFrame(boxes_rows, columns=["frame", "score", "cause", "whale_free_frame"])
    out_dir = detect.detect_cache_dir(DEV_VIDEO, args.run)
    d.to_csv(out_dir / f"maskframes_instances_{args.query.replace(' ', '_')}_{args.thr}.csv", index=False)
    print(f"\n### {args.run} query={args.query} thr={args.thr}: whale instances found on {d.frame.nunique()} mask frames\n")
    groups = [("all", d.index == d.index)] + [(g, d.grade == g) for g in ("clear", "hazy", "faint")] + \
             [("cut by frame edge", d.cut), ("behind bubbles", d.bubbles), ("small (<1 % of frame)", d.small)]
    t = pd.DataFrame([dict(group=n, n=int(m.sum()), found=int(d.found[m].sum()),
                           share=round(d.found[m].mean(), 3) if m.any() else np.nan,
                           median_box_iou=round(d.box_iou[m & d.found].median(), 3),
                           median_mask_iou=round(d.mask_iou[m & d.found].median(), 3),
                           mask_iou_ge_05=int((d.mask_iou[m & d.found] >= 0.5).sum())) for n, m in groups])
    print(to_md(t))
    print("\nUnmatched boxes at this threshold, by what they sit on (reference class map):\n")
    print(to_md(u.groupby(["cause", "whale_free_frame"]).size().rename("boxes").reset_index()))
    # back as whale: boxes covering >= 25 % back pixels at any score >= 0.1
    back_rows = []
    for f in mask_frames():
        cls, _, _ = load_masks(f)
        backpix = cls == CLASS_IDS["back"]
        if backpix.sum() < 1000:
            continue
        for q in qs + BACK_Q:
            sel = b[(b.frame == f) & (b["query"] == q) & ~b.clock & ~b.whole]
            best_on_back, best_back_iou = 0.0, 0.0
            for idx, r in sel.iterrows():
                x0, y0, x1, y1 = [int(round(v)) for v in (r.x0, r.y0, r.x1, r.y1)]
                sub = cls[max(y0, 0):y1, max(x0, 0):x1]
                if sub.size and (sub == CLASS_IDS["back"]).mean() >= 0.5:
                    best_on_back = max(best_on_back, r.score)
                if store is not None and f"{f}_{idx}" in store and r.score >= 0.12:
                    best_back_iou = max(best_back_iou, mask_iou(unpack(store, f"{f}_{idx}"), backpix))
            back_rows.append(dict(frame=f, query=q, best_score_on_back=best_on_back, best_sam_iou_with_back=best_back_iou))
    bk = pd.DataFrame(back_rows)
    print("\nBack: per query, frames with a back mask; best score of a box lying >= 50 % on back pixels\n")
    print(to_md(bk.groupby("query").agg(frames=("frame", "count"),
                                        median_best_score=("best_score_on_back", "median"),
                                        max_best_score=("best_score_on_back", "max"),
                                        frames_ge_thr=("best_score_on_back", lambda s: int((s >= args.thr).sum())),
                                        median_sam_iou_back=("best_sam_iou_with_back", "median")).reset_index()))
    bk.to_csv(out_dir / "maskframes_back.csv", index=False)


if __name__ == "__main__":
    main()
