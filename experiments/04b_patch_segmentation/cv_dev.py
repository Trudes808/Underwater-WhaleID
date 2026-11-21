"""Within-clip check on the dev clip: leave one time block out (4 blocks, 3.3 s buffer), single frames.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04b_patch_segmentation/cv_dev.py \
        compare            # input sizes x classifier heads -> results/cv_compare.json
    ... cv_dev.py curve --config w1920 [--hidden 0]   # learning curve 5 / 10 / 20 / all frames of the pool
    ... cv_dev.py final --config w1920 [--hidden 0]   # save the 4 fold classifiers and the all-31 classifier

Fitted on: dev clip mask frames only (training blocks). Scored on: the held-out block's mask frames.
The dev clip's "fish" are not fish (user, 2026-11-20): code 3 is ignored in training and scoring (common.DEV).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from uwid import patchseg as ps
import common as C

OUT = C.HERE / "results"


def run_folds(feats, targets, frames, cfg, train_select=None, purity=0.75, band=12):
    sc = C.Scorer(band)
    per_fold = []
    for b in range(len(C.BLOCKS)):
        test = [i for i, f in enumerate(frames) if C.block_of(f) == b]
        pool = [frames.index(f) for f in C.train_frames_for(b, frames)]
        train = train_select(pool) if train_select else pool
        X, Y = C.xy_from(feats, targets, train, purity)
        clf = ps.train_classifier(X, Y, cfg)
        fs = C.Scorer(band)
        for i in test:
            probs = ps.predict_probs(clf, torch.from_numpy(feats[i]).cuda())
            pred = C.pred_codes(probs)
            cm, inst, meta = C.read_mask(C.DEV, frames[i])
            sc.add(pred, cm, inst, meta); fs.add(pred, cm, inst, meta)
        per_fold.append({"block": b, "train_frames": [frames[i] for i in train],
                         "test_frames": [frames[i] for i in test], "iou": fs.summary()["iou"]})
    s = sc.summary(); s["folds"] = per_fold
    return s


def cmd_compare(a):
    res = {}
    for config in ["w960", "w1280", "w1920", "w1280_L4", "w1920_t2"]:
        feats, targets, frames = C.load_dev(config)
        heads = [0, 256] if config in ("w1280", "w1920", "w1920_t2") else [0]
        for hidden in heads:
            t = time.time()
            cfg = ps.TrainConfig(hidden=hidden, epochs=40 if config.endswith("t2") else 60)
            s = run_folds(feats, targets, frames, cfg)
            key = f"{config}_h{hidden}"
            res[key] = s
            iou = s["iou_band"]
            print(key, f"{time.time()-t:.0f}s", "mIoU(band, 7 cls)",
                  f"{np.mean([v for k, v in iou.items() if v is not None]):.3f}",
                  {k: C.fmt(v, 2) for k, v in iou.items()},
                  "whale rec", {g: round(v["pixel_recall"], 2) for g, v in s["whale_recall_by_grade"].items()},
                  "fish", s["fish"], flush=True)
        del feats
    OUT.mkdir(exist_ok=True)
    (OUT / "cv_compare.json").write_text(json.dumps(res, indent=1))


def cmd_curve(a):
    feats, targets, frames = C.load_dev(a.config)
    cfg = ps.TrainConfig(hidden=a.hidden)
    res = {}
    for n in [5, 10, 20, 999]:
        runs = []
        starts = [0] if n == 999 else [0, 1, 2]
        for s0 in starts:
            def sel(pool, n=n, s0=s0):
                if n >= len(pool):
                    return pool
                # a contiguous block of time inside the training pool; three starting points spread over the pool
                start = int(round(s0 * (len(pool) - n) / 2))
                return pool[start:start + n]
            runs.append(run_folds(feats, targets, frames, cfg, sel))
        key = "all" if n == 999 else str(n)
        res[key] = runs
        mi = [np.mean([v for v in r["iou_band"].values() if v is not None]) for r in runs]
        print(key, "mIoU", [round(x, 3) for x in mi], "whale IoU", [round(r["iou_band"]["whale"], 3) for r in runs],
              "back", [round(r["iou_band"]["back"], 3) for r in runs], flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"cv_curve_{a.config}_h{a.hidden}.json").write_text(json.dumps(res, indent=1))


def cmd_final(a):
    feats, targets, frames = C.load_dev(a.config)
    cfg = ps.TrainConfig(hidden=a.hidden)
    for b in range(len(C.BLOCKS)):
        pool = [frames.index(f) for f in C.train_frames_for(b, frames)]
        clf = ps.train_classifier(*C.xy_from(feats, targets, pool), cfg)
        ps.save_classifier(clf, ps.CACHE / f"clf_{a.config}_h{a.hidden}_fold{b}.pt",
                           {"config": a.config, "fitted_on": "dev clip mask frames", "frames": [frames[i] for i in pool],
                            "held_out_block": C.BLOCKS[b]})
    allidx = list(range(len(frames)))
    X, Y = C.xy_from(feats, targets, allidx)
    clf = ps.train_classifier(X, Y, cfg)
    ps.save_classifier(clf, ps.CACHE / f"clf_{a.config}_h{a.hidden}_all.pt",
                       {"config": a.config, "fitted_on": "dev clip, all 31 mask frames", "frames": frames,
                        "class_counts_patches": torch.bincount(Y, minlength=len(ps.CLASSES)).tolist()})
    print("saved; patches per class (all 31):", torch.bincount(Y, minlength=len(ps.CLASSES)).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--config", default="w1920"); ap.add_argument("--hidden", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8)
    {"compare": cmd_compare, "curve": cmd_curve, "final": cmd_final}[a.cmd](a)
