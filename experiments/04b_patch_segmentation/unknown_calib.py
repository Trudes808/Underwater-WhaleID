"""Threshold for "unknown" (patch unlike anything in training), set on the dev clip only.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04b_patch_segmentation/unknown_calib.py

For each held-out time block: cosine distance of its patches to the nearest labelled patch of the training blocks
(bank of up to 60k patches). thr_p99 / thr_p995 = 99th / 99.5th percentile over all held-out patches, so about 1 %
of dev patches from unseen moments are called unknown. Writes results/unknown_calib.json.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from uwid import patchseg as ps
import common as C


def main():
    feats, targets, frames = C.load_dev("w1280")
    d_all, cls_all = [], []
    for b in range(len(C.BLOCKS)):
        pool = [frames.index(f) for f in C.train_frames_for(b, frames)]
        X, _ = C.xy_from(feats, targets, pool)
        bank = ps.FeatureBank(X)
        for i, f in enumerate(frames):
            if C.block_of(f) != b:
                continue
            d = bank.distance(torch.from_numpy(feats[i])).cpu().numpy()
            d_all.append(d.ravel())
            cls_all.append(ps.hard_labels(targets[i], 0.5).ravel())
    d = np.concatenate(d_all); c = np.concatenate(cls_all)
    res = {"thr_p99": float(np.percentile(d, 99)), "thr_p995": float(np.percentile(d, 99.5)),
           "thr_p95": float(np.percentile(d, 95)), "median": float(np.median(d)),
           "per_class_median": {ps.CLASSES[k]: float(np.median(d[c == k])) for k in range(len(ps.CLASSES)) if (c == k).any()},
           "per_class_share_over_p99": {ps.CLASSES[k]: float((d[c == k] > np.percentile(d, 99)).mean())
                                        for k in range(len(ps.CLASSES)) if (c == k).any()},
           "n_patches": int(len(d))}
    (C.HERE / "results").mkdir(exist_ok=True)
    (C.HERE / "results" / "unknown_calib.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
