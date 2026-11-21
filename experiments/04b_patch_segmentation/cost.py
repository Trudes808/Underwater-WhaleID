"""End-to-end cost: decode in order + DINOv3 at 1280 px + linear classifier + nearest-patch distance.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04b_patch_segmentation/cost.py

Runs on the first 120 s of test recording 00005 (.mov, the tag's own HEVC file, and .m4v), classifying every 5th
frame (6 per 1 s window) and every frame. Writes results/cost.json.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from uwid import patchseg as ps
import common as C
from predict_video import build_bank, run


def main():
    torch.set_num_threads(8)
    clf = ps.load_classifier(ps.CACHE / "clf_w1280_h0_all.pt")[0]
    _, _, dev_frames = C.load_dev("w1280")
    bank = build_bank("w1280", list(range(len(dev_frames))))
    fps = 29.95; secs = 120
    res = {}
    for ext in ["m4v", "mov"]:
        for stride in [5, 1]:
            n = int(secs * fps)
            want = {f: int(f / fps) for f in range(0, n, stride)}
            torch.cuda.reset_peak_memory_stats()
            _, _, _, _, t = run(C.TEST_DIR / f"{C.TEST_STEMS[1]}.{ext}", want, 1280, lambda f: clf, lambda f: bank)
            t["video_s"] = n / fps
            t["gpu_h_per_video_h"] = t["wall_s"] / t["video_s"]
            t["classified_fps"] = t["frames_classified"] / t["wall_s"]
            res[f"{ext}_stride{stride}"] = t
            print(ext, stride, t, flush=True)
    (C.HERE / "results" / "cost.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
