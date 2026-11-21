"""Speed and GPU memory of each model on 1920x1080 dev frames (model time only; decoding runs in a separate thread
in the real runs). Writes ``cost.json`` next to this script.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04_whale_fish_detection/cost.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import DEV_VIDEO, PROMPTS, detect


def bench(fn, n_warm=2, n=12):
    import torch
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n, torch.cuda.max_memory_allocated() / 1e9


def main():
    import torch
    frames = [570, 900, 1485, 2175]
    imgs = [im for _, im in detect.iter_frames_cv(DEV_VIDEO, frames=frames)]
    res = {}
    owl = detect.OwlRunner(PROMPTS["owl_all"], batch=4)
    dt, mem = bench(lambda: owl.detect(imgs), n=4)
    res["owlv2-base (17 prompts, batch 4, fp16)"] = dict(s_per_frame=dt / 4, frames_per_s=4 / dt, peak_gb=mem)
    owl1 = detect.OwlRunner(["whale"], batch=4)
    dt, mem = bench(lambda: owl1.detect(imgs), n=4)
    res["owlv2-base (1 prompt, batch 4, fp16)"] = dict(s_per_frame=dt / 4, frames_per_s=4 / dt, peak_gb=mem)
    del owl, owl1
    torch.cuda.empty_cache()
    sam = detect.SamMasker()
    boxes = np.array([[400, 280, 1100, 690], [0, 180, 430, 620], [1500, 250, 1910, 700]], np.float32)
    dt, mem = bench(lambda: sam.masks(imgs[0], boxes), n=6)
    res["sam2.1-hiera-large (3 boxes, fp32)"] = dict(s_per_frame=dt, frames_per_s=1 / dt, peak_gb=mem)
    del sam
    torch.cuda.empty_cache()
    llm = detect.GroundingRunner(PROMPTS["llmdet_A"], batch=1)
    dt, mem = bench(lambda: llm.detect(imgs[:1]), n=6)
    res["llmdet-large (2 prompts, batch 1, fp16)"] = dict(s_per_frame=dt, frames_per_s=1 / dt, peak_gb=mem)
    del llm
    torch.cuda.empty_cache()
    for mk, mid in (("ade", "facebook/mask2former-swin-large-ade-semantic"),
                    ("coco", "facebook/mask2former-swin-large-coco-panoptic")):
        seg = detect.Mask2FormerSeg(mid)
        dt, mem = bench(lambda: seg.segment(imgs[0]), n=6)
        res[f"mask2former-swin-large-{mk} (540 px short side, fp16)"] = dict(s_per_frame=dt, frames_per_s=1 / dt,
                                                                          peak_gb=mem)
        del seg
        torch.cuda.empty_cache()
    res["gpu"] = torch.cuda.get_device_name(0)
    out = Path(__file__).resolve().parent / "cost.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
