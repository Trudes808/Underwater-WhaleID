"""Mask2Former (ADE20k, Swin-L, cached; not fitted on anything here) as a source of seafloor pixels.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04b_patch_segmentation/m2f_seafloor.py

Runs on the 31 dev mask frames (control: what does it call the back and the open water?) and on WP3's sampled frames
of every test recording (frames dumped by WP3 as JPEG under underwater_cache/reference/<stem>/sample/).
Input 1138 x 640. Writes underwater_cache/patchseg/m2f_<name>.npz: frames, labels (N, 180, 320) uint8 ADE ids.
Seafloor = sand, rock, earth, floor, mountain.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from uwid import patchseg as ps
import common as C

MODEL = "facebook/mask2former-swin-large-ade-semantic"
SEAFLOOR_IDS = [46, 34, 13, 3, 16]  # sand, rock, earth, floor, mountain


def run(model, proc, paths):
    labs = []
    for p in paths:
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (1138, 640), interpolation=cv2.INTER_AREA)
        x = proc(images=im, return_tensors="pt", do_resize=False)
        with torch.no_grad():
            out = model(pixel_values=x["pixel_values"].cuda().half())
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[(180, 320)])[0]
        labs.append(seg.cpu().numpy().astype(np.uint8))
    return np.stack(labs)


def main():
    torch.set_num_threads(8)
    proc = Mask2FormerImageProcessor.from_pretrained(MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).cuda().half().eval()
    jobs = {"dev": sorted((C.REF / "2025-12-20_CG" / "frames").glob("*.jpg"))}
    _, _, dev_frames = C.load_dev("w960")
    jobs["dev"] = [p for p in jobs["dev"] if int(p.stem) in dev_frames]
    for s in C.TEST_STEMS:
        d = C.REF / s / "sample"
        if d.exists():
            jobs[s] = sorted(d.glob("*.jpg"))
    for name, paths in jobs.items():
        out = ps.CACHE / f"m2f_{name}.npz"
        if out.exists() or not paths:
            continue
        t = time.time()
        labs = run(model, proc, paths)
        np.savez_compressed(out, frames=np.array([int(p.stem) for p in paths]), labels=labs)
        print(name, len(paths), f"{(time.time() - t) / len(paths):.2f} s/frame", flush=True)


if __name__ == "__main__":
    main()
