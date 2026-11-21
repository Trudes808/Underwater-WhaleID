"""Cache DINOv3 patch features of the label folders' mask frames for several input sizes.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04b_patch_segmentation/extract_dev.py \
        [--configs w960,w1280,w1920,w1280_L4,w1920_t2] [--folders dev,<test stem>,...]

Dev frames are decoded in order from the video (OpenCV, the frame index of WP3's labels); test folders read WP3's
full-size JPEGs. Cache: underwater_cache/patchseg/labels_<folder>_<config>.npz (features only; targets are rebuilt
from the masks at load time, so repainted masks take effect without re-extraction).
"""
from __future__ import annotations

import argparse
import time

import torch

from uwid import patchseg as ps
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="w960,w1280,w1920,w1280_L4,w1920_t2")
    ap.add_argument("--folders", default="dev")
    a = ap.parse_args()
    torch.set_num_threads(8)
    model = ps.load_backbone()
    for name in a.folders.split(","):
        folder = C.DEV if name == "dev" else C.test_folder(name)
        for cfg in a.configs.split(","):
            t = time.time()
            d = C.load_folder(folder, cfg, model=model)
            print(name, cfg, d["feats"].shape, f"{time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
