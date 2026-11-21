"""Class maps for the frames of 1 s windows of a video (decode in order, no seeking).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04b_patch_segmentation/predict_video.py dev
    ... predict_video.py test --stem CC-11-96-20250203-101236-000-00004

Windows are the 1 s grid of the video ([k, k+1) s by frame index / fps). Inside a window the frames at ``--stride``
(default 5: 6 frames per window) are classified, plus every labelled frame in it.
- dev: every window of the clip; each frame is classified by the fold classifier whose held-out time block contains
  it (so no frame is scored by a classifier fitted on its own block), "unknown" distance to that fold's training
  patches.
- test: the windows holding WP3's 120 sampled frames of the recording (from its index.json, the .m4v, OpenCV frame
  index as WP3); classifier fitted on all 31 dev mask frames.
Writes underwater_cache/patchseg/pred_<name>_<clf>.npz: frames, window (per frame), probs (N, h, w, 8) fp16,
dist (N, h, w) fp16, plus timing.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import numpy as np
import torch

from uwid import patchseg as ps
import common as C


def window_frames(k: int, fps: float, stride: int, n_total: int) -> list[int]:
    a = math.ceil(k * fps - 1e-9); b = min(math.ceil((k + 1) * fps - 1e-9), n_total)
    return list(range(a, b, stride))


def build_bank(config: str, frame_idx: list[int]) -> ps.FeatureBank:
    feats, targets, frames = C.load_dev(config)
    X, _ = C.xy_from(feats, targets, frame_idx)
    return ps.FeatureBank(X)


def run(video, want: dict[int, int], width: int, clf_for, bank_for, batch: int = 8):
    """want: frame index -> window k. clf_for / bank_for: frame -> classifier / FeatureBank."""
    model = ps.load_backbone()
    cap = cv2.VideoCapture(str(video))
    last = max(want)
    i = 0; buf = []
    out_f, out_p, out_d = [], [], []
    t_dec = t_gpu = 0.0
    t_all = time.time()

    def flush():
        nonlocal t_gpu
        if not buf:
            return
        t = time.time()
        f = ps.extract(model, [b[1] for b in buf], width)
        for j, (fi, _) in enumerate(buf):
            p = ps.predict_probs(clf_for(fi), f[j])
            d = bank_for(fi).distance(f[j])
            out_f.append(fi); out_p.append(p.half().cpu().numpy()); out_d.append(d.half().cpu().numpy())
        torch.cuda.synchronize(); t_gpu += time.time() - t
        buf.clear()

    while i <= last:
        t = time.time()
        if i in want:
            ok, fr = cap.read()
            if not ok:
                break
            buf.append((i, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        else:
            ok = cap.grab()
            if not ok:
                break
        t_dec += time.time() - t
        i += 1
        if len(buf) >= batch:
            flush()
    flush()
    timing = {"decode_s": t_dec, "gpu_s": t_gpu, "wall_s": time.time() - t_all, "frames_classified": len(out_f),
              "frames_decoded": i, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    return (np.array(out_f), np.array([want[f] for f in out_f]), np.stack(out_p), np.stack(out_d), timing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which"); ap.add_argument("--stem"); ap.add_argument("--config", default="w1280")
    ap.add_argument("--hidden", type=int, default=0); ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--clf", default=None, help="classifier file for test (default: the all-31 one)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(8)
    width = {"w960": 960, "w1280": 1280, "w1920": 1920}[a.config]
    name = f"{a.config}_h{a.hidden}"
    if a.which == "dev":
        n_total = 2857
        labelled = set(range(0, n_total, 15))
        want = {}
        for k in range(int(n_total / C.DEV_FPS) + 1):
            for f in window_frames(k, C.DEV_FPS, a.stride, n_total) + [f for f in labelled if int(f / C.DEV_FPS) == k]:
                want[f] = k
        clfs = [ps.load_classifier(ps.CACHE / f"clf_{name}_fold{b}.pt")[0] for b in range(4)]
        _, _, dev_frames = C.load_dev(a.config)
        banks = [build_bank(a.config, [dev_frames.index(f) for f in C.train_frames_for(b, dev_frames)]) for b in range(4)]
        res = run(C.DEV_VIDEO, want, width, lambda f: clfs[C.block_of(f)], lambda f: banks[C.block_of(f)])
        out = ps.CACHE / f"pred_dev_{name}{a.tag}.npz"
    else:
        idx = json.loads((C.REF / a.stem / "index.json").read_text())
        fps = idx["fps"]; n_total = idx["frames_decoded"]
        want = {}
        for s in map(int, idx["sample"]):
            k = int(s / fps)
            for f in window_frames(k, fps, a.stride, n_total) + [s]:
                want[f] = k
        clf_path = a.clf or (ps.CACHE / f"clf_{name}_all.pt")
        clf = ps.load_classifier(clf_path)[0]
        _, _, dev_frames = C.load_dev(a.config)
        bank = build_bank(a.config, list(range(len(dev_frames))))
        res = run(C.TEST_DIR / f"{a.stem}.m4v", want, width, lambda f: clf, lambda f: bank)
        out = ps.CACHE / f"pred_{a.stem}_{name}{a.tag}.npz"
    frames, wins, probs, dist, timing = res
    np.savez(out, frames=frames, window=wins, probs=probs, dist=dist, timing=json.dumps(timing))
    print(out.name, timing, flush=True)


if __name__ == "__main__":
    main()
