"""Experiments scored on WP3's test mask frames (single frames, 1280 px features from WP3's full-size JPEGs).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04b_patch_segmentation/test_masks.py features
    ... test_masks.py curve      # dev-only learning curve: 5 / 10 / 20 / 31 dev frames -> test masks
    ... test_masks.py adapt      # FITTED ON THE TEST DEPLOYMENT: dev 31 + k mask frames of ONE recording,
                                 # scored on the mask frames of the OTHER recordings (leave-one-recording-in)

Features cached as underwater_cache/patchseg/labels_<stem>_w1280.npz and results/test_curve.json,
results/test_adapt.json.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch

from uwid import patchseg as ps
import common as C

OUT = C.HERE / "results"


def done_stems():
    return [s for s in C.TEST_STEMS if (C.REF / s / "DONE").exists() and (C.REF / s / "masks").exists()]


def cmd_features(a):
    model = ps.load_backbone()
    for s in done_stems():
        d = C.load_folder(C.test_folder(s), "w1280", model=model)
        print(s, len(d["frames"]), "mask frames")


def load_test(s):
    d = C.load_folder(C.test_folder(s), "w1280")
    return d["feats"], d["targets"], d["frames"]


def score(clf, stems):
    sc = C.Scorer()
    for s in stems:
        feats, _, frames = load_test(s)
        for i, f in enumerate(frames):
            cm, inst, meta = C.read_mask(C.test_folder(s), f)
            sc.add(C.pred_codes(ps.predict_probs(clf, torch.from_numpy(feats[i]).cuda()), cm.shape), cm, inst, meta)
    return sc.summary()


def short(s):
    return {"iou_band": s["iou_band"], "iou": s["iou"], "whale_recall_by_grade": s["whale_recall_by_grade"],
            "confusions": s["confusions"], "frames": s["frames"], "seafloor_row": s["conf"][7], "fish": s["fish"],
            "back_row": s["conf"][2]}


def cmd_curve(a):
    feats, targets, frames = C.load_dev("w1280")
    stems = done_stems()
    res = {}
    for n in [5, 10, 20, 31]:
        runs = []
        for s0 in ([0] if n == 31 else [0, 1, 2]):
            start = int(round(s0 * (31 - n) / 2))
            idx = list(range(start, start + n))  # a contiguous block of time of the dev mask frames
            clf = ps.train_classifier(*C.xy_from(feats, targets, idx), ps.TrainConfig())
            r = short(score(clf, stems)); r["dev_frames"] = [frames[i] for i in idx]
            runs.append(r)
        res[str(n)] = runs
        print(n, [round(r["iou_band"]["whale"] or 0, 3) for r in runs], [round(r["iou_band"]["back"] or 0, 3) for r in runs])
    OUT.mkdir(exist_ok=True)
    (OUT / "test_curve.json").write_text(json.dumps({"stems": stems, "runs": res}, indent=1, default=float))


def _fit(Xd, Yd, extra):
    X, Y = [Xd], [Yd]
    for tf, tt, pick in extra:
        if pick:
            Xt, Yt = C.xy_from(tf, tt, pick)
            X.append(Xt); Y.append(Yt)
    X, Y = torch.cat(X), torch.cat(Y)
    return ps.train_classifier(X, Y, ps.TrainConfig()), Y


def cmd_adapt(a):
    """A: dev 31 + k mask frames of ONE recording (00004, the one with most mask frames), scored on the mask frames
    of all other recordings. B: leave one recording out: dev 31 + the mask frames of all other recordings, scored
    on the held-out recording, pooled over recordings."""
    feats, targets, frames = C.load_dev("w1280")
    Xd, Yd = C.xy_from(feats, targets, list(range(len(frames))))
    stems = done_stems()
    tests = {s: load_test(s) for s in stems}
    src = max(stems, key=lambda s: len(tests[s][2]))
    res = {"stems": stems, "runs": [], "lomo": {}}
    others = [s for s in stems if s != src]
    tf, tt, tfr = tests[src]
    for k in sorted({0, 1, 2, len(tfr)}):
        pick = [] if k == 0 else sorted(set(np.linspace(0, len(tfr) - 1, k).round().astype(int).tolist()))
        clf, Y = _fit(Xd, Yd, [(tf, tt, pick)])
        r = short(score(clf, others))
        r.update({"source": src, "k": len(pick), "source_frames": [tfr[i] for i in pick], "scored_on": others,
                  "patches_added": torch.bincount(Y, minlength=len(ps.CLASSES)).tolist()})
        res["runs"].append(r)
        print("A", src[-5:], len(pick), {c: C.fmt(v, 2) for c, v in r["iou_band"].items()}, r["fish"], flush=True)
    base, lomo = C.Scorer(), C.Scorer()
    clf0, _ = _fit(Xd, Yd, [])
    for s in stems:
        extra = [(tests[o][0], tests[o][1], list(range(len(tests[o][2])))) for o in stems if o != s]
        clf, Y = _fit(Xd, Yd, extra)
        for sc, c in [(base, clf0), (lomo, clf)]:
            f_, _, fr = tests[s]
            for i, f in enumerate(fr):
                cm, inst, meta = C.read_mask(C.test_folder(s), f)
                sc.add(C.pred_codes(ps.predict_probs(c, torch.from_numpy(f_[i]).cuda()), cm.shape), cm, inst, meta)
    res["lomo"] = {"dev_only": short(base.summary()), "dev_plus_other_recordings": short(lomo.summary())}
    for k, v in res["lomo"].items():
        print("B", k, {c: C.fmt(x, 2) for c, x in v["iou_band"].items()}, v["fish"], v["confusions"], flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "test_adapt.json").write_text(json.dumps(res, indent=1, default=float))


def cmd_lomo_save(a):
    """Save the leave-one-recording-out classifiers (dev 31 + mask frames of all OTHER recordings) for window
    prediction on the held-out recording: underwater_cache/patchseg/clf_lomo_<stem>.pt. FITTED ON THE TEST DEPLOYMENT."""
    feats, targets, frames = C.load_dev("w1280")
    Xd, Yd = C.xy_from(feats, targets, list(range(len(frames))))
    stems = done_stems()
    tests = {s: load_test(s) for s in stems}
    for s in stems:
        extra = [(tests[o][0], tests[o][1], list(range(len(tests[o][2])))) for o in stems if o != s]
        clf, Y = _fit(Xd, Yd, extra)
        ps.save_classifier(clf, ps.CACHE / f"clf_lomo_{s}.pt",
                           {"fitted_on": "dev 31 + mask frames of the other 2025-02-03 recordings", "held_out": s,
                            "class_counts_patches": torch.bincount(Y, minlength=len(ps.CLASSES)).tolist()})
        print("saved", s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    a = ap.parse_args()
    torch.set_num_threads(8)
    {"features": cmd_features, "curve": cmd_curve, "adapt": cmd_adapt, "lomo_save": cmd_lomo_save}[a.cmd](a)
