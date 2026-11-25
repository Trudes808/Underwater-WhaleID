"""GPU steps of 09_seafloor_sand (under the GPU lock; resumable; each call stops after ~13 min).

    heads   patch features of the 2025-01-02 mask frames (uwid.patchseg.cached_features, 1280 px), five linear patch
            classifiers (folds below), pixel predictions on every fold's held-out mask frames
    frames  every head on: the two frames of every 1 s window of the dense recordings (decode.py's temporary JPEGs),
            and every labelled frame of the three sources -> per frame argmax maps (45 x 80), mean class
            probabilities, frame embedding (mean patch feature of the frame and its quarters), grey levels

Folds (what each head was fitted on -> what it is scored on):
    test17   dev + d1 + d2 masks of 00023, 00024  -> d2 00017
    test23   dev + d1 + d2 masks of 00017, 00024  -> d2 00023
    test_d1  dev + d2 (all)                       -> d1 (no seafloor: false alarms)
    test_dev d1 + d2 (all)                        -> dev clip (no seafloor: false alarms)
    all      dev + d1 + d2                        -> nothing (the delivered head)

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/09_seafloor_sand/gpu.py heads frames
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "08_whale_fish_boxes"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import data as D  # noqa: E402
from uwid import patchseg as ps  # noqa: E402

BUDGET_S = 13 * 60
OUT = D.SF_CACHE
HEADS = ("test17", "test23", "test_d1", "test_dev", "all")
DENSE_TMP = OUT / "frames_tmp"


def folders():
    """name -> (source, LabelFolder)."""
    out = {"dev": ("dev", ps.LabelFolder("dev", D.REF / "2025-12-20_CG" / "masks",
                                         D.VIDEO / "2025-12-20 CG.m4v", ignore_codes=(ps.FISH,)))}
    for rec in D.recordings(("d1", "d2")):
        if rec.mask_frames():
            out[rec.stem] = (rec.source, ps.LabelFolder(rec.stem, rec.ref / "masks", rec.ref))
    return out


def fold_members(name: str, fs: dict) -> tuple[list[str], list[str]]:
    """(training folders, test folders) of a head."""
    d2 = [k for k, (s, _) in fs.items() if s == "d2"]
    d1 = [k for k, (s, _) in fs.items() if s == "d1"]
    s17 = [k for k in d2 if k.endswith("00017")]; s23 = [k for k in d2 if k.endswith("00023")]
    if name == "test17":
        return ["dev"] + d1 + [k for k in d2 if k not in s17], s17
    if name == "test23":
        return ["dev"] + d1 + [k for k in d2 if k not in s23], s23
    if name == "test_d1":
        return ["dev"] + d2, d1
    if name == "test_dev":
        return d1 + d2, ["dev"]
    return ["dev"] + d1 + d2, []


def step_heads(t0: float, model=None) -> bool:
    hd = OUT / "heads"; hd.mkdir(parents=True, exist_ok=True)
    if (hd / "DONE").exists():
        return True
    fs = folders()
    model = model or ps.load_backbone()
    data = {k: ps.cached_features(f, 1280, model=model, tag="w1280") for k, (_, f) in fs.items()}
    preds = {}
    for name in HEADS:
        train, test = fold_members(name, fs)
        X, Y = ps.training_set([data[k] for k in train])
        clf = ps.train_classifier(X, Y, ps.TrainConfig(), device="cuda")
        counts = {ps.CLASSES[c]: int(n) for c, n in enumerate(torch.bincount(Y, minlength=len(ps.CLASSES)))}
        ps.save_classifier(clf, hd / f"head_{name}.pt", {"trained_on": train, "tested_on": test,
                                                          "patches": counts, "width": 1280})
        for k in test:
            d = data[k]
            P = ps.predict_probs(clf, torch.from_numpy(d["feats"]).to("cuda"))
            preds[f"{name}|{k}"] = (np.array(d["frames"]), P.argmax(-1).to(torch.uint8).cpu().numpy(),
                                    P[..., ps.SEAFLOOR].float().cpu().numpy())
        print(f"head {name}: {len(Y)} patches {counts}", flush=True)
    np.savez_compressed(hd / "heldout_preds.npz", **{f"{k}|frames": v[0] for k, v in preds.items()},
                        **{f"{k}|map": v[1] for k, v in preds.items()}, **{f"{k}|psea": v[2] for k, v in preds.items()})
    (hd / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return True


def frame_sets():
    """(set name, stem, [(frame, t, path)])."""
    sets = []
    for d in sorted(DENSE_TMP.glob("*")) if DENSE_TMP.exists() else []:
        if not (d / "DONE").exists():
            continue
        rows = [l.split(",") for l in (d / "index.csv").read_text().split("\n")[1:] if l]
        sets.append(("dense", d.name, [(int(f), float(t), d / f"{int(f):06d}.jpg") for f, t in rows]))
    for rec in D.recordings():
        items = []
        fc = rec.frames_csv()
        for f, t in zip(fc.frame, fc.time_s):
            p = rec.image_path(int(f))
            if p is not None:
                items.append((int(f), float(t), p))
        sets.append(("label", rec.stem, items))
    return sets


def step_frames(t0: float, model=None) -> bool:
    fd = OUT / "frames"; fd.mkdir(parents=True, exist_ok=True)
    heads = {n: ps.load_classifier(OUT / "heads" / f"head_{n}.pt")[0] for n in HEADS}
    model = model or ps.load_backbone()
    timing = json.loads((fd / "timing.json").read_text()) if (fd / "timing.json").exists() else {}
    for kind, stem, items in frame_sets():
        f_out = fd / f"{kind}_{stem}.npz"
        if f_out.exists() or not items:
            continue
        if time.time() - t0 > BUDGET_S:
            print("frames: out of time, run again", flush=True); return False
        acc = {k: [] for k in ("frame", "t", "maps", "probs", "emb", "grey_mean", "grey_p95", "grey_std")}
        t_set = time.time(); model_s = 0.0
        for i in range(0, len(items), 8):
            chunk = items[i:i + 8]
            ims = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for _, _, p in chunk]
            tm = time.time()
            res = ps.frame_classes(model, ims, heads)
            torch.cuda.synchronize(); model_s += time.time() - tm
            for (f, t, _), r in zip(chunk, res):
                acc["frame"].append(f); acc["t"].append(t)
                acc["maps"].append(np.stack([r["maps"][n] for n in HEADS]))
                acc["probs"].append(np.stack([r["probs"][n] for n in HEADS]))
                acc["emb"].append(r["emb"].astype(np.float16))
                for k in ("grey_mean", "grey_p95", "grey_std"):
                    acc[k].append(r[k])
        np.savez_compressed(f_out, heads=np.array(HEADS), **{k: np.asarray(v) for k, v in acc.items()})
        timing[f"{kind}_{stem}"] = {"frames": len(items), "model_s": model_s, "wall_s": time.time() - t_set,
                                    "frame_size": list(ims[0].shape[:2])}
        (fd / "timing.json").write_text(json.dumps(timing, indent=1))
        print(f"frames {kind} {stem}: {len(items)} in {time.time() - t_set:.0f} s (model {model_s:.0f} s)", flush=True)
    return True


if __name__ == "__main__":
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8.5e9 / torch.cuda.get_device_properties(0).total_memory))
    t0 = time.time()
    model = ps.load_backbone()
    ok = True
    for step in sys.argv[1:]:
        ok = {"heads": step_heads, "frames": step_frames}[step](t0, model) and ok
    if ok:
        (OUT / f"gpu_{'_'.join(sys.argv[1:])}.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("done" if ok else "partial", f"{time.time() - t0:.0f} s", flush=True)
