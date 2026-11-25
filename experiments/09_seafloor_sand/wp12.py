"""WP12 step A: seafloor masks on user-confirmed frames of 2025-01-02 00017, a re-fitted patch head (v2), measured
leave-one-episode-out inside 00017 plus false calls everywhere else.

    frames   decode the mask frames (1920 x 1080, shown-frame index = round(t * 30)) to underwater_cache/seafloor/wp12/frames
    grids    my labelling view: 4 frames per sheet with a 100 px grid (cache, not for the user)
    (masks)  experiments/02_reference_labels/masktool.py run --frames <wp12/frames> --prompts wp12_prompts.json
             --out <wp12/masks> --cache <reference/samcache>      (GPU, under the lock)
    heads    (GPU) features of the new mask frames + v2 heads (folds below)        -> seafloor/heads/head_v2_*.pt
    frames2  (GPU) every v2 head on the window frames (decode.py's temporary JPEGs) and the labelled frames
    score    pixel and window scores -> results/wp12_*.json
    sheets   the user's sheets

Episodes of visible bottom in 00017 (user-confirmed sheet, `seafloor_spans.csv`): E1 292-327, E2 405-422,
E3 465-487, E4 510-524 + 527-533 (one episode). Fold k: the head sees no mask and no window within 5 s of episode k.
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

import data as D  # noqa: E402

S17 = "CC-09-85-HRA-20250102-141304-000-00017"
W12 = D.SF_CACHE / "wp12"
EPISODES = {1: [(292, 327)], 2: [(405, 422)], 3: [(465, 487)], 4: [(510, 524), (527, 533)]}
BUFFER = 5.0
# mask frames (s): the user-confirmed sheet frames first, spread over the episodes
MASK_T = [310, 316, 320, 324,                      # E1 (298 and 304 too faint to outline)
          406, 409, 413, 417, 420,                  # E2
          466, 470, 474, 478, 482, 486,             # E3
          512, 514, 516, 518, 520, 528, 531]        # E4
FPS = 30.0


def episode_of(t: float, buffer: float = 0.0) -> int:
    for k, rs in EPISODES.items():
        if any(a - buffer <= t < b + buffer for a, b in rs):
            return k
    return 0


def frame_index(t: float) -> int:
    return int(round(t * FPS))


def decode_frames():
    from uwid import video
    out = W12 / "frames"; out.mkdir(parents=True, exist_ok=True)
    idx = [frame_index(t) for t in MASK_T]
    print("decoding", len(idx), "mask frames", flush=True)
    for fr in video.iter_frames_at(D.VIDEO / "TestSubset 2025-1-2 CATS -85" / "Video" / f"{S17}.mov", idx,
                                   size=(1920, 1080)):
        cv2.imwrite(str(out / f"{fr.index:06d}.jpg"), cv2.cvtColor(fr.image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("done", len(list(out.glob("*.jpg"))), flush=True)


def grids():
    out = W12 / "grids"; out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for t in MASK_T:
        k = frame_index(t)
        im = cv2.imread(str(W12 / "frames" / f"{k:06d}.jpg"))
        s = cv2.resize(im, (960, 540))
        for x in range(0, 1920, 100):
            cv2.line(s, (x // 2, 0), (x // 2, 539), (0, 255, 255), 1)
            cv2.putText(s, str(x), (x // 2 + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        for y in range(0, 1080, 100):
            cv2.line(s, (0, y // 2), (959, y // 2), (0, 255, 255), 1)
            cv2.putText(s, str(y), (2, y // 2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.rectangle(s, (380, 0), (580, 22), (0, 0, 0), -1)
        cv2.putText(s, f"{k:06d}  t {t}s", (384, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(s)
    for i in range(0, len(tiles), 4):
        tt = tiles[i:i + 4]
        while len(tt) < 4:
            tt.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"g_{i // 4 + 1:02d}.jpg"), np.vstack([np.hstack(tt[:2]), np.hstack(tt[2:])]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


# ------------------------------------------------------------------------------------------------ heads (GPU)

V2_HEADS = ("v2_fold1", "v2_fold2", "v2_fold3", "v2_fold4", "v2_test_d1", "v2_test_dev", "v2_all")


def folders():
    """name -> (source, LabelFolder). Old 2025-01-02 masks: the withdrawn seafloor paint is ignored (code 7 -> 255)."""
    from uwid import patchseg as ps
    out = {"dev": ("dev", ps.LabelFolder("dev", D.REF / "2025-12-20_CG" / "masks", D.VIDEO / "2025-12-20 CG.m4v",
                                         ignore_codes=(ps.FISH,)))}
    for rec in D.recordings(("d1", "d2")):
        if rec.mask_frames():
            ign = (ps.SEAFLOOR,) if rec.source == "d2" else ()
            out[rec.stem] = (rec.source, ps.LabelFolder(rec.stem, rec.ref / "masks", rec.ref, ignore_codes=ign))
    out["wp12"] = ("wp12", ps.LabelFolder("wp12_00017", W12 / "masks", W12 / "frames"))
    return out


def frame_time(name: str, frame: int) -> float:
    return frame / FPS


def select(name, data, head):
    """Positions of the frames of folder ``name`` a head is trained on."""
    fr = data[name]["frames"]
    allp = list(range(len(fr)))
    if head.startswith("v2_fold"):
        k = int(head[-1])
        if name in ("wp12", S17):
            return [i for i, f in enumerate(fr) if episode_of(frame_time(name, f), BUFFER) != k]
        return allp
    if head == "v2_test_d1" and name.startswith("CC-11"):
        return []
    if head == "v2_test_dev" and name == "dev":
        return []
    return allp


def step_heads():
    import torch
    from uwid import patchseg as ps
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8.5e9 / torch.cuda.get_device_properties(0).total_memory))
    hd = D.SF_CACHE / "heads"
    fs = folders()
    print("heads: features", flush=True)
    model = ps.load_backbone()
    data = {k: ps.cached_features(f, 1280, model=model, tag="w1280") for k, (_, f) in fs.items()}
    preds = {}
    old = {n: ps.load_classifier(hd / f"head_{n}.pt")[0] for n in ("test_d1", "test_dev", "all")}
    for head in V2_HEADS:
        names = [k for k in data if select(k, data, head)]
        X, Y = ps.training_set([data[k] for k in names], [select(k, data, head) for k in names])
        clf = ps.train_classifier(X, Y, ps.TrainConfig(), device="cuda")
        counts = {ps.CLASSES[c]: int(n) for c, n in enumerate(torch.bincount(Y, minlength=len(ps.CLASSES)))}
        excluded = {k: len(data[k]["frames"]) - len(select(k, data, head)) for k in data if len(select(k, data, head)) < len(data[k]["frames"])}
        ps.save_classifier(clf, hd / f"head_{head}.pt", {"wp12": True, "frames_left_out": excluded, "patches": counts,
                                                         "width": 1280})
        print(f"head {head}: {len(Y)} patches, seafloor {counts['seafloor']}, left out {excluded}", flush=True)
        # predictions on every mask folder (argmax maps), kept small
        for k, d in data.items():
            P = ps.predict_probs(clf, torch.from_numpy(d["feats"]).to("cuda"))
            preds[f"{head}|{k}"] = P.argmax(-1).to(torch.uint8).cpu().numpy()
    for n, clf in old.items():
        for k, d in data.items():
            P = ps.predict_probs(clf, torch.from_numpy(d["feats"]).to("cuda"))
            preds[f"old_{n}|{k}"] = P.argmax(-1).to(torch.uint8).cpu().numpy()
    np.savez_compressed(W12 / "mask_preds.npz", **preds, **{f"frames|{k}": np.array(d["frames"]) for k, d in data.items()})
    (W12 / "heads.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("heads done", flush=True)


def step_frames2():
    """Every v2 head on the window frames and the labelled frames (as gpu.py step_frames) -> seafloor/frames_v2/."""
    import torch
    import gpu as G
    from uwid import patchseg as ps
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8.5e9 / torch.cuda.get_device_properties(0).total_memory))
    t0 = time.time()
    fd = D.SF_CACHE / "frames_v2"; fd.mkdir(exist_ok=True)
    heads = {n: ps.load_classifier(D.SF_CACHE / "heads" / f"head_{n}.pt")[0] for n in V2_HEADS}
    print("frames2: loading backbone", flush=True)
    model = ps.load_backbone()
    for kind, stem, items in G.frame_sets():
        f_out = fd / f"{kind}_{stem}.npz"
        if f_out.exists() or not items:
            continue
        if time.time() - t0 > 12 * 60:
            print("frames2: out of time, run again", flush=True); return
        print(f"frames2: {kind} {stem} ({len(items)} frames)", flush=True)
        acc = {k: [] for k in ("frame", "t", "maps", "grey_mean")}
        for i in range(0, len(items), 8):
            chunk = items[i:i + 8]
            ims = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for _, _, p in chunk]
            res = ps.frame_classes(model, ims, heads)
            for (f, t, _), r in zip(chunk, res):
                acc["frame"].append(f); acc["t"].append(t); acc["grey_mean"].append(r["grey_mean"])
                acc["maps"].append(np.stack([r["maps"][n] for n in V2_HEADS]))
        np.savez_compressed(f_out, heads=np.array(V2_HEADS), **{k: np.asarray(v) for k, v in acc.items()})
    (fd / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("frames2 done", flush=True)


if __name__ == "__main__":
    {"frames": decode_frames, "grids": grids, "heads": step_heads, "frames2": step_frames2}[sys.argv[1]]()
