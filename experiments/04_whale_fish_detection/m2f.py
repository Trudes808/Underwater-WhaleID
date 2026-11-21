"""Mask2Former Swin-L (ADE20K semantic, COCO panoptic) on labelled frames: what does it call the back, the open water,
the whales, the bubbles, the surface, and does it ever say sand / rock (seafloor) here?

    # label maps (GPU, cached per frame at 1/4 size):
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04_whale_fish_detection/m2f.py run \
        --frames-dir /home/sat3737/NOAA_whales/underwater_cache/reference/2025-12-20_CG/frames --tag 2025-12-20_CG
    # tables (CPU):
    .venv/bin/python experiments/04_whale_fish_detection/m2f.py dev
    .venv/bin/python experiments/04_whale_fish_detection/m2f.py test --stem CC-11-96-20250203-101236-000-00004

Seafloor-like classes are fixed before looking at results: ADE `sand, rock, earth, floor, ground, mountain, hill,
land, field, dirt track`; COCO `sand, rock-merged, dirt-merged, gravel, mountain-merged, floor-other-merged`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import CLASS_IDS, DEV_LABELS, detect, ep, load_masks, mask_frames, to_md

MODELS = {"ade": "facebook/mask2former-swin-large-ade-semantic",
          "coco": "facebook/mask2former-swin-large-coco-panoptic"}
SEAFLOOR = {"ade": ["sand", "rock", "earth", "floor", "ground", "mountain", "hill", "land", "field", "dirt track"],
            "coco": ["sand", "rock-merged", "dirt-merged", "gravel", "mountain-merged", "floor-other-merged"]}
REF = Path("/home/sat3737/NOAA_whales/underwater_cache/reference")
DS = 4   # label maps stored at 1/4 size (480 x 270)


def cache_dir(tag, model):
    return detect.paths.cache_dir("detect") / tag / f"m2f_{model}"


def run(frames_dir: Path, tag: str, models=("ade", "coco"), budget_s: float = 700.0):
    """Returns True when every frame has a label map for every model (False: budget reached, run again)."""
    import cv2
    t_all = time.time()
    files = sorted(frames_dir.rglob("*.jpg"))
    for mk in models:
        out = cache_dir(tag, mk)
        out.mkdir(parents=True, exist_ok=True)
        todo = [f for f in files if not (out / f"{f.stem}.npy").exists()]
        if not todo:
            continue
        seg = detect.Mask2FormerSeg(MODELS[mk])
        (out / "id2label.json").write_text(pd.Series(seg.id2label).to_json())
        t0 = time.time()
        for f in todo:
            if time.time() - t_all > budget_s:
                print("budget reached; run again")
                return False
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            lab = seg.segment(img)
            np.save(out / f"{f.stem}.npy", lab[DS // 2::DS, DS // 2::DS].astype(np.int16))
        dt = time.time() - t0
        print(f"{mk}: {len(todo)} frames in {dt:.0f} s ({len(todo) / dt:.1f} frames/s)")
        del seg
    return True


def load(tag, mk, frame):
    import json
    out = cache_dir(tag, mk)
    p = out / f"{frame:06d}.npy"
    if not p.exists():
        return None, None
    id2 = {int(k): v for k, v in json.loads((out / "id2label.json").read_text()).items()}
    return np.load(p), id2


def dev():
    tag = "2025-12-20_CG"
    for mk in ("ade", "coco"):
        # 1) mask frames: what each reference class is called
        rows = []
        for f in mask_frames():
            lab, id2 = load(tag, mk, f)
            if lab is None:
                continue
            cls, _, _ = load_masks(f)
            cls = cls[DS // 2::DS, DS // 2::DS]
            for cname, cid in CLASS_IDS.items():
                m = cls == cid
                if m.sum() < 20:
                    continue
                ids, n = np.unique(lab[m], return_counts=True)
                for i, k in zip(ids, n):
                    rows.append(dict(ref=cname, m2f=id2.get(int(i), str(i)), px=int(k)))
        d = pd.DataFrame(rows)
        tab = []
        for ref, g in d.groupby("ref"):
            tot = g.px.sum()
            top = g.groupby("m2f").px.sum().sort_values(ascending=False)
            sf = top[top.index.isin(SEAFLOOR[mk])].sum()
            tab.append(dict(reference_class=ref, pixels=int(tot),
                            top_labels=", ".join(f"{k} {v / tot:.0%}" for k, v in top.head(4).items()),
                            seafloor_like=f"{sf / tot:.1%}"))
        print(f"\n### Mask2Former {mk}: reference class -> its labels (31 mask frames, pixels at 1/4 size)\n")
        print(to_md(pd.DataFrame(tab)))
        # 2) labelled frames: how often seafloor-like pixels appear (dev has no seafloor: every one is a false alarm)
        labels = ep.load_frame_labels(DEV_LABELS)
        fr = []
        for f in labels.frame:
            lab, id2 = load(tag, mk, int(f))
            if lab is None:
                continue
            names = np.vectorize(lambda i: id2.get(int(i), ""))(lab)
            sfm = np.isin(names, SEAFLOOR[mk])
            fr.append(dict(frame=int(f), seafloor_frac=float(sfm.mean()),
                           seafloor_frac_bottom=float(sfm[int(0.7 * sfm.shape[0]):].mean())))
        fr = pd.DataFrame(fr).merge(labels[["frame", "state", "back_present"]], on="frame")
        print(f"\n{mk}: labelled frames {len(fr)}; frames with seafloor-like pixels > 1 % / 5 % / 20 % of the frame: "
              f"{(fr.seafloor_frac > 0.01).sum()} / {(fr.seafloor_frac > 0.05).sum()} / {(fr.seafloor_frac > 0.2).sum()} "
              f"(underwater with back: {((fr.seafloor_frac > 0.05) & fr.back_present & (fr.state == 'underwater')).sum()}"
              f" of {(fr.back_present & (fr.state == 'underwater')).sum()} at > 5 %)")
        fr.to_csv(cache_dir(tag, mk) / "seafloor_frac_dev.csv", index=False)


def test(stem):
    """Seafloor-like share on each labelled test frame against WP3's seafloor label."""
    lab_csv = Path(__file__).resolve().parents[1] / "02_reference_labels" / "labels" / stem / "frames.csv"
    labels = ep.load_frame_labels(lab_csv)
    for mk in ("ade", "coco"):
        fr = []
        for f in labels.frame:
            lab, id2 = load(stem, mk, int(f))
            if lab is None:
                continue
            names = np.vectorize(lambda i: id2.get(int(i), ""))(lab)
            fr.append(dict(frame=int(f), seafloor_frac=float(np.isin(names, SEAFLOOR[mk]).mean())))
        if not fr:
            print(mk, "no label maps for", stem)
            continue
        fr = pd.DataFrame(fr).merge(labels, on="frame")
        uw = fr[fr.state == "underwater"]
        c = ep.roc(uw.seafloor_frac, uw.seafloor_present)
        print(f"{stem} {mk}: underwater frames {len(uw)}, seafloor visible {int(uw.seafloor_present.sum())}, "
              f"unsure {int(uw.seafloor_unsure.sum())}; AUC {ep.auc(c):.3f}; at > 5 %: hits "
              f"{int(((uw.seafloor_frac > 0.05) & uw.seafloor_present).sum())}, false "
              f"{int(((uw.seafloor_frac > 0.05) & ~uw.seafloor_present & ~uw.seafloor_unsure).sum())}")
        fr.to_csv(cache_dir(stem, mk) / "seafloor_frac_test.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "dev", "test"])
    ap.add_argument("--frames-dir")
    ap.add_argument("--tag")
    ap.add_argument("--stem")
    a = ap.parse_args()
    if a.cmd == "run":
        print("DONE" if run(Path(a.frames_dir), a.tag) else "NOT DONE")
    elif a.cmd == "dev":
        dev()
    else:
        test(a.stem)


if __name__ == "__main__":
    main()
