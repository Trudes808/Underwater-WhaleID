"""GPU steps of 08_whale_fish_boxes (run under the GPU lock; each step resumable per recording).

    owl   OWLv2 with the scan's prompts and settings (uwid.detect.OwlRunner, patch scores, min score 0.03, GPU
          preprocessing) on every labelled frame and its partner frame (+0.5 s) -> CACHE/owl/<stem>.parquet
          (column ``kind`` = label / partner) and <stem>.frames.npy
    feat  DINOv3 crop embeddings (tight and 2x context square crops, class token + mean patch token) of every
          unique patch box with a whale / fish prompt score >= 0.2 on the labelled frames -> CACHE/feat/<stem>.npz

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/08_whale_fish_boxes/extract.py owl feat
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402
from uwid import boxclf as B  # noqa: E402

BUDGET_S = 13 * 60


def _read(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def step_owl(t_start: float):
    from uwid import detect
    out = D.CACHE / "owl"; out.mkdir(parents=True, exist_ok=True)
    runner = None
    for rec in D.recordings():
        f_out = out / f"{rec.stem}.parquet"
        if f_out.exists():
            continue
        if time.time() - t_start > BUDGET_S:
            print("owl: out of time, run again"); return False
        runner = runner or detect.OwlRunner(list(B.PROMPTS), patch_scores=True, batch=4, gpu_preprocess=True)
        items = []
        for f in rec.labelled_frames():
            p = rec.image_path(f)
            if p is not None:
                items.append((f, "label", p))
            q = rec.partner_path(f)
            if q is not None and rec.source != "dev":
                items.append((f + D.PARTNER_STEP, "partner", q))
        rows, done = [], []
        t = time.time()
        for i in range(0, len(items), 8):
            chunk = items[i:i + 8]
            ims = [_read(p) for _, _, p in chunk]
            dets = runner.detect(ims)
            for (f, kind, _), df in zip(chunk, dets):
                df = df.copy(); df.insert(0, "frame", int(f)); df["kind"] = kind
                rows.append(df); done.append((f, kind))
        df = pd.concat(rows, ignore_index=True)
        df.to_parquet(f_out, index=False)
        np.save(out / f"{rec.stem}.frames.npy", np.array([(f, 0 if k == "label" else 1) for f, k in done]))
        print(f"owl {rec.stem}: {len(items)} frames, {len(df)} rows, {time.time() - t:.0f} s", flush=True)
    return True


def step_feat(t_start: float):
    import torch
    from uwid import patchseg
    out = D.CACHE / "feat"; out.mkdir(parents=True, exist_ok=True)
    model = None
    for rec in D.recordings():
        f_out = out / f"{rec.stem}.npz"
        if f_out.exists():
            continue
        if time.time() - t_start > BUDGET_S:
            print("feat: out of time, run again"); return False
        model = model or patchseg.load_backbone("cuda", torch.float16)
        boxes = pd.read_parquet(D.CACHE / "owl" / f"{rec.stem}.parquet")
        boxes = boxes[boxes.kind == "label"]
        t = time.time()
        keep = {k: [] for k in ("frame", "x0", "y0", "x1", "y1", "tight", "ctx", "partner_tight", "has_partner")}
        for f, g in boxes.groupby("frame"):
            u = B.unique_boxes(g.drop(columns=["kind"]))
            if not len(u):
                continue
            img = _read(rec.image_path(int(f)))
            xy = u[["x0", "y0", "x1", "y1"]].to_numpy(float)
            e = B.crop_embeddings(model, img, xy, (1.0, 2.0))
            # step 4 cue: the same square 0.5 s later (a fish that moved leaves; a whale part holds)
            pp = rec.partner_path(int(f))
            if pp is not None:
                ep = B.crop_embeddings(model, _read(pp), xy, (1.0,))[1.0]
            else:
                ep = np.zeros_like(e[1.0])
            for k in ("x0", "y0", "x1", "y1"):
                keep[k].append(u[k].to_numpy(float))
            keep["frame"].append(np.full(len(u), int(f)))
            keep["tight"].append(e[1.0].astype(np.float16)); keep["ctx"].append(e[2.0].astype(np.float16))
            keep["partner_tight"].append(ep.astype(np.float16)); keep["has_partner"].append(np.full(len(u), pp is not None))
        np.savez(f_out, **{k: np.concatenate(v) if v else np.zeros(0) for k, v in keep.items()})
        print(f"feat {rec.stem}: {sum(len(x) for x in keep['frame'])} boxes, {time.time() - t:.0f} s", flush=True)
    return True


if __name__ == "__main__":
    import torch
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(min(1.0, 8.5e9 / torch.cuda.get_device_properties(0).total_memory))
    t0 = time.time()
    ok = True
    for step in sys.argv[1:]:
        ok = {"owl": step_owl, "feat": step_feat}[step](t0) and ok
    if ok:
        (D.CACHE / f"extract_{'_'.join(sys.argv[1:])}.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("done" if ok else "partial", f"{time.time() - t0:.0f} s")
