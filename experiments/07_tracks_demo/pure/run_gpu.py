"""WP8c GPU work (run under the GPU lock):
  backward  - the tracker run backwards over clips 01, 07 and the dev breath (forward / backward check)
  features  - DINOv3 patch features every 3rd frame of every reused run: patch classes per frame (for the
              bubble / surface / start screens) and, per track and sampled frame, the mean feature inside its mask
              (appearance drift) and its class shares. Written to <run dir>/pure_feats.npz and
              underwater_cache/tracks/pure/labels_<stem>.npz.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as RU  # noqa: E402
from runs import tracks  # noqa: E402

from uwid import video as uv  # noqa: E402

OUT = Path("/home/sat3737/NOAA_whales/underwater_cache/tracks/pure")
STEP = 3


def cmd_backward():
    for k, (v, t0, t1) in RU.BACKWARD.items():
        tracks.run_tracks(v, t0, t1, reverse=True, name=RU.backward_dir(k).name)


def cmd_features():
    import torch
    from uwid import patchseg as ps
    from uwid.detect import _torch_setup
    _torch_setup()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    model = ps.load_backbone()
    clf = ps.load_classifier(ps.CACHE / "clf_w1280_h0_all.pt")[0]
    runs = []
    for d in RU.all_dirs():
        if (d / "pure_feats.npz").exists():
            continue
        tr = tracks.load_tracks(d)
        runs.append(tr)
    by_video: dict[str, list] = {}
    for tr in runs:
        by_video.setdefault(tr.meta["video"], []).append(tr)
    for video, trs in by_video.items():
        src = video.replace(".mov", ".m4v")
        frames = sorted({f for tr in trs for f in range(tr.meta["frames"][0], tr.meta["frames"][0] + tr.n) if f % STEP == 0})
        acc = {id(tr): dict(i=[], obj=[], emb=[], cls=[]) for tr in trs}
        lab_f, lab = [], []
        batch = []

        def flush():
            if not batch:
                return
            with torch.no_grad():
                f = ps.extract(model, [b.image for b in batch], 1280)
                for j, fr in enumerate(batch):
                    feats = f[j].float()                                   # (45, 80, 768)
                    probs = ps.predict_probs(clf, f[j])
                    labels = probs.argmax(-1).cpu().numpy().astype(np.uint8)
                    lab_f.append(fr.index)
                    lab.append(labels)
                    gh, gw = labels.shape
                    for tr in trs:
                        i = fr.index - tr.meta["frames"][0]
                        if not (0 <= i < tr.n):
                            continue
                        for k, p in tr.packed.items():
                            if not p[i].any():
                                continue
                            m = tr.mask(k, i).astype(np.float32)
                            w = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)
                            if w.sum() < 0.5:
                                w = (w > 0).astype(np.float32)
                            wt = torch.from_numpy(w).to(feats.device)
                            e = (feats * wt[..., None]).sum((0, 1)) / wt.sum()
                            cls = np.bincount(labels.ravel(), weights=w.ravel(), minlength=probs.shape[-1]) / w.sum()
                            a = acc[id(tr)]
                            a["i"].append(i); a["obj"].append(k)
                            a["emb"].append(e.half().cpu().numpy()); a["cls"].append(cls.astype(np.float16))
                        if tr.back is not None and tr.back[i].any():
                            pass
            batch.clear()

        for fr in uv.iter_frames_at(src, frames):
            batch.append(fr)
            if len(batch) == 8:
                flush()
        flush()
        for tr in trs:
            a = acc[id(tr)]
            np.savez_compressed(tr.dir / "pure_feats.npz", i=np.array(a["i"], np.int32), obj=np.array(a["obj"], np.int32),
                                emb=np.stack(a["emb"]) if a["emb"] else np.zeros((0, 768), np.float16),
                                cls=np.stack(a["cls"]) if a["cls"] else np.zeros((0, 8), np.float16))
        np.savez_compressed(OUT / f"labels_{tracks.stem_of(video)}.npz", frames=np.array(lab_f), labels=np.stack(lab))
        print(Path(video).name, len(frames), "frames,", len(trs), "runs", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backward", "features"])
    a = ap.parse_args()
    {"backward": cmd_backward, "features": cmd_features}[a.cmd]()
