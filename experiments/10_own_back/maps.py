"""GPU (under the lock, pieces of at most ~4 min): the tracker's patch classifier (WP5b `clf_w1280_h0_all.pt`, as
trackstage.patch_pass) on the frames sample.py decoded -> ownback/maps/<stem>.npz (frames, labels 45 x 80 uint8,
probabilities of back 45 x 80 float16). Resumable per recording; writes ownback/maps/DONE when all are done.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 300 nice -n 10 .venv/bin/python experiments/10_own_back/maps.py
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
import torch  # noqa: E402

import common as C  # noqa: E402
from uwid import patchseg as ps  # noqa: E402

BUDGET_S = 200


def main():
    t0 = time.time()
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8.5e9 / torch.cuda.get_device_properties(0).total_memory))
    idx = pd.read_csv(C.OUT / "frames" / "index.csv")
    md = C.OUT / "maps"; md.mkdir(exist_ok=True)
    print("maps: loading", flush=True)
    model = ps.load_backbone()
    clf, _ = ps.load_classifier(C.WP5B_HEAD)
    for stem, g in idx.groupby("stem"):
        out = md / f"{stem}.npz"
        if out.exists():
            continue
        if time.time() - t0 > BUDGET_S:
            print("maps: out of time, run again", flush=True); return
        frames = sorted(set(int(f) for f in g.frame))
        labs, pb = [], []
        for i in range(0, len(frames), 8):
            ims = [cv2.cvtColor(cv2.imread(str(C.OUT / "frames" / stem / f"{f:06d}.jpg")), cv2.COLOR_BGR2RGB)
                   for f in frames[i:i + 8]]
            with torch.no_grad():
                feats = ps.extract(model, ims, 1280)
                for j in range(len(ims)):
                    p = ps.predict_probs(clf, feats[j])
                    labs.append(p.argmax(-1).to(torch.uint8).cpu().numpy())
                    pb.append(p[..., ps.BACK].half().cpu().numpy())
        np.savez_compressed(out, frames=np.array(frames), labels=np.stack(labs), p_back=np.stack(pb))
        print(f"maps {stem}: {len(frames)} frames", flush=True)
    (md / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("maps done", flush=True)


if __name__ == "__main__":
    main()
