"""Does running the OWLv2 processor on the GPU change the scores? 120 frames (40 per piece of 2025-02-03 00004 /
00006 / 00009, every 15th frame from a fixed start), "whale" score per frame with CPU vs GPU preprocessing.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/06_clips_e2e/gpu_preprocess_check.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from uwid import paths, video  # noqa: E402

if __name__ == "__main__":
    VD = paths.footage_dir() / "2025-2-3 Mini-mother 11-96" / "Video"
    ims = []
    for piece, f0 in (("00004", 20000), ("00006", 30000), ("00009", 12000)):
        v = next(p for p in VD.glob(f"*-{piece}.mov") if not p.name.startswith("._"))
        ims += [f.image for f in video.iter_frames_isolated(v, indices=list(range(f0, f0 + 15 * 40, 15)))]
    from uwid import detect
    out = {}
    for gp in (False, True):
        r = detect.OwlRunner(["whale", "fish"], gpu_preprocess=gp, patch_scores=True)
        r.detect(ims[:4])
        t = time.time()
        d = r.detect(ims)
        out[gp] = (time.time() - t, [float(x[x["query"] == "whale"].score.max()) if (x["query"] == "whale").any() else 0.0
                                     for x in d])
    a, b = np.array(out[False][1]), np.array(out[True][1])
    res = {"n_frames": len(ims), "cpu_s_per_frame": out[False][0] / len(ims), "gpu_s_per_frame": out[True][0] / len(ims),
           "max_abs_diff": float(np.abs(a - b).max()), "mean_abs_diff": float(np.abs(a - b).mean()),
           "decisions_at_0.25_differ": int(((a >= 0.25) != (b >= 0.25)).sum())}
    print(json.dumps(res, indent=1))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "gpu_preprocess_check.json").write_text(json.dumps(res, indent=1))
