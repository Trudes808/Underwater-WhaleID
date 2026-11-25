"""NOT RUN (kept for reference). Its first launch (2026-11-22) hung for 1 h 43 min holding the shared GPU lock: the
script had no ``__main__`` guard, the spawned decode child died at start-up and the parent waited on its pipe for
ever (ended by the orchestrator). window_classes now decodes through patchseg._iter_decoded, which raises when the
child dies or stalls. The cost figures in tables.md come from gpu.py / decode.py / extract.py timings instead.
If it is ever run: ``flock <lock> timeout 900 nice -n 10 .venv/bin/python experiments/09_seafloor_sand/cost.py``.

Cost of the delivered calls (GPU, under the lock): uwid.patchseg.window_classes on the first 120 s of a 4K
recording of 2025-01-02 (00023) and of a 1080p recording of 2025-02-03 (00006), and uwid.boxclf.classify on the
extra boxes' frames. Writes results/cost.json. Timing only: nothing is scored here.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/09_seafloor_sand/cost.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "08_whale_fish_boxes"))
import data as D  # noqa: E402
from uwid import boxclf as B, patchseg as ps  # noqa: E402



def main():
    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8.5e9 / torch.cuda.get_device_properties(0).total_memory))
    out = {}
    model = ps.load_backbone()
    for name, path in (("d2 00023 4K", D.VIDEO / "TestSubset 2025-1-2 CATS -85" / "Video" / "CC-09-85-HRA-20250102-154322-000-00023.mov"),
                       ("d1 00006 1080p", D.VIDEO / "2025-2-3 Mini-mother 11-96" / "Video" / "CC-11-96-20250203-111250-000-00006.mov")):
        torch.cuda.reset_peak_memory_stats()
        df = ps.window_classes(path, model=model, windows=np.arange(0, 120.0))
        out[name] = {**df.attrs, "windows": len(df), "seafloor": int(df.seafloor.sum()), "sand": int(df.sand.sum()),
                     "dark": int(df.dark.sum()), "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                     "gpu_s_per_video_h": df.attrs["gpu_model_s"] / 120 * 3600, "wall_s_per_video_h": df.attrs["wall_s"] / 120 * 3600}
        print(name, out[name], flush=True)
    # classify: the extra boxes' frames with the scan's detector rows
    import cv2, glob, pandas as pd  # noqa: E402,E401
    s = pd.read_csv(D.CACHE / "extra" / "chosen.csv")
    SCAN = Path("/home/sat3737/NOAA_whales/underwater_cache/scan")
    names = {"d1": "2025-2-3 Mini-mother 11-96", "d2": "TestSubset 2025-1-2 CATS -85"}
    ts = []
    B._LOADED["model"] = model
    for r in s.head(40).itertuples():
        rows = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(SCAN / names[r.source] / "cache" / "det" / f"{r.stem}.mov" / "part_*.parquet")))])
        g = rows[rows.frame == r.frame]
        img = cv2.cvtColor(cv2.imread(str(D.CACHE / "extra" / f"{r.stem}_{r.frame:06d}.jpg")), cv2.COLOR_BGR2RGB)
        torch.cuda.synchronize(); t = time.time()
        res = B.classify(g, img)
        torch.cuda.synchronize(); ts.append(time.time() - t)
    out["classify"] = {"frames": len(ts), "median_s": float(np.median(ts[3:])), "mean_s": float(np.mean(ts[3:]))}
    print(out["classify"])
    (HERE / "results" / "cost.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":   # window_classes decodes in a spawned process
    main()
