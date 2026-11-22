"""WP8b GPU queue, run under the GPU lock: patch classes for the 2025-02-03 cases, the longer dev base stretch, the
real cases (variants a, b, c), then the synthetic cut runs. Finished items are skipped; a new item starts only while
less than --budget-s seconds have passed, so one locked command stays short. Run again until it prints 'all done'.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/07_tracks_demo/gap/run_gpu.py --budget-s 540
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blocked as B  # noqa: E402
import real as R  # noqa: E402
import synthetic as S  # noqa: E402
from cases import BASES, REAL, tracks  # noqa: E402


def done(d: Path) -> bool:
    return (d / "meta.json").exists() and json.loads((d / "meta.json").read_text()).get("done", False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, default=540)
    a = ap.parse_args()
    t_start = time.time()
    items = [("patch", None)]
    items += [("base", b) for b in BASES]
    items += [("real", (cid, var)) for cid in REAL for var in ("a", "b", "c")]
    items += [("cut", p) for p in S.plan()]
    left = 0
    for kind, x in items:
        if kind == "patch":
            if all((B.CACHE / f"patch_{c}.npz").exists() for c, r in REAL.items() if r[0] != tracks.DEV_VIDEO):
                continue
        elif kind == "base":
            v, t0, t1 = BASES[x]
            if done(tracks.tracks_dir(v, t0, t1, "owl_sam3")):
                continue
        elif kind == "real":
            if done(R.run_dir(*x)):
                continue
        else:
            if done(tracks.tracks_dir(x["video"], 0, 0, "owl_sam3").with_name(x["name"])):
                continue
        if time.time() - t_start > a.budget_s:
            left += 1
            continue
        print(time.strftime("%H:%M:%S"), kind, x if kind != "cut" else x["name"], flush=True)
        if kind == "patch":
            B.cmd_patch()
            import torch
            torch.cuda.empty_cache()
            B.cmd_rule()
        elif kind == "base":
            v, t0, t1 = BASES[x]
            tracks.run_tracks(v, t0, t1)
        elif kind == "real":
            R.run(*x)
        else:
            tracks.run_tracks(x["video"], 0, 0, frames=(x["start"], x["end"]), hide=[(x["ga"], x["gb"])], name=x["name"])
    print("all done" if left == 0 else f"{left} items left", flush=True)


if __name__ == "__main__":
    main()
