"""WP13: does the frame store (trackstage frame_store flag) give the same result as decoding? Two spans already
finished today by the decode path are run again from a store and compared frame by frame (masks, table, patch
labels, blocked frames, numbers).

    nice -n 10 .venv/bin/python wp13_store_check.py decode [i]        # no lock: fill the store (span i only)
    flock <lock> timeout 900 nice -n 10 .venv/bin/python -u wp13_store_check.py run
    .venv/bin/python wp13_store_check.py compare
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from uwid import paths, trackstage as T, tracks  # noqa: E402

STORE = paths.cache_dir("tracks") / "tmp_frames"
CHECK = [("TestSubset 2025-1-2 CATS -85", "/home/sat3737/NOAA_whales/underwater_video/TestSubset 2025-1-2 CATS -85/"
          "Video/CC-09-85-HRA-20250102-141304-000-00017.mov", 290.0, 291.0),
         ("2025-2-3 Mini-mother 11-96", tracks.TEST_DIR + "/CC-11-96-20250203-104242-000-00005.mov", 1565.15, 1579.5)]


def ref_dir(d: Path) -> Path:
    return d.with_name(d.name + "_ref_decoded")


def cmd_decode():
    for dep, src, a, b in CHECK:
        f0, f1, _ = T.span_start(dep, src, a, b)
        m = json.loads((T.span_dir(dep, Path(src).name, a, b) / "meta.json").read_text())
        assert [f0, f1] == m["frames"], (f0, f1, m["frames"])
        print(T.store_frames(src, f0, f1, STORE), flush=True)


def cmd_run():
    print("store check run start", flush=True)
    for dep, src, a, b in CHECK:
        d = T.span_dir(dep, Path(src).name, a, b)
        r = ref_dir(d)
        if not r.exists():
            shutil.copytree(d, r)
        for f in ("patch.npz", "meta.json"):
            (d / f).unlink(missing_ok=True)
        T.run_span(dep, src, a, b, force=True, frame_store=STORE)


def cmd_compare():
    for dep, src, a, b in CHECK:
        d = T.span_dir(dep, Path(src).name, a, b)
        r = ref_dir(d)
        A, B = tracks.read_run(r), tracks.read_run(d)
        same = dict(frames=A.meta["frames"] == B.meta["frames"], blocked=A.meta["blocked"] == B.meta["blocked"],
                    objects=sorted(A.packed) == sorted(B.packed),
                    from_store=B.meta.get("frames_from"))
        za, zb = np.load(r / "patch.npz"), np.load(d / "patch.npz")
        same["patch_labels"] = bool(np.array_equal(za["frames"], zb["frames"]) and np.array_equal(za["labels"], zb["labels"]))
        diff_frames = 0
        for k in set(A.packed) | set(B.packed):
            pa, pb = A.packed.get(k), B.packed.get(k)
            if pa is None or pb is None or pa.shape != pb.shape:
                diff_frames += max(len(pa) if pa is not None else 0, len(pb) if pb is not None else 0)
                continue
            diff_frames += int((pa.reshape(len(pa), -1) != pb.reshape(len(pb), -1)).any(1).sum())
        same["mask_frames_differing"] = diff_frames
        ta, tb = A.table.drop(columns=["score"]), B.table.drop(columns=["score"])
        same["table"] = ta.shape == tb.shape and bool((ta.values == tb.values).all())
        same["scores_max_abs_diff"] = float(np.abs(A.table.score.values - B.table.score.values).max()) \
            if len(A.table) == len(B.table) else None
        same["endings"] = A.meta.get("endings") == B.meta.get("endings")
        same["refused"] = A.meta.get("refused") == B.meta.get("refused")
        la, lb = T.load_online(r), T.load_online(d)
        same["numbers"] = (la.number, lb.number)
        same["seconds"] = (A.meta["seconds_total"], B.meta["seconds_total"])
        print(d.parent.name[-5:], d.name, same, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:                       # one span at a time (the store stays under 3 GB)
        CHECK[:] = [CHECK[int(sys.argv[2])]]
    {"decode": cmd_decode, "run": cmd_run, "compare": cmd_compare}[sys.argv[1]]()
