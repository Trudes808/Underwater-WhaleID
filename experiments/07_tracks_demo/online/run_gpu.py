"""WP11a GPU queue (run under the GPU lock; re-run until it prints 'all done'): the 10 selected spans
(results/selection.csv) and the 2 old stretches with WP5a start boxes, each with `uwid.trackstage.run_span`.
A new span starts only within the first --budget-s seconds of a locked command."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from uwid import trackstage as T, tracks  # noqa: E402

OLD = [("2025-12-20_CG", tracks.DEV_VIDEO, 24.0, 36.0),
       ("2025-2-3 Mini-mother 11-96", tracks.TEST_DIR + "/CC-11-96-20250203-111250-000-00006.mov", 456.0, 476.0)]


# whole encounters for the user's videos (kept stretch = first appearance - 2 s to last view + 2 s), run in pieces
ENCOUNTERS = [("TestSubset 2025-1-2 CATS -85", "/home/sat3737/NOAA_whales/underwater_video/TestSubset 2025-1-2 CATS -85/"
               "Video/CC-09-85-HRA-20250102-160604-000-00024.mov", 75.0, 164.0),
              ("2025-2-3 Mini-mother 11-96", tracks.TEST_DIR + "/CC-11-96-20250203-121302-000-00008.mov", 25.0, 103.0)]


def items():
    sel = pd.read_csv(HERE / "results" / "selection.csv")
    out = [(r.deployment, r.source, float(r.t0), float(r.t1), "scan") for r in sel.itertuples()]
    out += [(d, s, a, b, "wp5a") for d, s, a, b in OLD]
    for d, s, a, b in ENCOUNTERS:
        out += [(d, s, x, y, "scan") for x, y in T.pieces(a, b)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, default=540)
    a = ap.parse_args()
    print(time.strftime("%H:%M:%S"), "WP11a online queue start", flush=True)
    t0 = time.time()
    left = 0
    for dep, src, s, e, start in items():
        d = T.span_dir(dep, Path(src).name, s, e)
        if (d / "meta.json").exists() and json.loads((d / "meta.json").read_text()).get("done"):
            continue
        if time.time() - t0 > a.budget_s:
            left += 1
            continue
        print(time.strftime("%H:%M:%S"), dep, Path(src).name, s, e, start, flush=True)
        T.run_span(dep, src, s, e, start_source=start)
    for dep in {i[0] for i in items()}:
        T.write_index(dep)
    print("all done" if left == 0 else f"{left} spans left", flush=True)


if __name__ == "__main__":
    main()
