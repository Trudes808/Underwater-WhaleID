"""WP8c: forward / backward check. The same stretch tracked backwards (last frame first) gives a second set of numbers;
a forward number that coincides (IoU >= 0.5) with one backward number for >= 10 frames and with a different backward
number for >= 10 other frames maps to different objects in the two directions and is flagged at the frame where the
mapping changes. Run on demo clips 01, 07 and the dev breath (straight through), where WP8 / WP8b saw moves by eye.

    .venv/bin/python experiments/07_tracks_demo/pure/fb.py      # -> results/forward_backward.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as RU  # noqa: E402
from runs import tracks  # noqa: E402

PAIRS = {"01": (RU.demo_dir("01"), RU.backward_dir("01")), "07": (RU.demo_dir("07"), RU.backward_dir("07")),
         "R02a": (RU.real_dir("02", "a"), RU.backward_dir("R02"))}


def flags(fw, bw, min_frames: int = 10, iou: float = 0.5, step: int = 2) -> list[dict]:
    out = []
    fdisp = [k for k, r in fw.roles.items() if r in ("whale", "small")]
    bdisp = [k for k, r in bw.roles.items() if r in ("whale", "small", "duplicate")]
    lab = lambda tr, k: str(tr.number[k]) if k in tr.number else f"s{tr.small_number.get(k, '?')}"  # noqa: E731
    for k in fdisp:
        seq = []
        for i in range(0, fw.n, step):
            if not fw.visible(k, i):
                continue
            m = fw.mask(k, i)
            sc = {b: tracks._mask_iou(m, bw.mask(b, i)) for b in bdisp if bw.packed[b][i].any()}
            b = max(sc, key=sc.get) if sc else None
            seq.append((i, b if b is not None and sc[b] >= iou else None))
        cnt = pd.Series([b for _, b in seq if b is not None]).value_counts()
        big = [b for b, n in cnt.items() if n * step >= min_frames]
        if len(big) >= 2:
            # first frame where the dominant backward number changes between two big ones
            last, change = None, None
            for i, b in seq:
                if b in big:
                    if last is not None and b != last and change is None:
                        change = i
                    last = b
            t = fw.meta["t0"] + (change or 0) / fw.meta["fps"]
            out.append(dict(track=lab(fw, k), backward_numbers="/".join(lab(bw, b) for b in big),
                            change_t=round(float(fw.table[fw.table.i == change].t.iloc[0]) if change is not None
                                           else t, 2)))
    return out


def main():
    rows = []
    for key, (f, b) in PAIRS.items():
        fw, bw = tracks.load_tracks(f), tracks.load_tracks(b)
        for r in flags(fw, bw):
            r["run"] = key
            rows.append(r)
        print(key, "forward ids", len(fw.number), "backward ids", len(bw.number))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "results" / "forward_backward.csv", index=False)
    print(d.to_string())


if __name__ == "__main__":
    main()
