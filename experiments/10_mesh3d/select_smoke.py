"""WP15 smoke test, step 1: apply the frame rule (tables.md S0) to the existing tracker runs and draw a contact sheet
of the best passing frame per (run, number, second) for the by-eye viewpoint pick.

    .venv/bin/python experiments/10_mesh3d/select_smoke.py

Writes underwater_cache/mesh3d/smoke/candidates.csv and sheet_NN.jpg there (a working sheet, not a review sheet).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import framerule as fr  # noqa: E402

SOURCES = [
    (fr.TRACKS / "2025-2-3 Mini-mother 11-96", ("00004", "00005", "00006", "00007")),
    (fr.TRACKS / "TestSubset 2025-1-2 CATS -85", ("00024",)),
    (fr.TRACKS / "2025-12-20_CG", ()),
]
OUT = fr.MESH / "smoke"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("select_smoke: start", flush=True)
    dfs, thumbs = [], []
    for dep, recs in SOURCES:
        for d in fr.run_dirs(dep, recs):
            frames: dict = {}
            df = fr.evaluate_run(d, frames_out=frames)
            print(d.parent.name, d.name, "rows", len(df), "final", int(df.final.sum()) if len(df) else 0, flush=True)
            if not len(df):
                continue
            dfs.append(df)
            fin = df[df.final].copy()
            fin["sec"] = fin.t.astype(int)
            best = fin.sort_values("sharp", ascending=False).groupby(["number", "sec"]).head(1)
            from uwid import trackstage
            tr = trackstage.load_online(d)
            for r in best.sort_values(["number", "t"]).itertuples():
                img = frames[r.frame].copy()
                m = tr.mask(r.obj, r.i).astype(np.uint8)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, (255, 60, 60), 2)
                img = cv2.resize(img, (480, 270))
                lab = f"{len(thumbs):03d} {r.video[-9:-4]} {r.t:.1f}s n{r.number} a{r.area_share:.2f}"
                cv2.putText(img, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                thumbs.append((img, dict(r._asdict())))
    allc = pd.concat(dfs, ignore_index=True)
    allc.to_csv(OUT / "candidates.csv", index=False)
    pd.DataFrame([t[1] for t in thumbs]).to_csv(OUT / "sheet_index.csv", index=False)
    per = 24
    for s in range(0, len(thumbs), per):
        tiles = [t[0] for t in thumbs[s:s + per]]
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
        cv2.imwrite(str(OUT / f"sheet_{s // per + 1:02d}.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("select_smoke: done", len(allc), "rows,", int(allc.final.sum()), "final,", len(thumbs), "thumbs", flush=True)


if __name__ == "__main__":
    main()
