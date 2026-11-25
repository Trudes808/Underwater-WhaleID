"""WP15: write the model inputs for a list of picked frames (repo .venv, CPU).

    .venv/bin/python experiments/10_mesh3d/prep_inputs.py PICKS.csv OUTDIR

PICKS.csv columns: id (NN_what), run (run folder), obj, i, frame. Writes OUTDIR/<id>.png (RGBA: frame + tracker
outline as alpha, 1920 x 1080; with --crop only the outline's box + 16 px) and OUTDIR/<id>_overlay.jpg (frame with the outline drawn, for the sheets).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import framerule as fr  # noqa: E402
from uwid import trackstage  # noqa: E402


def main(picks: str, outdir: str, crop: bool = False):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(picks)
    cache: dict = {}
    for r in df.itertuples():
        if (out / f"{r.id}.png").exists():
            continue
        tr = cache.setdefault(r.run, trackstage.load_online(r.run))
        rgba = fr.rgba_input(tr.meta["video"], int(r.frame), tr, int(r.obj), int(r.i))
        model_in = rgba
        if crop:   # TripoSG crops to the alpha box itself, so a crop with a margin is the same model input
            ys, xs = np.nonzero(rgba[..., 3])
            m = 16
            model_in = rgba[max(0, ys.min() - m):ys.max() + m + 1, max(0, xs.min() - m):xs.max() + m + 1]
        cv2.imwrite(str(out / f"{r.id}.png"), cv2.cvtColor(model_in, cv2.COLOR_RGBA2BGRA))
        ov = np.ascontiguousarray(rgba[..., :3])
        cs, _ = cv2.findContours((rgba[..., 3] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cs, -1, (255, 70, 70), 3)
        if crop:
            ov = cv2.resize(ov, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{r.id}_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(r.id, "written", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], crop="--crop" in sys.argv[3:])
