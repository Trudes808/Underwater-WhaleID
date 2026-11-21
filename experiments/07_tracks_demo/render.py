"""Render finished stretches as outline videos + sheets, and write per-clip numbers (track summary, overlap with the
reference masks, cost) to results/<clip>_<method>.json. CPU only.

    .venv/bin/python experiments/07_tracks_demo/render.py 01 --method owl_sam3 [--name-suffix _sam2] [--no-video]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from uwid import tracks  # noqa: E402
from clips import CLIPS  # noqa: E402

SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
REF = Path("/home/sat3737/NOAA_whales/underwater_cache/reference")


def ref_dir(video: str) -> Path:
    return REF / tracks.stem_of(video).replace("2025-12-20_CG", "2025-12-20_CG") / "masks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--method", default="owl_sam3")
    ap.add_argument("--name-suffix", default="")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--crf", type=int, default=21)
    a = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    for k in a.clips:
        v, t0, t1, short, kind, run = CLIPS[k]
        d = tracks.tracks_dir(v, t0, t1, a.method)
        back_from = tracks.tracks_dir(v, t0, t1, "owl_sam3") if a.method == "text_sam3" else None
        tr = tracks.load_tracks(d, back_from=back_from)
        summ = tracks.track_summary(tr)
        rd = ref_dir(v)
        ov = tracks.reference_overlap(tr, rd) if rd.exists() else None
        meta = tr.meta
        dur = meta["n_frames"] / meta["fps"]
        res = dict(clip=k, video=Path(v).name, t0=t0, t1=t1, method=a.method, kind=kind, stage1_run=run,
                   n_frames=meta["n_frames"], seconds_video=round(dur, 2), seconds_total=meta["seconds_total"],
                   seconds_model=meta["seconds_model"], gpu_s_per_video_s=round(meta["seconds_total"] / dur, 2),
                   gpu_peak_gb=meta["gpu_peak_gb"], n_started=len(meta.get("events", [])),
                   roles={str(o): r for o, r in tr.roles.items()}, n_whale_tracks=len(tr.number),
                   tracks=summ.to_dict("records"),
                   overlap=None if ov is None or ov.empty else ov.to_dict("records"))
        if ov is not None and not ov.empty:
            w = ov[ov.what == "whale"]
            res["overlap_summary"] = dict(n_ref_whales=int(len(w)), matched=int((w.iou > 0).sum()),
                                          median_iou_all=round(float(w.iou.median()), 3) if len(w) else None,
                                          median_iou_matched=round(float(w[w.iou > 0].iou.median()), 3) if (w.iou > 0).any() else None,
                                          back_iou=[float(x) for x in ov[ov.what == "back"].iou])
        print(json.dumps({x: res[x] for x in ("clip", "method", "n_whale_tracks", "gpu_s_per_video_s", "gpu_peak_gb")}))
        print(summ.to_string())
        if ov is not None and not ov.empty:
            print(ov.to_string())
        if not a.no_video:
            name = f"demo_tracks_{k}_{short}{a.name_suffix}"
            r = tracks.render_tracks(tr, SHEETS / f"{name}.mp4", SHEETS / f"{name}.jpg", crf=a.crf)
            res.update(r)
        (HERE / "results" / f"{k}_{a.method}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
