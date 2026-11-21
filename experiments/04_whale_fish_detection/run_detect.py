"""Run one text-prompted detector over a video and cache the boxes (WP5a). Resumable: run the same command again
until it prints DONE. Every call must hold the GPU lock:

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04_whale_fish_detection/run_detect.py \
        --video "/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v" --model owl --prompts owl_all --stride 5

Run names: ``<model>_<prompt set>_s<stride>[_tiles]`` (or ``_lab`` when only labelled frames +- window are run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from uwid import detect  # noqa: E402


def frames_around_labels(labels_csv: Path, fps: float, stride: int, n_frames: int) -> list[int]:
    """Every ``stride``-th frame (on the global grid) inside the 1 s windows holding a labelled frame, plus the
    labelled frames themselves."""
    import pandas as pd
    from uwid.eval_presence import window_of
    lab = pd.read_csv(labels_csv)
    wins = set()
    for w in window_of(lab["frame"], fps):
        wins.add(int(w))
    out = set()
    for w in wins:
        a, b = int(round(w * fps)), int(round((w + 1) * fps))
        out.update(f for f in range(max(a, 0), min(b, n_frames)) if f % stride == 0)
    out.update(int(f) for f in lab["frame"])
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", choices=["owl", "llmdet", "sam3", "yoloe"], required=True)
    ap.add_argument("--prompts", required=True, help="key in prompts.json")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--tiles", action="store_true", help="2x2 tiles (small fish)")
    ap.add_argument("--labels", help="frames.csv: run only the windows around labelled frames")
    ap.add_argument("--frame-range", help="a:b, only frames a..b-1")
    ap.add_argument("--extra-ref", help="WP3 reference dir of this video: also run every frame it saved as JPEG "
                                        "(the labelled frames), on top of the stride grid")
    ap.add_argument("--budget", type=float, default=780.0, help="seconds before stopping (resume later)")
    ap.add_argument("--min-score", type=float, default=None)
    args = ap.parse_args()

    queries = json.loads((HERE / "prompts.json").read_text())[args.prompts]
    info = detect.video_info(args.video)
    if args.model == "owl":
        runner = detect.OwlRunner(queries, min_score=args.min_score or 0.03)
    elif args.model == "yoloe":
        runner = detect.YoloeRunner(queries, min_score=args.min_score or 0.05)
    elif args.model == "sam3":
        runner = detect.Sam3Runner(queries, min_score=args.min_score or 0.1)
    else:
        runner = detect.GroundingRunner(queries, min_score=args.min_score or 0.05, batch=1)
    name = f"{args.model}_{args.prompts}_s{args.stride}"
    if args.tiles:
        runner = detect.TiledRunner(runner)
        name += "_tiles"
    frames = None
    if args.labels:
        frames = frames_around_labels(Path(args.labels), info["fps"], args.stride, info["n_listed"])
        name += "_lab"
    if args.extra_ref:
        extra = sorted(int(p.stem) for p in Path(args.extra_ref).rglob("*.jpg") if p.stem.isdigit())
        frames = sorted(set(range(0, info["n_listed"], args.stride)) | set(extra))
    if args.frame_range:
        a, b = map(int, args.frame_range.split(":"))
        base = frames if frames is not None else range(0, info["n_listed"], args.stride)
        frames = [f for f in base if a <= f < b]
        name += f"_f{a}-{b}"
    done = detect.run_cached(args.video, runner, name, frames=frames, stride=args.stride, time_budget_s=args.budget)
    print("DONE" if done else "NOT DONE (run again)", name)


if __name__ == "__main__":
    main()
