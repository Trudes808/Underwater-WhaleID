"""WP8c: every tracker run the purity work reuses (all cached by WP8 / WP8b; only the backward runs are new)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gap"))

from uwid import tracks  # noqa: E402
from clips import CLIPS  # noqa: E402
import cases  # noqa: E402

BACKWARD = {"01": CLIPS["01"][:3], "07": CLIPS["07"][:3], "R02": cases.REAL["02"][:3]}


def demo_dir(c: str) -> Path:
    v, t0, t1 = CLIPS[c][:3]
    return tracks.tracks_dir(v, t0, t1, "owl_sam3")


def real_dir(c: str, var: str) -> Path:
    import real
    return real.run_dir(c, var)


def backward_dir(key: str) -> Path:
    v, t0, t1 = BACKWARD[key]
    return tracks.tracks_dir(v, t0, t1, "owl_sam3").with_name(tracks.tracks_dir(v, t0, t1, "owl_sam3").name + "_rev")


def gap_dirs() -> list[Path]:
    import synthetic
    return [tracks.tracks_dir(p["video"], 0, 0, "owl_sam3").with_name(p["name"]) for p in synthetic.plan()]


def all_dirs() -> list[Path]:
    import synthetic
    out = [demo_dir(c) for c in CLIPS]
    out += [real_dir(c, v) for c in cases.REAL for v in "abc"]
    out += [synthetic.base_dir("B1")] + gap_dirs()
    out += [backward_dir(k) for k in BACKWARD]
    seen, res = set(), []
    for d in out:
        if d not in seen and (d / "meta.json").exists():
            seen.add(d)
            res.append(d)
    return res
