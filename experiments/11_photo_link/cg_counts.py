"""WP16 Q7: how many whales are outlined at once in the comp-group dev clip (2025-12-20 CG.m4v), from cached tracks.

Reads the three OWLv2 -> SAM 3 runs that cover the clip (5-22, 24-48, 61-76 s) after the shipped strict purity
guards (WP8c), and the AI by-eye reference frames (every 15th frame). CPU only, nothing fitted. Prints JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07_tracks_demo" / "pure"))
from uwid import tracks  # noqa: E402
import evaluate as EV  # noqa: E402  (blocked_for / feats_for of WP8c)

RUNS = Path("/home/sat3737/NOAA_whales/underwater_cache/tracks/2025-12-20_CG")
REF = Path(__file__).resolve().parents[1] / "02_reference_labels/labels/2025-12-20_CG/frames.csv"
EDGE = 0.02  # outline within 2 % of the frame height of the top edge


def per_run(d: Path, cfg, purify: bool) -> dict:
    print("run", d.name, "purify", purify, file=sys.stderr, flush=True)
    tr = EV.apply(d, cfg) if purify else tracks.load_tracks(d)
    print(" loaded", file=sys.stderr, flush=True)
    fps = tr.meta["fps"]
    h = int(tr.meta["store_size"][1])
    whales = [k for k, r in tr.roles.items() if r == "whale" and k in tr.number]
    n = tr.n
    vis = np.zeros((n, len(whales)), bool)
    top = np.zeros((n, len(whales)), bool)
    edge_rows = max(1, int(EDGE * h))
    for j, k in enumerate(whales):
        p = tr.packed[k]                                   # (n, h, w/8) packed bits
        vis[:, j] = p.reshape(n, -1).any(1)
        top[:, j] = p[:, :edge_rows].reshape(n, -1).any(1)
    for (i, o) in tr.hidden:
        if o in whales:
            vis[i, whales.index(o)] = False
    cnt = vis.sum(1)
    sec = (np.arange(n) / fps).astype(int)
    per_s = pd.Series(cnt).groupby(sec).max()
    lens, ends_top, ends_inside, ends_runend = [], 0, 0, 0
    for j in range(len(whales)):
        idx = np.nonzero(vis[:, j])[0]
        if not len(idx):
            continue
        lens.append((idx[-1] - idx[0] + 1) / fps)
        if idx[-1] >= n - 2:
            ends_runend += 1
        elif top[idx[-1], j]:
            ends_top += 1
        else:
            ends_inside += 1
    return dict(run=d.name, purified=purify, t0=tr.meta["t0"], t1=tr.meta["t1"], whale_numbers=len(whales),
                max_at_once=int(cnt.max()), median_per_frame=float(np.median(cnt)),
                median_of_per_second_max=float(per_s.median()), max_of_per_second_max=int(per_s.max()),
                seconds_with_ge2=int((per_s >= 2).sum()), seconds_with_ge3=int((per_s >= 3).sum()), seconds=int(len(per_s)),
                number_life_s=dict(median=round(float(np.median(lens)), 2), max=round(float(np.max(lens)), 2),
                                   n_ge_3s=int(sum(x >= 3 for x in lens))),
                ends_top_edge=ends_top, ends_elsewhere=ends_inside, alive_at_run_end=ends_runend)


def main():
    cfg = tracks.PurityConfig()
    names = ("0005.00-0022.00_owl_sam3", "0024.00-0048.00_owl_sam3", "0061.00-0076.00_owl_sam3")
    out = {"runs": [], "runs_purified": []}
    part = Path("/home/sat3737/NOAA_whales/underwater_cache/photo_link/cg_counts_partial.json")
    for key, pur in (("runs", False), ("runs_purified", True)):
        for r in names:
            out[key].append(per_run(RUNS / r, cfg, pur))
            part.write_text(json.dumps(out, indent=1, default=str))  # survives a kill
    f = pd.read_csv(REF)
    uw = f[f.state.astype(str).str.startswith("under")] if "state" in f else f
    ow = pd.to_numeric(uw.other_whales, errors="coerce").fillna(0)
    fw = pd.to_numeric(uw.faint_whales, errors="coerce").fillna(0)
    out["reference_by_eye"] = dict(frames=int(len(f)), underwater_frames=int(len(uw)),
                                   states=f.state.value_counts().to_dict(),
                                   other_whales=dict(max=int(ow.max()), median=float(ow.median()),
                                                     hist=ow.astype(int).value_counts().sort_index().to_dict()),
                                   other_plus_faint=dict(max=int((ow + fw).max()), median=float((ow + fw).median())),
                                   clock_frame0=str(f.clock_est.iloc[0]), clip_s=float(f.time_s.max()))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
