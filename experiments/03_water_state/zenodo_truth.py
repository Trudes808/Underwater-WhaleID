"""Window truth for the public CATS deployment mn200312-58 (Zenodo 4902325, Gough and Cade, CC0) from its depth.

The downloaded files are untrusted: run with ``python -I`` from outside the download folder, e.g.
    .venv/bin/python -I experiments/03_water_state/zenodo_truth.py [--out-lag 0.0 --in-lead 0.0] [--strips]

Time base (CATS toolbox, ``*movieTimes.mat`` and ``*_10Hzprh.mat``): sensor time = ``DN`` (MATLAB datenum,
local time, 10 Hz); video ``i`` starts at ``vidDN[i]`` (its file-name time) and a frame at presentation time
``pts`` is at ``vidDN[i] + pts / 86400`` (``frameTimes`` equal the .mov presentation times; checked).
Depth ``p`` is the toolbox's calibrated depth (m, 0 at the surface).

Port out of the water: ``p < THR`` (THR = 0.10 m: the air peak of ``p`` on the whale is centred at -0.01 m with
its upper tail ending near 0.10 m); re-immersions shorter than 1 s are merged, exposures shorter than 1 s dropped
(the same rules as WP1's surfacings). The lens lags (``--out-lag``, ``--in-lead``) come from a by-eye check of
frame strips around events (``--strips`` writes them to the scratch folder given by ``--strip-dir``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

Z = Path("/home/sat3737/NOAA_whales/underwater_cache/public/zenodo_4902325")
OUT = Path("/home/sat3737/NOAA_whales/underwater_cache/waterstate/zenodo_truth")
VIDEOS = {10: "CATS_58-20200312-181746-359-00011", 12: "CATS_58-20200312-183732-336-00013"}
THR = 0.10


def runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def cross(t, p, i, thr):
    """Interpolated time where p crosses thr between samples i-1 and i."""
    if i <= 0 or i >= len(p):
        return float(t[min(max(i, 0), len(t) - 1)])
    a, b = p[i - 1], p[i]
    f = (thr - a) / (b - a) if b != a else 0.5
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-lag", type=float, default=0.0)
    ap.add_argument("--in-lead", type=float, default=0.0)
    ap.add_argument("--strips", action="store_true")
    ap.add_argument("--strip-dir", default=None)
    a = ap.parse_args()
    import scipy.io as sio
    d = sio.loadmat(Z / "mn200312-58_10Hzprh.mat", squeeze_me=True, struct_as_record=False)
    p = np.asarray(d["p"], float)
    on = np.asarray(d["tagon"]).astype(bool)
    DN = np.asarray(d["DN"], float)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for i, name in VIDEOS.items():
        v0 = float(d["vidDN"][i])
        dur = float(d["vidDurs"][i])
        t = (DN - v0) * 86400.0           # seconds of this video's presentation time
        sel = (t > -60) & (t < dur + 60)
        tt, pp, oo = t[sel], p[sel], on[sel]
        exp = (pp < THR) & oo
        # merge re-immersions < 1 s, drop exposures < 1 s
        r = runs(exp)
        merged = []
        for s, e in r:
            if merged and tt[s] - tt[merged[-1][1] - 1] < 1.0:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        ex = []
        for s, e in merged:
            t_out = cross(tt, pp, s, THR)
            t_in = cross(tt, pp, e, THR)
            if t_in - t_out >= 1.0:
                ex.append((t_out, t_in))
        seg, evs = [], []
        cur = 0.0
        state_at0 = "air" if any(o <= 0 < i_ for o, i_ in ex) else "underwater"
        for t_out, t_in in ex:
            a0, b0 = t_out + a.out_lag, t_in - a.in_lead
            if b0 <= 0 or a0 >= dur or b0 <= a0:
                continue
            if a0 > cur:
                seg.append((cur, a0, "underwater"))
                evs.append((a0, "exit"))
            seg.append((max(a0, 0.0), min(b0, dur), "air"))
            if b0 < dur:
                evs.append((b0, "entry"))
            cur = min(b0, dur)
        if cur < dur:
            seg.append((cur, dur, "underwater"))
        segs = pd.DataFrame(seg, columns=["t0", "t1", "state"])
        segs = segs[segs.t1 > segs.t0]
        evdf = pd.DataFrame(evs, columns=["t", "kind"])
        evdf = evdf[(evdf.t > 0) & (evdf.t < dur)]
        segs.to_csv(OUT / f"{name}_segments.csv", index=False)
        evdf.to_csv(OUT / f"{name}_events.csv", index=False)
        raw = pd.DataFrame(ex, columns=["port_out_t", "port_in_t"])
        raw.to_csv(OUT / f"{name}_port_exposures.csv", index=False)
        meta = {"duration": dur, "vidDN": v0, "thr_m": THR, "out_lag": a.out_lag, "in_lead": a.in_lead,
                "n_exposures": len(ex), "air_s": float(sum(min(b, dur) - max(o, 0) for o, b in ex)),
                "state_at_0": state_at0, "tagon_frac": float(oo[(tt >= 0) & (tt < dur)].mean())}
        (OUT / f"{name}_meta.json").write_text(json.dumps(meta, indent=1))
        summary[name] = meta
        print(name, json.dumps(meta))
        print(raw.round(2).to_string())
        if a.strips:
            from uwid import video as uv, sheets
            sd = Path(a.strip_dir)
            sd.mkdir(parents=True, exist_ok=True)
            with uv.VideoReader(Z / f"{name}.mov", size=(320, 180)) as rd:
                for j, (t_out, t_in) in enumerate(ex[:8]):
                    for kind, te in (("exit", t_out), ("entry", t_in)):
                        if not (1.5 < te < dur - 1.5):
                            continue
                        ts = list(np.arange(te - 1.5, te + 1.51, 0.1))
                        fr = rd.frames_at(ts)
                        sheets.make_sheet([f.image for f in fr], [f"{f.t - te:+.1f}" for f in fr], cols=8,
                                          tile_width=240, title=f"{name} {kind} depth t={te:.2f}",
                                          out=sd / f"{name}_{j:02d}_{kind}.jpg")


if __name__ == "__main__":
    main()
