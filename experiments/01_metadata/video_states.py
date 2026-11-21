"""Seconds of each tag state inside each video piece (sensor log only: pieces from 'Camera time').

Also lists the off-whale stretches as (piece, pts_from, pts_to, state) for the clip tool. The on-whale period is
taken from impact of the attached drop to the release estimate (gyro jump at depth), not from the depth > 1 m rule.

    .venv/bin/python experiments/01_metadata/video_states.py "2025-2-3 Mini-mother 11-96"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import tagdata as td  # noqa: E402


def main(dep):
    out = td.cache_dir(dep)
    df = td.load(dep)
    st = pd.read_parquet(out / "states_50hz.parquet")
    N = json.loads((out / "sensor_numbers.json").read_text())
    drops = pd.read_csv(out / "drops.csv")
    t = st.t.to_numpy()
    s = st.state.astype(str).to_numpy().copy()
    meta = td.load_meta(dep)
    t0_local = pd.Timestamp(meta["first_utc"]) + pd.Timedelta(hours=meta["utc_offset_h"])
    rel = (pd.Timestamp(t0_local.date().isoformat() + " " + N["tag_off"]["release_est_local"]) - t0_local).total_seconds()
    on = float(drops[drops.attached].impact_t.iloc[0])
    # refine: whale = [impact, release]; in-water-after-impact but before 'whale' and ascent after release -> whale / floating
    s[(t >= on) & (t <= rel)] = "whale"
    s[(t > rel) & (s == "whale")] = "floating"   # the tag rising after release is not on the whale
    s[(t > rel) & (t < rel + 60) & (s == "air")] = "floating"  # breaking the surface, before the floating rule
    for r in drops[~drops.attached].itertuples():  # gap between impact and the floating rule kicking in
        k = (t >= r.impact_t) & (t < r.impact_t + 15) & (s == "air")
        s[k] = "floating"
    pieces = td.first_frames_from_camera_time(df)
    fits = pd.read_csv(out / "clock_fits_mov.csv")
    dur = float(fits.duration_s.median())
    rows, cuts = [], []
    for p in pieces.itertuples():
        a, b = p.first_frame_t, min(p.first_frame_t + dur, t[-1])
        k = (t >= a) & (t < b)
        row = {"piece": p.piece, "first_frame_t": a, "seconds": b - a}
        for name in ["air", "flight", "floating", "whale"]:
            row[name + "_s"] = float((s[k] == name).sum() / 50.0)
        rows.append(row)
        kk = np.nonzero(k)[0]
        ss = s[kk]
        ch = np.nonzero(ss[1:] != ss[:-1])[0] + 1
        for x, y in zip(np.concatenate([[0], ch]), np.concatenate([ch, [len(ss)]])):
            cuts.append({"piece": p.piece, "state": ss[x], "pts_from": t[kk[x]] - a, "pts_to": t[kk[y - 1]] - a,
                         "seconds": t[kk[y - 1]] - t[kk[x]]})
    R = pd.DataFrame(rows)
    C = pd.DataFrame(cuts)
    R.to_csv(out / "video_state_seconds.csv", index=False)
    C.to_csv(out / "video_state_spans.csv", index=False)
    tot = R[[c for c in R.columns if c.endswith("_s")]].sum()
    print(R.round(1).to_string())
    print(C[C.seconds > 1].round(1).to_string())
    print("total video s:", round(R.seconds.sum(), 1), "by state:", tot.round(1).to_dict(),
          "share not on whale:", round(1 - tot["whale_s"] / R.seconds.sum(), 3))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-2-3 Mini-mother 11-96")
