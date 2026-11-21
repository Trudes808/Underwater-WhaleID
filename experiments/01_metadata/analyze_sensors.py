"""Sensor-only analysis of one deployment: channels, tag-state timeline, drone drops, surfacings, dives, camera.

Needs the cache from build_cache.py. Writes to underwater_cache/tagdata/<deployment>/:
states.csv (timeline), drops.csv, surfacings.csv, dives.csv, camera_log.csv, sensor_numbers.json

    .venv/bin/python experiments/01_metadata/analyze_sensors.py "2025-2-3 Mini-mother 11-96"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import tagdata as td  # noqa: E402

VIDEO_ROOT = Path("/home/sat3737/NOAA_whales/underwater_video")


def main(dep: str):
    out = td.cache_dir(dep)
    meta = td.load_meta(dep)
    csv = meta["csv"]
    t0 = np.datetime64(meta["first_utc"])
    off_h = meta["utc_offset_h"]
    df = td.clean(td.load(dep))
    t = df.t.to_numpy()
    fs = 50.0
    loc = lambda tt: str((t0 + np.timedelta64(int(round(tt * 1e6)), "us") + np.timedelta64(int(off_h * 3600), "s")))[11:22]
    N = {}

    # ---------------- channels
    stats = json.loads((out / "channel_stats.json").read_text())
    N["channel_rates_hz"] = {k: round(v["rate_hz"], 3) for k, v in stats.items()}
    N["depth_glitches"] = {"bins_50hz": int((df.depth >= td.DEPTH_GLITCH_M).sum()),
                           "value": float(df.depth[df.depth >= td.DEPTH_GLITCH_M].max())}
    upd = td.load_depth_updates(dep)
    upd = upd[upd.depth < td.DEPTH_GLITCH_M]
    steps = np.unique(np.round(np.abs(np.diff(upd.depth.to_numpy())), 4))
    N["depth_step_m"] = float(steps[steps > 0][0])

    # ---------------- states
    st = td.tag_states(df)
    seg = td.state_segments(st)
    # merge short blips (< 5 s) into their neighbours for the readable timeline, keep the raw one too
    seg["start_local"] = seg.start_t.map(loc)
    seg["end_local"] = seg.end_t.map(loc)
    seg.to_csv(out / "states_raw.csv", index=False)
    s = st["state"].astype(str).to_numpy().copy()
    # smoothing: segments shorter than 5 s take the state of the previous segment
    ch = np.nonzero(s[1:] != s[:-1])[0] + 1
    starts = np.concatenate([[0], ch])
    ends = np.concatenate([ch, [len(s)]])
    for a, b in zip(starts, ends):
        if (b - a) / fs < 5.0 and a > 0:
            s[a:b] = s[a - 1]
    st["state"] = pd.Categorical(s, categories=["air", "flight", "floating", "whale"])
    seg = td.state_segments(st)
    seg["start_local"] = seg.start_t.map(loc)
    seg["end_local"] = seg.end_t.map(loc)
    # camera on during each segment
    cam_on = (df.camera.to_numpy() == 12)
    seg["camera_on_frac"] = [cam_on[(t >= a) & (t <= b)].mean() for a, b in zip(seg.start_t, seg.end_t)]
    seg.to_csv(out / "states.csv", index=False)
    N["states"] = seg[["state", "start_local", "end_local", "duration_s", "camera_on_frac"]].round(3).to_dict("records")
    st[["t", "state"]].to_parquet(out / "states_50hz.parquet", index=False)

    # features per state (for the 'not on a whale' question)
    feat = {}
    whale = (st.state == "whale").to_numpy()
    d = df.depth_clean.to_numpy()
    groups = {"air": (st.state == "air").to_numpy(), "flight": (st.state == "flight").to_numpy(),
              "floating": (st.state == "floating").to_numpy(), "whale_exposed": whale & (d < 0.04),
              "whale_0-1m": whale & (d >= 0.04) & (d < 1), "whale_>5m": whale & (d > 5)}
    for k, m in groups.items():
        x = pd.DataFrame({"depth_raw": d[m], "acc_hf": df.acc_hf[m], "gyro_mag": df.gmag[m], "acc_mag": df.amag[m],
                          "light1": df.light1_clean[m], "light2": df.light2_clean[m], "temp": df.temp[m]})
        feat[k] = {"seconds": round(m.sum() / fs, 1), **{c: [round(float(v), 3) for v in x[c].quantile([.05, .5, .95])]
                                                          for c in x.columns}}
    N["features_by_state_q05_q50_q95"] = feat

    # ---------------- air offset and noise
    air = (st.state == "air").to_numpy()
    quiet = air & (df.acc_hf.to_numpy() < 0.3) & (df.gmag.to_numpy() < 100)
    pre = quiet & (t < seg[seg.state == "flight"].start_t.min())
    N["air_depth_pre_deploy"] = {"median": float(np.median(d[pre])), "mean": float(d[pre].mean()), "std": float(d[pre].std()),
                                 "q01_q99": [float(np.quantile(d[pre], .01)), float(np.quantile(d[pre], .99))],
                                 "seconds": float(pre.sum() / fs)}
    post_start = seg[seg.state == "floating"].end_t.max()
    post = quiet & (t > post_start)
    N["air_depth_post_recovery"] = {"median": float(np.median(d[post])), "min": float(d[post].min()),
                                    "q01_q99": [float(np.quantile(d[post], .01)), float(np.quantile(d[post], .99))],
                                    "seconds": float(post.sum() / fs)}
    air_off = N["air_depth_pre_deploy"]["median"]

    # ---------------- drops
    flights = seg[seg.state == "flight"][["start_t", "end_t"]].rename(columns={"start_t": "fstart", "end_t": "end_t"})
    drops = td.find_drops(csv, flights.rename(columns={"end_t": "end_t"}))
    drops["flight_start_t"] = flights.fstart.to_numpy()
    outcome = []
    for _, r in drops.iterrows():
        k = (t > r.impact_t) & (t < r.impact_t + 600)
        dmax = float(d[k].max())
        nxt = seg[(seg.start_t >= r.impact_t - 5)].iloc[0].state if (seg.start_t >= r.impact_t - 5).any() else ""
        first_deep = t[k][np.argmax(d[k] > 1.0)] if (d[k] > 1.0).any() else np.nan
        outcome.append({"max_depth_10min_m": dmax, "first_depth_gt_1m_after_impact_s": first_deep - r.impact_t,
                        "attached": bool(dmax > 1.0), "state_after": nxt})
    drops = pd.concat([drops, pd.DataFrame(outcome)], axis=1)
    for c in ["flight_start_t", "ff_start_t", "impact_t"]:
        drops[c.replace("_t", "_local")] = drops[c].map(loc)
    drops.to_csv(out / "drops.csv", index=False)
    N["drops"] = drops.round(3).to_dict("records")

    # tag off the whale: release at depth = gyro jump during the final ascent
    wsegs = seg[seg.state == "whale"]
    w_end = wsegs.end_t.max()
    i_end = np.searchsorted(t, w_end)
    gm = td.rolling_median(df.gmag.to_numpy(), 25)
    j = i_end
    base = np.median(df.gmag.to_numpy()[i_end - 50 * 120:i_end - 50 * 60])
    while j > i_end - 50 * 120 and not (gm[j] < 3 * base and d[j] > d[i_end - 50 * 60] - 0.5):
        j -= 1
    N["tag_off"] = {"last_whale_sample_local": loc(w_end), "release_est_local": loc(t[j]), "depth_at_release_m": float(d[j]),
                    "surface_reached_local": loc(t[i_end + np.argmax(d[i_end:] < 0.1)]),
                    "ascent_mps": float((d[j] - 0.1) / max(t[i_end + np.argmax(d[i_end:] < 0.1)] - t[j], 1e-3)),
                    "gyro_baseline_mrads": float(base)}

    # ---------------- surfacings and dives (on the whale)
    p = td.SurfacingParams(air_offset_m=air_off)
    ex = td.exposures(t, d, whale, p, light2=df.light2_clean.to_numpy())
    ex.insert(0, "surfacing_id", np.arange(len(ex)))
    ex["start_utc"] = [str(t0 + np.timedelta64(int(round(x * 1e6)), "us")) for x in ex.start_t]
    ex["end_utc"] = [str(t0 + np.timedelta64(int(round(x * 1e6)), "us")) for x in ex.end_t]
    ex["start_local"] = ex.start_t.map(loc)
    ex["end_local"] = ex.end_t.map(loc)
    # air reading of the depth sensor during each surfacing (offset drift)
    ex["air_reading_m"] = [float(np.median(df.depth.to_numpy()[(t >= a) & (t <= b)])) if b > a else np.nan
                           for a, b in zip(ex.start_t, ex.end_t)]
    ex["temp_c"] = [float(np.median(df.temp.to_numpy()[(t >= a) & (t <= b + 0.1)])) for a, b in zip(ex.start_t, ex.end_t)]
    dv = td.dives(t, d, whale, p)
    dv["start_local"] = dv.start_t.map(loc)
    dv["deep"] = dv.max_depth_m > 10
    dv.to_csv(out / "dives.csv", index=False)
    N["surfacings"] = {"n": int(len(ex)), "n_bouts": int(ex.bout_id.nunique()),
                       "duration_q05_q50_q95": [round(float(x), 2) for x in ex.duration_s.quantile([.05, .5, .95])],
                       "n_short_lt_1s": int((ex.duration_s < 1).sum()), "n_multi_part": int((ex.n_parts > 1).sum()),
                       "ascent_mps_q50": float(ex.ascent_mps.median()), "descent_mps_q50": float(ex.descent_mps.median()),
                       "air_reading_first_last": [float(ex.air_reading_m.iloc[:5].median()), float(ex.air_reading_m.iloc[-5:].median())],
                       "air_reading_vs_time_slope_m_per_h": float(np.polyfit(ex.start_t / 3600, ex.air_reading_m, 1)[0]),
                       "light2_sat_found": int(ex.light2_sat_start_t.notna().sum()),
                       "light2_start_minus_depth_start_s_q05_q50_q95": [round(float(x), 3) for x in (ex.light2_sat_start_t - ex.start_t).quantile([.05, .5, .95])],
                       "light2_end_minus_depth_end_s_q05_q50_q95": [round(float(x), 3) for x in (ex.light2_sat_end_t - ex.end_t).quantile([.05, .5, .95])]}
    N["dives"] = {"n_gt_1m": int(len(dv)), "n_gt_10m": int(dv.deep.sum()), "max_depth_m": float(dv.max_depth_m.max()),
                  "deep_dive_duration_min_q50": float(dv[dv.deep].duration_s.median() / 60),
                  "deep_dive_max_depth_q50": float(dv[dv.deep].max_depth_m.median())}

    # ---------------- camera from the log
    cam = df.camera.to_numpy()
    rec = df.cc_status.astype(str).str.startswith("R").to_numpy()
    pieces = []
    starting = td.runs(cam == 11)
    for k, (a, b) in enumerate(starting):
        r12 = np.nonzero(cam[b:] == 12)[0]
        s12 = b + r12[0] if r12.size else None
        e12 = s12 + np.argmax(cam[s12:] != 12) if s12 is not None and (cam[s12:] != 12).any() else len(cam) - 1
        rr = np.nonzero(rec[a:])[0]
        r0 = a + rr[0] if rr.size else None
        r1 = r0 + np.argmax(~rec[r0:]) if r0 is not None and (~rec[r0:]).any() else len(cam) - 1
        pieces.append({"piece": k + 1, "cam11_start_t": t[a], "cam12_start_t": t[s12], "cam12_end_t": t[e12],
                       "cc_rec_start_t": t[r0], "cc_rec_end_t": t[r1], "camera_time_at_cam12": float(df.camera_time.iloc[s12]),
                       "camera_time_at_cam12_end": float(df.camera_time.iloc[e12]),
                       "vid_kb_final": float(df.cc_vid_kb.iloc[max(e12 - 1, 0)]), "log_end": bool(e12 == len(cam) - 1)})
    cl = pd.DataFrame(pieces)
    for c in [c for c in cl.columns if c.endswith("_t")]:
        cl[c.replace("_t", "_local")] = cl[c].map(loc)
    cl["cam12_s"] = cl.cam12_end_t - cl.cam12_start_t
    cl["cc_rec_s"] = cl.cc_rec_end_t - cl.cc_rec_start_t
    cl.to_csv(out / "camera_log.csv", index=False)
    N["camera_pieces"] = cl[["piece", "cam11_start_local", "cam12_start_local", "cam12_end_local", "cc_rec_start_local",
                             "cam12_s", "camera_time_at_cam12", "vid_kb_final", "log_end"]].round(2).to_dict("records")
    # Camera time: seconds counter? compare its increments with wall time
    ct = df.camera_time.to_numpy()
    ctu = np.nonzero(np.diff(ct) != 0)[0] + 1
    N["camera_time"] = {"first_nonzero_local": loc(t[ctu[0]]), "value_first": float(ct[ctu[0]]), "last_value": float(ct[-1]),
                        "wall_s_first_to_last": float(t[-1] - t[ctu[0]]),
                        "increment_values": {str(k): int(v) for k, v in zip(*np.unique(np.diff(ct[ctu]), return_counts=True))},
                        "update_interval_s_q": [float(x) for x in np.quantile(np.diff(t[ctu]), [.01, .5, .99])]}
    # camera time during the roll gaps: does it stop?
    gaps = []
    for _, r in cl.iterrows():
        k = (t > r.cam12_end_t - 5) & (t < r.cam12_end_t + 60)
        gaps.append(float(np.ptp(ct[k])))
    N["camera_time_change_over_65s_around_rolls"] = gaps

    ex.to_csv(out / "surfacings.csv", index=False)
    (out / "sensor_numbers.json").write_text(json.dumps(N, indent=1, default=float))
    print(json.dumps(N, indent=1, default=float)[:20000])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-2-3 Mini-mother 11-96")
