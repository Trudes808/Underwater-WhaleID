"""Combine the time-link measurements and add video times to surfacings.csv.

Inputs (cache): clock_fits_mov.csv, sync_events_mov.csv (sync_video.py), camera_log.csv, surfacings.csv
(analyze_sensors.py). The drone-drop moments below were read by eye from full-rate frames of 00004.mov
(AI by-eye reading, 2 moments, +-1 frame).

Writes sync_summary.json and rewrites surfacings.csv with: piece (video sequence number), in_video (False if the
surfacing falls in a roll gap), first_frame_t (sensor time of that file's first frame, from 'Camera time'),
pts_exit_pred / pts_entry_pred (expected picture exit / entry in the .mov presentation time, using the burned-clock
fit when the file was read, else Camera time), and the measured picture times where available.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import tagdata as td  # noqa: E402

DROP_FILE = "CC-11-96-20250203-101236-000-00004.mov"
DROP_RELEASE_PTS = 90.83   # last frame with the drone still in place 90.814, drone moving away 90.847
DROP_IMPACT_PTS = 91.63    # last calm frame 91.615, first splash 91.648


def main(dep):
    out = td.cache_dir(dep)
    fits = pd.read_csv(out / "clock_fits_mov.csv")
    ev = pd.read_csv(out / "sync_events_mov.csv")
    drops = pd.read_csv(out / "drops.csv")
    surf = pd.read_csv(out / "surfacings.csv")
    surf = surf[[c for c in surf.columns if not c.startswith(("piece", "in_video", "first_frame", "pts_", "pic_"))]]
    df = td.load(dep)
    S = {}
    # 1. burned clock vs accelerometer at the drone drop
    f4 = fits[fits.file == DROP_FILE].iloc[0]
    att = drops[drops.attached].iloc[0]
    rel = f4.clock_at_pts0_t + f4.clock_rate * DROP_RELEASE_PTS - att.ff_start_t
    imp = f4.clock_at_pts0_t + f4.clock_rate * DROP_IMPACT_PTS - att.impact_t
    S["clock_minus_sensor_from_drop_s"] = {"release": rel, "impact": imp, "mean": (rel + imp) / 2,
                                           "picture_fall_s": DROP_IMPACT_PTS - DROP_RELEASE_PTS, "accelerometer_fall_s": att.ff_s}
    # 2. burned clock vs depth at surfacings
    ok = ev.dropna(subset=["pic_exit_pts"])
    for k in ["exit", "entry"]:
        x = ok[f"lag_{k}_clock_s"]
        S[f"clock_minus_depth_{k}_s"] = {"n": int(len(x)), "median": float(x.median()), "mad": float((x - x.median()).abs().median()),
                                         "q10_q90": [float(x.quantile(.1)), float(x.quantile(.9))], "min_max": [float(x.min()), float(x.max())],
                                         "per_file_median": {f: float(g.median()) for f, g in ok.groupby("file")[f"lag_{k}_clock_s"]}}
        y = ok[f"lag_{k}_name_s"]
        S[f"name_minus_depth_{k}_s"] = {"median": float(y.median()), "mad": float((y - y.median()).abs().median()),
                                        "per_file_median": {f: float(g.median()) for f, g in ok.groupby("file")[f"lag_{k}_name_s"]}}
    off = S["clock_minus_sensor_from_drop_s"]["mean"]
    # saturation cue (automatic, biased: frames looking up at the surface from just below count as 'air')
    S["depth_exit_picture_lag_s_satcue"] = S["clock_minus_depth_exit_s"]["median"] - off
    S["depth_entry_picture_lag_s_satcue"] = S["clock_minus_depth_entry_s"]["median"] - off
    # AI by-eye reading of frame strips (9 exits, 9 entries, see tables.md): lens fully out of the water about
    # 0.3 s after the depth trace crosses air+0.10 m; picture clearly under water again about 0.4 s before it
    S["depth_exit_picture_lag_s"] = td.DEPTH_EXIT_PIC_LAG_S
    S["depth_entry_picture_lag_s"] = td.DEPTH_ENTRY_PIC_LAG_S
    S["picture_air_minus_sensor_exposure_s_median"] = float(((ok.pic_entry_pts - ok.pic_exit_pts) - (ok.sensor_end_t - ok.sensor_start_t)).median())
    # 3. file name and Camera time against the clock-based first frame
    fits["first_frame_t"] = fits.clock_at_pts0_t - off
    ff = td.first_frames_from_camera_time(df)
    m = fits.merge(ff, left_on="seq", right_on="piece", how="left", suffixes=("", "_ct"))
    S["name_minus_first_frame_s"] = (m.t_name - m.first_frame_t).round(3).tolist()
    S["camtime_first_frame_minus_clock_first_frame_s"] = (m.first_frame_t_ct - m.first_frame_t).round(3).tolist()
    S["clock_drift_ppm_per_file"] = fits.clock_drift_ppm.round(2).tolist()
    S["clock_fit_rms_s_per_file"] = fits.clock_fit_rms_s.round(4).tolist()
    S["video_minus_name_s"] = float(np.median(m.first_frame_t - m.t_name))
    S["files"] = fits.file.tolist()
    S["pieces_from_camera_time"] = ff.round(2).to_dict("records")
    (out / "sync_summary.json").write_text(json.dumps(S, indent=1, default=float))

    # 4. video columns for the surfacings
    dur_default = float(fits.duration_s.median())
    rows = []
    for s in surf.itertuples():
        p = ff[(ff.first_frame_t <= s.start_t)].tail(1)
        r = {"piece": np.nan, "in_video": False, "first_frame_t": np.nan, "pts_exit_pred": np.nan, "pts_entry_pred": np.nan,
             "pts_source": "", "pic_exit_pts": np.nan, "pic_entry_pts": np.nan}
        if len(p):
            p = p.iloc[0]
            fr = fits[fits.seq == p.piece]
            if len(fr):
                f = fr.iloc[0]
                t0v, rate, dur, src = f.first_frame_t, f.clock_rate, f.duration_s, "burned clock"
            else:
                t0v, rate, dur, src = p.first_frame_t, 1.0, dur_default, "camera time"
            pe = (s.start_t + S["depth_exit_picture_lag_s"] - t0v) / rate
            pn = (s.end_t + S["depth_entry_picture_lag_s"] - t0v) / rate
            r.update(piece=int(p.piece), first_frame_t=t0v, pts_exit_pred=pe, pts_entry_pred=pn, pts_source=src,
                     in_video=bool(pe > 0 and pn < dur))
            e = ev[(ev.surfacing_id == s.surfacing_id)]
            if len(e):
                r.update(pic_exit_pts=e.iloc[0].pic_exit_pts, pic_entry_pts=e.iloc[0].pic_entry_pts)
        rows.append(r)
    surf = pd.concat([surf.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    surf.to_csv(out / "surfacings.csv", index=False)
    print(json.dumps({k: v for k, v in S.items() if k != "pieces_from_camera_time"}, indent=1, default=float))
    print("surfacings in video:", int(surf.in_video.sum()), "of", len(surf), "; in a roll gap:",
          int((~surf.in_video & surf.piece.notna()).sum()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-2-3 Mini-mother 11-96")
