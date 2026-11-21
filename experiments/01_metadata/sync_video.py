"""Time link between the tag's sensor log and its videos.

For every stable video of the deployment:
1. burned-in clock: read every `step` s (12 consecutive frames each) and fit clock = a + b * pts;
2. compare the clock at pts 0 with the file-name start time and with the camera states in the log;
3. physical check: around every sensor surfacing (surfacings.csv) decode the frames, compute the mean HSV
   saturation of a 192x108 copy (underwater frames 150-220, frames with the lens in air 30-70) and take the longest
   run of frames below `s_thr` (gaps of up to 5 frames bridged) as the picture-side exposure. Lags are given against
   the sensor-side start / end, with the video mapped to sensor time by (i) the file name, (ii) the burned clock.

Writes underwater_cache/tagdata/<deployment>/clock_fits.csv, sync_events.csv and per-event frame features
(sync_frames.parquet).

    .venv/bin/python experiments/01_metadata/sync_video.py "2025-2-3 Mini-mother 11-96" [--suffix .mov]
"""
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uwid import tagdata as td  # noqa: E402
import clock_ocr  # noqa: E402

VIDEO_ROOT = Path("/home/sat3737/NOAA_whales/underwater_video")
S_THR = 90.0


def stable(p: Path, wait=60.0) -> bool:
    return (time.time() - p.stat().st_mtime) > wait


def frame_features(video: str, windows: list[tuple[float, float]]) -> pd.DataFrame:
    """Per-frame features in the given pts windows (sorted, non-overlapping)."""
    c = av.open(video)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    rows = []
    for k, (p0, p1) in enumerate(windows):
        try:
            c.seek(int(max(p0, 0) / s.time_base), stream=s, backward=True)
        except av.error.PermissionError:  # some .m4v refuse a seek near pts 0
            c.close()
            c = av.open(video)
            s = c.streams.video[0]
            s.thread_type = "AUTO"
        for fr in c.decode(s):
            if fr.time < p0:
                continue
            if fr.time > p1:
                break
            im = fr.to_ndarray(format="bgr24", width=192, height=108)
            im = im[:100]  # drop the clock strip at the bottom
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            rows.append({"win": k, "pts": fr.time, "sat": float(hsv[..., 1].mean()), "val": float(hsv[..., 2].mean()),
                         "b_minus_r": float(im[..., 0].mean() - im[..., 2].mean())})
    c.close()
    return pd.DataFrame(rows)


def picture_exposure(f: pd.DataFrame, thr=S_THR, bridge=5):
    air = (f.sat < thr).to_numpy().copy()
    # bridge short non-air gaps
    rr = td.runs(~air)
    for a, b in rr:
        if a > 0 and b < len(air) and (b - a) <= bridge:
            air[a:b] = True
    rr = td.runs(air)
    if not rr:
        return np.nan, np.nan, 0
    a, b = max(rr, key=lambda x: x[1] - x[0])
    p = f.pts.to_numpy()
    return float(p[a]), float(p[b - 1]), len(rr)


def main(dep: str, suffix: str = ".mov", step: float = 60.0):
    out = td.cache_dir(dep)
    meta = td.load_meta(dep)
    t0 = np.datetime64(meta["first_utc"])
    off = meta["utc_offset_h"]
    sod0 = ((t0 + np.timedelta64(int(off * 3600), "s")) - t0.astype("datetime64[D]")) / np.timedelta64(1, "s")
    surf = pd.read_csv(out / "surfacings.csv")
    cam = pd.read_csv(out / "camera_log.csv")
    files = sorted(f for f in (VIDEO_ROOT / dep / "Video").glob(f"CC-*{suffix}") if not f.name.startswith("._"))
    fits, events, frames = [], [], []
    tmpl = clock_ocr.load_templates()
    for f in files:
        if not stable(f):
            print("skip (copying)", f.name)
            continue
        t_start = time.time()
        start_utc, seq = td.video_start_from_name(f.name, off)
        t_name = (start_utc - t0) / np.timedelta64(1, "s")
        ck = clock_ocr.read_video(str(f), step_s=step, frames_per_probe=12, tmpl=tmpl)
        a, b, rms, nchg, maxres = clock_ocr.fit_clock(ck)
        t_clock0 = a - sod0          # sensor-time reading of the burned clock at pts 0
        c = av.open(str(f))
        dur = (c.duration or 0) / 1e6
        c.close()
        piece = cam[cam.piece == seq]
        row = {"file": f.name, "seq": seq, "duration_s": dur, "t_name": t_name, "clock_at_pts0_t": t_clock0,
               "clock_minus_name_s": t_clock0 - t_name, "clock_rate": b, "clock_drift_ppm": (b - 1) * 1e6,
               "clock_fit_rms_s": rms, "clock_fit_maxres_s": maxres, "clock_n_changes": nchg,
               "clock_frames_read": len(ck), "clock_bad_reads": int((ck.worst_hamming > 6).sum())}
        if len(piece):
            pc = piece.iloc[0]
            row.update(cam11_minus_clock0=pc.cam11_start_t - t_clock0, cam12_minus_clock0=pc.cam12_start_t - t_clock0,
                       cc_rec_minus_clock0=pc.cc_rec_start_t - t_clock0,
                       clock_end_minus_cam12_end=t_clock0 + b * dur - pc.cam12_end_t,
                       camera_time_at_cam12=pc.camera_time_at_cam12)
        # physical check
        t_end_clock = t_clock0 + b * dur
        sel = surf[(surf.start_t > t_clock0 + 6) & (surf.end_t < t_end_clock - 6)]
        wins = [((r.start_t - t_clock0) / b - 6.0, (r.end_t - t_clock0) / b + 6.0) for r in sel.itertuples()]
        ff = frame_features(str(f), wins) if wins else pd.DataFrame()
        for k, r in enumerate(sel.itertuples()):
            fk = ff[ff.win == k]
            p_exit, p_entry, nruns = picture_exposure(fk)
            ev = {"file": f.name, "surfacing_id": r.surfacing_id, "sensor_start_t": r.start_t, "sensor_end_t": r.end_t,
                  "pic_exit_pts": p_exit, "pic_entry_pts": p_entry, "pic_air_runs": nruns,
                  "sat_min": float(fk.sat.min()) if len(fk) else np.nan, "sat_median_window": float(fk.sat.median()) if len(fk) else np.nan}
            # lags: picture moment (mapped to sensor time) minus sensor moment
            ev["lag_exit_clock_s"] = t_clock0 + b * p_exit - r.start_t
            ev["lag_entry_clock_s"] = t_clock0 + b * p_entry - r.end_t
            ev["lag_exit_name_s"] = t_name + p_exit - r.start_t
            ev["lag_entry_name_s"] = t_name + p_entry - r.end_t
            events.append(ev)
        if len(ff):
            ff["file"] = f.name
            ff["surfacing_id"] = ff.win.map(dict(enumerate(sel.surfacing_id)))
            frames.append(ff)
        fits.append(row)
        print(f.name, f"{time.time() - t_start:.0f}s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    tag = suffix.strip(".")
    pd.DataFrame(fits).to_csv(out / f"clock_fits_{tag}.csv", index=False)
    ev = pd.DataFrame(events)
    ev.to_csv(out / f"sync_events_{tag}.csv", index=False)
    if frames:
        pd.concat(frames).to_parquet(out / f"sync_frames_{tag}.parquet", index=False)
    if len(ev):
        pd.set_option("display.width", 250)
        print(ev.round(3).to_string())
        ok = ev.dropna(subset=["lag_exit_clock_s"])
        for c in ["lag_exit_clock_s", "lag_entry_clock_s", "lag_exit_name_s", "lag_entry_name_s"]:
            q = ok[c].quantile([.1, .5, .9]).round(3).tolist()
            print(c, "n", len(ok), "median", round(ok[c].median(), 3), "q10/50/90", q, "MAD", round((ok[c] - ok[c].median()).abs().median(), 3))


if __name__ == "__main__":
    dep = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "2025-2-3 Mini-mother 11-96"
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ".mov"
    main(dep, suffix)
