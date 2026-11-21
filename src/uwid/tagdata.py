"""Camera-tag sensor log (CATS-style CSV): compact cache, depth, surfacings, camera intervals, time link to video.

The tag's CSV is large (3 GB for 6.7 h at 400 rows/s) and most channels are step-held copies of slower sensors.
``build_cache`` reads it once in chunks and writes, under ``underwater_cache/tagdata/<deployment>/``:

- ``sensors_50hz.parquet``: one row per 20 ms (8 CSV rows). Motion channels are means over the 8 rows (plus the
  rms of the 400 Hz acceleration around that mean, ``acc_hf``); every other channel is the last value in the bin
  (the most recent update, since the CSV holds values until the next update).
- ``depth_updates.parquet``: every row where the depth value changed, with its exact CSV time (the real 10 Hz
  depth samples, for timing work finer than the 50 Hz grid).
- ``transitions.csv``: every change of the categorical columns (Camera, Flags, LED, GPS, CC status, System error).
- ``channel_stats.json``: per channel the number of value changes (real update rate), min, max, smallest step.
- ``meta.json``: source path and size, first / last sample, row count, time gaps, UTC offset.

Times: ``t`` is seconds since the first CSV sample; ``utc`` is a naive UTC datetime64. Video file names carry
LOCAL time (UTC-10 for Hawaii); see ``video_start_from_name``.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_ROOT = Path("/home/sat3737/NOAA_whales/underwater_cache/tagdata")
ROW_HZ = 400
BIN = 8  # CSV rows per 50 Hz bin

# CSV column index -> short name. Columns 2, 3 (local date / time) are skipped: local = UTC + utc_offset.
COLS = {
    0: "date_utc", 1: "time_utc",
    4: "ax", 5: "ay", 6: "az", 7: "gx", 8: "gy", 9: "gz", 10: "mx", 11: "my", 12: "mz",
    13: "temp_mag", 14: "gps_date", 15: "gps_time", 16: "gps3", 17: "gps4",
    18: "temp", 19: "depth", 20: "temp_depth", 21: "light1", 22: "light2", 23: "hydro",
    24: "sys_error", 25: "batt_v", 26: "batt_ma", 27: "batt_mah", 28: "camera", 29: "flags", 30: "led",
    31: "camera_time", 32: "gps_flag", 33: "cc_status", 34: "cc_vid_kb", 35: "cpu_temp",
}
MOTION = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]
NUMERIC_HELD = ["temp_mag", "gps3", "gps4", "temp", "depth", "temp_depth", "light1", "light2", "hydro",
                "batt_v", "batt_ma", "batt_mah", "camera", "led", "camera_time", "cc_vid_kb", "cpu_temp"]
CATEGORICAL = ["sys_error", "flags", "gps_flag", "cc_status", "camera", "led"]
STRINGS = ["date_utc", "time_utc", "gps_date", "gps_time", "sys_error", "flags", "gps_flag", "cc_status"]


def cache_dir(deployment: str) -> Path:
    return CACHE_ROOT / deployment


def _parse_utc(date: pd.Series, tim: pd.Series) -> np.ndarray:
    """'03.02.2025' + '18:36:52.002500' -> datetime64[ns]."""
    return pd.to_datetime(date + " " + tim, format="%d.%m.%Y %H:%M:%S.%f").to_numpy()


def build_cache(csv_path: str | os.PathLike, deployment: str, chunk_rows: int = 400 * 600,
                utc_offset_h: float | None = None, force: bool = False, log=print) -> Path:
    """Read the tag CSV once and write the compact cache. Returns the cache folder."""
    csv_path = Path(csv_path)
    out = cache_dir(deployment)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "DONE"
    if done.exists() and not force:
        return out
    assert chunk_rows % (BIN * 40) == 0
    size0 = csv_path.stat().st_size
    if utc_offset_h is None:
        utc_offset_h = _utc_offset_from_txt(csv_path.with_suffix(".txt"))

    usecols = sorted(COLS)
    dtypes = {i: (str if COLS[i] in STRINGS else np.float64) for i in usecols}
    reader = pd.read_csv(csv_path, encoding="latin-1", header=0, usecols=usecols, dtype=dtypes,
                         chunksize=chunk_rows, skipinitialspace=True, na_values=["", " "], keep_default_na=False)
    numeric_all = MOTION + NUMERIC_HELD
    stats = {c: {"changes": 0, "min": np.inf, "max": -np.inf, "min_step": np.inf} for c in numeric_all}
    prev_row = None
    parts, depth_parts, trans = [], [], []
    t_first = None
    n_rows = 0
    gaps = []
    last_time = None
    t_start = time.time()
    for k, ch in enumerate(reader):
        ch.columns = [COLS[i] for i in usecols]
        utc = _parse_utc(ch["date_utc"], ch["time_utc"])
        if t_first is None:
            t_first = utc[0]
        tsec = (utc - t_first).astype("timedelta64[ns]").astype(np.int64) / 1e9
        # time gaps (expected step 2.5 ms)
        d = np.diff(np.concatenate([[last_time] if last_time is not None else [], tsec]))
        bad = np.nonzero(np.abs(d - 1.0 / ROW_HZ) > 1e-4)[0]
        for b in bad[:50]:
            gaps.append({"after_t": float((np.concatenate([[last_time], tsec]) if last_time is not None else tsec)[b]),
                         "step_s": float(d[b])})
        last_time = tsec[-1]

        # change statistics on numeric channels (vs the previous row, carrying over the chunk boundary)
        for c in numeric_all:
            v = ch[c].to_numpy(dtype=np.float64)
            vv = np.concatenate([[prev_row[c]], v]) if prev_row is not None else v
            dv = np.diff(vv)
            nz = dv[np.isfinite(dv) & (dv != 0)]
            s = stats[c]
            s["changes"] += int(nz.size)
            if np.isfinite(v).any():
                s["min"] = min(s["min"], float(np.nanmin(v)))
                s["max"] = max(s["max"], float(np.nanmax(v)))
            if nz.size:
                s["min_step"] = min(s["min_step"], float(np.min(np.abs(nz))))
        # exact depth updates
        dep = ch["depth"].to_numpy(dtype=np.float64)
        dd = np.concatenate([[prev_row["depth"]], dep]) if prev_row is not None else np.concatenate([[np.nan], dep])
        idx = np.nonzero(np.diff(dd) != 0)[0]
        depth_parts.append(pd.DataFrame({"t": tsec[idx], "depth": dep[idx]}))
        # transitions of categorical columns
        for c in CATEGORICAL:
            v = ch[c].astype(str).to_numpy()
            pv = str(prev_row[c]) if prev_row is not None else None
            vv = np.concatenate([[pv], v]) if pv is not None else v
            ii = np.nonzero(vv[1:] != vv[:-1])[0] + 1
            if pv is None:
                ii = np.concatenate([[0], ii])
                for i in ii:
                    trans.append((float(tsec[i]), c, "" if i == 0 else vv[i - 1], vv[i]))
            else:
                for i in ii:
                    trans.append((float(tsec[i - 1]), c, vv[i - 1], vv[i]))
        prev_row = ch.iloc[-1]

        # 50 Hz bins
        n = len(ch) // BIN * BIN
        g = ch.iloc[:n]
        nb = n // BIN
        rec = {"t": tsec[:n].reshape(nb, BIN)[:, 0], "utc": utc[:n].reshape(nb, BIN)[:, 0]}
        for c in MOTION:
            a = g[c].to_numpy(dtype=np.float64).reshape(nb, BIN)
            rec[c] = a.mean(1).astype(np.float32)
            if c in ("ax", "ay", "az"):
                rec.setdefault("_hf", np.zeros(nb))
                rec["_hf"] = rec["_hf"] + ((a - a.mean(1, keepdims=True)) ** 2).mean(1)
        rec["acc_hf"] = np.sqrt(rec.pop("_hf")).astype(np.float32)
        last = g.iloc[BIN - 1::BIN]
        for c in NUMERIC_HELD:
            rec[c] = last[c].to_numpy(dtype=np.float64).astype(np.float32 if c not in ("camera_time", "batt_mah") else np.float64)
        for c in ["flags", "gps_flag", "cc_status", "sys_error", "gps_date", "gps_time"]:
            rec[c] = last[c].astype(str).to_numpy()
        parts.append(pd.DataFrame(rec))
        if len(ch) % BIN:
            log(f"warning: chunk {k} has {len(ch)} rows, {len(ch) % BIN} dropped at the end")
        n_rows += len(ch)
        if k % 10 == 0:
            log(f"chunk {k}: {n_rows / ROW_HZ / 3600:.2f} h read, {time.time() - t_start:.0f} s")
    df = pd.concat(parts, ignore_index=True)
    for c in ["flags", "gps_flag", "cc_status", "sys_error"]:
        df[c] = df[c].astype("category")
    df.to_parquet(out / "sensors_50hz.parquet", index=False)
    pd.concat(depth_parts, ignore_index=True).to_parquet(out / "depth_updates.parquet", index=False)
    pd.DataFrame(trans, columns=["t", "column", "old", "new"]).to_csv(out / "transitions.csv", index=False)
    dur = float(last_time)
    for c, s in stats.items():
        s["rate_hz"] = s["changes"] / dur if dur > 0 else float("nan")
        for key in ("min", "max", "min_step"):
            if not np.isfinite(s[key]):
                s[key] = None
    (out / "channel_stats.json").write_text(json.dumps(stats, indent=1))
    meta = {"csv": str(csv_path), "csv_size": size0, "rows": n_rows, "first_utc": str(t_first),
            "duration_s": dur, "utc_offset_h": utc_offset_h, "gaps": gaps[:200],
            "build_seconds": time.time() - t_start}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    done.write_text("ok\n")
    return out


def _utc_offset_from_txt(txt: Path) -> float:
    try:
        m = re.search(r"utc_offset=([-+]?\d+(\.\d+)?)", txt.read_text(errors="replace"))
        return float(m.group(1)) if m else 0.0
    except OSError:
        return 0.0


def read_raw(csv_path: str | os.PathLike, t_start: float, t_end: float, columns: list[str] | None = None) -> pd.DataFrame:
    """Raw 400 Hz rows between t_start and t_end (s since the first sample), assuming no gaps (meta.json: none here)."""
    i0 = max(0, int(np.floor(t_start * ROW_HZ)))
    n = int(np.ceil((t_end - t_start) * ROW_HZ)) + 1
    names = {v: k for k, v in COLS.items()}
    cols = columns or list(COLS.values())
    use = sorted({names[c] for c in cols} | {0, 1})
    df = pd.read_csv(csv_path, encoding="latin-1", header=None, skiprows=1 + i0, nrows=n, usecols=use,
                     dtype={i: (str if COLS[i] in STRINGS else np.float64) for i in use}, skipinitialspace=True)
    df.columns = [COLS[i] for i in use]
    utc = _parse_utc(df["date_utc"], df["time_utc"])
    first = pd.read_csv(csv_path, encoding="latin-1", header=None, skiprows=1, nrows=1, usecols=[0, 1], dtype=str)
    t0 = _parse_utc(first[0], first[1])[0]
    df.insert(0, "t", (utc - t0).astype("timedelta64[ns]").astype(np.int64) / 1e9)
    df.insert(1, "utc", utc)
    return df.drop(columns=["date_utc", "time_utc"])


def load(deployment: str) -> pd.DataFrame:
    return pd.read_parquet(cache_dir(deployment) / "sensors_50hz.parquet")


def load_meta(deployment: str) -> dict:
    return json.loads((cache_dir(deployment) / "meta.json").read_text())


def load_depth_updates(deployment: str) -> pd.DataFrame:
    return pd.read_parquet(cache_dir(deployment) / "depth_updates.parquet")


# ---------------------------------------------------------------- cleaning and derived channels


DEPTH_GLITCH_M = 1000.0  # this log has isolated 1529.54 m readings (27 updates, all with the tag in air)
LIGHT_SAT = 37177.0      # both light channels clip here in daylight (65535 = rare invalid readings)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns: depth_clean (glitches interpolated), amag, gmag (|acc| m/s2, |gyro| mrad/s)."""
    df = df.copy()
    d = df["depth"].astype(np.float64).where(df["depth"] < DEPTH_GLITCH_M)
    df["depth_clean"] = d.interpolate(limit_direction="both").to_numpy()
    df["amag"] = np.sqrt(df.ax.astype(np.float64) ** 2 + df.ay ** 2 + df.az ** 2)
    df["gmag"] = np.sqrt(df.gx.astype(np.float64) ** 2 + df.gy ** 2 + df.gz ** 2)
    for c in ("light1", "light2"):
        df[c + "_clean"] = df[c].where(df[c] < 65535).ffill()
    return df


def rolling_median(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, center=True, min_periods=1).median().to_numpy()


@dataclass
class PhaseParams:
    """Thresholds for the tag-state timeline (justified from this log in experiments/01_metadata/tables.md)."""
    fs: float = 50.0
    flight_hf: float = 1.5        # rms of 400 Hz acceleration around its 20 ms mean (m/s2): drone motors 2-10, else < 1
    flight_min_s: float = 20.0
    float_gyro: float = 350.0     # median |gyro| over 10 s (mrad/s): floating 270-1570 (5-95 %), whale at surface 60-320
    float_depth_max: float = 0.35  # raw depth (m) stays below this while floating (sensor 0-20 cm under the surface)
    float_win_s: float = 10.0
    float_hf_max: float = 0.12    # median acc_hf over 10 s: floating 0.04-0.10, hand / deck handling 0.1-0.5
    float_amag_max: float = 16.0  # no knocks above this (m/s2) inside the 10 s window
    air_offset_m: float = -0.062  # raw depth in air (deck median)
    float_wet_m: float = 0.04     # median raw depth over 10 s at least air_offset + this: pressure port under water
    float_fill_s: float = 10.0    # gaps shorter than this between floating stretches are floating
    float_min_s: float = 20.0     # floating stretches shorter than this are dropped
    dive_m: float = 1.0           # a tag in the water that goes deeper than this is on an animal (it floats otherwise)


def tag_states(df: pd.DataFrame, p: PhaseParams = PhaseParams()) -> pd.DataFrame:
    """Per-sample tag state: 'flight' (under the drone), 'whale', 'floating', 'air' (deck / hand / anything else dry).

    Rules, in order: flight = rolling-1 s median acc_hf > p.flight_hf for at least p.flight_min_s; floating =
    over a 10 s window: median |gyro| > p.float_gyro (bobbing), median acc_hf < p.float_hf_max (no handling
    vibration), no knock > p.float_amag_max, depth max < p.float_depth_max and median >= air offset + p.float_wet_m
    (port wet), outside flights, short gaps filled, short stretches dropped;
    whale = from the first sample deeper than p.dive_m after a flight to the last sample deeper than p.dive_m
    before a floating stretch; air = the rest. Returns df with a 'state' column (categorical).
    """
    fs = p.fs
    hf = rolling_median(df["acc_hf"].to_numpy(np.float64), int(fs))
    flight = np.zeros(len(df), bool)
    for a, b in runs(hf > p.flight_hf):
        if (b - a) / fs >= p.flight_min_s:
            flight[a:b] = True
    w = int(p.float_win_s * fs)
    g = rolling_median(df["gmag"].to_numpy(), w)
    dmax = pd.Series(df["depth_clean"].to_numpy()).rolling(w, center=True, min_periods=1).max().to_numpy()
    ser = lambda x: pd.Series(x).rolling(w, center=True, min_periods=1)
    hf10 = ser(df["acc_hf"].to_numpy(np.float64)).median().to_numpy()
    amax10 = ser(df["amag"].to_numpy()).max().to_numpy()
    dmed10 = ser(df["depth_clean"].to_numpy()).median().to_numpy()
    floating = ((g > p.float_gyro) & (dmax < p.float_depth_max) & (hf10 < p.float_hf_max) & (amax10 < p.float_amag_max)
                & (dmed10 >= p.air_offset_m + p.float_wet_m) & ~flight)
    for a, b in runs(~floating):
        if a > 0 and b < len(floating) and (b - a) / fs < p.float_fill_s and not flight[a:b].any():
            floating[a:b] = True
    for a, b in runs(floating):
        if (b - a) / fs < p.float_min_s:
            floating[a:b] = False
    d = df["depth_clean"].to_numpy()
    whale = np.zeros(len(df), bool)
    deep = d > p.dive_m
    # whale: spans between deep samples that are not separated by floating or flight
    blockers = floating | flight
    seg_id = np.cumsum(np.concatenate([[0], (blockers[1:] != blockers[:-1]).astype(int)]))
    for sid in np.unique(seg_id[deep & ~blockers]):
        idx = np.nonzero((seg_id == sid) & deep)[0]
        whale[idx[0]:idx[-1] + 1] = True
    state = np.full(len(df), "air", dtype=object)
    state[flight] = "flight"
    state[floating] = "floating"
    state[whale] = "whale"
    out = df.copy()
    out["state"] = pd.Categorical(state, categories=["air", "flight", "floating", "whale"])
    return out


def state_segments(df: pd.DataFrame, min_s: float = 0.0) -> pd.DataFrame:
    s = df["state"].astype(str).to_numpy()
    t = df["t"].to_numpy()
    ch = np.nonzero(s[1:] != s[:-1])[0] + 1
    starts = np.concatenate([[0], ch])
    ends = np.concatenate([ch, [len(s)]])
    rows = [{"state": s[a], "start_t": t[a], "end_t": t[b - 1], "duration_s": t[b - 1] - t[a]} for a, b in zip(starts, ends)]
    seg = pd.DataFrame(rows)
    return seg[seg.duration_s >= min_s].reset_index(drop=True)


def find_drops(csv_path, flights: pd.DataFrame, search_s: float = 30.0, ff_g: float = 0.35,
               impact_g: float = 2.5, smooth_n: int = 20, min_ff_s: float = 0.1, g0: float = 9.81) -> pd.DataFrame:
    """Free fall + impact at the end of each drone flight, from the raw 400 Hz accelerometer.

    The raw magnitude carries +-1 g of vibration and spin even in free fall, so each axis is first averaged over
    smooth_n samples (50 ms). Free fall = smoothed |a| < ff_g * g for at least min_ff_s; impact = first raw sample
    with |a| > impact_g * g after the free-fall run (within 0.5 s). The accelerometer clips at about 4 g per axis
    (|a| 67.9 m/s2), so impact peaks are lower bounds. Searches the last search_s s of each flight plus 10 s after.
    """
    rows = []
    for _, f in flights.iterrows():
        r = read_raw(csv_path, f.end_t - search_s, f.end_t + 10.0, ["ax", "ay", "az", "depth"])
        sm = lambda x: pd.Series(x.to_numpy()).rolling(smooth_n, center=True, min_periods=1).mean().to_numpy()
        a_s = np.sqrt(sm(r.ax) ** 2 + sm(r.ay) ** 2 + sm(r.az) ** 2)
        a = np.sqrt(r.ax ** 2 + r.ay ** 2 + r.az ** 2).to_numpy()
        t = r.t.to_numpy()
        best = None
        for s0, e0 in runs(a_s < ff_g * g0):
            if (e0 - s0) / ROW_HZ < min_ff_s:
                continue
            after = np.nonzero(a[e0:e0 + ROW_HZ // 2] > impact_g * g0)[0]
            if after.size:
                best = (s0, e0, e0 + after[0])
        if best is None:
            rows.append({"flight_end_t": f.end_t, "ff_start_t": np.nan, "impact_t": np.nan})
            continue
        s0, e0, k = best
        rows.append({"flight_end_t": f.end_t, "ff_start_t": t[s0], "impact_t": t[k], "ff_s": t[k] - t[s0],
                     "ff_mean_ms2": float(a_s[s0:e0].mean()), "impact_peak_ms2": float(a[k:k + 80].max()),
                     "impact_clipped": bool(a[k:k + 80].max() > 67.0),
                     "drop_height_m_est": 0.5 * g0 * (t[k] - t[s0]) ** 2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- depth


def depth_series(df: pd.DataFrame, offset: float | None = None, air_mask: np.ndarray | None = None) -> np.ndarray:
    """Depth with the pressure sensor's zero offset removed (offset = median depth while in air, if given)."""
    d = df["depth"].to_numpy(dtype=np.float64)
    if offset is None:
        offset = float(np.median(d[air_mask])) if air_mask is not None and air_mask.any() else 0.0
    return d - offset


@dataclass
class SurfacingParams:
    air_offset_m: float = -0.062  # raw depth with the sensor in air (median on deck before the flights)
    exposed_m: float = 0.10       # depth above the air offset below which the sensor is out of the water
    merge_gap_s: float = 1.0      # re-immersions shorter than this (a wave over the tag) do not split a surfacing
    bout_m: float = 1.0           # a surfacing bout is a stretch shallower than this; between bouts = dives


def exposures(t: np.ndarray, depth_raw: np.ndarray, on_whale: np.ndarray, p: SurfacingParams = SurfacingParams(),
              light2: np.ndarray | None = None) -> pd.DataFrame:
    """Surfacings of the tag (pressure port out of the water) while on the whale.

    start_t / end_t: interpolated crossings of (air_offset + exposed_m). bout_*: the enclosing stretch shallower
    than bout_m. prev_dive_*: the excursion between the previous surfacing and this one. light2_sat_*: first / last
    sample with light2 clipped (daylight on the sensor) inside [start-2 s, end+2 s], a second sensor-side clock.
    """
    d = depth_raw - p.air_offset_m
    fs = 1.0 / np.median(np.diff(t))
    expo = (d < p.exposed_m) & on_whale
    rr = runs(expo)
    merged = []
    for a, b in rr:
        if merged and (t[a] - t[merged[-1][1] - 1]) < p.merge_gap_s and d[merged[-1][1]:a].max() < p.bout_m:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    bout = runs((d < p.bout_m) & on_whale)
    bout_of = np.full(len(t), -1)
    for i, (a, b) in enumerate(bout):
        bout_of[a:b] = i
    rows = []
    prev_end = None
    for a, b in merged:
        ts = _cross(t, d, a - 1, a, p.exposed_m)
        te = _cross(t, d, b - 1, b, p.exposed_m)
        bi = bout_of[a]
        ba, bb = bout[bi] if bi >= 0 else (a, b)
        row = {"start_t": ts, "end_t": te, "duration_s": te - ts, "n_parts": sum(1 for x, y in rr if x >= a and y <= b),
               "min_depth_m": float(d[a:b].min()), "bout_id": int(bi), "bout_start_t": _cross(t, d, ba - 1, ba, p.bout_m),
               "bout_end_t": _cross(t, d, bb - 1, bb, p.bout_m)}
        # ascent / descent speed over the last / first metre
        row["ascent_mps"] = p.bout_m / max(ts - row["bout_start_t"], 1e-3) if row["bout_start_t"] < ts else np.nan
        row["descent_mps"] = p.bout_m / max(row["bout_end_t"] - te, 1e-3) if row["bout_end_t"] > te else np.nan
        if prev_end is not None:
            seg = d[prev_end:a]
            row["prev_gap_s"] = ts - t[prev_end - 1]
            row["prev_max_depth_m"] = float(seg.max())
        else:
            row["prev_gap_s"] = np.nan
            row["prev_max_depth_m"] = np.nan
        if light2 is not None:
            lo, hi = np.searchsorted(t, ts - 2.0), np.searchsorted(t, te + 2.0)
            sat = np.nonzero(light2[lo:hi] >= LIGHT_SAT)[0]
            row["light2_sat_start_t"] = t[lo + sat[0]] if sat.size else np.nan
            row["light2_sat_end_t"] = t[lo + sat[-1]] if sat.size else np.nan
        rows.append(row)
        prev_end = b
    return pd.DataFrame(rows)


def dives(t: np.ndarray, depth_raw: np.ndarray, on_whale: np.ndarray, p: SurfacingParams = SurfacingParams(),
          min_depth_m: float = 1.0) -> pd.DataFrame:
    """Excursions deeper than p.bout_m between surfacing bouts, with their maximum depth."""
    d = depth_raw - p.air_offset_m
    rows = []
    for a, b in runs((d >= p.bout_m) & on_whale):
        k = a + int(np.argmax(d[a:b]))
        if d[k] < min_depth_m:
            continue
        rows.append({"start_t": _cross(t, d, a - 1, a, p.bout_m), "end_t": _cross(t, d, b - 1, b, p.bout_m),
                     "max_depth_m": float(d[k]), "max_t": t[k]})
    out = pd.DataFrame(rows)
    if len(out):
        out["duration_s"] = out.end_t - out.start_t
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _cross(t, d, i0, i1, thr):
    if i0 < 0 or i1 >= len(t):
        return float(t[max(i0, 0)])
    d0, d1 = d[i0], d[i1]
    if d1 == d0:
        return float(t[i1])
    f = (thr - d0) / (d1 - d0)
    return float(t[i0] + np.clip(f, 0, 1) * (t[i1] - t[i0]))


# ---------------------------------------------------------------- camera


def camera_intervals(df: pd.DataFrame, recording_values=(12,)) -> pd.DataFrame:
    """Intervals where the Camera column holds a recording value (12 in this log while CC status is 'R--')."""
    rec = df["camera"].isin(recording_values).to_numpy() & df["cc_status"].astype(str).str.startswith("R").to_numpy()
    t = df["t"].to_numpy()
    return pd.DataFrame([{"start_t": t[a], "end_t": t[b - 1], "duration_s": t[b - 1] - t[a]} for a, b in runs(rec)])


def video_files(df: pd.DataFrame) -> pd.DataFrame:
    """Video pieces as the log sees them: a new piece starts where 'CC vid. size' drops back."""
    t = df["t"].to_numpy()
    kb = df["cc_vid_kb"].to_numpy()
    cam = df["camera"].to_numpy()
    rec = df["cc_status"].astype(str).str.startswith("R").to_numpy()
    out = []
    for a, b in runs(rec):
        k = kb[a:b]
        drops = np.nonzero(np.diff(k) < -1000)[0] + 1
        edges = np.concatenate([[0], drops, [b - a]])
        for s, e in zip(edges[:-1], edges[1:]):
            out.append({"start_t": t[a + s], "end_t": t[a + e - 1], "duration_s": t[a + e - 1] - t[a + s],
                        "final_kb": float(k[e - 1]), "camera_val": float(np.median(cam[a + s:a + e]))})
    return pd.DataFrame(out)


_NAME = re.compile(r"(\d{8})-(\d{6})-(\d{3})-(\d{5})")


def video_start_from_name(name: str, utc_offset_h: float = -10.0) -> tuple[np.datetime64, int]:
    """'CC-11-96-20250203-101236-000-00004.mov' -> (UTC start as datetime64, sequence number)."""
    m = _NAME.search(name)
    if not m:
        raise ValueError(name)
    local = pd.Timestamp(f"{m.group(1)} {m.group(2)}")
    return np.datetime64(local - pd.Timedelta(hours=utc_offset_h)), int(m.group(4))


@dataclass
class VideoClock:
    """Linear map between sensor time t (s since the first CSV sample) and a video's presentation time v (s).

    t = t0 + v * (1 + drift). Fill t0 from the file name, the log, or a measured sync; drift in s/s.
    """
    t0: float
    drift: float = 0.0
    source: str = ""

    def to_sensor(self, v):
        return self.t0 + np.asarray(v) * (1.0 + self.drift)

    def to_video(self, t):
        return (np.asarray(t) - self.t0) / (1.0 + self.drift)

    def frame_index(self, t, fps: float):
        return np.floor(self.to_video(t) * fps + 1e-6).astype(int)


# Measured on CC-11-96, 2025-02-03 (experiments/01_metadata/tables.md):
CLOCK_MINUS_SENSOR_S = 2.03       # burned-in clock minus sensor-log time (drone release + impact frames vs accelerometer)
CAMTIME_RESUME_MINUS_FIRST_FRAME_S = 6.87  # 'Camera time' resumes counting this long after a file's first frame
DEPTH_EXIT_PIC_LAG_S = 0.3        # lens fully out of the water this long after depth crosses air+0.10 m (by eye, n=9)
DEPTH_ENTRY_PIC_LAG_S = -0.4      # picture clearly under water this long before depth crosses back (by eye, n=9)


def clock_from_burned(clock_sod_at_pts0: float, clock_rate: float, first_utc, utc_offset_h: float,
                      clock_minus_sensor_s: float = CLOCK_MINUS_SENSOR_S) -> VideoClock:
    """Video clock from a fit of the burned-in clock (seconds of the local day at pts 0, rate per pts second)."""
    first = pd.Timestamp(first_utc) + pd.Timedelta(hours=utc_offset_h)
    sod0 = (first - first.normalize()).total_seconds()
    return VideoClock(clock_sod_at_pts0 - sod0 - clock_minus_sensor_s, clock_rate - 1.0, "burned clock")


def first_frames_from_camera_time(df: pd.DataFrame, min_pause_s: float = 10.0,
                                  lead_s: float = CAMTIME_RESUME_MINUS_FIRST_FRAME_S) -> pd.DataFrame:
    """Sensor time of each video file's first frame from the 'Camera time' column alone (no video needed).

    'Camera time' counts recorded seconds; it pauses ~23-24 s at every file roll. A file's first frame is lead_s
    before the counter resumes (calibrated on 4 files, spread 0.08 s; the counter itself ticks once a second).
    """
    t = df.t.to_numpy()
    ct = df.camera_time.to_numpy()
    up = np.nonzero(np.diff(ct) != 0)[0] + 1
    tu, cu = t[up], ct[up]
    starts = [0] + list(np.nonzero(np.diff(tu) > min_pause_s)[0] + 1)
    rows = []
    for k, s in enumerate(starts):
        e = starts[k + 1] - 1 if k + 1 < len(starts) else len(tu) - 1
        rows.append({"piece": k + 1, "first_frame_t": tu[s] - lead_s,
                     "camera_time_first": cu[s], "camera_time_last": cu[e], "last_tick_t": tu[e]})
    return pd.DataFrame(rows)


def video_clock(deployment: str, video_name: str) -> VideoClock:
    """Best available sensor<->video map for one file of a deployment.

    1. burned-clock fit from experiments/01_metadata/sync_video.py (clock_fits_<ext>.csv): accuracy +-0.05 s;
    2. else the log's 'Camera time' (file sequence number -> piece): +-0.1 s, .mov presentation time only
       (the .m4v conversions drift up to -800 ppm against real time, so they need the clock fit);
    3. else the file name: +-1 s.
    """
    out = cache_dir(deployment)
    meta = load_meta(deployment)
    ext = Path(video_name).suffix.lstrip(".").lower()
    f = out / f"clock_fits_{ext}.csv"
    if f.exists():
        fits = pd.read_csv(f)
        r = fits[fits.file == Path(video_name).name]
        if len(r):
            r = r.iloc[0]
            return VideoClock(float(r.clock_at_pts0_t) - CLOCK_MINUS_SENSOR_S, float(r.clock_rate) - 1.0, "burned clock")
    _, seq = video_start_from_name(video_name, meta["utc_offset_h"])
    ff = first_frames_from_camera_time(load(deployment))
    r = ff[ff.piece == seq]
    if len(r) and ext == "mov":
        return VideoClock(float(r.first_frame_t.iloc[0]), 0.0, "camera time")
    return clock_from_name(video_name, meta["first_utc"], meta["utc_offset_h"])


def clock_from_name(name: str, first_utc: str | np.datetime64, utc_offset_h: float = -10.0, offset_s: float = 0.0,
                    drift: float = 0.0) -> VideoClock:
    """Video clock from the file-name start time, plus a measured correction offset_s (s, video later than name)."""
    start, _ = video_start_from_name(name, utc_offset_h)
    t0 = (start - np.datetime64(first_utc)) / np.timedelta64(1, "s")
    return VideoClock(float(t0) + offset_s, drift, f"name{'+offset' if offset_s else ''}")


# ---------------------------------------------------------------- derived tables for a new deployment (WP7)


def smooth_states(st: pd.DataFrame, min_s: float = 5.0, fs: float = 50.0) -> pd.DataFrame:
    """State runs shorter than ``min_s`` take the state of the run before them (as in experiments/01_metadata)."""
    s = st["state"].astype(str).to_numpy().copy()
    ch = np.nonzero(s[1:] != s[:-1])[0] + 1
    for a, b in zip(np.concatenate([[0], ch]), np.concatenate([ch, [len(s)]])):
        if (b - a) / fs < min_s and a > 0:
            s[a:b] = s[a - 1]
    out = st.copy()
    out["state"] = pd.Categorical(s, categories=["air", "flight", "floating", "whale"])
    return out


def release_time(df: pd.DataFrame, whale_end_t: float, fs: float = 50.0) -> float:
    """Sensor time at which the tag came off the whale before ``whale_end_t`` (end of the last 'whale' run).

    Tags release at depth and rise; the gyro jumps from the resting baseline at release. Walk back from the end of
    the whale state while the 0.5 s median |gyro| stays above 3x the baseline of the minute before, or the tag is
    shallower than it was a minute before the end (the rise). Same rule as experiments/01_metadata (one release
    seen: 14:38:44 on 2025-02-03, 30 s before the depth passes 1 m).
    """
    t = df["t"].to_numpy()
    d = df["depth_clean"].to_numpy()
    i_end = int(np.searchsorted(t, whale_end_t))
    i_end = min(i_end, len(t) - 1)
    n1, n2 = int(fs * 120), int(fs * 60)
    if i_end - n1 < 0:
        return float(whale_end_t)
    gm = rolling_median(df["gmag"].to_numpy(), 25)
    base = float(np.median(df["gmag"].to_numpy()[i_end - n1:i_end - n2]))
    j = i_end
    while j > i_end - n1 and not (gm[j] < 3 * base and d[j] > d[i_end - n2] - 0.5):
        j -= 1
    return float(t[j])


def derive_tables(deployment: str, csv_path: str | os.PathLike | None = None, force: bool = False, log=print) -> dict:
    """Make ``states.csv``, ``drops.csv`` and ``surfacings.csv`` in the deployment's cache when they are missing
    (the part of experiments/01_metadata/analyze_sensors.py the clip pipeline needs). Returns the three tables.

    states: :func:`tag_states` then runs shorter than 5 s merged into the run before; drops: :func:`find_drops` at
    the end of each flight, ``attached`` = deeper than 1 m within 10 min of the impact; surfacings:
    :func:`exposures` on the whale with the air offset = median depth on quiet deck time before the first flight.
    """
    out = cache_dir(deployment)
    files = {k: out / f"{k}.csv" for k in ("states", "drops", "surfacings")}
    if not force and all(f.exists() for f in files.values()):
        return {k: pd.read_csv(f) for k, f in files.items()}
    meta = load_meta(deployment)
    csv_path = csv_path or meta["csv"]
    log(f"{deployment}: deriving tag states, drops and surfacings from the sensor cache")
    df = clean(load(deployment))
    t = df.t.to_numpy()
    st = smooth_states(tag_states(df))
    seg = state_segments(st)
    seg.to_csv(files["states"], index=False)
    st[["t", "state"]].to_parquet(out / "states_50hz.parquet", index=False)
    d = df.depth_clean.to_numpy()
    flights = seg[seg.state == "flight"][["start_t", "end_t"]]
    drops = find_drops(csv_path, flights) if len(flights) else pd.DataFrame(columns=["flight_end_t", "impact_t"])
    att = []
    for r in drops.itertuples():
        if not np.isfinite(getattr(r, "impact_t", np.nan)):
            att.append(False)
            continue
        k = (t > r.impact_t) & (t < r.impact_t + 600)
        att.append(bool(k.any() and d[k].max() > 1.0))
    drops["attached"] = att
    drops.to_csv(files["drops"], index=False)
    air = (st.state == "air").to_numpy()
    quiet = air & (df.acc_hf.to_numpy() < 0.3) & (df.gmag.to_numpy() < 100)
    first_flight = seg[seg.state == "flight"].start_t.min() if (seg.state == "flight").any() else np.inf
    pre = quiet & (t < first_flight)
    air_off = float(np.median(d[pre])) if pre.sum() > 50 * 60 else SurfacingParams().air_offset_m
    whale = (st.state == "whale").to_numpy()
    ex = exposures(t, d, whale, SurfacingParams(air_offset_m=air_off), light2=df.light2_clean.to_numpy())
    ex.insert(0, "surfacing_id", np.arange(len(ex)))
    ex["air_reading_m"] = [float(np.median(df.depth.to_numpy()[(t >= a) & (t <= b)])) if b > a else np.nan
                           for a, b in zip(ex.start_t, ex.end_t)]
    ex.to_csv(files["surfacings"], index=False)
    return {k: pd.read_csv(f) for k, f in files.items()}
