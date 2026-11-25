"""WP16: do the tag deployments and the boat photos of the same day line up in time? Read-only feasibility check.

Nothing is fitted. Reads (never writes): tag caches in underwater_cache/tagdata + scan, the tag's raw GNSS file
(.ubx, counted only), the photo project's per-day EXIF times / clusters / catalogue labels, the field-notes export.
Per-photo and per-surfacing rows go to underwater_cache/photo_link/ (outside the repo: photo names carry people's
names). Prints a JSON of counts; tables.md is written from it by hand.

Clocks (see tables.md): tag `t` = s since the first CSV sample, whose UTC stamp is the tag's own clock; photo times
= EXIF DateTimeOriginal + SubSec, the camera's own clock, assumed Hawaii time (UTC-10, no DST). A photo matches a
surfacing at shift s when |photo_utc + s - surfacing interval| <= d (0 inside the interval).

Run: .venv/bin/python experiments/11_photo_link/photo_link.py
"""
from __future__ import annotations

import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/sat3737/NOAA_whales")
TAG = ROOT / "underwater_cache/tagdata"
SCAN = ROOT / "underwater_cache/scan"
OUT = ROOT / "underwater_cache/photo_link"
DAYS_ALL = ROOT / "Dinov3-WhaleID/experiments/days_all"
LABELS = ROOT / "Dinov3-WhaleID/model_fine_tuning/splits/labels_all_days.json"
GPS_JSON = ROOT / "Dinov3-WhaleID/experiments/label_assist/gps"
NOTES = ROOT / "moonshot_cache/notes"
HST = -10.0

DEPS = {
    "2025-02-03": dict(dep="2025-2-3 Mini-mother 11-96", tag="CC-11-96", ubx="Data/20250203-164014-CC-11-96.ubx"),
    "2025-01-02": dict(dep="TestSubset 2025-1-2 CATS -85", tag="CC-09-85", ubx="Data/20250102-225123-CC-09-85-HRA.ubx"),
}
SHIFTS = np.arange(-600, 601, 1)
# Clock offsets from one photographed tag-on event per day (AI by-eye on the photos, sensor spike on the tag):
# 2025-02-03 drone drop: tag impact 10:14:07.785 (tag clock, drops.csv); photo 10:17:00.96 shows the tag in the air,
#   10:17:01.30 on the back -> shift = -173.3 +- 0.2 s.
# 2025-01-02 pole tagging: acceleration spike 12:31:25.0 (tag clock, |a| 33 m/s2, depth leaves 0 m 2 s later);
#   photos 12:34:13.0-17.2 show the pole placing the tag, contact between 12:34:14.7 and 12:34:15.8 -> -170.2 +- 0.6 s.
EVENT_SHIFT = {"2025-02-03": -173.3, "2025-01-02": -170.2}


# ------------------------------------------------------------------------------------------------ tag side

def tag_side(day: str, cfg: dict) -> dict:
    d = TAG / cfg["dep"]
    meta = json.loads((d / "meta.json").read_text())
    t0 = pd.Timestamp(meta["first_utc"])
    st = pd.read_csv(d / "states.csv")
    wh = st[st.state == "whale"]
    on0, on1 = float(wh.start_t.min()), float(wh.end_t.max())
    s = pd.read_csv(d / "surfacings.csv")
    s = s[(s.start_t >= on0) & (s.end_t <= on1)].copy()
    s["start_utc"] = t0 + pd.to_timedelta(s.start_t, unit="s")
    s["end_utc"] = t0 + pd.to_timedelta(s.end_t, unit="s")
    # bouts (stretches shallower than 1 m around a surfacing)
    bouts = s.groupby("bout_id").agg(bout_start_t=("bout_start_t", "first"), bout_end_t=("bout_end_t", "first"),
                                    n=("start_t", "size")).reset_index()
    # recordings on disk, in sensor time, from the scan windows table
    rec = []
    wdir = SCAN / cfg["dep"] / "strict" / "windows"
    for f in sorted(wdir.glob("*.csv")):
        w = pd.read_csv(f, usecols=["recording", "t0", "t1", "sensor_t0", "tag_state", "water"])
        lens_out = int((w.water != "underwater").sum())
        rec.append(dict(recording=f.name.split(".")[0][-5:], sensor_t0=float(w.sensor_t0.min()),
                        sensor_t1=float(w.sensor_t0.max() + 1), windows=len(w), whale_windows=int((w.tag_state == "whale").sum()),
                        not_underwater_windows=lens_out))
    rec = pd.DataFrame(rec)
    # tag clock against the GNSS time the tag logged next to it
    sen = pd.read_parquet(d / "sensors_50hz.parquet", columns=["t", "utc", "gps_date", "gps_time"])
    ch = sen.gps_time.notna() & sen.gps_time.ne(sen.gps_time.shift())
    g = sen[ch]
    gutc = pd.to_datetime(g.gps_date + " " + g.gps_time, format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")
    diff = (g.utc - gutc).dt.total_seconds().dropna()
    return dict(t0=t0, on0=on0, on1=on1, surf=s, bouts=bouts, rec=rec, utc_offset_h=meta["utc_offset_h"],
                tag_minus_gnss=dict(n=int(len(diff)), median=float(diff.median()) if len(diff) else None,
                                    p05=float(diff.quantile(.05)) if len(diff) else None,
                                    p95=float(diff.quantile(.95)) if len(diff) else None,
                                    first_t=float(g.t.iloc[0]) if len(g) else None, last_t=float(g.t.iloc[-1]) if len(g) else None))


def ubx_epochs(path: Path) -> pd.DataFrame:
    """UBX RXM-RAWX epochs (raw GNSS snapshots): GPS time -> UTC, number of measurements. Positions are not
    computed (the tag stores raw measurements, not fixes)."""
    b = path.read_bytes()
    rows = []
    for m in re.finditer(rb"\xb5\x62\x02\x15", b):
        i = m.start()
        if i + 6 + 16 > len(b):
            continue
        ln = struct.unpack_from("<H", b, i + 4)[0]
        if ln < 16 or i + 6 + ln > len(b):
            continue
        tow, week, leap, nmeas = struct.unpack_from("<dHbB", b, i + 6)
        if not (2000 < week < 2600):
            continue
        utc = pd.Timestamp("1980-01-06") + pd.Timedelta(days=7 * week) + pd.Timedelta(seconds=tow - leap)
        rows.append((utc, nmeas))
    return pd.DataFrame(rows, columns=["utc", "nmeas"])


# ------------------------------------------------------------------------------------------------ photo side

def photo_side(day: str) -> pd.DataFrame:
    dd = DAYS_ALL / day
    ph = json.loads((dd / "safe" / "photos.json").read_text())
    ex = json.loads((dd / "safe" / "exif.json").read_text())
    lab = json.loads(LABELS.read_text())["days"].get(day, {})
    labels, cap = lab.get("photos", {}), lab.get("capture_time", {})
    cl = json.loads((dd / "safe" / "clusters.json").read_text())["clusters"]
    cid = {p: k for k, c in enumerate(cl) for p in c["photos"]}
    gps = json.loads((GPS_JSON / f"{day}.json").read_text()) if (GPS_JSON / f"{day}.json").exists() else {}
    rows = []
    for k, v in ph.items():
        e = ex.get(k, {})
        dto, sub = e.get("DateTimeOriginal"), e.get("SubSecTimeOriginal") or "0"
        local = pd.Timestamp(pd.to_datetime(dto, format="%Y:%m:%d %H:%M:%S")) + pd.Timedelta(seconds=float("0." + sub.strip()))
        lk = k.split("__")[-1]
        rows.append(dict(key=k, camera=v["camera"], local=local, label=labels.get(lk), cluster=cid.get(k),
                         cap_label=cap.get(lk), has_gps=k in gps, status=v.get("status")))
    df = pd.DataFrame(rows).sort_values("local").reset_index(drop=True)
    df["utc"] = df.local - pd.Timedelta(hours=HST)
    # cross-check: catalogue capture_time string = EXIF local time
    c = df.dropna(subset=["cap_label"])
    df.attrs["cap_check"] = dict(n=int(len(c)), max_abs_s=float((pd.to_datetime(c.cap_label) - c.local).dt.total_seconds().abs().max()) if len(c) else None)
    df.attrs["gps_distinct"] = len({tuple(v) for v in gps.values()})
    return df


def notes_day(day: str) -> dict:
    sys.path.insert(0, str(NOTES))
    from tables import tables, xsec  # noqa: E402
    T = tables()
    s = T["sighting"]
    s = s[s.day == day]
    a = T["animals"]
    a = a[a.day == day]
    groups = []
    for r in s.to_dict("records"):
        an = a[a.Sighting_ID == r["Sighting_ID"]]
        groups.append(dict(g=int(r["Sighting#"]), start_s=xsec(r["Start time"]), end_s=xsec(r["End time"]),
                           type=r["Group type"], size_avg=r["Gr size avg"], size_actual=r["Actual Gr size"],
                           tagged=r["Tagged?"], video_platform=r["Video_platform"],
                           has_start_pos=bool(pd.notna(r["Start location"])),
                           has_end_pos=bool(pd.notna(r["End location"])),
                           whales=[dict(n=int(w["Whale #"]), role=w["Ind_role"], age=w["Age class"], sex=w["Sex"],
                                        comment=w["Animal comment"]) for _, w in an.iterrows()]))
    return dict(groups=groups)


# ------------------------------------------------------------------------------------------------ matching

def dist_to_intervals(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance (s) of each x to the nearest [a, b] interval (0 inside); a, b sorted by a, non-overlapping."""
    if not len(a):
        return np.full(len(x), np.inf)
    j = np.searchsorted(a, x, side="right") - 1
    d = np.full(len(x), np.inf)
    ok = j >= 0
    d[ok] = np.where(x[ok] <= b[j[ok]], 0.0, x[ok] - b[j[ok]])
    k = np.clip(j + 1, 0, len(a) - 1)
    nxt = np.where(j + 1 < len(a), a[k] - x, np.inf)
    return np.minimum(d, np.maximum(nxt, 0))


def nearest_idx(x, a, b):
    da = np.abs(x[:, None] - a[None, :])
    db = np.abs(x[:, None] - b[None, :])
    inside = (x[:, None] >= a[None, :]) & (x[:, None] <= b[None, :])
    dd = np.where(inside, 0, np.minimum(da, db))
    return dd.argmin(1)


def cover_share(a, b, lo, hi, d):
    """Share of [lo, hi] within d s of a surfacing interval."""
    grid = np.arange(lo, hi, 0.1)
    return float((dist_to_intervals(grid, a, b) <= d).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    notes_cache = {}
    for day, cfg in DEPS.items():
        tg = tag_side(day, cfg)
        ph = photo_side(day)
        try:
            notes_cache[day] = notes_day(day)
        except Exception as e:  # noqa: BLE001
            notes_cache[day] = dict(error=str(e))
        t0 = tg["t0"]
        # photo time in tag seconds (t), shift 0
        ph["t"] = (ph.utc - t0).dt.total_seconds()
        s = tg["surf"].sort_values("start_t")
        a, b = s.start_t.to_numpy(), s.end_t.to_numpy()
        on0, on1 = tg["on0"], tg["on1"]
        hours = (on1 - on0) / 3600
        on_utc0, on_utc1 = t0 + pd.Timedelta(seconds=on0), t0 + pd.Timedelta(seconds=on1)
        inside = (ph.t >= on0) & (ph.t <= on1)
        r = dict(tag=cfg["tag"], tag_utc_offset_h=tg["utc_offset_h"],
                 on_whale_utc=[str(on_utc0), str(on_utc1)],
                 on_whale_hst=[str(on_utc0 + pd.Timedelta(hours=HST)), str(on_utc1 + pd.Timedelta(hours=HST))],
                 on_whale_h=round(hours, 3), tag_minus_gnss_s=tg["tag_minus_gnss"],
                 recordings_on_disk=len(tg["rec"]),
                 recordings_on_disk_whale_s=int(tg["rec"].whale_windows.sum()) if len(tg["rec"]) else 0,
                 photos=len(ph), photos_status=dict(Counter(ph.status)), cap_check=ph.attrs["cap_check"],
                 photo_hst=[str(ph.local.min()), str(ph.local.max())],
                 cameras={c: dict(n=int(len(g)), first=str(g.local.min().time()), last=str(g.local.max().time()),
                                  inside_on_whale=int(((g.t >= on0) & (g.t <= on1)).sum()))
                          for c, g in ph.groupby("camera")},
                 overlap_min=round(max(0.0, (min(on1, ph.t.max()) - max(on0, ph.t.min())) / 60), 1),
                 photos_inside_on_whale=int(inside.sum()),
                 surfacings=int(len(s)), surfacings_per_h=round(len(s) / hours, 2),
                 surfacing_dur_s=dict(median=round(float(s.duration_s.median()), 2), sum=round(float(s.duration_s.sum()), 1)),
                 bouts=int(len(tg["bouts"])), bouts_per_h=round(len(tg["bouts"]) / hours, 2),
                 photos_with_gps=int(ph.has_gps.sum()), photo_gps_distinct=ph.attrs["gps_distinct"])
        # notes: photos inside each group's interval (clock check against the boat's notes)
        ng = notes_cache[day].get("groups", [])
        sod = (ph.local - ph.local.dt.normalize()).dt.total_seconds().to_numpy()
        grp_lab = ph.label.fillna("").str.extract(r"_G(\d\d)A")[0]
        chk = []
        for g in ng:
            if not (np.isfinite(g["start_s"]) and np.isfinite(g["end_s"])):
                continue
            m = (sod >= g["start_s"]) & (sod <= g["end_s"])
            lab_same = int(((grp_lab == f"{g['g']:02d}") & m).sum())
            chk.append(dict(g=g["g"], start=f"{int(g['start_s'])//3600:02d}:{int(g['start_s'])%3600//60:02d}",
                            end=f"{int(g['end_s'])//3600:02d}:{int(g['end_s'])%3600//60:02d}",
                            photos_in=int(m.sum()), labelled_this_group=int((grp_lab == f"{g['g']:02d}").sum()),
                            labelled_this_group_in=lab_same))
        r["notes_group_check"] = chk
        # coarse camera-zone check: photos labelled G_k inside notes interval G_k at shifts of whole hours
        hour_chk = {}
        for hsh in (-2, -1, 0, 1, 2):
            n_in = 0
            n_lab = 0
            for g in ng:
                if not (np.isfinite(g["start_s"]) and np.isfinite(g["end_s"])):
                    continue
                sel = (grp_lab == f"{g['g']:02d}").to_numpy()
                n_lab += int(sel.sum())
                x = sod[sel] + 3600 * hsh
                n_in += int(((x >= g["start_s"]) & (x <= g["end_s"])).sum())
            hour_chk[str(hsh)] = [n_in, n_lab]
        r["labelled_photos_in_own_group_interval_by_hour_shift"] = hour_chk
        # Q2: photos near surfacings, shift 0
        x = ph.t.to_numpy()
        dist0 = dist_to_intervals(x, a, b)
        r["near_surfacing_shift0"] = {}
        for dthr in (3, 5, 15):
            share = cover_share(a, b, on0, on1, dthr)
            n = int((dist0 <= dthr).sum())
            r["near_surfacing_shift0"][str(dthr)] = dict(photos=n, of=int(inside.sum()), share_of_time=round(share, 4),
                                                         chance=round(share * int(inside.sum()), 1))
        # Q3: offset scan
        scan = []
        for sh in SHIFTS:
            dd = dist_to_intervals(x + sh, a, b)
            m3 = dd <= 3
            # distinct surfacings hit
            hit = len(set(nearest_idx((x + sh)[m3], a, b))) if m3.any() else 0
            scan.append((int(sh), int(m3.sum()), hit))
        sc = pd.DataFrame(scan, columns=["shift_s", "photos_3s", "surfacings_hit"])
        sc.to_csv(OUT / f"shift_scan_{day}.csv", index=False)
        med = float(sc.photos_3s.median())
        mad = float((sc.photos_3s - med).abs().median())
        best = sc.loc[sc.photos_3s.idxmax()]
        top = sc.sort_values("photos_3s", ascending=False).head(5)
        r["shift_scan"] = dict(best_shift_s=int(best.shift_s), best_photos=int(best.photos_3s),
                               best_surfacings_hit=int(best.surfacings_hit), median=med, mad=mad,
                               p95=float(sc.photos_3s.quantile(.95)), p99=float(sc.photos_3s.quantile(.99)),
                               peak_over_median=round(float(best.photos_3s) / med, 2) if med > 0 else None,
                               robust_z=round((float(best.photos_3s) - med) / (1.4826 * mad), 2) if mad > 0 else None,
                               at_zero=int(sc.loc[sc.shift_s == 0, "photos_3s"].iloc[0]),
                               top5=top.values.tolist(),
                               surf_hit_median=float(sc.surfacings_hit.median()),
                               surf_hit_max=int(sc.surfacings_hit.max()),
                               surf_hit_best_shift=int(sc.loc[sc.surfacings_hit.idxmax(), "shift_s"]))
        # per camera scans
        r["shift_scan_per_camera"] = {}
        for c, g in ph.groupby("camera"):
            xc = g.t.to_numpy()
            cnt = np.array([(dist_to_intervals(xc + sh, a, b) <= 3).sum() for sh in SHIFTS])
            mc = float(np.median(cnt))
            r["shift_scan_per_camera"][c.split("|")[0][-6:] + "|" + c.split("|")[1]] = dict(
                n=int(len(g)), best_shift_s=int(SHIFTS[cnt.argmax()]), best=int(cnt.max()), median=mc,
                p99=float(np.quantile(cnt, .99)))
        # Q4: labels / clusters at best shift and at 0
        bs = int(best.shift_s)
        ph["dist_shift0"] = dist0
        ph["dist_best"] = dist_to_intervals(x + bs, a, b)
        near_i = nearest_idx(x + bs, a, b)
        ph["surf_id_best"] = s.surfacing_id.to_numpy()[near_i]
        inn = inside.to_numpy()
        base = Counter(ph.label[inn].fillna("unlabelled"))
        r["labels_inside_on_whale"] = dict(base.most_common())
        for nm, col in (("best", "dist_best"), ("shift0", "dist_shift0")):
            for dthr in (3, 5, 15):
                m = ph[col] <= dthr
                r[f"labels_{nm}_{dthr}s"] = dict(Counter(ph.label[m].fillna("unlabelled")).most_common())
                r[f"clusters_{nm}_{dthr}s"] = dict(n_photos=int(m.sum()), n_clusters=int(ph.cluster[m].nunique()),
                                                     top=[[int(k), v] for k, v in Counter(ph.cluster[m].dropna()).most_common(5)])
        # at the event-synced shift: counts, chance, labels, signed lag photo - surfacing start
        es = EVENT_SHIFT[day]
        de = dist_to_intervals(x + es, a, b)
        ph["dist_event"] = de
        ev = dict(shift_s=es, scan_count_at_round_shift=int(sc.loc[sc.shift_s == int(round(es)), "photos_3s"].iloc[0]))
        for dthr in (3, 5, 15):
            share = cover_share(a, b, on0, on1, dthr)
            m = de <= dthr
            ev[f"{dthr}s"] = dict(photos=int(m.sum()), of=int(inside.sum()), chance=round(share * int(inside.sum()), 1),
                                  surfacings_hit=int(len(set(nearest_idx((x + es)[m], a, b)))) if m.any() else 0,
                                  labels=dict(Counter(ph.label[m].fillna("unlabelled")).most_common()))
        j = nearest_idx(x[inn] + es, a, b)
        lag = (x[inn] + es) - a[j]
        lag = lag[np.abs(lag) <= 30]
        ev["lag_photo_minus_surfacing_start_s"] = dict(n_within_30s=int(len(lag)),
                                                       hist_5s=np.histogram(lag, bins=np.arange(-30, 31, 5))[0].tolist(),
                                                       median=round(float(np.median(lag)), 2) if len(lag) else None)
        # per animal: share of its on-whale photos within 5 s of a surfacing at the event shift
        ev["per_animal_within_5s"] = {an: [int((g.dist_event <= 5).sum()), int(len(g))]
                                      for an, g in ph[inn].groupby(ph.label[inn].fillna("unlabelled"))}
        r["event_shift"] = ev
        # per-animal shift scans (is there an animal whose photos peak at a shift?)
        per_an = {}
        for an, g in ph[inn].groupby(ph.label[inn].fillna("unlabelled")):
            if len(g) < 5:
                continue
            xc = g.t.to_numpy()
            cnt = np.array([(dist_to_intervals(xc + sh, a, b) <= 3).sum() for sh in SHIFTS])
            per_an[an] = dict(n=int(len(g)), best_shift_s=int(SHIFTS[cnt.argmax()]), best=int(cnt.max()),
                              median=float(np.median(cnt)), p99=float(np.quantile(cnt, .99)),
                              at_zero=int(cnt[SHIFTS == 0][0]))
        r["per_animal_scan"] = per_an
        # GNSS snapshots on the tag
        ub = ubx_epochs(ROOT / "underwater_video" / cfg["dep"] / cfg["ubx"])
        if len(ub):
            ub["t"] = (ub.utc - t0).dt.total_seconds()  # GNSS UTC on the tag's t axis (shift by tag-gnss offset ignored)
            ub = ub[ub.nmeas >= 4]
            ow = ub[(ub.t >= on0) & (ub.t <= on1)].copy()
            # sessions: epochs closer than 2 s
            sess = (ow.t.diff() > 2).cumsum() if len(ow) else pd.Series(dtype=int)
            ss = ow.groupby(sess).t.agg(["min", "max", "size"]) if len(ow) else pd.DataFrame()
            off = tg["tag_minus_gnss"]["median"] or 0.0
            sd = dist_to_intervals(ss["min"].to_numpy() + off, a, b) if len(ss) else np.array([])
            r["tag_gnss"] = dict(epochs_all=int(len(ub)), epochs_on_whale=int(len(ow)), sessions_on_whale=int(len(ss)),
                                 sessions_per_h=round(len(ss) / hours, 2),
                                 sessions_within_3s_of_surfacing=int((sd <= 3).sum()),
                                 surfacings_with_session=int(len(set(nearest_idx(ss["min"].to_numpy() + off, a, b)[sd <= 3]))) if len(ss) else 0,
                                 kind="UBX RXM-RAWX raw measurements (no position solution stored)")
        else:
            r["tag_gnss"] = dict(epochs_all=0)
        # write per-photo / per-surfacing tables (outside the repo)
        keep = ph[["key", "camera", "local", "utc", "t", "label", "cluster", "dist_shift0", "dist_best", "dist_event", "surf_id_best"]]
        keep.to_csv(OUT / f"photos_{day}.csv", index=False)
        s.to_csv(OUT / f"surfacings_{day}.csv", index=False)
        tg["rec"].to_csv(OUT / f"recordings_{day}.csv", index=False)
        r["notes"] = notes_cache[day]
        res[day] = r
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
