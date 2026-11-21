"""End to end (WP7): a folder of camera-tag video -> per-window table, keep / drop intervals, clips.

``uwid scan DIR`` runs :func:`scan`. For every recording under ``DIR``:

1. **Sensors first.** When the recording's deployment folder holds the tag's sensor log (a CATS-style CSV), the tag
   state (``air`` on deck / in hand, ``flight`` under the drone, ``floating`` free in the sea, ``whale``) and the
   lens state (under water / surfaced, from depth with the measured lens lags) come from :mod:`uwid.tagdata`.
   Sensor time is linked to the video by the clock burned into the picture: sensor time = clock - ``clock_offset_s``
   (default 2.03 s, measured on tag CC-11-96). An **offset check** compares the picture's own exits / entries
   (:mod:`uwid.waterstate`) with the depth's and reports the median difference, so a wrong offset on a new tag is
   seen (``--clock-offset auto`` adopts the measured value).
   Without a log, the picture water state (:func:`uwid.waterstate.predict_video`) stands in.
2. **Whale detector** (OWLv2, prompt "whale", :mod:`uwid.detect`) runs only on windows that are on the whale and
   under water (or crossing), on two frames per 1 s window (at +0.25 s and +0.75 s). Boxes in the burned-in clock
   and boxes covering more than 60 % of the frame are dropped (WP5a operating point, fixed on the dev clip).
3. **Keep rule**, preset ``strict`` (both frames have a "whale" box >= 0.25) or ``sensitive`` (either frame does).
   Whale windows are padded by 2 s on each side, then stretches less than 5 s apart (after padding) are merged.
   Nothing is dropped for being short. Every other window gets one reason from :data:`REASONS`.
4. Writes, under ``<cache>/scan/<deployment or video name>/<run label>/``: ``windows/<stem>.csv`` (one row per
   second), ``intervals.json`` (format ``uwid-intervals/1``), ``dropped.csv`` (stretches with reasons),
   ``summary.json``; ``offset_check.json`` next to it. With ``cut=True`` the keep intervals are cut by
   :mod:`uwid.clips` into ``<cache>/clips/<name>/``.

Re-running reuses the clock fits, picture water states and detector boxes cached under ``<name>/cache/`` (the
detector cache keeps every box with the scores of all prompts at its patch, so the options below re-score without
running the model again). Source footage is only read.

Options (off by default, each reported once in experiments/06_clips_e2e as NOT a clean test, because the idea came
from failures seen on the test pieces): ``near_whale_score`` keeps a box covering most of the frame when its score
is at least this (0.6 = above every whole-frame box on the dev clip, max 0.58); ``fish_veto`` counts a "whale" box
only when its whale score beats the best fish-prompt score of the same box. The fish score of each window is written
as information and never changes the keep decision.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import paths, video

REASONS = ("tag in air", "drone flight", "tag floating", "surfaced", "underwater with no whale",
           "small (fish or far whale)", "unreadable")
PRESETS = {"strict": "both frames of the window have a whale box >= threshold",
           "sensitive": "either frame of the window has a whale box >= threshold"}
FORMAT = "uwid-intervals/1"
TAG_REASON = {"air": "tag in air", "flight": "drone flight", "floating": "tag floating"}


@dataclass
class ScanConfig:
    preset: str = "sensitive"
    threshold: float = 0.25              # WP5a, chosen on the dev clip
    whale_query: str = "whale"
    fish_queries: tuple = ("fish", "school of fish", "small fish")
    bubble_query: str = "air bubbles"    # for the quality tier only
    whole_frame: float = 0.6             # boxes covering more than this share of the frame are dropped ...
    near_whale_score: float | None = 0.6  # ... unless their score is at least this (None = always dropped)
    min_whale_px: float = 300.0          # whale boxes shorter than this (longest side, 1920 px frame) are 'small'
    hq_threshold: float = 0.6            # hq_whale: quality score at least this (first guess) ...
    hq_min_cover: float = 0.10           # ... and the whale box covers at least this share of the frame
    fish_veto: bool = False              # option: whale box only if its whale score beats its fish score
    frame_offsets: tuple = (0.25, 0.75)  # frames per window, seconds after the window start
    pad_s: float = 2.0
    merge_gap_s: float = 5.0             # after padding, stretches closer than this are merged
    clock_offset_s: float | str = 2.03   # burned-in clock minus sensor time; "auto" = take the offset check's value
    lens_out_lag_s: float = 0.3          # lens out this long after depth says surfaced (tag CC-11-96, by eye)
    lens_in_lead_s: float = 0.4          # lens under again this long before depth says submerged
    scan_all: bool = False               # run the detector on every window (cost comparison only)
    picture_check: bool = True           # run the picture water model with a log too (offset check)
    min_age_s: float = 60.0              # files changed more recently are being copied: skipped
    gpu_lock: str | None = "/var/tmp/sat3737-parts-cache/gpu_train.lock"
    lock_budget_s: float = 720.0         # hold the GPU lock at most this long at a time
    gpu_need_gb: float = 2.0
    detector_batch: int = 4
    clock_probes: int = 60

    def label(self) -> str:
        s = self.preset
        if self.near_whale_score is None:
            s += "+nonear"
        elif self.near_whale_score != 0.6:
            s += f"+near{self.near_whale_score:g}"
        if self.min_whale_px != 300.0:
            s += f"+px{self.min_whale_px:g}"
        if self.fish_veto:
            s += "+fishveto"
        if self.scan_all:
            s += "+scanall"
        return s


# ------------------------------------------------------------------------------------------- discovery

@dataclass
class Recording:
    path: Path
    form: str                           # file extension without the dot

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def key(self) -> str:
        return f"{self.path.stem.replace(' ', '_')}.{self.form}"


@dataclass
class Deployment:
    name: str
    folder: Path | None
    sensor_csv: Path | None
    recordings: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def _is_debris(p: Path) -> bool:
    return p.name.startswith("._") or p.name in {".DS_Store", "Thumbs.db", "desktop.ini"}


def _stable(p: Path, min_age_s: float) -> bool:
    try:
        return (time.time() - p.stat().st_mtime) >= min_age_s
    except OSError:
        return False


def find_sensor_log(folder: Path, recursive: bool = True) -> Path | None:
    """The tag's sensor CSV in ``folder``: a CSV whose header names Depth and Accelerometer columns (the largest one
    if several). ``recursive=False`` looks only in ``folder`` and ``folder/Data``."""
    if recursive:
        cands = folder.rglob("*.csv")
    else:
        cands = [p for d in (folder, folder / "Data") if d.is_dir() for p in d.glob("*.csv")]
    best = None
    for p in sorted(cands):
        if _is_debris(p) or not p.is_file():
            continue
        try:
            with open(p, "r", encoding="latin-1", errors="replace") as fh:
                head = fh.readline()
        except OSError:
            continue
        if "Depth" in head and "Accelerometer" in head:
            if best is None or p.stat().st_size > best.stat().st_size:
                best = p
    return best


def _deployment_folder(v: Path, root: Path) -> Path:
    parent = v.parent
    if parent.name.lower() in ("video", "videos") and parent != root:
        return parent.parent
    return parent


def discover(target: str | os.PathLike, min_age_s: float = 60.0) -> list[Deployment]:
    """Group the videos under ``target`` (a folder or one video file) into deployments.

    A deployment is a folder with a sensor log (videos in it or in its ``Video/`` folder); a video whose folder
    has no log is its own deployment, named after the video. When a recording exists as ``.mov`` and as another
    form (the ``.m4v`` conversions), only the ``.mov`` is used. Files changed less than ``min_age_s`` ago are
    skipped (still being copied)."""
    target = Path(target)
    if target.is_file():
        vids = [target]
        root = target.parent.parent if target.parent.name.lower() in ("video", "videos") else target.parent
    else:
        vids = sorted(p for p in target.rglob("*") if p.is_file() and video.is_video_file(p) and not _is_debris(p))
        root = target
    groups: dict[Path | str, Deployment] = {}
    log_cache: dict[Path, Path | None] = {}
    for v in vids:
        folder = _deployment_folder(v, root) if not target.is_file() else root
        if folder not in log_cache:
            # a folder holding loose videos is searched shallowly: a log deeper down belongs to another deployment
            log_cache[folder] = find_sensor_log(folder, recursive=folder != (target if target.is_dir()
                                                                             else target.parent))
        csv = log_cache[folder]
        if csv is not None:
            key: Path | str = folder
            if key not in groups:
                groups[key] = Deployment(folder.name, folder, csv)
        else:
            key = str(v.with_suffix(""))
            if key not in groups:
                groups[key] = Deployment(v.stem.replace(" ", "_"), None, None)
        groups[key].recordings.append(v)
    out = []
    for dep in groups.values():
        by_stem: dict[str, list[Path]] = {}
        for v in dep.recordings:
            by_stem.setdefault(v.stem, []).append(v)
        recs = []
        for stem, forms in sorted(by_stem.items()):
            forms = sorted(forms, key=lambda p: (p.suffix.lower() != ".mov", p.suffix.lower()))
            pick = forms[0]
            for other in forms[1:]:
                dep.skipped.append({"path": str(other), "reason": f"other form of {pick.name} (the .mov is used)"})
            if not _stable(pick, min_age_s):
                dep.skipped.append({"path": str(pick), "reason": f"modified less than {min_age_s:.0f} s ago (copying)"})
                continue
            recs.append(Recording(pick, pick.suffix.lower().lstrip(".")))
        dep.recordings = recs
        if dep.sensor_csv is not None and not _stable(dep.sensor_csv, min_age_s):
            dep.skipped.append({"path": str(dep.sensor_csv), "reason": "sensor log still being copied: not used"})
            dep.sensor_csv = None
        out.append(dep)
    return out


# ------------------------------------------------------------------------------------------- GPU slot

def _gpu_free_gb() -> float | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        used, total = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return (total - used) / 1024.0
    except Exception:
        return None


_HELD: dict = {}   # the GPU lock this process holds: {"fh": file, "since": time, "runner": model kept loaded}


def release_gpu() -> None:
    """Release the shared GPU lock if this process holds it (and drop the kept model)."""
    import fcntl
    fh = _HELD.pop("fh", None)
    _HELD.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if fh:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


@contextlib.contextmanager
def gpu_slot(cfg: ScanConfig, log=print, check_memory: bool = True):
    """Hold the shared GPU lock (flock on ``cfg.gpu_lock``) with enough free memory; yields the waiting time.

    The lock is kept after the block, so consecutive GPU stages (several recordings) share one hold, until it has
    been held for ``cfg.lock_budget_s``: then it is released before the next block waits for it again. Call
    :func:`release_gpu` when the GPU work is done."""
    import fcntl
    if _HELD.get("fh") is not None and time.time() - _HELD["since"] < cfg.lock_budget_s:
        yield 0.0
        return
    release_gpu()
    fh = open(cfg.gpu_lock, "a") if cfg.gpu_lock else None
    t0 = time.time()
    while True:
        if fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
        free = _gpu_free_gb() if check_memory else None
        if free is None or free - 1.5 >= cfg.gpu_need_gb:
            break
        if fh:
            fcntl.flock(fh, fcntl.LOCK_UN)
        log(f"GPU: {free:.1f} GB free, waiting")
        time.sleep(60)
    _HELD.update(fh=fh, since=time.time())
    yield time.time() - t0


def lock_time_left(cfg: ScanConfig) -> float:
    return cfg.lock_budget_s - (time.time() - _HELD["since"]) if _HELD.get("fh") is not None else cfg.lock_budget_s


# ------------------------------------------------------------------------------------------- clock fit

def fit_burned_clock(path: Path, n_probes: int = 60, frames_per_probe: int = 6) -> dict:
    """Fit clock (seconds of the day) = a + b * pts from the burned-in clock.

    As in experiments/01_metadata: at ``n_probes`` evenly spaced times, ``frames_per_probe`` consecutive frames are
    read; where the tenths change between two frames, the change time is the midpoint of the two frames and the
    value is the new reading. Without changes the plain readings are fitted."""
    from . import names, timecode
    tab = video.frame_table(path)
    try:
        glyphs = timecode.load_glyphs()
    except FileNotFoundError:
        return {"ok": False, "why": "no glyphs"}
    hint = names.date_hint(path)
    xs, ys, plain_x, plain_y, n_read, n_try = [], [], [], [], 0, 0
    date = None
    times = np.linspace(0.5, max(tab.duration - 1.0, 0.6), n_probes)
    with video.VideoReader(path, fmt="gray") as r:
        for t in times:
            i0 = tab.index_at(float(t))
            prev = None
            for i in range(i0, min(i0 + frames_per_probe, tab.n_frames)):
                fr = r.frame_by_index(i)
                n_try += 1
                rd = timecode.read_clock(fr.image, glyphs, hint) if fr is not None else None
                if rd is None:
                    prev = None
                    continue
                n_read += 1
                date = date or (rd.datetime.date().isoformat() if rd.datetime else None)
                plain_x.append(fr.t)
                plain_y.append(rd.seconds_of_day)
                if prev is not None and 0.05 < rd.seconds_of_day - prev[1] < 0.15 and fr.t - prev[0] < 0.05:
                    xs.append((fr.t + prev[0]) / 2)
                    ys.append(rd.seconds_of_day)
                prev = (fr.t, rd.seconds_of_day)
    out = {"n_frames_tried": n_try, "n_read": n_read, "share_read": n_read / max(n_try, 1), "date": date}
    use_x, use_y, kind = (xs, ys, "tenth changes") if len(xs) >= 5 else (plain_x, plain_y, "plain readings")
    if len(use_x) < 3 or out["share_read"] < 0.5:
        out.update(ok=False, why="clock not readable")
        return out
    fit = timecode.robust_line_fit(np.array(use_x), np.array(use_y))
    fit.pop("residuals", None)
    fit.pop("inliers", None)
    out.update(ok=True, kind=kind, n_points=len(use_x), clock_at_pts0=fit["intercept"], rate=fit["slope"],
               rms_s=fit["rms_residual_inliers_s"], n_outliers=fit["n_outliers"])
    return out


def clock_text(sod: float) -> str:
    sod = sod % 86400
    h, rem = divmod(sod, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:04.1f}"


# ------------------------------------------------------------------------------------------- sensors

SURFACE_AIR_M = -0.062   # raw depth reading in air (tagdata.SurfacingParams); depth_m = raw - this


class SensorTimeline:
    """Tag state and lens-in-air intervals in sensor time (s since the first log sample) for one deployment."""

    def __init__(self, deployment: str, csv_path: Path, cfg: ScanConfig, log=print):
        from . import tagdata
        self.deployment = deployment
        if not (tagdata.cache_dir(deployment) / "DONE").exists():
            log(f"{deployment}: building the sensor cache from {csv_path.name} (one pass over the CSV)")
            tagdata.build_cache(csv_path, deployment, log=log)
        tabs = tagdata.derive_tables(deployment, csv_path, log=log)
        self.meta = tagdata.load_meta(deployment)
        states, drops, surf = tabs["states"], tabs["drops"], tabs["surfacings"]
        segs = [[float(r.start_t), float(r.end_t) + 0.02, str(r.state)] for r in states.itertuples()]
        air = []
        self.notes = []
        dep_upd = None
        # an attaching drop: the tag is on the whale from the impact (states.csv keeps 'flight' until depth > 1 m);
        # the lens is in air until the depth first passes the surfacing threshold
        if len(drops) and "attached" in drops.columns:
            thr = float(np.nanmedian(surf.air_reading_m)) + 0.10 if len(surf) and "air_reading_m" in surf else 0.04
            for d in drops[drops.attached.astype(bool)].itertuples():
                imp = float(d.impact_t)
                if dep_upd is None:
                    dep_upd = tagdata.load_depth_updates(deployment)
                after = dep_upd[(dep_upd.t > imp) & (dep_upd.depth > thr) & (dep_upd.depth < tagdata.DEPTH_GLITCH_M)]
                t_in = (float(after.t.iloc[0]) if len(after) else imp) - cfg.lens_in_lead_s
                new = []
                for a, b, s in segs:
                    if s == "flight" and a < imp < b:
                        new += [[a, imp, "flight"], [imp, b, "whale"]]
                    else:
                        new.append([a, b, s])
                segs = new
                if t_in > imp:
                    air.append((imp, t_in))
                self.notes.append(f"attaching drop: impact at t={imp:.2f}, on the whale from there, lens in air "
                                  f"until t={t_in:.2f}")
        df = tagdata.clean(tagdata.load(deployment))
        self.depth_t = df.t.to_numpy()[::5]
        self.depth_m = (df.depth_clean.to_numpy() - SURFACE_AIR_M)[::5]
        # release: the tag leaves the whale at depth and rises; from the gyro jump on it is not on the whale
        wh = [s for s in segs if s[2] == "whale"]
        if wh:
            last = max(wh, key=lambda s: s[1])
            nxt = [s for s in segs if s[0] >= last[1] - 0.05]
            if nxt and min(nxt, key=lambda s: s[0])[2] in ("floating", "air"):
                rel = tagdata.release_time(df, last[1])
                if last[0] < rel < last[1] - 1.0:
                    segs.remove(last)
                    segs += [[last[0], rel, "whale"], [rel, last[1], "floating"]]
                    self.notes.append(f"release: off the whale at t={rel:.2f} (gyro jump), rising until t={last[1]:.2f}"
                                      " counted as floating")
        self.segments = pd.DataFrame(sorted(segs), columns=["t0", "t1", "state"])
        for r in surf.itertuples():
            a, b = float(r.start_t) + cfg.lens_out_lag_s, float(r.end_t) - cfg.lens_in_lead_s
            if b > a:
                air.append((a, b))
        self.air = pd.DataFrame(sorted(air), columns=["t0", "t1"])
        self.surfacings = surf
        self.drops = drops
        self.first_utc = self.meta["first_utc"]
        self.utc_offset_h = float(self.meta["utc_offset_h"])
        self.t_end = float(self.meta["duration_s"])

    def depth_events(self, cfg: ScanConfig) -> pd.DataFrame:
        """Lens exits / entries predicted from depth (surfacing start + lag, end - lead), sensor time."""
        s = self.surfacings
        ev = [(float(r.start_t) + cfg.lens_out_lag_s, "exit") for r in s.itertuples()]
        ev += [(float(r.end_t) - cfg.lens_in_lead_s, "entry") for r in s.itertuples()]
        return pd.DataFrame(sorted(ev), columns=["t", "kind"])

    def sod_of_first_sample(self) -> float:
        first = pd.Timestamp(self.first_utc) + pd.Timedelta(hours=self.utc_offset_h)
        return float((first - first.normalize()).total_seconds())


def _overlap(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Overlap (s) of each interval [a0, a1) with the union of the intervals [b0, b1) (assumed disjoint)."""
    out = np.zeros(len(a0))
    if len(b0) == 0:
        return out
    for x0, x1 in zip(b0, b1):
        out += np.clip(np.minimum(a1, x1) - np.maximum(a0, x0), 0, None)
    return out


# ------------------------------------------------------------------------------------------- picture water

def picture_water(rec: Recording, cache: Path, cfg: ScanConfig, log=print, water_fn: Callable | None = None,
                  timings: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Picture water state per 1 s window (cached): columns k, state, score (p air); and the picture's events."""
    wf, ef = cache / f"{rec.key}.water.csv", cache / f"{rec.key}.water_events.csv"
    if wf.exists() and ef.exists():
        return pd.read_csv(wf), pd.read_csv(ef)
    t0 = time.time()
    if water_fn is not None:
        win, ev = water_fn(rec.path)
        held = time.time() - t0
    else:
        from . import waterstate
        with gpu_slot(cfg, log) as waited:
            t1 = time.time()
            win, ev, _ = waterstate.predict_video(rec.path)
            held = time.time() - t1
        log(f"  picture water state: {held:.0f} s on the GPU (waited {waited:.0f} s)")
    cache.mkdir(parents=True, exist_ok=True)
    win[["k", "t0", "t1", "state", "score"]].to_csv(wf, index=False)
    ev.to_csv(ef, index=False)
    if timings is not None:
        timings["water_picture_gpu_s"] = timings.get("water_picture_gpu_s", 0.0) + held
    (cache / f"{rec.key}.water_timing.json").write_text(json.dumps({"gpu_held_s": held, "wall_s": time.time() - t0}))
    return pd.read_csv(wf), pd.read_csv(ef)


# ------------------------------------------------------------------------------------------- offset check

def offset_check(pic_events: pd.DataFrame, depth_events: pd.DataFrame, search_s: float = 30.0,
                 tol_s: float = 0.5, pair_s: float = 1.0, min_pairs: int = 5) -> dict:
    """Compare picture exits / entries with depth exits / entries, both in sensor time (the picture's mapped with
    the offset in use). Finds the shift (within +-search_s, 0.1 s steps) under which most depth events have a
    picture event of the same kind within tol_s, then pairs events within pair_s of that shift and reports the
    median of picture minus depth: the offset in use is off by this much (positive = the burned-in clock is further
    ahead of the sensors than assumed)."""
    res: dict = {"n_depth_events": int(len(depth_events)), "n_picture_events": int(len(pic_events))}
    if len(depth_events) == 0 or len(pic_events) == 0:
        res.update(verdict="no events to compare", n_pairs=0)
        return res
    best = (-1, 0.0)
    for lag in np.arange(-search_s, search_s + 1e-9, 0.1):
        n = 0
        for kind in ("exit", "entry"):
            d = depth_events[depth_events.kind == kind].t.to_numpy() + lag
            p = np.sort(pic_events[pic_events.kind == kind].t.to_numpy())
            if len(d) == 0 or len(p) == 0:
                continue
            j = np.clip(np.searchsorted(p, d), 1, len(p) - 1) if len(p) > 1 else np.zeros(len(d), int)
            near = np.minimum(np.abs(p[j] - d), np.abs(p[np.maximum(j - 1, 0)] - d))
            n += int((near <= tol_s).sum())
        if n > best[0] or (n == best[0] and abs(lag) < abs(best[1])):
            best = (n, float(lag))
    lag = best[1]
    pairs = []
    for kind in ("exit", "entry"):
        p = pic_events[pic_events.kind == kind].t.to_numpy()
        for t in depth_events[depth_events.kind == kind].t.to_numpy():
            if len(p) == 0:
                continue
            dd = p - t
            i = int(np.argmin(np.abs(dd - lag)))
            if abs(dd[i] - lag) <= pair_s:
                pairs.append((kind, float(t), float(dd[i])))
    diffs = np.array([x[2] for x in pairs])
    res.update(best_shift_s=lag, n_matched_at_shift=best[0], n_pairs=len(pairs))
    if len(pairs):
        res["median_picture_minus_depth_s"] = float(np.median(diffs))
        res["mad_s"] = float(np.median(np.abs(diffs - np.median(diffs))))
        for kind in ("exit", "entry"):
            k = np.array([x[2] for x in pairs if x[0] == kind])
            res[f"median_{kind}_s"] = float(np.median(k)) if len(k) else None
            res[f"n_{kind}"] = int(len(k))
    if len(pairs) < min_pairs:
        res["verdict"] = f"too few matched events ({len(pairs)}) to check the offset"
    elif abs(res["median_picture_minus_depth_s"]) <= 0.3:
        res["verdict"] = "ok"
    else:
        res["verdict"] = "OFFSET LOOKS WRONG"
    res["pairs"] = pairs
    return res


# ------------------------------------------------------------------------------------------- detector

def _detector_dir(cache: Path, rec: Recording) -> Path:
    return cache / "det" / rec.key


def video_size(path: Path) -> tuple[int, int]:
    vs = [st for st in video.probe(path)["streams"] if st["type"] == "video"]
    return (int(vs[0].get("width") or 1920), int(vs[0].get("height") or 1080)) if vs else (1920, 1080)


def detector_frame_size(cache: Path, rec: Recording) -> tuple[int, int]:
    """Size of the frames the detector saw (boxes are in these pixels): 4K footage is scaled to 1920 wide."""
    f = _detector_dir(cache, rec) / "meta.json"
    if f.exists():
        m = json.loads(f.read_text())
        if m.get("decode_size"):
            return tuple(m["decode_size"])
    return video_size(rec.path)


def load_boxes(cache: Path, rec: Recording) -> tuple[pd.DataFrame, np.ndarray]:
    d = _detector_dir(cache, rec)
    parts = sorted(d.glob("part_*.parquet"))
    dfs = [pd.read_parquet(p) for p in parts]
    dfs = [x for x in dfs if len(x)]
    boxes = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["frame", "query", "score"])
    fr = [np.load(p) for p in sorted(d.glob("part_*.frames.npy"))]
    return boxes, (np.unique(np.concatenate(fr)) if fr else np.zeros(0, np.int64))


def run_detector(rec: Recording, frames: np.ndarray, cache: Path, cfg: ScanConfig, log=print,
                 detector_factory: Callable | None = None, timings: dict | None = None) -> np.ndarray:
    """Run the detector on the frames not yet in the cache (resumable, at most ``cfg.lock_budget_s`` per hold of
    the GPU lock). Returns the processed frame indices."""
    d = _detector_dir(cache, rec)
    d.mkdir(parents=True, exist_ok=True)
    queries = [cfg.whale_query] + [q for q in cfg.fish_queries if q != cfg.whale_query] + [cfg.bubble_query]
    meta_f = d / "meta.json"
    vw, vh = video_size(rec.path)
    dsize = (1920, int(round(1920 * vh / vw / 2)) * 2) if vw > 1920 else None
    meta = {"model": "owlv2-base-patch16-ensemble", "queries": queries, "patch_scores": True, "min_score": 0.03,
            "decode_size": list(dsize) if dsize else [vw, vh]}
    if detector_factory is not None:
        meta["model"] = "injected"
    if meta_f.exists():
        old = json.loads(meta_f.read_text())
        if (old.get("queries") != queries or old.get("model") != meta["model"]
                or old.get("decode_size", meta["decode_size"]) != meta["decode_size"]):
            raise RuntimeError(f"{d}: cached with other settings {old}; remove it to re-run")
    meta_f.write_text(json.dumps(meta, indent=1))
    _, done = load_boxes(cache, rec)
    todo = np.setdiff1d(np.asarray(frames, np.int64), done)
    tab = video.frame_table(rec.path)
    while len(todo):
        slot = gpu_slot(cfg, log) if detector_factory is None else contextlib.nullcontext(0.0)
        with slot as waited:
            t_hold = time.time()
            budget = lock_time_left(cfg) if detector_factory is None else cfg.lock_budget_s
            if detector_factory is not None:
                runner = detector_factory(queries)
            elif _HELD.get("runner") is not None and _HELD.get("runner_queries") == queries:
                runner = _HELD["runner"]
            else:
                from . import detect
                runner = detect.OwlRunner(queries, patch_scores=True, batch=cfg.detector_batch)
                _HELD.update(runner=runner, runner_queries=queries)
            q: queue.Queue = queue.Queue(maxsize=16)
            stop = threading.Event()
            dec_time = [0.0]

            def worker():
                t_d = time.time()
                try:
                    frames_it = (video.iter_frames_at(rec.path, todo, size=dsize) if detector_factory is not None
                                 else video.iter_frames_isolated(rec.path, indices=todo, size=dsize))
                    for fr in frames_it:
                        if stop.is_set():
                            break
                        q.put((fr.index, fr.image))
                finally:
                    dec_time[0] = time.time() - t_d
                    q.put(None)

            th = threading.Thread(target=worker, daemon=True)
            th.start()
            buf_f, buf_i, part_rows, part_frames = [], [], [], []
            model_s = 0.0

            def flush_batch():
                nonlocal model_s, buf_f, buf_i
                if not buf_f:
                    return
                tm = time.time()
                dets = runner.detect(buf_i)
                model_s += time.time() - tm
                for f, df, im in zip(buf_f, dets, buf_i):
                    if len(df):
                        df = add_box_quality(df, im, cfg.whale_query)
                        df.insert(0, "frame", int(f))
                        df.insert(1, "t", float(tab.times[f]))
                        part_rows.append(df)
                    part_frames.append(int(f))
                buf_f, buf_i = [], []

            def write_part():
                nonlocal part_rows, part_frames
                if not part_frames:
                    return
                stamp = f"{int(time.time() * 1000):d}_{part_frames[0]:07d}"
                df = pd.concat(part_rows, ignore_index=True) if part_rows else pd.DataFrame(columns=["frame", "query",
                                                                                                    "score"])
                df.to_parquet(d / f"part_{stamp}.parquet", index=False)
                np.save(d / f"part_{stamp}.frames.npy", np.asarray(part_frames, np.int64))
                part_rows, part_frames = [], []

            out_of_time = False
            while True:
                item = q.get()
                if item is None:
                    break
                buf_f.append(item[0])
                buf_i.append(item[1])
                if len(buf_f) >= cfg.detector_batch:
                    flush_batch()
                if len(part_frames) >= 240:
                    write_part()
                if time.time() - t_hold > budget:
                    out_of_time = True
                    stop.set()
                    break
            flush_batch()
            write_part()
            stop.set()
            while th.is_alive():
                try:
                    q.get(timeout=0.1)
                except queue.Empty:
                    pass
            th.join(timeout=5)
            held = time.time() - t_hold
            del runner
            if out_of_time and detector_factory is None:
                release_gpu()
        if timings is not None:
            timings["detector_gpu_s"] = timings.get("detector_gpu_s", 0.0) + held
            timings["detector_model_s"] = timings.get("detector_model_s", 0.0) + model_s
            timings["detector_decode_s"] = timings.get("detector_decode_s", 0.0) + dec_time[0]
            timings["detector_wait_s"] = timings.get("detector_wait_s", 0.0) + float(waited)
        _, done = load_boxes(cache, rec)
        before = len(todo)
        todo = np.setdiff1d(np.asarray(frames, np.int64), done)
        log(f"  detector: {before - len(todo)} frames in {held:.0f} s (model {model_s:.0f} s), {len(todo)} left")
        if not out_of_time:
            break           # the rest could not be decoded
    return load_boxes(cache, rec)[1]


QUALITY_MIN_SCORE = 0.15   # whale boxes at least this good get the picture measures


def add_box_quality(df: pd.DataFrame, image: np.ndarray, whale_query: str = "whale") -> pd.DataFrame:
    """Picture measures inside each whale box (score >= QUALITY_MIN_SCORE), on a 960 px wide grey copy:
    ``q_bright`` mean grey (0-1), ``q_contrast`` |mean grey inside - mean grey of a ring around the box| (0-1),
    ``q_sharp`` log10 variance of the Laplacian inside the box."""
    import cv2
    df = df.copy()
    for c in ("q_bright", "q_contrast", "q_sharp"):
        df[c] = np.nan
    sel = (df["query"] == whale_query) & (df.score >= QUALITY_MIN_SCORE)
    if not sel.any():
        return df
    h0, w0 = image.shape[:2]
    sc = 960.0 / w0
    g = cv2.cvtColor(cv2.resize(image, (960, int(round(h0 * sc))), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    H, W = g.shape
    for i in df.index[sel]:
        r = df.loc[i]
        x0, y0 = int(max(r.x0 * sc, 0)), int(max(r.y0 * sc, 0))
        x1, y1 = int(min(r.x1 * sc, W)), int(min(r.y1 * sc, H))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        inside = g[y0:y1, x0:x1]
        mx, my = max(int(0.25 * (x1 - x0)), 4), max(int(0.25 * (y1 - y0)), 4)
        X0, Y0, X1, Y1 = max(x0 - mx, 0), max(y0 - my, 0), min(x1 + mx, W), min(y1 + my, H)
        outer = g[Y0:Y1, X0:X1]
        ring_sum = outer.sum() - inside.sum()
        ring_n = outer.size - inside.size
        df.loc[i, "q_bright"] = float(inside.mean())
        df.loc[i, "q_contrast"] = float(abs(inside.mean() - ring_sum / ring_n)) if ring_n > 0 else np.nan
        df.loc[i, "q_sharp"] = float(np.log10(np.var(cv2.Laplacian(inside, cv2.CV_32F)) + 1e-8))
    return df


FRAME_COLUMNS = ["whale", "small", "fish", "cover", "top_score", "whale_px", "bright", "contrast", "sharp", "bubbles"]


def frame_scores(boxes: pd.DataFrame, frames: np.ndarray, cfg: ScanConfig, width: int, height: int) -> pd.DataFrame:
    """Per processed frame (0 / NaN when there is no box):

    - ``whale``: best "whale" box that survives the filters (not in the clock; covering <= ``whole_frame`` of the
      frame unless its score is >= ``near_whale_score``; with ``fish_veto`` its whale score beats its fish score)
      and is large (longest side >= ``min_whale_px`` in a 1920 px frame); ``small``: the same for the small boxes;
    - ``fish``: best fish-prompt box (clock and whole-frame filters; information only);
    - for the largest large whale box with score >= threshold: ``cover`` (share of the frame), ``top_score``,
      ``whale_px`` (longest side, 1920 px frame), ``bright`` / ``contrast`` / ``sharp`` (:func:`add_box_quality`);
    - ``bubbles``: share of the frame of the largest "air bubbles" box >= 0.3."""
    from . import detect
    out = pd.DataFrame({"frame": np.asarray(frames, np.int64)})
    for c in FRAME_COLUMNS:
        out[c] = 0.0 if c in ("whale", "small", "fish", "bubbles") else np.nan
    if len(boxes) == 0:
        return out
    b = boxes.copy()
    area = ((b.x1 - b.x0).clip(lower=0) * (b.y1 - b.y0).clip(lower=0)) / float(width * height)
    b["area"] = area
    b["px"] = np.maximum(b.x1 - b.x0, b.y1 - b.y0) * (1920.0 / width)
    clock = detect.in_clock(b, width, height)
    whole = area > cfg.whole_frame
    w = b[(b["query"] == cfg.whale_query) & ~clock]
    w_whole = whole[w.index]
    if cfg.near_whale_score is not None:
        w = w[~w_whole | (w.score >= cfg.near_whale_score)]
    else:
        w = w[~w_whole]
    if cfg.fish_veto:
        cols = [detect.patch_score_column(q) for q in cfg.fish_queries]
        cols = [c for c in cols if c in w.columns]
        if cols:
            w = w[w.score > w[cols].max(axis=1)]
    large = w[w.px >= cfg.min_whale_px]
    small = w[w.px < cfg.min_whale_px]
    f = b[b["query"].isin(cfg.fish_queries) & ~clock & ~whole]
    bub = b[(b["query"] == cfg.bubble_query) & (b.score >= 0.3)]
    out["whale"] = out.frame.map(large.groupby("frame").score.max()).fillna(0.0).astype(float)
    out["small"] = out.frame.map(small.groupby("frame").score.max()).fillna(0.0).astype(float)
    out["fish"] = out.frame.map(f.groupby("frame").score.max()).fillna(0.0).astype(float)
    out["bubbles"] = out.frame.map(bub.groupby("frame").area.max()).fillna(0.0).astype(float)
    top = large[large.score >= cfg.threshold].sort_values("area").groupby("frame").tail(1).set_index("frame")
    for c, src in (("cover", "area"), ("top_score", "score"), ("whale_px", "px"), ("bright", "q_bright"),
                   ("contrast", "q_contrast"), ("sharp", "q_sharp")):
        if src in top.columns:
            out[c] = out.frame.map(top[src]).astype(float)
    return out


# Picture-quality scales of the hq_whale tier: part = 0 at ``lo``, 1 at ``hi`` (clipped). Set from the dev clip
# (whale boxes >= 0.25, 10th / 90th percentiles; see experiments/06_clips_e2e/hq_scales.py), cover / score / depth
# by hand. A first guess for the user to judge on the scan_hq_* sheets, not fitted to any truth.
# dev clip, 638 whale boxes on 164 frames: bright 0.23 / 0.40, contrast 0.003 / 0.057, sharp -4.58 / -2.67.
HQ_SCALES = {"cover": (0.02, 0.30), "score": (0.25, 0.70), "bright": (0.23, 0.40), "contrast": (0.003, 0.057),
             "sharp": (-4.58, -2.67), "bubbles": (0.5, 0.0), "depth": (30.0, 0.0)}


def hq_parts(df: pd.DataFrame, scales: dict = HQ_SCALES) -> pd.DataFrame:
    """Quality parts (0-1) from columns cover, top_score, bright, contrast, sharp, bubbles and depth_m (if present)
    and their mean ``hq_score`` (depth only where it is known)."""
    src = {"cover": "cover", "score": "top_score", "bright": "bright", "contrast": "contrast", "sharp": "sharp",
           "bubbles": "bubbles", "depth": "depth_m"}
    out = pd.DataFrame(index=df.index)
    for k, c in src.items():
        if c not in df.columns:
            continue
        lo, hi = scales[k]
        out[f"hq_{k}"] = ((df[c].astype(float) - lo) / (hi - lo)).clip(0, 1)
    out["hq_score"] = out.mean(axis=1, skipna=True)
    out.loc[df["cover"].isna(), "hq_score"] = np.nan
    return out


# ------------------------------------------------------------------------------------------- windows

def window_grid(duration: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.ceil(duration - 1e-6))
    k = np.arange(n)
    return k.astype(float), np.minimum(k + 1.0, duration)


def window_frames(tab, t0: np.ndarray, t1: np.ndarray, offsets=(0.25, 0.75)) -> np.ndarray:
    """Frame index per window and offset (-1 when the time is past the window end): shape (n windows, n offsets)."""
    out = np.full((len(t0), len(offsets)), -1, np.int64)
    for j, o in enumerate(offsets):
        tt = t0 + o
        ok = tt < t1
        idx = np.searchsorted(tab.times, tt)
        idx = np.clip(idx, 1, len(tab.times) - 1)
        left = tab.times[idx - 1]
        right = tab.times[idx]
        pick = np.where(np.abs(tt - left) <= np.abs(right - tt), idx - 1, idx)
        out[ok, j] = pick[ok]
    return out


def keep_intervals(whale: np.ndarray, t0: np.ndarray, t1: np.ndarray, pad: float, merge_gap: float,
                   duration: float) -> list[tuple[float, float]]:
    """Whale windows -> padded, merged keep intervals (merge when the gap after padding is below ``merge_gap``)."""
    from . import clips
    ivs = [clips.Interval(float(t0[i]), float(t1[i])) for i in np.nonzero(whale)[0]]
    # runs first (adjacent windows), then pad and merge
    merged = clips.pad_and_merge(ivs, pad=pad, merge_gap=max(merge_gap - 1e-6, 0.0), t_min=0.0, t_max=duration)
    return [(it.start, it.end) for it in merged]


def drop_reason(tag_state: str, water: str, scanned: bool, decoded: bool) -> str:
    if tag_state in TAG_REASON:
        return TAG_REASON[tag_state]
    if water in ("air", "surfaced"):
        return "surfaced"
    if water == "unknown":
        return "unreadable"
    if scanned and not decoded:
        return "unreadable"
    if scanned:
        return "underwater with no whale"
    return "unreadable"


def build_windows(rec: Recording, cfg: ScanConfig, sensors: SensorTimeline | None, vclock, clock_fit: dict | None,
                  water_pic: pd.DataFrame | None) -> pd.DataFrame:
    """The per-window table before detection: times, clock, tag state, water state, which windows to scan."""
    tab = video.frame_table(rec.path)
    duration = tab.duration
    t0, t1 = window_grid(duration)
    n = len(t0)
    w = pd.DataFrame({"recording": rec.stem, "k": np.arange(n), "t0": t0, "t1": t1})
    if clock_fit and clock_fit.get("ok"):
        w["clock"] = [clock_text(clock_fit["clock_at_pts0"] + clock_fit["rate"] * x) for x in t0]
    else:
        w["clock"] = ""
    if water_pic is not None and len(water_pic):
        wp = water_pic.set_index("k")
        w["water_picture"] = w.k.map(wp["state"]).fillna("unknown")
        w["p_air_picture"] = w.k.map(wp["score"])
    else:
        w["water_picture"] = "unknown"
        w["p_air_picture"] = np.nan
    if sensors is not None and vclock is not None:
        s0, s1 = vclock.to_sensor(t0), vclock.to_sensor(t1)
        w["sensor_t0"] = s0
        seg = sensors.segments
        ov = {st: _overlap(s0, s1, seg[seg.state == st].t0.to_numpy(), seg[seg.state == st].t1.to_numpy())
              for st in ("air", "flight", "floating", "whale")}
        cover = sum(ov.values())
        length = s1 - s0
        tag = np.array(["air", "flight", "floating", "whale"], object)[np.argmax(np.stack([ov[s] for s in
                                                                                         ("air", "flight", "floating",
                                                                                          "whale")]), 0)]
        tag = np.where(ov["whale"] > 0, "whale", tag)
        tag = np.where(cover < 0.5 * length, "unknown", tag)
        air = _overlap(s0, s1, sensors.air.t0.to_numpy(), sensors.air.t1.to_numpy())
        frac_air = np.where(ov["whale"] > 0, air / np.maximum(ov["whale"], 1e-9), np.nan).clip(0, 1)
        water = np.where(tag == "whale", np.where(frac_air >= 0.999, "surfaced",
                                                  np.where(frac_air <= 0.0, "underwater", "crossing")),
                         np.where(np.isin(tag, ["air", "flight"]), "air",
                                  np.where(tag == "floating", "waterline", "unknown")))
        src = np.where(tag == "whale", "depth", np.where(tag == "unknown", "picture", "tag state"))
        # no sensor coverage: the picture stands in
        unk = tag == "unknown"
        water = np.where(unk, w.water_picture.to_numpy(), water)
        w["tag_state"] = tag
        w["tag_source"] = np.where(unk, "none", "sensors")
        dm = np.interp((s0 + s1) / 2, sensors.depth_t, sensors.depth_m) if hasattr(sensors, "depth_t") else np.nan
        w["depth_m"] = np.where(tag == "whale", dm, np.nan)
        w["water"] = water
        w["water_source"] = src
        w["frac_air"] = frac_air
    else:
        w["sensor_t0"] = np.nan
        w["depth_m"] = np.nan
        w["tag_state"] = "unknown"
        w["tag_source"] = "none"
        w["water"] = w.water_picture
        w["water_source"] = "picture"
        w["frac_air"] = np.nan
    w.loc[(w.tag_state == "unknown") & (w.water == "air"), "water"] = "surfaced"
    if cfg.scan_all:
        w["scanned"] = True
    else:
        w["scanned"] = w.water.isin(["underwater", "crossing"]) & w.tag_state.isin(["whale", "unknown"])
    fr = window_frames(tab, t0, t1, cfg.frame_offsets)
    for j in range(fr.shape[1]):
        w[f"frame_{j}"] = fr[:, j]
    return w


def apply_detections(w: pd.DataFrame, fs: pd.DataFrame, processed: np.ndarray, cfg: ScanConfig,
                     duration: float) -> tuple[pd.DataFrame, list[tuple[float, float]]]:
    """Fill whale / fish scores, the preset rule, the quality tier, the kept flag and reasons. Returns the table and
    the keep intervals."""
    import warnings
    w = w.copy()
    score = fs.set_index("frame")
    proc = set(int(x) for x in processed)
    fcols = [c for c in w.columns if c.startswith("frame_")]
    per = {c: [] for c in FRAME_COLUMNS}
    dec = []
    for j, c in enumerate(fcols):
        idx = w[c].to_numpy()
        ok = np.array([(i >= 0) and (int(i) in proc) for i in idx]) & w.scanned.to_numpy()
        for col in FRAME_COLUMNS:
            v = pd.Series(idx).map(score[col]).to_numpy(dtype=float) if col in score.columns else np.full(len(idx), np.nan)
            if col in ("whale", "small", "fish", "bubbles"):
                v = np.nan_to_num(v, nan=0.0)
            per[col].append(np.where(ok, v, np.nan))
        w[f"whale_{j}"] = per["whale"][-1]
        dec.append(ok)
    S = np.stack(per["whale"], 1)
    SM = np.stack(per["small"], 1)
    decoded = np.stack(dec, 1).any(1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if cfg.preset == "strict":
            rule = np.nanmin(S, 1)
        elif cfg.preset == "sensitive":
            rule = np.nanmax(S, 1)
        else:
            raise ValueError(f"unknown preset {cfg.preset!r} (strict, sensitive)")
        fish = np.nanmax(np.stack(per["fish"], 1), 1)
        small = np.nanmax(SM, 1)
        cover = np.stack(per["cover"], 1)
        best = np.where(np.isnan(cover).all(1), 0, np.nanargmax(np.nan_to_num(cover, nan=-1), 1))
    rows = np.arange(len(w))
    w["whale_score"] = np.where(decoded, rule, np.nan)
    w["small_score"] = np.where(decoded, small, np.nan)
    w["fish_score"] = np.where(decoded, fish, np.nan)
    w["whale"] = decoded & (np.nan_to_num(rule, nan=-1) >= cfg.threshold)
    w["small"] = decoded & ~w.whale.to_numpy() & (np.nan_to_num(small, nan=-1) >= cfg.threshold)
    for col in ("cover", "top_score", "whale_px", "bright", "contrast", "sharp", "bubbles"):
        w[col] = np.stack(per[col], 1)[rows, best]
    for col in ("cover", "top_score", "whale_px", "bright", "contrast", "sharp"):
        w.loc[~w.whale, col] = np.nan
    parts = hq_parts(w)
    for c in parts.columns:
        w[c] = parts[c]
    w["hq_whale"] = w.whale & (w.cover.fillna(0) >= cfg.hq_min_cover) & (w.hq_score.fillna(0) >= cfg.hq_threshold)
    keep = keep_intervals(w.whale.to_numpy(), w.t0.to_numpy(), w.t1.to_numpy(), cfg.pad_s, cfg.merge_gap_s, duration)
    mid = (w.t0.to_numpy() + w.t1.to_numpy()) / 2
    kept = np.zeros(len(w), bool)
    for a, b in keep:
        kept |= (mid >= a) & (mid < b)
    w["kept"] = kept
    why = []
    for r, dk in zip(w.itertuples(), decoded):
        if r.kept:
            why.append("whale" if r.whale else "padding")
        elif r.small:
            why.append("small (fish or far whale)")
        else:
            why.append(drop_reason(r.tag_state, r.water, bool(r.scanned), bool(dk)))
    w["why"] = why
    return w, keep


def drop_stretches(w: pd.DataFrame, keep: list[tuple[float, float]]) -> list[dict]:
    """Runs of dropped windows with one reason, trimmed to the keep intervals."""
    out = []
    d = w[~w.kept]
    if d.empty:
        return out
    cur = None
    for r in d.itertuples():
        a, b = float(r.t0), float(r.t1)
        for ka, kb in keep:
            if ka < b and kb > a:
                if ka <= a:
                    a = max(a, kb)
                else:
                    b = min(b, ka)
        if b <= a:
            continue
        if cur and cur["reason"] == r.why and abs(cur["end"] - a) < 1e-6:
            cur["end"] = b
        else:
            if cur:
                out.append(cur)
            cur = {"start": a, "end": b, "reason": r.why}
    if cur:
        out.append(cur)
    return out


# ------------------------------------------------------------------------------------------- main

def _video_clock(rec: Recording, sensors: SensorTimeline, fit: dict, offset: float):
    from . import tagdata
    if fit.get("ok"):
        return tagdata.VideoClock(fit["clock_at_pts0"] - sensors.sod_of_first_sample() - offset, fit["rate"] - 1.0,
                                  f"burned clock - {offset:.2f} s")
    try:
        vc = tagdata.video_clock(sensors.deployment, rec.path.name)
        return vc
    except Exception:
        return None


def scan(target: str | os.PathLike, cfg: ScanConfig | None = None, cut: bool = False, mode: str = "copy",
         log=print, detector_factory: Callable | None = None, water_fn: Callable | None = None,
         out_root: Path | None = None, clip_root: Path | None = None) -> list[dict]:
    """Scan every deployment under ``target``; returns one summary dict per deployment (see module docstring)."""
    cfg = cfg or ScanConfig()
    if cfg.preset not in PRESETS:
        raise ValueError(f"preset must be one of {list(PRESETS)}")
    out_root = Path(out_root) if out_root else paths.cache_dir("scan")
    clip_root = Path(clip_root) if clip_root else paths.cache_dir("clips")
    paths.check_writable_target(out_root)
    summaries = []
    for dep in discover(target, cfg.min_age_s):
        if not dep.recordings:
            log(f"{dep.name}: no recordings ready ({len(dep.skipped)} skipped)")
            continue
        summaries.append(scan_deployment(dep, cfg, cut, mode, log, detector_factory, water_fn, out_root, clip_root))
    return summaries


def scan_deployment(dep: Deployment, cfg: ScanConfig, cut: bool, mode: str, log, detector_factory, water_fn,
                    out_root: Path, clip_root: Path) -> dict:
    t_start = time.time()
    base = out_root / dep.name
    cache = base / "cache"
    run_dir = base / cfg.label()
    (run_dir / "windows").mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    log(f"== {dep.name}: {len(dep.recordings)} recordings, sensor log: {dep.sensor_csv.name if dep.sensor_csv else 'none'}")
    timings: dict = {}
    sensors = None
    if dep.sensor_csv is not None:
        ts = time.time()
        sensors = SensorTimeline(dep.name, dep.sensor_csv, cfg, log)
        timings["sensors_s"] = time.time() - ts
        for nmsg in sensors.notes:
            log("  " + nmsg)
    # pass 1: clock and picture water state per recording
    fits, water, pic_events = {}, {}, {}
    for rec in dep.recordings:
        cf = cache / f"{rec.key}.clock.json"
        if cf.exists():
            fits[rec.key] = json.loads(cf.read_text())
        else:
            ts = time.time()
            fits[rec.key] = fit_burned_clock(rec.path, cfg.clock_probes)
            fits[rec.key]["seconds"] = time.time() - ts
            cf.write_text(json.dumps(fits[rec.key], indent=1))
            timings["clock_s"] = timings.get("clock_s", 0.0) + fits[rec.key]["seconds"]
        if sensors is None or cfg.picture_check:
            ts = time.time()
            water[rec.key], pic_events[rec.key] = picture_water(rec, cache, cfg, log, water_fn, timings)
            timings["water_picture_wall_s"] = timings.get("water_picture_wall_s", 0.0) + time.time() - ts
    # offset check
    offset = cfg.clock_offset_s
    check = None
    if sensors is not None:
        nominal = 2.03 if offset == "auto" else float(offset)
        pe, de = [], []
        dev = sensors.depth_events(cfg)
        for rec in dep.recordings:
            vc = _video_clock(rec, sensors, fits[rec.key], nominal)
            if vc is None or rec.key not in pic_events:
                continue
            dur = video.frame_table(rec.path).duration
            a, b = float(vc.to_sensor(0.0)), float(vc.to_sensor(dur))
            ev = pic_events[rec.key]
            if len(ev):
                # events within 1 s of a video edge are where the picture model starts, not real exits / entries
                ev = ev[(ev.t > 1.0) & (ev.t < dur - 1.0)]
                pe.append(pd.DataFrame({"t": vc.to_sensor(ev.t.to_numpy()), "kind": ev.kind.to_numpy(),
                                        "recording": rec.stem}))
            de.append(dev[(dev.t > a + 1.0) & (dev.t < b - 1.0)])
        pe = pd.concat(pe, ignore_index=True) if pe else pd.DataFrame(columns=["t", "kind"])
        de = pd.concat(de, ignore_index=True) if de else pd.DataFrame(columns=["t", "kind"])
        check = offset_check(pe, de)
        check["offset_in_use_s"] = nominal
        if "median_picture_minus_depth_s" in check:
            check["implied_offset_s"] = nominal + check["median_picture_minus_depth_s"]
        (base / "offset_check.json").write_text(json.dumps(check, indent=1))
        msg = (f"  offset check: burned clock - sensor time in use {nominal:.2f} s; picture minus depth median "
               f"{check.get('median_picture_minus_depth_s', float('nan')):+.2f} s over {check['n_pairs']} events "
               f"-> {check['verdict']}")
        log(msg)
        if offset == "auto":
            offset = check["implied_offset_s"] if check.get("verdict") in ("ok", "OFFSET LOOKS WRONG") else nominal
            log(f"  using offset {offset:.2f} s")
        elif check["verdict"] == "OFFSET LOOKS WRONG":
            log(f"  WARNING: the picture says the offset is {check['implied_offset_s']:.2f} s; re-run with "
                f"--clock-offset {check['implied_offset_s']:.2f} (or auto) if the sheets agree")
        offset = float(offset)
    # pass 2: windows, detector, keep rule
    videos_json, drop_rows, rec_sum = [], [], []
    for rec in dep.recordings:
        tab = video.frame_table(rec.path)
        vc = _video_clock(rec, sensors, fits[rec.key], offset) if sensors is not None else None
        w = build_windows(rec, cfg, sensors, vc, fits.get(rec.key), water.get(rec.key))
        fcols = [c for c in w.columns if c.startswith("frame_")]
        want = np.unique(w.loc[w.scanned, fcols].to_numpy().ravel())
        want = want[want >= 0]
        log(f"  {rec.path.name}: {len(w)} windows, {int(w.scanned.sum())} to scan ({len(want)} frames)")
        processed = run_detector(rec, want, cache, cfg, log, detector_factory, timings) if len(want) else np.zeros(0)
        boxes, _ = load_boxes(cache, rec)
        size = detector_frame_size(cache, rec)
        fs = frame_scores(boxes, processed, cfg, size[0], size[1])
        w, keep = apply_detections(w, fs, processed, cfg, tab.duration)
        w.to_csv(run_dir / "windows" / f"{rec.key}.csv", index=False)
        drops = drop_stretches(w, keep)
        keep_rows = []
        for a, b in keep:
            m = (w.t0 < b) & (w.t1 > a)
            n_hq = int((m & w.hq_whale).sum())
            keep_rows.append({"start": round(a, 4), "end": round(b, 4), "reason": "whale in view",
                              "tier": "hq_whale" if n_hq else "whale", "hq_windows": n_hq,
                              "max_hq_score": round(float(w.loc[m, "hq_score"].max()), 3)
                              if w.loc[m, "hq_score"].notna().any() else None,
                              "whale_windows": int((m & w.whale).sum()),
                              "max_whale_score": round(float(w.loc[m, "whale_score"].max()), 3),
                              "max_fish_score": round(float(w.loc[m, "fish_score"].max()), 3)
                              if w.loc[m, "fish_score"].notna().any() else None})
        hq = keep_intervals(w.hq_whale.to_numpy(), w.t0.to_numpy(), w.t1.to_numpy(), cfg.pad_s, cfg.merge_gap_s,
                            tab.duration)
        videos_json.append({"source": str(rec.path), "duration": round(tab.duration, 4),
                            "keep": keep_rows, "hq": [{"start": round(a, 4), "end": round(b, 4)} for a, b in hq], "drop": [{k: (round(v, 4) if isinstance(v, float) else v)
                                                         for k, v in dr.items()} for dr in drops]})
        for dr in drops:
            drop_rows.append({"recording": rec.stem, **dr, "seconds": dr["end"] - dr["start"]})
        by = {r: 0.0 for r in REASONS}
        for dr in drops:
            by[dr["reason"]] += dr["end"] - dr["start"]
        kept_s = sum(b - a for a, b in keep)
        rec_sum.append({"recording": rec.stem, "form": rec.form, "seconds": tab.duration, "kept_s": kept_s,
                        "hq_windows": int(w.hq_whale.sum()), "hq_stretch_s": sum(b - a for a, b in hq),
                        "n_hq_clips": sum(1 for r in keep_rows if r["tier"] == "hq_whale"),
                        "n_clips": len(keep), "dropped_s": by, "windows": int(len(w)),
                        "windows_scanned": int(w.scanned.sum()), "whale_windows": int(w.whale.sum()),
                        "clock": {k: fits[rec.key].get(k) for k in ("ok", "kind", "rms_s", "share_read")},
                        "video_clock": vc.source if vc is not None else None})
        log(f"    kept {kept_s / 60:.1f} of {tab.duration / 60:.1f} min in {len(keep)} stretches")
    release_gpu()
    iv = {"format": FORMAT, "created": _dt.datetime.now().isoformat(timespec="seconds"),
          "tool": "uwid scan", "config": {**asdict(cfg), "clock_offset_used_s": offset if sensors else None},
          "videos": videos_json}
    (run_dir / "intervals.json").write_text(json.dumps(iv, indent=1, default=str))
    pd.DataFrame(drop_rows, columns=["recording", "start", "end", "reason", "seconds"]).to_csv(
        run_dir / "dropped.csv", index=False)
    total = sum(r["seconds"] for r in rec_sum)
    summary = {"deployment": dep.name, "label": cfg.label(), "sensor_log": str(dep.sensor_csv) if dep.sensor_csv
               else None, "skipped": dep.skipped, "recordings": rec_sum, "seconds": total,
               "kept_s": sum(r["kept_s"] for r in rec_sum),
               "hq_windows": sum(r["hq_windows"] for r in rec_sum),
               "hq_stretch_s": sum(r["hq_stretch_s"] for r in rec_sum),
               "dropped_s": {k: sum(r["dropped_s"][k] for r in rec_sum) for k in REASONS},
               "offset_check": {k: v for k, v in (check or {}).items() if k != "pairs"} or None,
               "timings_this_run": timings, "wall_s_this_run": time.time() - t_start,
               "outputs": {"windows": str(run_dir / "windows"), "intervals": str(run_dir / "intervals.json"),
                           "dropped": str(run_dir / "dropped.csv")}}
    if cut:
        from . import clips
        cdir = clip_root / dep.name
        cdir.mkdir(parents=True, exist_ok=True)
        for f in ("clips_index.csv", "dropped.csv"):
            if (cdir / f).exists():
                (cdir / f).unlink()      # rewritten in full; existing clips are kept and not re-cut
        tc = time.time()
        res = clips.run(run_dir / "intervals.json", out_dir=cdir, mode=mode, pad=0.0, merge_gap=0.0, log=log)
        summary["clips"] = {"n": len(res["clips"]), "out_dir": str(cdir), "seconds": time.time() - tc,
                            "bytes": int(sum(Path(r["clip_file"]).stat().st_size for r in res["clips"]
                                             if Path(r["clip_file"]).exists()))}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with open(base / "run_history.jsonl", "a") as fh:     # every run, with its timings (summary.json is replaced)
        fh.write(json.dumps({"finished": _dt.datetime.now().isoformat(timespec="seconds"), "label": cfg.label(),
                             "kept_s": summary["kept_s"], "seconds": total, "timings": timings,
                             "wall_s": summary["wall_s_this_run"]}, default=str) + "\n")
    log(f"  {dep.name}: kept {summary['kept_s'] / 60:.1f} of {total / 60:.1f} min; dropped "
        + ", ".join(f"{k} {v / 60:.1f}" for k, v in summary["dropped_s"].items() if v > 0) + f" -> {run_dir}")
    return summary
