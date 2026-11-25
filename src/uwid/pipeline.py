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
3. **Keep rule**, preset ``strict`` (both frames have a large "whale" box >= 0.25) or ``sensitive`` (either frame
   does). A box shorter than ``min_whale_px`` (300 px of a 1920 px frame) is ``small`` (fish or far whale, user
   2026-11-20) and keeps nothing. A box over 60 % of the frame counts when its score is >= 0.6 and, with a sensor
   log, the tag is >= 1 m deep (shallower it is the tagged whale's back mirrored in the surface). Whale windows are
   padded by 2 s on each side, then stretches less than 5 s apart (after padding) are merged. Nothing is dropped for
   being short. Every other window gets one reason from :data:`REASONS`. Each window and each kept stretch also gets
   the quality tier ``hq_whale`` (:func:`hq_parts`: share of the frame covered by the whale box, detector score,
   brightness, contrast and sharpness in the box, no bubbles, shallow depth; first guess, for the user to judge).
4. Writes, under ``<cache>/scan/<deployment or video name>/<run label>/``: ``windows/<stem>.csv`` (one row per
   second), ``intervals.json`` (format ``uwid-intervals/1``), ``dropped.csv`` (stretches with reasons),
   ``summary.json``; ``offset_check.json`` next to it. With ``cut=True`` the keep intervals are cut by
   :mod:`uwid.clips` into ``<cache>/clips/<name>/``.

Re-running reuses the clock fits, picture water states and detector boxes cached under ``<name>/cache/`` (the
detector cache keeps every box with the scores of all prompts at its patch, so the options below re-score without
running the model again). Source footage is only read.

Options, each reported once in experiments/06_clips_e2e as NOT a clean test: ``fish_veto`` (off by default) counts
a "whale" box only when its whale score beats the best fish-prompt score of the same box; ``near_whale_score``
(default 0.6, None = WP5a's plain 60 % filter). The fish score of each window is written as information and never
changes the keep decision.
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

# drop reasons (WP9, user 2026-11-22: every window with anything detected is kept, so "small (fish or far whale)"
# is no longer a drop reason, and "underwater with no whale" became "nothing detected")
REASONS = ("tag in air", "drone flight", "tag floating", "surfaced", "nothing detected", "unreadable")
KEPT_WHY = ("whale", "small", "fish", "padding")
PRESETS = {"strict": "both frames of the window have something detected (a whale or fish box >= threshold)",
           "sensitive": "either frame of the window has something detected"}
FORMAT = "uwid-intervals/1"
# burned-in clock minus sensor-log time per tag serial, measured: CC-11-96 2.03 s (WP1, drone impact frames, and 156
# surfacing events); CC-09-85 is not listed on purpose: it is the test of the estimate for a new tag (-0.09 s, WP9)
KNOWN_CLOCK_OFFSETS = {"CC-11-96": 2.03}


class ClockOffsetError(RuntimeError):
    """The clock-to-sensor offset of a new tag could not be estimated and none was given."""
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
    near_whale_min_depth_m: float = 1.0  # ... and (with a sensor log) the tag is at least this deep: just under the
    #                                      surface the tagged whale's own back, mirrored in the surface, fills the frame
    min_whale_px: float = 300.0          # whale boxes shorter than this (longest side, 1920 px frame) are 'small'
    hq_threshold: float = 0.6            # hq_whale: quality score at least this (first guess) ...
    hq_min_cover: float = 0.10           # ... and the whale box covers at least this share of the frame
    ok_min_px: float = 400.0             # tier 'ok': a whale window whose whale box is at least this long (1920 px
    #                                      frame), else 'low'; chosen on the dev clip only (WP9, tables.md)
    sil_min_px: float = 576.0            # silhouette: whale box at least this long (0.3 of a 1920 px frame) ...
    sil_max_bright: float = 0.40         # ... mean grey inside at most this ...
    sil_min_lift: float = 0.02           # ... and the ring around it at least this much brighter (dev clip, WP9;
    #                                      weak: 14 of 29 dev silhouettes, 44 of 116 others; see tables.md)
    min_tier: str | None = None          # with cut: clips whose best tier is below this are listed, not cut
    #                                      (default: every kept stretch is cut; user 2026-11-22)
    fish_veto: bool = False              # option: whale box only if its whale score beats its fish score
    back_filter: bool = True             # a "whale" box across the full width resting on the bottom edge and
    #                                      starting below 30 % of the height is the tagged whale's own back (CC-09-85:
    #                                      the back fills the lower half; WP9, found on that test deployment)
    fish_threshold: float = 0.5          # a fish-prompt box at least this good is activity (WP5a: 0.5 set from the
    #                                      dev clip's false-alarm rate)
    keep_activity: bool = True           # keep small whales and fish too (user 2026-11-22); False = whales only
    frame_offsets: tuple = (0.25, 0.75)  # frames per window, seconds after the window start
    pad_s: float = 2.0                   # buffer before / after a clip (user 2026-11-23: 1-2 s)
    merge_gap_s: float = 5.0             # hq spans: closer than this (after no padding) are joined
    guard_s: float = 30.0                # a clip ends only after this long without any active window (user
    #                                      2026-11-23); it never crosses tag in air / flight or a long lens-out
    max_breath_s: float = 20.0           # lens out of the water longer than this ends a clip (a breath does not)
    boxclf: bool = True                  # veto large "whale" boxes with uwid.boxclf (WP10): whale / fish / other
    boxclf_weights: str | None = None    # default uwid.boxclf.DEFAULT_WEIGHTS
    window_classes: bool = True          # sand / dark per scanned window (uwid.patchseg, WP10): labels only;
    #                                      seafloor withdrawn 2026-11-23 (the class sat on backs and haze)
    run_budget_s: float | None = None    # stop the GPU stages after this many seconds of the run (resumable)
    clock_offset_s: float | str = "tag"  # burned-in clock minus sensor time (s); "tag" = the measured value of a
    #                                      known tag (KNOWN_CLOCK_OFFSETS), else estimated like "auto"; "auto" = the
    #                                      offset check's estimate, refused when too few events match
    min_offset_events: int = 10          # "auto" needs this many matched exits + entries (at least 3 of each)
    lens_out_lag_s: float = 0.3          # lens out this long after depth says surfaced (tag CC-11-96, by eye)
    lens_in_lead_s: float = 0.4          # lens under again this long before depth says submerged
    scan_all: bool = False               # run the detector on every window (cost comparison only)
    scan_floating: bool = True           # run the detector on 'tag floating' time where the picture shows the lens
    #                                      under water (whales come to a floating tag; seen on 2025-02-03 00012)
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
        if not self.scan_floating:
            s += "+nofloat"
        if not self.keep_activity:
            s += "+whaleonly"
        if not self.back_filter:
            s += "+noback"
        if not self.boxclf:
            s += "+noclf"
        if self.guard_s != 30.0:
            s += f"+guard{self.guard_s:g}"
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
        glyphs = timecode.glyphs_for(path)
    except FileNotFoundError:
        return {"ok": False, "why": "no glyphs"}
    if glyphs is None:
        return {"ok": False, "why": "no clock layout found or learned"}
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
    out = {"n_frames_tried": n_try, "n_read": n_read, "share_read": n_read / max(n_try, 1), "date": date,
           "clock_format": glyphs.layout.fmt}
    # a clock in tenths is timed by its changes; one in milliseconds by its plain readings
    tenths = glyphs.layout.resolution_s >= 0.1
    use_x, use_y, kind = (xs, ys, "tenth changes") if tenths and len(xs) >= 5 else (plain_x, plain_y, "plain readings")
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
        # only surfacings where the lens is out at all (the same rule as the air intervals); brushes of the
        # surface shorter than the lags (wave wash, 0.1-0.9 s, many on CC-09-85) have no picture event
        s = s[(s.end_t - cfg.lens_in_lead_s) - (s.start_t + cfg.lens_out_lag_s) > 0.2]
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
    has_rec = "recording" in depth_events.columns
    for kind in ("exit", "entry"):
        p = pic_events[pic_events.kind == kind].t.to_numpy()
        dk = depth_events[depth_events.kind == kind]
        recs = dk.recording.to_numpy() if has_rec else [""] * len(dk)
        for t, rec in zip(dk.t.to_numpy(), recs):
            if len(p) == 0:
                continue
            dd = p - t
            i = int(np.argmin(np.abs(dd - lag)))
            if abs(dd[i] - lag) <= pair_s:
                pairs.append((kind, float(t), float(dd[i]), str(rec)))
    diffs = np.array([x[2] for x in pairs])
    res.update(best_shift_s=lag, n_matched_at_shift=best[0], n_pairs=len(pairs))
    if len(pairs):
        res["median_picture_minus_depth_s"] = float(np.median(diffs))
        ke = {k: np.array([x[2] for x in pairs if x[0] == k]) for k in ("exit", "entry")}
        # the midpoint of the exit and entry medians: does not depend on the mix of kinds (a tag whose lens lags
        # differ from the assumed ones shifts exits and entries in opposite directions)
        if len(ke["exit"]) >= 3 and len(ke["entry"]) >= 3:
            res["midpoint_s"] = float((np.median(ke["exit"]) + np.median(ke["entry"])) / 2)
            res["half_difference_s"] = float((np.median(ke["entry"]) - np.median(ke["exit"])) / 2)
        res["mad_s"] = float(np.median(np.abs(diffs - np.median(diffs))))
        for kind in ("exit", "entry"):
            k = np.array([x[2] for x in pairs if x[0] == kind])
            res[f"median_{kind}_s"] = float(np.median(k)) if len(k) else None
            res[f"n_{kind}"] = int(len(k))
    if len(pairs) < min_pairs:
        res["verdict"] = f"too few matched events ({len(pairs)}) to check the offset"
    elif abs(res.get("midpoint_s", res["median_picture_minus_depth_s"])) <= 0.3:
        res["verdict"] = "ok"
    else:
        res["verdict"] = "OFFSET LOOKS WRONG"
    if has_rec:
        per = {}
        for rec, g in depth_events.groupby("recording"):
            k = np.array([x[2] for x in pairs if x[3] == rec])
            kx = np.array([x[2] for x in pairs if x[3] == rec and x[0] == "exit"])
            kn = np.array([x[2] for x in pairs if x[3] == rec and x[0] == "entry"])
            per[str(rec)] = {"n_depth_events": int(len(g)), "n_pairs": int(len(k)), "n_exit": int(len(kx)),
                             "n_entry": int(len(kn)),
                             "median_s": float(np.median(k)) if len(k) else None,
                             "mad_s": float(np.median(np.abs(k - np.median(k)))) if len(k) else None,
                             "median_exit_s": float(np.median(kx)) if len(kx) else None,
                             "median_entry_s": float(np.median(kn)) if len(kn) else None,
                             "midpoint_s": float((np.median(kx) + np.median(kn)) / 2) if len(kx) and len(kn)
                             else None}
        res["per_recording"] = per
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
                runner = detect.OwlRunner(queries, patch_scores=True, batch=cfg.detector_batch, gpu_preprocess=True)
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


# ------------------------------------------------------------------------------------------- picture classes (WP11b)

def _pclass_dir(cache: Path, rec: Recording) -> Path:
    return cache / "pclass" / rec.key


def _box_key(df: pd.DataFrame) -> pd.Series:
    r = lambda c: [round(float(x), 1) for x in df[c]]
    return pd.Series(list(zip(df.frame.astype(int), r("x0"), r("y0"), r("x1"), r("y1"))), index=df.index)


def load_picture_classes(cache: Path, rec: Recording) -> tuple[pd.DataFrame, pd.DataFrame, set, set]:
    """Cached box classes (frame, x0..y1, boxclf_class, p_whale, p_fish, p_other), frame classes (frame, grey_mean,
    share_<class>, sand_p), and the sets of frames done for each."""
    d = _pclass_dir(cache, rec)
    bx = [pd.read_parquet(f) for f in sorted(d.glob("box_*.parquet"))] if d.is_dir() else []
    fr = [pd.read_parquet(f) for f in sorted(d.glob("frm_*.parquet"))] if d.is_dir() else []
    bdone = set(int(x) for f in sorted(d.glob("box_*.frames.npy")) for x in np.load(f)) if d.is_dir() else set()
    boxes = pd.concat([x for x in bx if len(x)], ignore_index=True) if any(len(x) for x in bx) else pd.DataFrame(
        columns=["frame", "x0", "y0", "x1", "y1", "boxclf_class", "p_whale", "p_fish", "p_other"])
    frames = pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=["frame", "grey_mean", "sand_p"])
    return boxes, frames, bdone, set(int(x) for x in frames.frame) if len(frames) else set()


def boxclf_frames(boxes: pd.DataFrame, cfg: ScanConfig, width: int, height: int) -> np.ndarray:
    """Frames that hold a rule-A "whale" box the veto concerns (score >= threshold, >= min_whale_px, not clock)."""
    from . import detect
    if len(boxes) == 0:
        return np.zeros(0, np.int64)
    b = boxes[(boxes["query"] == cfg.whale_query) & (boxes.score >= cfg.threshold)]
    px = np.maximum(b.x1 - b.x0, b.y1 - b.y0) * (1920.0 / width)
    b = b[(px >= cfg.min_whale_px) & ~detect.in_clock(b, width, height)]
    return np.unique(b.frame.to_numpy(np.int64))


def boxclf_weights(cfg: ScanConfig) -> Path | None:
    from . import boxclf
    w = Path(cfg.boxclf_weights) if cfg.boxclf_weights else boxclf.DEFAULT_WEIGHTS
    return w if w.exists() else None


def picture_classes(rec: Recording, cache: Path, cfg: ScanConfig, win_frames: np.ndarray, box_frames: np.ndarray,
                    boxes: pd.DataFrame, log=print, deadline: float | None = None, timings: dict | None = None) -> bool:
    """WP10's models on frames the scan already chose: the box classifier on ``box_frames`` (with the detector's
    rows of those frames, in detector pixels) and the seafloor / sand / dark frame classes on ``win_frames`` (the
    scanned windows' frames, at 1280 px). Each frame is decoded once (in a child process that cannot hang the parent:
    ``uwid.patchseg._iter_decoded``); results are cached per frame under ``<cache>/pclass/<recording>/``, so a
    re-score costs nothing. Stops at ``deadline`` (time.time()) and returns False when frames are left."""
    import cv2
    from . import patchseg
    _, _, bdone, fdone = load_picture_classes(cache, rec)
    wts = boxclf_weights(cfg) if cfg.boxclf else None
    todo_b = sorted(set(int(x) for x in box_frames) - bdone) if wts is not None else []
    have_sf = patchseg.SEAFLOOR_CLF.exists()
    todo_w = sorted(set(int(x) for x in win_frames) - fdone) if (cfg.window_classes and have_sf) else []
    if not todo_b and not todo_w:
        return True
    if deadline is not None and time.time() > deadline:
        return False
    d = _pclass_dir(cache, rec)
    d.mkdir(parents=True, exist_ok=True)
    dsize = detector_frame_size(cache, rec)
    vw, vh = video_size(rec.path)
    size = tuple(dsize) if (vw, vh) != tuple(dsize) else None
    want_b, want_w = set(todo_b), set(todo_w)
    log(f"  picture classes: {len(todo_b)} frames for the box classifier, {len(todo_w)} for seafloor / sand / dark")
    with gpu_slot(cfg, log) as waited:
        import torch
        from . import boxclf
        t0 = time.time()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if _HELD.get("pmodel") is None:
            _HELD["pmodel"] = patchseg.load_backbone(dev, torch.float16 if dev == "cuda" else torch.float32)
        model = _HELD["pmodel"]
        head = sand = None
        if todo_w:
            head, _ = patchseg.load_classifier(patchseg.SEAFLOOR_CLF, dev)
            sand = patchseg.SandModel.load(patchseg.SAND_CLF) if patchseg.SAND_CLF.exists() else None
        brows, bframes, frows = [], [], []
        buf_i, buf_im = [], []
        budget = lock_time_left(cfg)
        stopped = False

        def flush_w():
            nonlocal buf_i, buf_im
            if not buf_im:
                return
            res = patchseg.frame_classes(model, buf_im, {"h": head}, 1280)
            for i, r in zip(buf_i, res):
                m = r["maps"]["h"]
                sh = np.bincount(m.ravel(), minlength=len(patchseg.CLASSES))[:len(patchseg.CLASSES)] / m.size
                row = {"frame": int(i), "grey_mean": r["grey_mean"]}
                row.update({f"share_{c}": float(sh[k]) for k, c in enumerate(patchseg.CLASSES)})
                row["sand_p"] = float(sand.proba(r["emb"][None])[0]) if sand is not None else np.nan
                frows.append(row)
            buf_i, buf_im = [], []

        def write():
            nonlocal brows, bframes, frows
            stamp = f"{int(time.time() * 1000):d}"
            if bframes:
                df = pd.DataFrame(brows, columns=["frame", "x0", "y0", "x1", "y1", "boxclf_class", "p_whale", "p_fish",
                                                  "p_other"])
                tmp = d / f"box_{stamp}.parquet.tmp"
                df.to_parquet(tmp, index=False)
                tmp.rename(d / f"box_{stamp}.parquet")
                np.save(d / f"box_{stamp}.frames.npy", np.asarray(bframes, np.int64))
            if frows:
                tmp = d / f"frm_{stamp}.parquet.tmp"
                pd.DataFrame(frows).to_parquet(tmp, index=False)
                tmp.rename(d / f"frm_{stamp}.parquet")
            brows, bframes, frows = [], [], []

        n = 0
        for fr in patchseg._iter_decoded(rec.path, sorted(want_b | want_w), size, log=log):
            if fr.index in want_b:
                fb = boxes[boxes.frame == fr.index]
                res = boxclf.classify(fb.drop(columns=["frame"]), fr.image, model=model, clf=str(wts))
                for ix, r in res.iterrows():
                    if r.boxclf_class:
                        b0 = fb.loc[ix]
                        brows.append([int(fr.index), round(float(b0.x0), 1), round(float(b0.y0), 1),
                                      round(float(b0.x1), 1), round(float(b0.y1), 1), r.boxclf_class,
                                      float(r.p_whale), float(r.p_fish), float(r.p_other)])
                bframes.append(int(fr.index))
            if fr.index in want_w:
                h0, w0 = fr.image.shape[:2]
                buf_im.append(cv2.resize(fr.image, (1280, int(round(h0 * 1280 / w0))), interpolation=cv2.INTER_AREA))
                buf_i.append(fr.index)
                if len(buf_im) >= 8:
                    flush_w()
            n += 1
            if n % 200 == 0:
                flush_w()
                write()
            if time.time() - t0 > budget or (deadline is not None and time.time() > deadline):
                stopped = True
                break
        flush_w()
        write()
        held = time.time() - t0
    if timings is not None:
        timings["picture_classes_gpu_s"] = timings.get("picture_classes_gpu_s", 0.0) + held
        timings["picture_classes_frames"] = timings.get("picture_classes_frames", 0) + n
    log(f"  picture classes: {n} frames in {held:.0f} s (waited {waited:.0f} s){' - stopped, re-run' if stopped else ''}")
    if stopped:
        release_gpu()
    return not stopped


def attach_box_classes(boxes: pd.DataFrame, cls: pd.DataFrame) -> pd.DataFrame:
    """Add ``boxclf_class`` (and p_whale / p_fish / p_other) to detector rows by frame and box coordinates."""
    b = boxes.copy()
    if len(b) == 0 or len(cls) == 0:
        b["boxclf_class"] = ""
        return b
    dd = cls.drop_duplicates(["frame", "x0", "y0", "x1", "y1"])
    keys = list(_box_key(dd))
    k = list(_box_key(b))
    for c in ("boxclf_class", "p_whale", "p_fish", "p_other"):
        d = dict(zip(keys, dd[c].tolist()))      # a plain dict: tuple keys in a pandas index become a MultiIndex
        b[c] = [d.get(x) for x in k]
    b["boxclf_class"] = b["boxclf_class"].fillna("")
    return b


def fill_window_classes(w: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """``sand`` and ``dark`` per window from its frames' cached classes (uwid.patchseg.window_decisions' rules:
    every frame's mean grey < 20 = dark, mean p(sand) >= 0.5 and not dark = sand). ``seafloor`` stays "" (WP10's class
    withdrawn 2026-11-23, see below). Windows without classified frames stay "" (not measured). Labels only: no keep
    decision uses them; sand and dark windows are left out of the hq spans."""
    from . import patchseg
    w = w.copy()
    for c in ("seafloor", "sand", "dark"):
        w[c] = ""
    w["seafloor_share"] = np.nan
    if frames is None or len(frames) == 0:
        return w
    r = patchseg.WINDOW_RULES
    F = frames.drop_duplicates("frame").set_index("frame")
    fcols = [c for c in w.columns if c.startswith("frame_") and c[6:].isdigit()]
    sh = np.stack([w[c].map(F["share_seafloor"]).to_numpy(float) for c in fcols], 1)
    gr = np.stack([w[c].map(F["grey_mean"]).to_numpy(float) for c in fcols], 1)
    sp = np.stack([w[c].map(F["sand_p"]).to_numpy(float) for c in fcols], 1)
    have = np.isfinite(gr).any(1)
    import warnings
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            share = np.nanmean(sh, 1)
            dark = np.nanmax(gr, 1) < r["dark_grey"]
            sand = (np.nanmean(sp, 1) >= r["sand_p"]) & ~dark
            sea = (share >= r["seafloor_share"]) & ~dark
    w["seafloor_share"] = np.where(have, share, np.nan)

    def lab(mask, val):
        col = np.full(len(w), "", object)
        col[mask] = [bool(x) for x in val[mask]]
        return col
    # seafloor: NOT filled. The user (2026-11-23) found WP10's seafloor class on whale backs and haze: the labels it
    # learned from were wrong. The column stays empty (hook) until a model trained on the user's own examples exists.
    w["seafloor_share_model"] = w.pop("seafloor_share")      # WP10's raw share, kept for diagnosis only
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sand_known = np.isfinite(np.nanmean(sp, 1))
    w["sand"] = lab(have & sand_known, sand)
    w["dark"] = lab(have, dark)
    return w


def _truthy(col: pd.Series) -> np.ndarray:
    return col.astype(str).str.lower().isin(["true", "1"]).to_numpy()


QUALITY_MIN_SCORE = 0.15   # whale boxes at least this good get the picture measures


def add_box_quality(df: pd.DataFrame, image: np.ndarray, whale_query: str = "whale") -> pd.DataFrame:
    """Picture measures inside each whale box (score >= QUALITY_MIN_SCORE), on a 960 px wide grey copy:
    ``q_bright`` mean grey (0-1), ``q_contrast`` |mean grey inside - mean grey of a ring around the box| (0-1),
    ``q_sharp`` log10 variance of the Laplacian inside the box."""
    import cv2
    df = df.copy()
    for c in ("q_bright", "q_contrast", "q_sharp", "q_ring"):
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
        df.loc[i, "q_ring"] = float(ring_sum / ring_n) if ring_n > 0 else np.nan
        df.loc[i, "q_sharp"] = float(np.log10(np.var(cv2.Laplacian(inside, cv2.CV_32F)) + 1e-8))
    return df


QUALITY_COLUMNS = ["cover", "top_score", "whale_px", "bright", "contrast", "sharp", "ring", "bx0", "by0", "bx1", "by1"]
COUNT_COLUMNS = ("whale", "whale_part", "small", "fish", "bubbles", "n_whale", "n_fish", "veto_fish", "veto_other")
FRAME_COLUMNS = (["whale", "whale_part", "small", "fish", "bubbles", "n_whale", "n_fish", "veto_fish", "veto_other"]
                 + QUALITY_COLUMNS
                 + [c + "_part" for c in QUALITY_COLUMNS])


def is_back_box(b: pd.DataFrame, width: int, height: int) -> pd.Series:
    """Boxes on the tagged whale's own back: across >= 95 % of the width, reaching the bottom edge (>= 98 % of the
    height) and starting below 30 % of the height. Seen on CC-09-85 (the back fills the lower half of the frame):
    125 of 432 whale windows, 122 of them hq, rested on such a box before this filter."""
    return (((b.x1 - b.x0) >= 0.95 * width) & (b.y1 >= 0.98 * height) & (b.y0 >= 0.30 * height))


def _count_objects(b: pd.DataFrame, iou: float = 0.5, max_boxes: int = 300) -> pd.Series:
    """Number of distinct boxes per frame (greedy NMS: boxes overlapping a kept one by IoU >= ``iou`` are the same;
    at most the ``max_boxes`` best per frame are considered)."""
    out = {}
    for f, g in b.groupby("frame"):
        g = g.nlargest(max_boxes, "score")
        bx = g[["x0", "y0", "x1", "y1"]].to_numpy(float)
        area = (bx[:, 2] - bx[:, 0]).clip(0) * (bx[:, 3] - bx[:, 1]).clip(0)
        alive = np.ones(len(bx), bool)
        n = 0
        for i in range(len(bx)):
            if not alive[i]:
                continue
            n += 1
            ix = (np.minimum(bx[i, 2], bx[:, 2]) - np.maximum(bx[i, 0], bx[:, 0])).clip(0)
            iy = (np.minimum(bx[i, 3], bx[:, 3]) - np.maximum(bx[i, 1], bx[:, 1])).clip(0)
            inter = ix * iy
            un = area[i] + area - inter
            alive &= ~((un > 0) & (inter / np.maximum(un, 1e-9) >= iou))
        out[f] = n
    return pd.Series(out, dtype=float)


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
        out[c] = 0.0 if c in COUNT_COLUMNS else np.nan
    if len(boxes) == 0:
        return out
    b = boxes.copy()
    area = ((b.x1 - b.x0).clip(lower=0) * (b.y1 - b.y0).clip(lower=0)) / float(width * height)
    b["area"] = area
    b["px"] = np.maximum(b.x1 - b.x0, b.y1 - b.y0) * (1920.0 / width)
    clock = detect.in_clock(b, width, height)
    whole = area > cfg.whole_frame
    back = is_back_box(b, width, height) if cfg.back_filter else pd.Series(False, index=b.index)
    w = b[(b["query"] == cfg.whale_query) & ~clock & ~back]
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
    # box classifier veto (WP10 / WP11b): a large "whale" box stays a whale only when the classifier says whale;
    # "fish" makes it a fish object (activity), "other" (blow, surface, mirrored back, sediment) drops it. Boxes the
    # classifier has not seen (no weights, not yet run) keep the rule's answer. Under min_whale_px the size rule rules.
    if cfg.boxclf and "boxclf_class" in large.columns:
        cls = large["boxclf_class"].fillna("").astype(str)
        vf, vo = large[cls == "fish"], large[cls == "other"]
        large = large[~cls.isin(["fish", "other"])]
        out["veto_fish"] = out.frame.map(vf[vf.score >= cfg.threshold].groupby("frame").score.max()).fillna(0.0)
        out["veto_other"] = out.frame.map(vo[vo.score >= cfg.threshold].groupby("frame").score.max()).fillna(0.0)
    f = b[b["query"].isin(cfg.fish_queries) & ~clock & ~whole]
    bub = b[(b["query"] == cfg.bubble_query) & (b.score >= 0.3)]
    out["whale"] = out.frame.map(large.groupby("frame").score.max()).fillna(0.0).astype(float)
    out["small"] = out.frame.map(small.groupby("frame").score.max()).fillna(0.0).astype(float)
    out["fish"] = out.frame.map(f.groupby("frame").score.max()).fillna(0.0).astype(float)
    out["bubbles"] = out.frame.map(bub.groupby("frame").area.max()).fillna(0.0).astype(float)
    out["n_whale"] = out.frame.map(_count_objects(large[large.score >= cfg.threshold])).fillna(0).astype(float)
    out["n_fish"] = out.frame.map(_count_objects(f[f.score >= cfg.fish_threshold])).fillna(0).astype(float)
    part = large[~whole[large.index]]
    out["whale_part"] = out.frame.map(part.groupby("frame").score.max()).fillna(0.0).astype(float)
    for suffix, boxes_ in (("", large), ("_part", part)):
        top = boxes_[boxes_.score >= cfg.threshold].sort_values("area").groupby("frame").tail(1).set_index("frame")
        for c, src in (("cover", "area"), ("top_score", "score"), ("whale_px", "px"), ("bright", "q_bright"),
                       ("ring", "q_ring"), ("bx0", "x0"), ("by0", "y0"), ("bx1", "x1"), ("by1", "y1"),
                       ("contrast", "q_contrast"), ("sharp", "q_sharp")):
            if src in top.columns:
                out[c + suffix] = out.frame.map(top[src]).astype(float)
    return out


def ring_brightness(image: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> float:
    """Mean grey (0-1) of the ring around a box (a quarter of the box size each side), on a 960 px wide grey copy, as
    in :func:`add_box_quality`. ``image`` is in the detector's pixels."""
    import cv2
    h0, w0 = image.shape[:2]
    sc = 960.0 / w0
    g = cv2.cvtColor(cv2.resize(image, (960, int(round(h0 * sc))), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    H, W = g.shape
    a0, b0 = int(max(x0 * sc, 0)), int(max(y0 * sc, 0))
    a1, b1 = int(min(x1 * sc, W)), int(min(y1 * sc, H))
    if a1 - a0 < 4 or b1 - b0 < 4:
        return float("nan")
    inside = g[b0:b1, a0:a1]
    mx, my = max(int(0.25 * (a1 - a0)), 4), max(int(0.25 * (b1 - b0)), 4)
    outer = g[max(b0 - my, 0):min(b1 + my, H), max(a0 - mx, 0):min(a1 + mx, W)]
    n = outer.size - inside.size
    return float((outer.sum() - inside.sum()) / n) if n > 0 else float("nan")


def fill_ring(w: pd.DataFrame, rec: "Recording", cache: Path, size: tuple[int, int],
              decode: Callable | None = None) -> pd.DataFrame:
    """Brightness around the window's whale box where the detector cache predates ``q_ring`` (WP9): the best frames
    of the whale windows are decoded once and the values cached in ``<key>.ring.csv``."""
    need = w.whale.astype(bool) & w.ring.isna() & w.bx0.notna() & (w.best_frame >= 0)
    if not need.any():
        return w
    f = cache / f"{rec.key}.ring.csv"
    have = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["frame", "x0", "y0", "x1", "y1", "ring"])
    key = lambda fr, a, b, c, d: (int(fr), round(float(a), 1), round(float(b), 1), round(float(c), 1), round(float(d), 1))
    known = {key(r.frame, r.x0, r.y0, r.x1, r.y1): r.ring for r in have.itertuples()}
    rows = w[need]
    todo = {}
    for r in rows.itertuples():
        k = key(r.best_frame, r.bx0, r.by0, r.bx1, r.by1)
        if k not in known:
            todo.setdefault(int(r.best_frame), []).append(k)
    if todo:
        dsize = size if size != video_size(rec.path) else None
        frames = decode(rec.path, sorted(todo)) if decode else (
            video.iter_frames_isolated(rec.path, indices=sorted(todo), size=dsize) if video.torch_loaded()
            else video.iter_frames_at(rec.path, sorted(todo), size=dsize))
        new = []
        for fr in frames:
            for k in todo.get(fr.index, []):
                known[k] = ring_brightness(fr.image, *k[1:])
                new.append({"frame": k[0], "x0": k[1], "y0": k[2], "x1": k[3], "y1": k[4], "ring": known[k]})
        if new:
            pd.concat([have, pd.DataFrame(new)], ignore_index=True).to_csv(f, index=False)
    w = w.copy()
    w.loc[need, "ring"] = [known.get(key(r.best_frame, r.bx0, r.by0, r.bx1, r.by1), np.nan) for r in rows.itertuples()]
    return w


def silhouette_flag(w: pd.DataFrame, cfg: "ScanConfig") -> np.ndarray:
    """A large whale box whose inside is dark and low in detail against a brighter surround (a whale seen from below
    against the light). A label only: it changes neither the keep decision nor ``advanced_ok``. Limits chosen on
    the dev clip (experiments/06_clips_e2e/silhouette_rule_dev.py)."""
    whale = w.whale.to_numpy(bool)
    px = w.whale_px.fillna(0).to_numpy()
    lift = (w.ring - w.bright).to_numpy(dtype=float)
    return (whale & (px >= cfg.sil_min_px) & (np.nan_to_num(w.bright.to_numpy(dtype=float), nan=1) <= cfg.sil_max_bright)
            & (np.nan_to_num(lift, nan=-1) >= cfg.sil_min_lift))


DECISION_LABELS = ("whale", "small", "fish", "other")


def decision_objects(frame_boxes: pd.DataFrame, cfg: ScanConfig, width: int, height: int, shallow: bool = False,
                     fish_min: float | None = None, iou_same: float = 0.5) -> pd.DataFrame:
    """What the keep rule made of one frame's boxes, one row per object (for sheets).

    The detector lists every box once per prompt; here boxes of different prompts on the same region are one
    object (greedy, IoU >= ``iou_same``), labelled by the decision in the order whale > small > fish: ``whale`` = a
    "whale" box that passes the same filters as :func:`frame_scores` (not in the clock, the whole-frame rule with
    ``shallow`` = the window's depth limit, the fish veto if on) with score >= threshold and longest side >=
    ``min_whale_px``; ``small`` = the same below the size limit; ``fish`` = a fish-prompt box >= ``fish_min``
    (default the scan's ``fish_threshold``) or a large "whale" box the box classifier calls fish; ``other`` = a
    large "whale" box the classifier calls other (no whale). Boxes that play no part are left out. Columns: x0, y0,
    x1, y1 (detector pixels), label, score (the score that decided the label), px (longest side, 1920 px frame),
    whale_score, fish_score (best fish prompt at the same patch), clf (True when the classifier decided)."""
    from . import detect
    cols = ["x0", "y0", "x1", "y1", "label", "score", "px", "whale_score", "fish_score", "clf"]
    if frame_boxes is None or len(frame_boxes) == 0:
        return pd.DataFrame(columns=cols)
    b = frame_boxes.copy()
    b["area"] = ((b.x1 - b.x0).clip(lower=0) * (b.y1 - b.y0).clip(lower=0)) / float(width * height)
    b["px"] = np.maximum(b.x1 - b.x0, b.y1 - b.y0) * (1920.0 / width)
    fcols = [detect.patch_score_column(q) for q in cfg.fish_queries]
    fcols = [c for c in fcols if c in b.columns]
    b["fish_score"] = b[fcols].max(axis=1) if fcols else np.nan
    wcol = detect.patch_score_column(cfg.whale_query)
    b["whale_score"] = b[wcol] if wcol in b.columns else np.where(b["query"] == cfg.whale_query, b.score, np.nan)
    clock = detect.in_clock(b, width, height)
    whole = b.area > cfg.whole_frame
    back = is_back_box(b, width, height) if cfg.back_filter else pd.Series(False, index=b.index)
    w = b[(b["query"] == cfg.whale_query) & ~clock & ~back & (b.score >= cfg.threshold)]
    ok_whole = (cfg.near_whale_score is not None) and not shallow
    w = w[~whole[w.index] | (ok_whole & (w.score >= (cfg.near_whale_score or 9)))]
    if cfg.fish_veto and fcols:
        w = w[w.score > w[fcols].max(axis=1)]
    w = w.assign(label=np.where(w.px >= cfg.min_whale_px, "whale", "small"), clf=False)
    if cfg.boxclf and "boxclf_class" in w.columns:
        cls = w["boxclf_class"].fillna("").astype(str)
        big = w.px >= cfg.min_whale_px
        w.loc[big & cls.isin(["whale", "fish", "other"]), "clf"] = True   # the classifier decided this object
        w.loc[big & (cls == "fish"), "label"] = "fish"          # a "whale" box the classifier calls fish
        w.loc[big & (cls == "other"), "label"] = "other"        # blow, surface, back, sediment: no whale
    fmin = cfg.fish_threshold if fish_min is None else fish_min
    f = b[b["query"].isin(cfg.fish_queries) & ~clock & ~whole & (b.score >= fmin)]
    f = f.sort_values("score", ascending=False).assign(label="fish", clf=False)
    cand = pd.concat([w, f], ignore_index=True)
    if cand.empty:
        return pd.DataFrame(columns=cols)
    cand["rank"] = cand.label.map({"whale": 0, "small": 1, "fish": 2, "other": 3})
    cand = cand.sort_values(["rank", "score"], ascending=[True, False]).reset_index(drop=True)
    kept = []
    for i, r in cand.iterrows():
        dup = False
        for j in kept:
            q = cand.loc[j]
            ix = max(0.0, min(r.x1, q.x1) - max(r.x0, q.x0))
            iy = max(0.0, min(r.y1, q.y1) - max(r.y0, q.y0))
            inter = ix * iy
            union = (r.x1 - r.x0) * (r.y1 - r.y0) + (q.x1 - q.x0) * (q.y1 - q.y0) - inter
            if union > 0 and inter / union >= iou_same:
                dup = True
                break
        if not dup:
            kept.append(i)
    out = cand.loc[kept]
    return out[cols].reset_index(drop=True)


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


def mark_continuing(files: list, max_gap_s: float = 3.0) -> None:
    """Two recordings of one deployment back to back (the camera splits a session into files): a clip that reaches
    the end of one file and a clip that starts at the start of the next are one scene cut in two. Each file keeps its
    own clip; they are marked ``continues_in`` / ``continues_from`` (the other file's name)."""
    files = sorted(files, key=lambda f: f[0])
    for (c0, c1, dur, name, rows), (d0, _, _, name2, rows2) in zip(files, files[1:]):
        if not rows or not rows2 or abs(d0 - c1) > max_gap_s:
            continue
        if rows[-1]["end"] >= dur - 0.05 and rows2[0]["start"] <= 0.05:
            rows[-1]["continues_in"] = name2
            rows2[0]["continues_from"] = name


def hard_breaks(w: pd.DataFrame, cfg: ScanConfig) -> np.ndarray:
    """Windows a clip never runs across: tag in air or under the drone, and the lens out of the water for longer
    than ``max_breath_s`` (a breath of the tagged whale does not end a clip)."""
    hard = np.array(w.tag_state.isin(["air", "flight"]).to_numpy(), bool) if "tag_state" in w else np.zeros(len(w), bool)
    out = (w.water.isin(["surfaced", "air"]).to_numpy() if "water" in w else np.zeros(len(w), bool))
    i = 0
    n = len(out)
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) > cfg.max_breath_s:
                hard[i:j] = True
            i = j
        else:
            i += 1
    return hard


def clip_intervals(active: np.ndarray, t0: np.ndarray, t1: np.ndarray, hard: np.ndarray, pad: float, guard: float,
                   duration: float) -> list[tuple[float, float]]:
    """Active windows -> clips (user 2026-11-23): a clip starts at the first active window and ends only when no
    window is active for at least ``guard`` seconds (the gap between the end of one active window and the start of the
    next), and never across a ``hard`` window; then ``pad`` seconds of buffer on each side, clipped to the file and
    to hard windows."""
    active = np.asarray(active, bool)
    idx = np.nonzero(active)[0]
    if not len(idx):
        return []
    hard_cum = np.concatenate([[0], np.cumsum(np.asarray(hard, bool))])
    groups = [[idx[0], idx[0]]]
    for i in idx[1:]:
        a, b = groups[-1]
        gap = float(t0[i] - t1[b])
        crosses = hard_cum[i] - hard_cum[b + 1] > 0
        if gap < guard and not crosses:
            groups[-1][1] = i
        else:
            groups.append([i, i])
    hard_t0 = np.asarray(t0)[np.asarray(hard, bool)]
    hard_t1 = np.asarray(t1)[np.asarray(hard, bool)]
    out = []
    for a, b in groups:
        s, e = max(float(t0[a]) - pad, 0.0), min(float(t1[b]) + pad, duration)
        before = hard_t1[hard_t1 <= t0[a] + 1e-9]
        after = hard_t0[hard_t0 >= t1[b] - 1e-9]
        if len(before):
            s = max(s, float(before.max()))
        if len(after):
            e = min(e, float(after.min()))
        if out and s < out[-1][1]:
            s = out[-1][1]
        out.append((s, e))
    return out


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
        return "nothing detected"
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
        # floating: the sensors cannot say where the lens looks; the picture water state decides (under water ->
        # the detector runs, like on the whale)
        flo = tag == "floating"
        if cfg.scan_floating:
            pic = w.water_picture.to_numpy()
            water = np.where(flo & np.isin(pic, ["underwater", "crossing", "surfaced"]), pic, water)
            src = np.where(flo & (pic != "unknown"), "picture", src)
        w["tag_state"] = tag
        w["tag_source"] = np.where(unk, "none", "sensors")
        dm = np.interp((s0 + s1) / 2, sensors.depth_t, sensors.depth_m) if hasattr(sensors, "depth_t") else np.nan
        w["depth_m"] = np.where((tag == "whale") | flo, dm, np.nan)
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
        on = ["whale", "unknown"] + (["floating"] if cfg.scan_floating else [])
        w["scanned"] = w.water.isin(["underwater", "crossing"]) & w.tag_state.isin(on)
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
            if col in COUNT_COLUMNS:
                v = np.nan_to_num(v, nan=0.0)
            per[col].append(np.where(ok, v, np.nan))
        w[f"whale_{j}"] = per["whale"][-1]
        dec.append(ok)
    # with a sensor log: just under the surface (depth < near_whale_min_depth_m) a box over most of the frame is the
    # tagged whale's back mirrored in the surface, so those windows use the boxes that do not fill the frame
    shallow = (w["depth_m"].to_numpy(dtype=float) < cfg.near_whale_min_depth_m) if "depth_m" in w.columns else \
        np.zeros(len(w), bool)
    shallow = np.nan_to_num(shallow, nan=0).astype(bool)
    for j in range(len(fcols)):
        per["whale"][j] = np.where(shallow, per["whale_part"][j], per["whale"][j])
        for c in QUALITY_COLUMNS:
            per[c][j] = np.where(shallow, per[c + "_part"][j], per[c][j])
        w[f"whale_{j}"] = per["whale"][j]
    w["shallow_rule"] = shallow
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
        FI = np.stack(per["fish"], 1)
        fish = np.nanmax(FI, 1)
        small = np.nanmax(SM, 1)
        # activity per frame: a whale box of any size or a fish box; the preset combines the frames
        VF = np.stack(per["veto_fish"], 1)
        VO = np.stack(per["veto_other"], 1)
        act_f = (np.nan_to_num(S, nan=-1) >= cfg.threshold) | (np.nan_to_num(SM, nan=-1) >= cfg.threshold) | (
            np.nan_to_num(FI, nan=-1) >= cfg.fish_threshold) | (np.nan_to_num(VF, nan=-1) >= cfg.threshold)
        dec_f = np.stack(dec, 1)
        if cfg.preset == "strict":
            active = (act_f | ~dec_f).all(1) & dec_f.any(1)
        else:
            active = (act_f & dec_f).any(1)
        n_whale = np.nanmax(np.stack(per["n_whale"], 1), 1)
        n_fish = np.nanmax(np.stack(per["n_fish"], 1), 1)
        cover = np.stack(per["cover"], 1)
        best = np.where(np.isnan(cover).all(1), 0, np.nanargmax(np.nan_to_num(cover, nan=-1), 1))
    rows = np.arange(len(w))
    w["whale_score"] = np.where(decoded, rule, np.nan)
    w["small_score"] = np.where(decoded, small, np.nan)
    w["fish_score"] = np.where(decoded, fish, np.nan)
    w["whale"] = decoded & (np.nan_to_num(rule, nan=-1) >= cfg.threshold)
    w["small"] = decoded & ~w.whale.to_numpy() & (np.nan_to_num(small, nan=-1) >= cfg.threshold)
    vfish = np.nanmax(np.nan_to_num(VF, nan=0), 1)
    vother = np.nanmax(np.nan_to_num(VO, nan=0), 1)
    w["veto_fish_score"] = np.where(decoded, vfish, np.nan)       # large "whale" box the classifier calls fish
    w["veto_other_score"] = np.where(decoded, vother, np.nan)     # ... calls other (blow, surface, back, sediment)
    w["fish"] = decoded & ((np.nan_to_num(fish, nan=-1) >= cfg.fish_threshold) | (vfish >= cfg.threshold))
    w["active"] = w.whale.to_numpy() | (active & decoded) if cfg.keep_activity else w.whale.to_numpy()
    w["n_whale"] = np.where(decoded, n_whale, np.nan)
    w["n_fish"] = np.where(decoded, n_fish, np.nan)
    w["vetoed_other_only"] = decoded & ~w.active.to_numpy() & (vother >= cfg.threshold)
    # seafloor / sand / dark are filled by fill_window_classes() (uwid.patchseg, WP10); labels only, no keep rule
    for c in ("seafloor", "sand", "dark"):
        if c not in w.columns:
            w[c] = ""
    for col in ("cover", "top_score", "whale_px", "bright", "contrast", "sharp", "ring", "bx0", "by0", "bx1", "by1",
                "bubbles"):
        w[col] = np.stack(per[col], 1)[rows, best]
    w["best_frame"] = np.stack([w[c].to_numpy() for c in fcols], 1)[rows, best]   # the frame the quality is from
    for col in ("cover", "top_score", "whale_px", "bright", "contrast", "sharp", "ring", "bx0", "by0", "bx1", "by1"):
        w.loc[~w.whale, col] = np.nan
    parts = hq_parts(w)
    for c in parts.columns:
        w[c] = parts[c]
    w["hq_whale"] = w.whale & (w.cover.fillna(0) >= cfg.hq_min_cover) & (w.hq_score.fillna(0) >= cfg.hq_threshold)
    w["tier"] = window_tier(w, cfg)
    keep = clip_intervals(w.active.to_numpy(), w.t0.to_numpy(), w.t1.to_numpy(), hard_breaks(w, cfg), cfg.pad_s,
                          cfg.guard_s, duration)
    mid = (w.t0.to_numpy() + w.t1.to_numpy()) / 2
    kept = np.zeros(len(w), bool)
    for a, b in keep:
        kept |= (mid >= a) & (mid < b)
    w["kept"] = kept
    why = []
    for r, dk in zip(w.itertuples(), decoded):
        if r.kept:
            why.append(r.tier if r.tier in ("small", "fish") else "whale" if r.whale else
                       "padding")
        elif r.small and not cfg.keep_activity:
            why.append("nothing detected")       # whale-only option: a small whale is not kept
        else:
            why.append(drop_reason(r.tag_state, r.water, bool(r.scanned), bool(dk)))
    w["why"] = why
    return w, keep


TIERS = ("hq", "ok", "low", "small", "fish", "none")      # best first
TIER_RANK = {t: i for i, t in enumerate(TIERS)}


def window_tier(w: pd.DataFrame, cfg: ScanConfig) -> np.ndarray:
    """Label of a window (not a filter): hq = hq_whale (the only windows for 3D / identification work); ok = another
    whale window whose whale box is >= ``ok_min_px``; low = the other whale windows (>= ``min_whale_px``); small = a
    whale box under the size limit (fish or far whale); fish = a fish box >= ``fish_threshold`` and no whale box;
    none = nothing detected (padding, dropped)."""
    whale = w.whale.to_numpy(bool)
    ok = whale & (w.whale_px.fillna(0).to_numpy() >= cfg.ok_min_px)
    small = w.small.to_numpy(bool) if "small" in w else np.zeros(len(w), bool)
    fish = w.fish.to_numpy(bool) if "fish" in w else np.zeros(len(w), bool)
    return np.where(w.hq_whale.to_numpy(bool), "hq", np.where(ok, "ok", np.where(whale, "low", np.where(
        small, "small", np.where(fish, "fish", "none")))))


def best_tier(tiers) -> str:
    t = [x for x in tiers if x in TIER_RANK]
    return min(t, key=TIER_RANK.get) if t else "none"


def _word_count(n: float) -> str:
    return {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}.get(int(n), f"{int(n)}")


def _secs(x: float) -> str:
    """Seconds for a description: whole seconds, one decimal under 10 s when not whole."""
    return f"{x:.0f}" if x >= 10 or abs(x - round(x)) < 0.05 else f"{x:.1f}"


def describe_clip(w: pd.DataFrame, start: float, end: float) -> str:
    """One or two plain sentences about a kept stretch, built only from its window table (every word traces to a
    column: clock, depth_m, tier, whale_px, n_whale, n_fish, small, fish, water, tag_state)."""
    def clk(t):
        c = w.loc[(w.t0 <= t + 1e-6), "clock"]
        return str(c.iloc[-1])[:8] if len(c) and isinstance(c.iloc[-1], str) and c.iloc[-1] else f"{t:.0f} s"
    length = end - start
    parts = [f"{clk(start)} to {clk(max(end - 1.0, start))}, {_secs(length)} s"]
    d = w.depth_m.dropna() if "depth_m" in w else pd.Series(dtype=float)
    if len(d):
        parts[0] += f", depth {d.min():.0f} to {d.max():.0f} m" if d.max() - d.min() >= 1 else f", depth {d.mean():.0f} m"
    out = [parts[0] + "."]
    secs = {t: int((w.tier == t).sum()) for t in TIERS}
    nw = int(np.nanmax(w.n_whale)) if "n_whale" in w and w.n_whale.notna().any() else 0
    if secs["hq"] + secs["ok"] + secs["low"]:
        px = float(np.nanmax(w.whale_px)) if w.whale_px.notna().any() else np.nan
        who = "One whale" if nw <= 1 else f"{_word_count(nw)} whale boxes at once at most"
        q = ", ".join(f"{name} {secs[t]} s" for t, name in (("hq", "high quality"), ("ok", "ok"), ("low", "low"))
                      if secs[t])
        size = f", largest {px / 1920:.2f} of the frame width" if np.isfinite(px) else ""
        out.append(f"{who} in view{size} ({q}).")
        if "silhouette" in w and w.silhouette.astype(bool).any():
            out.append(f"Large whale in silhouette against the light for {int(w.silhouette.astype(bool).sum())} s.")
    if secs["small"]:
        out.append(f"Small or far whale-like shapes (fish-sized) for {secs['small']} s.")
    nf = int(np.nanmax(w.n_fish)) if "n_fish" in w and w.n_fish.notna().any() else 0
    nfs = int(w.fish.sum()) if "fish" in w else 0
    if nfs:
        amount = "a few" if nf <= 3 else "several" if nf <= 9 else "many"
        boxes = f"up to {nf} fish box{'es' if nf != 1 else ''} in a frame, " if nf else ""
        out.append(f"Fish: {amount} ({boxes}{nfs} s).")
    surf = w[w.water.isin(["surfaced", "crossing"]) & (w.tag_state == "whale")] if "water" in w else w.iloc[:0]
    if len(surf):
        out.append(f"Tagged whale surfaces at {str(surf.clock.iloc[0])[:8]}.")
    for col, text in (("sand", "View blocked by sand for {} s."), ("dark", "Lens dark for {} s.")):
        if col in w:
            k = int(_truthy(w[col]).sum())
            if k:
                out.append(text.format(k))
    if "active" in w:
        # seconds of the clip (not windows: a clip cut by the file's end holds a part window)
        ov = (np.minimum(w.t1.to_numpy(float), end) - np.maximum(w.t0.to_numpy(float), start)).clip(0)
        quiet = float(ov[~w.active.astype(bool).to_numpy()].sum())
        if quiet > 0.05:
            out.append(f"{_secs(quiet)} s of {_secs(length)} s without anything detected.")
    if "tag_state" in w and (w.tag_state == "floating").any():
        out.append(f"Tag floating (off the whale) for {int((w.tag_state == 'floating').sum())} s.")
    return " ".join(out)


def write_descriptions(path: Path, name: str, cfg: ScanConfig, videos_json: list) -> None:
    """One readable line block per kept stretch (the same text as the intervals file and the clip index)."""
    lines = [f"{name}: kept stretches of `uwid scan` [{cfg.label()}], with their best tier and description.",
             "Tiers: hq (advanced_ok: for 3D / identification), ok, low, small (fish-sized whale shapes), fish.", ""]
    for v in videos_json:
        lines.append(f"== {Path(v['source']).name}")
        for k in v["keep"]:
            hq = (" hq spans " + ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in k["hq_spans"])) if k["hq_spans"] else ""
            lines.append(f"  {k['start']:8.1f}-{k['end']:8.1f} s  [{k['tier']}{', advanced_ok' if k['advanced_ok'] else ''}"
                         f"{hq}]  {k['description']}")
        lines.append("")
    path.write_text("\n".join(lines))


def filter_intervals_by_tier(iv: dict, min_tier: str) -> dict:
    """A copy of an intervals file whose keep intervals below ``min_tier`` are moved to drop ("tier low: not cut")."""
    import copy
    out = copy.deepcopy(iv)
    for v in out["videos"]:
        keep, drop = [], list(v.get("drop", []))
        for k in v["keep"]:
            if TIER_RANK.get(k.get("tier", "none"), 9) <= TIER_RANK[min_tier]:
                keep.append(k)
            else:
                drop.append({"start": k["start"], "end": k["end"],
                             "reason": f"tier {k.get('tier')} below --min-tier {min_tier}: not cut"})
        v["keep"] = keep
        v["drop"] = sorted(drop, key=lambda d: d["start"])
    return out


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


def resolve_offset(cfg: ScanConfig, dep: "Deployment") -> tuple[str, float]:
    """(mode, starting offset): a number given -> ("given", it); "tag" with a known tag serial -> ("tag", its
    measured offset); otherwise ("auto", 0.0): estimated from the events (0.0 is only the start of the search)."""
    from . import names
    if not isinstance(cfg.clock_offset_s, str):
        return "given", float(cfg.clock_offset_s)
    if cfg.clock_offset_s == "tag":
        tags = {(names.parse_tag_video_name(r.path) or {}).get("tag") for r in dep.recordings}
        tags.discard(None)
        if len(tags) == 1 and next(iter(tags)) in KNOWN_CLOCK_OFFSETS:
            return "tag", KNOWN_CLOCK_OFFSETS[next(iter(tags))]
    elif cfg.clock_offset_s != "auto":
        return "given", float(cfg.clock_offset_s)
    return "auto", 0.0


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
    incomplete: list = []
    joins: list = []      # (clock at start, clock at end, duration, file name, keep rows) for back-to-back files
    deadline = t_start + cfg.run_budget_s if cfg.run_budget_s else None
    if cfg.boxclf and boxclf_weights(cfg) is None:
        log("  no box classifier weights (uwid.boxclf.DEFAULT_WEIGHTS or --boxclf-weights): the size rule decides "
            "alone, as before WP11b")
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
        if not cf.exists() or not fits[rec.key].get("ok"):     # unread clocks are tried again (new layouts)
            ts = time.time()
            fits[rec.key] = fit_burned_clock(rec.path, cfg.clock_probes)
            fits[rec.key]["seconds"] = time.time() - ts
            cf.write_text(json.dumps(fits[rec.key], indent=1))
            timings["clock_s"] = timings.get("clock_s", 0.0) + fits[rec.key]["seconds"]
    offset_mode, nominal_offset = resolve_offset(cfg, dep)

    def on_whale(rec) -> bool:
        """With a log, the picture model is only needed where the tag is on a whale (offset check) or floating
        (where the lens looks)."""
        vc = _video_clock(rec, sensors, fits[rec.key], nominal_offset)
        if vc is None:
            return True
        a, b = float(vc.to_sensor(0.0)), float(vc.to_sensor(video.frame_table(rec.path).duration))
        seg = sensors.segments
        want = ["whale"] + (["floating"] if cfg.scan_floating else [])
        return bool((seg.state.isin(want) & (seg.t1 > a) & (seg.t0 < b)).any())

    for rec in dep.recordings:          # GPU stage after every clock fit (CPU), so one GPU hold covers several
        if sensors is None or (cfg.picture_check and on_whale(rec)):
            ts = time.time()
            water[rec.key], pic_events[rec.key] = picture_water(rec, cache, cfg, log, water_fn, timings)
            timings["water_picture_wall_s"] = timings.get("water_picture_wall_s", 0.0) + time.time() - ts
    # offset check
    offset = nominal_offset
    check = None
    if sensors is not None:
        nominal = nominal_offset
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
            de.append(dev[(dev.t > a + 1.0) & (dev.t < b - 1.0)].assign(recording=rec.stem))
        pe = pd.concat(pe, ignore_index=True) if pe else pd.DataFrame(columns=["t", "kind"])
        de = pd.concat(de, ignore_index=True) if de else pd.DataFrame(columns=["t", "kind"])
        check = offset_check(pe, de)
        check["offset_mode"] = offset_mode
        check["offset_in_use_s"] = nominal
        est = check.get("midpoint_s", check.get("median_picture_minus_depth_s"))
        if est is not None:
            check["implied_offset_s"] = nominal + est
        enough = (check["n_pairs"] >= cfg.min_offset_events and check.get("n_exit", 0) >= 3
                  and check.get("n_entry", 0) >= 3)
        log(f"  offset check ({offset_mode}): burned clock - sensor time {nominal:.2f} s; picture minus depth "
            f"midpoint {check.get('midpoint_s', float('nan')):+.2f} s (exits {check.get('median_exit_s') or 0:+.2f}, "
            f"entries {check.get('median_entry_s') or 0:+.2f}) over {check['n_pairs']} of "
            f"{check['n_depth_events']} depth events -> {check['verdict']}")
        if offset_mode == "auto":
            if not enough:
                (base / "offset_check.json").write_text(json.dumps(check, indent=1))
                raise ClockOffsetError(
                    f"{dep.name}: only {check['n_pairs']} picture events match the depth's ({check.get('n_exit', 0)} "
                    f"exits, {check.get('n_entry', 0)} entries; need {cfg.min_offset_events}, 3 of each): the offset "
                    f"between the burned-in clock and the sensor log cannot be estimated. Give it with "
                    f"--clock-offset SECONDS (see {base / 'offset_check.json'}).")
            # one re-check at the estimate: the pairing window is centred on the offset in use
            offset = float(check["implied_offset_s"])
            check2 = offset_check(pe.assign(t=pe.t - est) if len(pe) else pe, de)
            check["after"] = {k: v for k, v in check2.items() if k != "pairs"}
            check["offset_used_s"] = offset
            log(f"  using the estimated offset {offset:.2f} s; per recording picture minus depth (midpoint): "
                + ", ".join(f"{k[-5:]} {v['midpoint_s']:+.2f}" for k, v in check2.get("per_recording", {}).items()
                            if v.get("midpoint_s") is not None))
        elif check["verdict"] == "OFFSET LOOKS WRONG":
            log(f"  WARNING: the picture says the offset is {check['implied_offset_s']:.2f} s; re-run with "
                f"--clock-offset {check['implied_offset_s']:.2f} (or auto) if the sheets agree")
        check["offset_used_s"] = float(offset)
        (base / "offset_check.json").write_text(json.dumps(check, indent=1))
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
        if (cfg.boxclf or cfg.window_classes) and detector_factory is None and len(processed):
            done_set = np.asarray(processed, np.int64)
            bfr = np.intersect1d(boxclf_frames(boxes, cfg, size[0], size[1]), done_set)
            wfr = np.intersect1d(want, done_set)
            if not picture_classes(rec, cache, cfg, wfr, bfr, boxes, log, deadline, timings):
                incomplete.append(rec.stem)
        pcls, pfrm, _, _ = load_picture_classes(cache, rec)
        if cfg.boxclf:
            boxes = attach_box_classes(boxes, pcls)
        fs = frame_scores(boxes, processed, cfg, size[0], size[1])
        w, keep = apply_detections(w, fs, processed, cfg, tab.duration)
        if cfg.window_classes:
            w = fill_window_classes(w, pfrm)
        w = fill_ring(w, rec, cache, size, None if detector_factory is None else
                      (lambda pth, idx: video.iter_frames_at(pth, idx, size=None)))
        w["silhouette"] = silhouette_flag(w, cfg)
        w.to_csv(run_dir / "windows" / f"{rec.key}.csv", index=False)
        drops = drop_stretches(w, keep)
        keep_rows = []
        for a, b in keep:
            m = (w.t0 < b) & (w.t1 > a)
            n_hq = int((m & w.hq_whale).sum())
            adv = w.hq_whale.to_numpy() & m.to_numpy() & ~_truthy(w.sand) & ~_truthy(w.dark)
            spans = [(x, y) for x, y in keep_intervals(adv, w.t0.to_numpy(), w.t1.to_numpy(), 0.0, 2.0, tab.duration)]
            tier = best_tier(w.loc[m, "tier"])
            keep_rows.append({"start": round(a, 4), "end": round(b, 4),
                              "reason": "whale in view" if tier in ("hq", "ok", "low") else f"{tier} in view",
                              "tier": tier, "hq_whale": bool(n_hq), "advanced_ok": bool(spans),
                              "active_s": int((m & w.active).sum()), "quiet_s": int((m & ~w.active).sum()),
                              "sand_s": int((m & _truthy(w.sand)).sum()), "dark_s": int((m & _truthy(w.dark)).sum()),
                              "silhouette": bool((m & w.silhouette).any()),
                              "silhouette_windows": int((m & w.silhouette).sum()),
                              "hq_spans": [[round(x, 3), round(y, 3)] for x, y in spans],
                              "description": describe_clip(w[m], a, b),
                              "hq_windows": n_hq,
                              "tier_windows": {t: int((m & (w.tier == t)).sum()) for t in TIERS[:5]},
                              "max_hq_score": round(float(w.loc[m, "hq_score"].max()), 3)
                              if w.loc[m, "hq_score"].notna().any() else None,
                              "whale_windows": int((m & w.whale).sum()),
                              "max_whale_score": round(float(w.loc[m, "whale_score"].max()), 3),
                              "max_fish_score": round(float(w.loc[m, "fish_score"].max()), 3)
                              if w.loc[m, "fish_score"].notna().any() else None})
        hq = keep_intervals(w.hq_whale.to_numpy(), w.t0.to_numpy(), w.t1.to_numpy(), cfg.pad_s, cfg.merge_gap_s,
                            tab.duration)
        f_ = fits.get(rec.key) or {}
        if f_.get("ok"):
            joins.append((float(f_["clock_at_pts0"]), float(f_["clock_at_pts0"]) + float(f_["rate"]) * tab.duration,
                          tab.duration, rec.path.name, keep_rows))
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
                        "n_hq_clips": sum(1 for r in keep_rows if r["hq_whale"]),
                        "tier_window_s": {t: float((w.tier == t).sum()) for t in TIERS[:5]},
                        "clip_s_by_tier": {t: sum(r["end"] - r["start"] for r in keep_rows if r["tier"] == t)
                                           for t in TIERS[:5]},
                        "n_clips_by_tier": {t: sum(1 for r in keep_rows if r["tier"] == t) for t in TIERS[:5]},
                        "n_clips": len(keep), "dropped_s": by, "windows": int(len(w)),
                        "windows_scanned": int(w.scanned.sum()), "whale_windows": int(w.whale.sum()),
                        "clock": {k: fits[rec.key].get(k) for k in ("ok", "kind", "rms_s", "share_read")},
                        "video_clock": vc.source if vc is not None else None})
        log(f"    kept {kept_s / 60:.1f} of {tab.duration / 60:.1f} min in {len(keep)} stretches")
    release_gpu()
    mark_continuing(joins)
    iv = {"format": FORMAT, "created": _dt.datetime.now().isoformat(timespec="seconds"),
          "tool": "uwid scan", "config": {**asdict(cfg), "clock_offset_used_s": offset if sensors else None},
          "videos": videos_json}
    (run_dir / "intervals.json").write_text(json.dumps(iv, indent=1, default=str))
    write_descriptions(run_dir / "clips.txt", dep.name, cfg, videos_json)
    pd.DataFrame(drop_rows, columns=["recording", "start", "end", "reason", "seconds"]).to_csv(
        run_dir / "dropped.csv", index=False)
    total = sum(r["seconds"] for r in rec_sum)
    summary = {"deployment": dep.name, "label": cfg.label(), "sensor_log": str(dep.sensor_csv) if dep.sensor_csv
               else None, "skipped": dep.skipped, "recordings": rec_sum, "seconds": total,
               "kept_s": sum(r["kept_s"] for r in rec_sum),
               "hq_windows": sum(r["hq_windows"] for r in rec_sum),
               "hq_stretch_s": sum(r["hq_stretch_s"] for r in rec_sum),
               "tier_window_s": {t: sum(r["tier_window_s"][t] for r in rec_sum) for t in TIERS[:5]},
               "clip_s_by_tier": {t: sum(r["clip_s_by_tier"][t] for r in rec_sum) for t in TIERS[:5]},
               "n_clips_by_tier": {t: sum(r["n_clips_by_tier"][t] for r in rec_sum) for t in TIERS[:5]},
               "dropped_s": {k: sum(r["dropped_s"][k] for r in rec_sum) for k in REASONS},
               "offset_check": {k: v for k, v in (check or {}).items() if k != "pairs"} or None,
               "timings_this_run": timings, "wall_s_this_run": time.time() - t_start,
               "complete": not incomplete, "incomplete_recordings": incomplete,
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
        ivf = run_dir / "intervals.json"
        if cfg.min_tier:
            ivf = run_dir / f"intervals_cut_{cfg.min_tier}.json"
            ivf.write_text(json.dumps(filter_intervals_by_tier(iv, cfg.min_tier), indent=1, default=str))
        res = clips.run(ivf, out_dir=cdir, mode=mode, pad=0.0, merge_gap=0.0, log=log)
        summary["clips"] = {"n": len(res["clips"]), "out_dir": str(cdir), "seconds": time.time() - tc,
                            "min_tier": cfg.min_tier,
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
