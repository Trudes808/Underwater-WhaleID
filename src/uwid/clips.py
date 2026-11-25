"""Cut keep-intervals of source videos into clips with the bundled ffmpeg (imageio-ffmpeg).

Intervals file (JSON, the interface the detection work fills)::

    {
      "format": "uwid-intervals/1",
      "videos": [
        {
          "source": "2025-12-20 CG.m4v",          # absolute path, or relative to the footage folder
          "keep": [ {"start": 20.0, "end": 41.5, "reason": "whale in view"} ],
          "drop": [ {"start": 0.0,  "end": 20.0, "reason": "camera in air"} ]
        }
      ]
    }

Times are presentation times in seconds as :mod:`uwid.video` reports them (frame ``t`` = pts after the edit list;
frame ``i`` of :func:`uwid.video.frame_table`). An interval holds the frames with ``start <= t < end``. ``reason``
is free text and is copied to the outputs. Optional keys per interval are kept in the CSVs as ``extra`` (JSON).

Modes
- ``copy``: the video is not re-encoded (the sound is, to AAC, so it is trimmed exactly). The clip starts on the keyframe at or before the wanted start, so nothing wanted is lost;
  the real cut times are measured on the written clip and reported.
- ``reencode``: frame exact (first and last wanted frame), H.264 CRF 17 (visually lossless), sound re-encoded AAC.
- ``auto``: ``copy`` when the keyframe before the start is at most ``max_lead`` seconds early, else ``reencode``.

Outputs (default ``<cache>/clips/``): one folder per source with the clips, plus ``clips_index.csv`` and
``dropped.csv`` in the output folder (appended rows of earlier runs are kept). Source footage is only read.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths, video

INTERVALS_FORMAT = "uwid-intervals/1"
INDEX_FIELDS = ["source", "clip_file", "mode", "start_s", "end_s", "start_frame", "end_frame", "n_frames",
                "wanted_start_s", "wanted_end_s", "lead_s", "clock_start", "clock_end", "reason", "tier", "advanced_ok", "silhouette",
                "hq_spans", "description", "extra",
                "verified_frames", "verified_duration_s", "status", "created"]
DROP_FIELDS = ["source", "start_s", "end_s", "reason", "extra", "created"]


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class Interval:
    start: float
    end: float
    reason: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class ClipPlan:
    source: Path
    mode: str
    wanted: Interval
    start_frame: int          # first frame in the clip (index among shown frames)
    end_frame: int            # one past the last frame
    cut_start_s: float        # time of the first frame in the clip
    cut_end_s: float          # time just after the last frame
    lead_s: float             # cut_start - wanted start (<= 0: the clip starts early)
    out: Path


def load_intervals(path: str | os.PathLike) -> list[dict]:
    """Read and check an intervals file. Returns the list of videos with resolved ``source`` paths."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        data = {"videos": data}
    fmt = data.get("format", INTERVALS_FORMAT)
    if fmt != INTERVALS_FORMAT:
        raise ValueError(f"unknown intervals format {fmt!r} (expected {INTERVALS_FORMAT})")
    vids = []
    for v in data.get("videos", []):
        src = Path(v["source"])
        if not src.is_absolute():
            src = paths.footage_dir() / src
        def ivs(key):
            out = []
            for it in v.get(key, []):
                s, e = float(it["start"]), float(it["end"])
                if not (e > s):
                    raise ValueError(f"{src.name}: {key} interval with end <= start: {it}")
                extra = {k: val for k, val in it.items() if k not in ("start", "end", "reason")}
                out.append(Interval(s, e, str(it.get("reason", "")), extra))
            return out
        vids.append({"source": src, "keep": ivs("keep"), "drop": ivs("drop")})
    return vids


def pad_and_merge(intervals: list[Interval], pad: float = 0.0, merge_gap: float = 0.0,
                  t_min: float = 0.0, t_max: float | None = None) -> list[Interval]:
    """Widen every interval by ``pad`` seconds on both sides (clamped), then merge intervals that overlap or are
    closer than ``merge_gap`` seconds. Reasons of merged intervals are joined with "; "."""
    padded = []
    for it in intervals:
        s, e = it.start - pad, it.end + pad
        s = max(s, t_min)
        if t_max is not None:
            e = min(e, t_max)
        if e > s:
            padded.append(Interval(s, e, it.reason, dict(it.extra)))
    padded.sort(key=lambda x: x.start)
    merged: list[Interval] = []
    for it in padded:
        if merged and it.start - merged[-1].end <= merge_gap:
            m = merged[-1]
            m.end = max(m.end, it.end)
            if it.reason and it.reason not in m.reason.split("; "):
                m.reason = f"{m.reason}; {it.reason}" if m.reason else it.reason
            m.extra.setdefault("merged", 1)
            m.extra["merged"] += 1
        else:
            merged.append(Interval(it.start, it.end, it.reason, dict(it.extra)))
    return merged


def plan_clips(source: Path, keep: list[Interval], mode: str = "auto", max_lead: float = 2.0, pad: float = 0.0,
               merge_gap: float = 0.0, out_dir: Path | None = None) -> tuple[list[ClipPlan], list[tuple[Interval, str]]]:
    """Turn keep-intervals into clip plans. Returns (plans, rejected keep-intervals with the reason)."""
    if mode not in ("copy", "reencode", "auto"):
        raise ValueError(f"mode must be copy, reencode or auto, not {mode!r}")
    tab = video.frame_table(source)
    if tab.n_frames == 0:
        return [], [(k, "no frames in source") for k in keep]
    step = tab.frame_step
    t_first, t_end = float(tab.times[0]), float(tab.times[-1]) + step
    merged = pad_and_merge(keep, pad, merge_gap, t_first, t_end)
    rejected = []
    for k in keep:
        if k.end <= t_first or k.start >= t_end:
            rejected.append((k, f"outside the video ({t_first:.3f}-{t_end:.3f} s)"))
    out_dir = Path(out_dir) if out_dir else paths.cache_dir("clips")
    stem = source.stem.replace(" ", "_")
    plans = []
    for it in merged:
        i0 = int(np.searchsorted(tab.times, it.start - 1e-6))
        i1 = int(np.searchsorted(tab.times, it.end - 1e-6))
        if i1 <= i0:
            rejected.append((it, "no frame inside the interval"))
            continue
        want_t0 = float(tab.times[i0])
        kf = tab.keyframe_at_or_before(want_t0)
        lead = math.inf if kf is None else want_t0 - kf
        m = mode
        if mode == "auto":
            m = "copy" if lead <= max_lead + 1e-9 else "reencode"
        if m == "copy" and kf is None:
            rejected.append((it, "copy impossible: no shown keyframe at or before the start (use reencode)"))
            continue
        s0 = i0 if m == "reencode" else int(tab.index_of(kf, tol=1e-3) if tab.index_of(kf, tol=1e-3) is not None
                                            else tab.index_at(kf))
        cut_start = float(tab.times[s0])
        cut_end = float(tab.times[i1 - 1]) + step
        out = out_dir / stem / f"{stem}_{s0:06d}-{i1:06d}_{m}.mp4"
        plans.append(ClipPlan(source=source, mode=m, wanted=it, start_frame=s0, end_frame=i1, cut_start_s=cut_start,
                              cut_end_s=cut_end, lead_s=cut_start - it.start, out=out))
    return plans, rejected


def _ffmpeg_cmd(plan: ClipPlan, crf: int = 17, preset: str = "medium", threads: int = 8) -> list[str]:
    ff = ffmpeg_exe()
    tab = video.frame_table(plan.source)
    step = tab.frame_step
    n = plan.end_frame - plan.start_frame
    base = [ff, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
    if plan.mode == "copy":
        # seek target rounded up to the microsecond: ffmpeg lands on the keyframe at or before it, i.e. this keyframe
        ss = math.ceil(plan.cut_start_s * 1e6) / 1e6
        dur = plan.cut_end_s - ss - 0.5 * step
        # video copied as is; sound re-encoded (cheap) because a stream-copied sound track is not trimmed reliably:
        # in the tag's fragmented .mov a copied sound track started at the file's beginning, not at the cut
        return base + ["-ss", f"{ss:.6f}", "-i", str(plan.source), "-t", f"{dur:.6f}", "-map", "0:v:0",
                       "-map", "0:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                       "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(plan.out)]
    ss = max(plan.cut_start_s - 0.25 * step, 0.0)
    return base + ["-threads", str(threads), "-ss", f"{ss:.6f}", "-i", str(plan.source), "-map", "0:v:0",
                   "-map", "0:a:0?", "-frames:v", str(n), "-t", f"{n * step:.6f}", "-c:v", "libx264", "-crf", str(crf),
                   "-preset", preset, "-threads", str(threads), "-c:a", "aac", "-b:a", "192k",
                   "-movflags", "+faststart", str(plan.out)]


def cut(plan: ClipPlan, overwrite: bool = False, **kw) -> dict:
    """Run ffmpeg for one plan and measure the clip. Returns a row for ``clips_index.csv``."""
    paths.check_writable_target(plan.out)
    if plan.out.resolve() == plan.source.resolve():
        raise PermissionError("output would replace the source")
    plan.out.parent.mkdir(parents=True, exist_ok=True)
    status = "ok"
    if plan.out.exists() and not overwrite:
        status = "exists (kept, not re-cut)"
    else:
        cmd = _ffmpeg_cmd(plan, **kw)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return _row(plan, status=f"ffmpeg error: {res.stderr.strip()[-300:]}")
    # measure the written clip: its frames and length decide the real end
    vt = video.frame_table(plan.out)
    row = _row(plan, status=status)
    row["verified_frames"] = vt.n_frames
    row["verified_duration_s"] = round(vt.duration, 4)
    src_tab = video.frame_table(plan.source)
    if plan.mode == "copy" and vt.n_frames:
        end_frame = min(plan.start_frame + vt.n_frames, src_tab.n_frames)
        row["end_frame"] = end_frame
        row["n_frames"] = vt.n_frames
        row["end_s"] = round(float(src_tab.times[end_frame - 1]) + src_tab.frame_step, 4)
        if end_frame != plan.end_frame and status == "ok":
            # name the file after the frames it really holds
            real = plan.out.with_name(plan.out.name.replace(f"-{plan.end_frame:06d}_", f"-{end_frame:06d}_"))
            if not real.exists() or overwrite:
                plan.out.replace(real)
                row["clip_file"] = str(real)
    if vt.n_frames != row["n_frames"]:
        row["status"] = f"{row['status']}; clip has {vt.n_frames} frames, planned {row['n_frames']}"
    return row


def _row(plan: ClipPlan, status: str = "planned") -> dict:
    return {"source": str(plan.source), "clip_file": str(plan.out), "mode": plan.mode,
            "start_s": round(plan.cut_start_s, 4), "end_s": round(plan.cut_end_s, 4),
            "start_frame": plan.start_frame, "end_frame": plan.end_frame,
            "n_frames": plan.end_frame - plan.start_frame,
            "wanted_start_s": round(plan.wanted.start, 4), "wanted_end_s": round(plan.wanted.end, 4),
            "lead_s": round(plan.lead_s, 4), "clock_start": "", "clock_end": "", "reason": plan.wanted.reason,
            "tier": plan.wanted.extra.get("tier", ""), "advanced_ok": plan.wanted.extra.get("advanced_ok", ""),
            "silhouette": plan.wanted.extra.get("silhouette", ""),
            "hq_spans": json.dumps(plan.wanted.extra["hq_spans"]) if "hq_spans" in plan.wanted.extra else "",
            "description": plan.wanted.extra.get("description", ""),
            "extra": json.dumps(plan.wanted.extra) if plan.wanted.extra else "", "verified_frames": "",
            "verified_duration_s": "", "status": status,
            "created": _dt.datetime.now().isoformat(timespec="seconds")}


def _append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def run(intervals_file: str | os.PathLike, out_dir: str | os.PathLike | None = None, mode: str = "auto",
        max_lead: float = 2.0, pad: float = 0.0, merge_gap: float = 0.0, dry_run: bool = False,
        overwrite: bool = False, read_clock: bool = True, crf: int = 17, preset: str = "medium",
        log=print) -> dict:
    """Cut all keep-intervals of an intervals file. With ``dry_run`` nothing is written; the plans are returned."""
    out_dir = Path(out_dir) if out_dir else paths.cache_dir("clips")
    paths.check_writable_target(out_dir)
    vids = load_intervals(intervals_file)
    index_rows, drop_rows = [], []
    now = _dt.datetime.now().isoformat(timespec="seconds")
    for v in vids:
        src: Path = v["source"]
        if not src.is_file():
            log(f"missing source: {src}")
            for k in v["keep"]:
                drop_rows.append({"source": str(src), "start_s": k.start, "end_s": k.end,
                                  "reason": f"NOT CUT (source missing); wanted: {k.reason}", "created": now})
            continue
        plans, rejected = plan_clips(src, v["keep"], mode, max_lead, pad, merge_gap, out_dir)
        for d in v["drop"]:
            drop_rows.append({"source": str(src), "start_s": d.start, "end_s": d.end, "reason": d.reason,
                              "extra": json.dumps(d.extra) if d.extra else "", "created": now})
        for k, why in rejected:
            drop_rows.append({"source": str(src), "start_s": k.start, "end_s": k.end,
                              "reason": f"NOT CUT ({why}); wanted: {k.reason}", "created": now})
        for p in plans:
            if dry_run:
                row = _row(p, status="dry run")
                log(f"[dry] {src.name} {p.mode:8s} frames {p.start_frame}-{p.end_frame} "
                    f"{p.cut_start_s:.3f}-{p.cut_end_s:.3f} s (wanted {p.wanted.start:.3f}-{p.wanted.end:.3f}, "
                    f"lead {p.lead_s:+.3f}) -> {p.out}")
            else:
                row = cut(p, overwrite=overwrite, crf=crf, preset=preset)
                log(f"{src.name} {p.mode:8s} {row['start_s']}-{row['end_s']} s, {row['n_frames']} frames: "
                    f"{row['status']} -> {p.out.name}")
            if read_clock and not row["status"].startswith("ffmpeg error"):
                from . import timecode
                step = video.frame_table(src).frame_step
                rds = timecode.clock_at(src, [float(row["start_s"]), float(row["end_s"]) - step])
                row["clock_start"] = rds[0].text if rds[0] else ""
                row["clock_end"] = rds[1].text if rds[1] else ""
            index_rows.append(row)
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        _append_csv(out_dir / "clips_index.csv", INDEX_FIELDS, index_rows)
        _append_csv(out_dir / "dropped.csv", DROP_FIELDS, drop_rows)
    return {"clips": index_rows, "dropped": drop_rows, "out_dir": str(out_dir)}
