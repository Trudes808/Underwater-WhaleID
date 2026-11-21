"""Video reading that is right about time.

Everything here is in *presentation time* in seconds, as the container shows it after its edit list (PyAV / ffmpeg
semantics): frame ``t = pts * time_base``. Frame indices count the frames a player would show, in presentation
order, starting at 0.

- :func:`probe`: metadata (container, tracks, sound, creation time, tags, MP4 edit lists and fragments).
- :func:`frame_table`: presentation times and keyframes of every shown frame, found by demuxing (fast, no decode).
  The header's frame count is not trusted: packets an edit list hides are left out.
- :func:`decode_check`: decode every frame once and count what really decodes (slow: about 1 min per 30 min of
  1080p video with 8 threads).
- :func:`iter_frames`: sequential decoding, optionally sampled at ``fps`` frames per second.
- :class:`VideoReader`: random access by time (keyframe seek, then decode forward to the exact frame).

Copes with fragmented QuickTime files (``moov`` describing only the first second), edit lists that hide frames, and
truncated tails (decode errors are counted, not raised).
"""

from __future__ import annotations

import datetime as _dt
import os
import struct
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import av
import numpy as np

VIDEO_EXTENSIONS = {".mov", ".m4v", ".mp4", ".avi", ".mkv", ".mts", ".lrv"}
DEFAULT_THREADS = 8
_MP4_FAMILY = {".mov", ".m4v", ".mp4", ".m4a", ".3gp", ".lrv"}


@dataclass
class Frame:
    """One decoded frame: index among shown frames, presentation time (s) and the image (H x W x 3 RGB uint8,
    or whatever ``fmt`` asked for)."""

    index: int
    t: float
    image: np.ndarray
    key: bool = False


# ----------------------------------------------------------------------------------------------------------------
# MP4 / QuickTime box structure (edit lists, fragments, truncation) without any external tool
# ----------------------------------------------------------------------------------------------------------------

_EPOCH_1904 = _dt.datetime(1904, 1, 1, tzinfo=_dt.timezone.utc)


def _iter_boxes(fh, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        hdr = fh.read(8)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack(">I4s", hdr)
        hlen = 8
        if size == 1:
            ext = fh.read(8)
            if len(ext) < 8:
                return
            size = struct.unpack(">Q", ext)[0]
            hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen:
            return
        yield typ.decode("latin1"), pos, hlen, size
        pos += size


def _read_full_box(fh, pos: int, hlen: int, size: int, limit: int = 1 << 20) -> bytes:
    fh.seek(pos + hlen)
    return fh.read(min(size - hlen, limit))


def _mp4_time(seconds: int) -> str | None:
    if not seconds:
        return None
    try:
        return (_EPOCH_1904 + _dt.timedelta(seconds=int(seconds))).isoformat()
    except OverflowError:
        return None


def mp4_structure(path: str | os.PathLike) -> dict:
    """Read the MP4 / QuickTime box tree far enough to report fragments, truncation, movie header and per-track edit
    lists. Never raises on a damaged file: problems are listed under ``problems``."""
    path = Path(path)
    file_size = path.stat().st_size
    out: dict = {"top_level": {}, "fragmented": False, "n_moof": 0, "truncated": False, "problems": [],
                 "movie": {}, "tracks": []}
    try:
        with open(path, "rb") as fh:
            last_end = 0
            for typ, pos, hlen, size in _iter_boxes(fh, 0, file_size):
                out["top_level"][typ] = out["top_level"].get(typ, 0) + 1
                last_end = pos + size
                if typ == "moof":
                    out["n_moof"] += 1
                if typ == "moov":
                    _parse_moov(fh, pos + hlen, min(pos + size, file_size), out)
            if last_end > file_size:
                out["truncated"] = True
                out["problems"].append(f"last box ends at byte {last_end}, file has {file_size} bytes")
    except OSError as exc:
        out["problems"].append(f"read error: {exc}")
    out["fragmented"] = out["n_moof"] > 0 or out.get("has_mvex", False)
    return out


def _parse_moov(fh, start: int, end: int, out: dict) -> None:
    for typ, pos, hlen, size in _iter_boxes(fh, start, end):
        if typ == "mvhd":
            b = _read_full_box(fh, pos, hlen, size)
            ver = b[0]
            if ver == 1:
                ctime, mtime, ts, dur = struct.unpack(">QQIQ", b[4:32])
            else:
                ctime, mtime, ts, dur = struct.unpack(">IIII", b[4:20])
            out["movie"] = {"timescale": ts, "duration_s": dur / ts if ts else None,
                            "creation_time": _mp4_time(ctime), "modification_time": _mp4_time(mtime)}
        elif typ == "mvex":
            out["has_mvex"] = True
        elif typ == "trak":
            out["tracks"].append(_parse_trak(fh, pos + hlen, pos + size, out["movie"].get("timescale")))


def _parse_trak(fh, start: int, end: int, movie_ts: int | None) -> dict:
    trak: dict = {"track_id": None, "handler": None, "media_timescale": None, "media_duration_s": None,
                  "edit_list": []}
    raw_edits = []

    def walk(s, e):
        for typ, pos, hlen, size in _iter_boxes(fh, s, e):
            if typ in ("edts", "mdia", "minf", "stbl"):
                walk(pos + hlen, pos + size)
            elif typ == "tkhd":
                b = _read_full_box(fh, pos, hlen, size)
                if b[0] == 1:
                    trak["track_id"] = struct.unpack(">I", b[20:24])[0]
                else:
                    trak["track_id"] = struct.unpack(">I", b[12:16])[0]
            elif typ == "mdhd":
                b = _read_full_box(fh, pos, hlen, size)
                if b[0] == 1:
                    ts, dur = struct.unpack(">IQ", b[20:32])
                else:
                    ts, dur = struct.unpack(">II", b[12:20])
                trak["media_timescale"] = ts
                trak["media_duration_s"] = dur / ts if ts else None
            elif typ == "hdlr":
                b = _read_full_box(fh, pos, hlen, size)
                trak["handler"] = b[8:12].decode("latin1")
            elif typ == "elst":
                b = _read_full_box(fh, pos, hlen, size)
                ver = b[0]
                n = struct.unpack(">I", b[4:8])[0]
                off = 8
                for _ in range(min(n, 1000)):
                    if ver == 1:
                        seg, mt = struct.unpack(">Qq", b[off:off + 16])
                        off += 16
                    else:
                        seg, mt = struct.unpack(">Ii", b[off:off + 8])
                        off += 8
                    rate_i, rate_f = struct.unpack(">hH", b[off:off + 4])
                    off += 4
                    raw_edits.append((seg, mt, rate_i + rate_f / 65536))

    walk(start, end)
    mts = trak["media_timescale"]
    for seg, mt, rate in raw_edits:
        trak["edit_list"].append({
            "segment_duration_s": seg / movie_ts if movie_ts else None,
            "media_time_s": (mt / mts if mts else None) if mt >= 0 else None,  # None = empty edit (a gap)
            "rate": rate,
        })
    return trak


# ----------------------------------------------------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------------------------------------------------

def _rate(r) -> float | None:
    try:
        return float(r) if r else None
    except (TypeError, ZeroDivisionError):
        return None


def probe(path: str | os.PathLike) -> dict:
    """Metadata of a video file from its headers (no decoding). Durations here are what the headers say; use
    :func:`frame_table` / :func:`decode_check` for what is really there."""
    path = Path(path)
    st = path.stat()
    out: dict = {
        "path": str(path), "name": path.name, "size_bytes": st.st_size,
        "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }
    with av.open(str(path)) as c:
        out["container"] = c.format.name
        out["duration_header_s"] = c.duration / 1e6 if c.duration else None
        out["start_time_s"] = c.start_time / 1e6 if c.start_time else 0.0
        out["bit_rate"] = c.bit_rate
        out["tags"] = dict(c.metadata)
        streams = []
        for s in c.streams:
            cc = s.codec_context
            info = {
                "index": s.index, "type": s.type, "codec": cc.name if cc else None,
                "profile": getattr(s, "profile", None), "time_base": str(s.time_base),
                "frames_header": s.frames or None,
                "duration_s": float(s.duration * s.time_base) if s.duration and s.time_base else None,
                "start_s": float(s.start_time * s.time_base) if s.start_time is not None and s.time_base else None,
                "bit_rate": cc.bit_rate if cc else None,
                "tags": dict(s.metadata),
            }
            if s.type == "video":
                info.update(width=cc.width, height=cc.height, pix_fmt=cc.pix_fmt,
                            avg_rate=_rate(s.average_rate), base_rate=_rate(s.base_rate))
            elif s.type == "audio":
                info.update(sample_rate=cc.sample_rate, channels=cc.channels,
                            layout=cc.layout.name if cc.layout else None)
            streams.append(info)
        out["streams"] = streams
    out["has_video"] = any(s["type"] == "video" for s in streams)
    out["has_audio"] = any(s["type"] == "audio" for s in streams)
    creation = out["tags"].get("creation_time")
    if path.suffix.lower() in _MP4_FAMILY:
        out["mp4"] = mp4_structure(path)
        creation = creation or out["mp4"].get("movie", {}).get("creation_time")
        out["edit_lists"] = {t.get("handler"): t["edit_list"] for t in out["mp4"]["tracks"] if t["edit_list"]}
    out["creation_time"] = creation
    return out


# ----------------------------------------------------------------------------------------------------------------
# frame table (demux only)
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class FrameTable:
    """Presentation times of the shown frames of one video track, from demuxing (no decoding)."""

    times: np.ndarray                    # seconds, sorted, one per shown frame
    keyframe_times: np.ndarray           # seconds, shown keyframes
    hidden_keyframe_times: np.ndarray    # keyframes of packets an edit list hides (decodable, not shown)
    n_packets: int                       # all video packets in the file
    n_hidden: int                        # packets an edit list hides
    n_no_pts: int                        # packets without a timestamp (left out)
    time_base: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(len(self.times))

    @property
    def duration(self) -> float:
        """Shown duration: last frame time minus first, plus one typical frame step."""
        if len(self.times) == 0:
            return 0.0
        return float(self.times[-1] - self.times[0] + self.frame_step)

    @property
    def frame_step(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return float(np.median(np.diff(self.times)))

    def index_at(self, t: float) -> int:
        """Index of the shown frame nearest to time ``t``."""
        if len(self.times) == 0:
            raise ValueError("no frames")
        i = int(np.searchsorted(self.times, t))
        if i <= 0:
            return 0
        if i >= len(self.times):
            return len(self.times) - 1
        return i if (self.times[i] - t) < (t - self.times[i - 1]) else i - 1

    def index_of(self, t: float, tol: float = 1e-4) -> int | None:
        """Index of the frame whose time is ``t`` (within ``tol``), else None."""
        i = self.index_at(t)
        return i if abs(self.times[i] - t) <= tol else None

    def keyframe_at_or_before(self, t: float, include_hidden: bool = False) -> float | None:
        kf = self.keyframe_times
        if include_hidden and len(self.hidden_keyframe_times):
            kf = np.sort(np.concatenate([kf, self.hidden_keyframe_times]))
        i = bisect_left(list(kf), t + 1e-6)
        return float(kf[i - 1]) if i > 0 else None

    def rate_summary(self) -> dict:
        """Frame rate figures: mean rate, step min / median / max, and whether the timestamps are constant."""
        if len(self.times) < 2:
            return {"mean_fps": None, "constant": None}
        d = np.diff(self.times)
        med = float(np.median(d))
        spread = float((d.max() - d.min()) / med) if med > 0 else None
        gaps = int((d > 1.5 * med).sum())
        return {
            "mean_fps": float((len(self.times) - 1) / (self.times[-1] - self.times[0])),
            "median_fps": 1.0 / med if med > 0 else None,
            "step_min_s": float(d.min()), "step_median_s": med, "step_max_s": float(d.max()),
            "n_gaps_over_1p5_steps": gaps,
            # constant: every step within 1 % of the median (a few doubled steps = dropped frames, listed as gaps)
            "constant": bool(np.mean(np.abs(d - med) <= 0.01 * med) > 0.999),
            "jitter_rel": spread,
        }

    def keyframe_summary(self) -> dict:
        kf = self.keyframe_times
        if len(kf) < 2:
            return {"n_keyframes": int(len(kf)), "spacing_median_s": None, "spacing_max_s": None}
        d = np.diff(kf)
        return {"n_keyframes": int(len(kf)), "n_hidden_keyframes": int(len(self.hidden_keyframe_times)),
                "spacing_min_s": float(d.min()), "spacing_median_s": float(np.median(d)),
                "spacing_max_s": float(d.max()), "first_keyframe_s": float(kf[0])}


_TABLE_CACHE: dict = {}


def _open(path, threads: int = DEFAULT_THREADS, options: dict | None = None):
    c = av.open(str(path), options=options or {})
    if c.streams.video:
        v = c.streams.video[0]
        v.thread_type = "AUTO"
        # with torchvision loaded in the process, threaded HEVC decoding of the tag's .mov deadlocks (PyAV 19 +
        # torchvision 0.21, seen 2026-11-20); one decoding thread is safe and still ~200 frames/s at 1080p
        v.codec_context.thread_count = 1 if torch_loaded() else threads
    return c


def frame_table(path: str | os.PathLike, stream: int = 0) -> FrameTable:
    """Demux the video track and return the presentation times of the shown frames (cached per file version)."""
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns, stream)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    times, kfs, hidden_kfs = [], [], []
    n_packets = n_hidden = n_no_pts = 0
    errors = []
    with av.open(str(path)) as c:
        v = c.streams.video[stream]
        tb = v.time_base
        try:
            for pkt in c.demux(v):
                if pkt.size == 0:
                    continue
                n_packets += 1
                if pkt.pts is None:
                    n_no_pts += 1
                    continue
                t = float(pkt.pts * tb)
                if pkt.is_discard:
                    n_hidden += 1
                    if pkt.is_keyframe:
                        hidden_kfs.append(t)
                    continue
                times.append(t)
                if pkt.is_keyframe:
                    kfs.append(t)
        except av.error.FFmpegError as exc:  # truncated or damaged tail: keep what was read
            errors.append(f"demux stopped at packet {n_packets}: {exc}")
    table = FrameTable(times=np.array(sorted(times)), keyframe_times=np.array(sorted(kfs)),
                       hidden_keyframe_times=np.array(sorted(hidden_kfs)), n_packets=n_packets, n_hidden=n_hidden,
                       n_no_pts=n_no_pts, time_base=str(tb), extra={"demux_errors": errors})
    _TABLE_CACHE[key] = table
    return table


def keyframes(path: str | os.PathLike, include_hidden: bool = False) -> list[float]:
    """Presentation times (s) of the keyframes; ``include_hidden`` adds those an edit list hides."""
    tab = frame_table(path)
    kf = tab.keyframe_times
    if include_hidden:
        kf = np.sort(np.concatenate([kf, tab.hidden_keyframe_times]))
    return [float(x) for x in kf]


# ----------------------------------------------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------------------------------------------

def _decode_frames(c, v, errors: list | None = None):
    """Decode every frame of stream ``v`` in container ``c``, skipping packets that fail to decode."""
    try:
        for pkt in c.demux(v):
            try:
                for f in pkt.decode():
                    yield f
            except av.error.FFmpegError as exc:
                if errors is not None:
                    errors.append(f"t={float(pkt.pts * v.time_base) if pkt.pts is not None else None}: {exc}")
    except av.error.FFmpegError as exc:
        if errors is not None:
            errors.append(f"demux: {exc}")


def _to_array(f, fmt: str, size: tuple[int, int] | None):
    if size is not None:
        return f.to_ndarray(format=fmt, width=size[0], height=size[1])
    return f.to_ndarray(format=fmt)


def decode_check(path: str | os.PathLike, grab_times: Sequence[float] = (), fmt: str = "rgb24",
                 size: tuple[int, int] | None = None, threads: int = DEFAULT_THREADS) -> dict:
    """Decode every frame once. Returns the number of decodable frames, first / last time, decode errors and, if
    ``grab_times`` is given, the frames nearest to those times (as :class:`Frame`), all in one pass."""
    targets = sorted(grab_times)
    grabbed: list[Frame] = []
    errors: list[str] = []
    n = 0
    first = last = None
    prev = None
    ti = 0
    with _open(path, threads) as c:
        v = c.streams.video[0]
        for f in _decode_frames(c, v, errors):
            t = float(f.time) if f.time is not None else None
            if t is None:
                continue
            if first is None:
                first = t
            # nearest-frame pick for each target
            while ti < len(targets) and t >= targets[ti]:
                if prev is not None and abs(prev[1] - targets[ti]) <= abs(t - targets[ti]):
                    pick = prev
                else:
                    pick = (n, t, f)
                if not grabbed or grabbed[-1].index != pick[0]:
                    grabbed.append(Frame(pick[0], pick[1], _to_array(pick[2], fmt, size), bool(pick[2].key_frame)))
                ti += 1
            prev = (n, t, f)
            last = t
            n += 1
        if ti < len(targets) and prev is not None and (not grabbed or grabbed[-1].index != prev[0]):
            grabbed.append(Frame(prev[0], prev[1], _to_array(prev[2], fmt, size), bool(prev[2].key_frame)))
    return {"n_decoded": n, "first_t": first, "last_t": last, "n_errors": len(errors), "errors": errors[:20],
            "frames": grabbed}


def iter_frames(path: str | os.PathLike, fps: float | None = None, start: float = 0.0, end: float | None = None,
                fmt: str = "rgb24", size: tuple[int, int] | None = None,
                threads: int = DEFAULT_THREADS) -> Iterator[Frame]:
    """Decode frames in order. With ``fps`` set, yield the frame nearest to each time ``start + k / fps``
    (no frame twice); otherwise every frame. ``start`` > 0 seeks to the keyframe before it first.
    Index = position among the shown frames (from :func:`frame_table`)."""
    tab = frame_table(path)
    with _open(path, threads) as c:
        v = c.streams.video[0]
        if start > 0:
            kf = tab.keyframe_at_or_before(start, include_hidden=True)
            if kf is not None and kf > 0:
                c.seek(int(kf / v.time_base), stream=v, backward=True, any_frame=False)
        step = 1.0 / fps if fps else None
        target = start
        prev = None
        last_yield = -1
        for f in _decode_frames(c, v):
            if f.time is None:
                continue
            t = float(f.time)
            if t < start - 1e-6:
                prev = None
                continue
            if end is not None and t > end + 1e-6:
                break
            idx = tab.index_at(t)
            if step is None:
                yield Frame(idx, t, _to_array(f, fmt, size), bool(f.key_frame))
                continue
            while t >= target - 1e-9:
                pick = (idx, t, f)
                if prev is not None and abs(prev[1] - target) < abs(t - target):
                    pick = prev
                if pick[0] != last_yield:
                    yield Frame(pick[0], pick[1], _to_array(pick[2], fmt, size), bool(pick[2].key_frame))
                    last_yield = pick[0]
                target += step
                if end is not None and target > end + 1e-6:
                    return
            prev = (idx, t, f)


class VideoReader:
    """Random access by time: keyframe seek, then decode forward to the exact frame. Requests in increasing time
    close together are served by decoding forward without a new seek."""

    def __init__(self, path: str | os.PathLike, threads: int = DEFAULT_THREADS, fmt: str = "rgb24",
                 size: tuple[int, int] | None = None, forward_limit_s: float = 4.0):
        self.path = Path(path)
        self.table = frame_table(self.path)
        self.fmt, self.size, self.forward_limit_s = fmt, size, forward_limit_s
        self._c = _open(self.path, threads)
        self._v = self._c.streams.video[0]
        self._gen = None
        self._pos = None  # time of the last decoded frame

    def close(self):
        self._c.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _seek(self, t: float):
        kf = self.table.keyframe_at_or_before(t, include_hidden=True)
        target = 0 if kf is None else int(kf / self._v.time_base)
        try:
            self._c.seek(max(target, 0) if kf is None or kf >= 0 else target, stream=self._v, backward=True,
                         any_frame=False)
        except av.error.FFmpegError:
            self._c.seek(0, stream=self._v, backward=True)
        self._gen = _decode_frames(self._c, self._v)
        self._pos = None

    def frame_at(self, t: float) -> Frame | None:
        """The shown frame nearest to time ``t`` (clamped to the video), or None if it cannot be decoded."""
        idx = self.table.index_at(t)
        return self.frame_by_index(idx)

    def frame_by_index(self, idx: int) -> Frame | None:
        want = float(self.table.times[idx])
        need_seek = (self._gen is None or self._pos is None or want <= self._pos - 1e-6
                     or want - self._pos > self.forward_limit_s)
        for attempt in range(2):
            if need_seek:
                self._seek(want)
            for f in self._gen:
                if f.time is None:
                    continue
                ft = float(f.time)
                self._pos = ft
                if ft >= want - 1e-6:
                    if ft > want + 1e-4 and attempt == 0:
                        break  # overshot (seek landed after the frame): retry from further back
                    return Frame(self.table.index_at(ft), ft, _to_array(f, self.fmt, self.size), bool(f.key_frame))
            else:
                return None  # end of stream before the frame: undecodable tail
            need_seek = True
            want_seek = want - 2 * self.forward_limit_s
            self._seek(max(want_seek, 0.0))
            need_seek = False
        return None

    def frames_at(self, times: Sequence[float]) -> list[Frame | None]:
        """Frames nearest to each of ``times`` (any order; served in time order)."""
        order = np.argsort(times)
        out: list[Frame | None] = [None] * len(times)
        for i in order:
            out[i] = self.frame_at(float(times[i]))
        return out


def read_frame_at(path: str | os.PathLike, t: float, fmt: str = "rgb24") -> Frame | None:
    with VideoReader(path, fmt=fmt) as r:
        return r.frame_at(t)


def is_video_file(path: str | os.PathLike) -> bool:
    p = Path(path)
    return p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith("._")


def iter_frames_at(path: str | os.PathLike, indices: Sequence[int], fmt: str = "rgb24",
                   size: tuple[int, int] | None = None, threads: int = DEFAULT_THREADS,
                   max_forward_s: float = 3.0) -> Iterator[Frame]:
    """Decode the shown frames with the given indices, in increasing order, converting only those frames.

    Decodes forward through short gaps and seeks to the keyframe before a target that is more than
    ``max_forward_s`` ahead. Frames that cannot be decoded are skipped (the caller sees which indices came back)."""
    tab = frame_table(path)
    want = sorted({int(i) for i in indices if 0 <= int(i) < tab.n_frames})
    if not want:
        return
    with _open(path, threads) as c:
        v = c.streams.video[0]
        k = 0
        pos = None   # time of the last decoded frame
        gen = None
        while k < len(want):
            target_t = float(tab.times[want[k]])
            if gen is None or pos is None or target_t - pos > max_forward_s or target_t < pos - 1e-6:
                kf = tab.keyframe_at_or_before(target_t, include_hidden=True)
                try:
                    c.seek(int(kf / v.time_base) if kf is not None and kf > 0 else 0, stream=v, backward=True,
                           any_frame=False)
                except av.error.FFmpegError:
                    c.seek(0, stream=v, backward=True)
                gen = _decode_frames(c, v)
                pos = None
            for f in gen:
                if f.time is None:
                    continue
                ft = float(f.time)
                pos = ft
                idx = tab.index_at(ft)
                while k < len(want) and want[k] < idx:
                    k += 1          # skipped (undecodable or seek landed late)
                if k >= len(want):
                    return
                if idx == want[k]:
                    yield Frame(idx, ft, _to_array(f, fmt, size), bool(f.key_frame))
                    k += 1
                    if k >= len(want):
                        return
                    if float(tab.times[want[k]]) - ft > max_forward_s:
                        break
            else:
                return      # end of stream


def _decode_child(q, kind: str, path: str, arg, fmt: str, size):
    try:
        if kind == "at":
            it = iter_frames_at(path, arg, fmt=fmt, size=size)
        else:
            it = iter_frames(path, fps=arg.get("fps"), start=arg.get("start", 0.0), end=arg.get("end"), fmt=fmt,
                             size=size)
        for fr in it:
            q.put((fr.index, fr.t, fr.image, fr.key))
    except Exception as exc:          # reported to the parent
        q.put(("error", repr(exc)))
    finally:
        q.put(None)


def iter_frames_isolated(path: str | os.PathLike, indices: Sequence[int] | None = None, fps: float | None = None,
                         start: float = 0.0, end: float | None = None, fmt: str = "rgb24",
                         size: tuple[int, int] | None = None, prefetch: int = 8) -> Iterator[Frame]:
    """:func:`iter_frames_at` (``indices`` given) or :func:`iter_frames`, decoded with all threads in a spawned
    process that never imports torch. Used by model code: with torchvision loaded, this process may only decode
    with one thread (see :func:`_open`), too slow for 4K footage."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=prefetch)
    if indices is not None:
        kind, arg = "at", [int(i) for i in indices]
    else:
        kind, arg = "seq", {"fps": fps, "start": start, "end": end}
    p = ctx.Process(target=_decode_child, args=(q, kind, str(path), arg, fmt, size), daemon=True)
    p.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            if item[0] == "error":
                raise RuntimeError(f"decoding {path} failed: {item[1]}")
            yield Frame(int(item[0]), float(item[1]), item[2], bool(item[3]))
    finally:
        if p.is_alive():
            p.terminate()
        p.join(timeout=5)


def torch_loaded() -> bool:
    """True when torchvision is imported in this process (see :func:`_open`)."""
    import sys
    return "torchvision" in sys.modules
