"""Decode the sound of a time range to a mono float32 array at the track's native sample rate."""

from __future__ import annotations

import os
from dataclasses import dataclass

import av
import numpy as np


@dataclass
class Audio:
    samples: np.ndarray     # mono float32, nominally in [-1, 1]
    sample_rate: int
    t0: float               # presentation time (s) of samples[0]
    channels: int           # channels in the file before the mono mix
    codec: str

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0


def has_audio(path: str | os.PathLike) -> bool:
    with av.open(str(path)) as c:
        return bool(c.streams.audio)


def read_audio(path: str | os.PathLike, start: float = 0.0, end: float | None = None,
               stream: int = 0) -> Audio | None:
    """Sound between ``start`` and ``end`` seconds (presentation time) as a mono mix at the native rate.

    Returns None when the file has no sound track. Packets that fail to decode are skipped (their samples are
    missing, not zero-filled); ``t0`` is the time of the first returned sample."""
    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None
        a = c.streams.audio[stream]
        sr = a.codec_context.sample_rate
        ch = a.codec_context.channels
        codec = a.codec_context.name
        if start > 0:
            try:
                c.seek(int(max(start - 1.0, 0) / a.time_base), stream=a, backward=True)
            except av.error.FFmpegError:
                c.seek(0, stream=a)
        chunks: list[np.ndarray] = []
        t_first = None
        try:
            for pkt in c.demux(a):
                try:
                    frames = pkt.decode()
                except av.error.FFmpegError:
                    continue
                for f in frames:
                    if f.pts is None:
                        continue
                    ft = float(f.pts * f.time_base)
                    arr = f.to_ndarray()
                    if arr.dtype.kind in "iu":
                        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
                    arr = arr.astype(np.float32)
                    # planar: (channels, n); packed: (1, n * channels)
                    if f.format.is_planar:
                        mono = arr.mean(axis=0)
                    else:
                        mono = arr.reshape(-1, len(f.layout.channels)).mean(axis=1)
                    n = len(mono)
                    f_end = ft + n / sr
                    if f_end <= start:
                        continue
                    if end is not None and ft >= end:
                        raise StopIteration
                    i0 = max(0, int(round((start - ft) * sr)))
                    i1 = n if end is None else min(n, int(round((end - ft) * sr)))
                    if i1 <= i0:
                        continue
                    if t_first is None:
                        t_first = ft + i0 / sr
                    chunks.append(mono[i0:i1])
        except StopIteration:
            pass
        except av.error.FFmpegError:
            pass  # truncated tail: keep what was decoded
    samples = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return Audio(samples=samples, sample_rate=sr, t0=t_first if t_first is not None else start, channels=ch,
                 codec=codec)
