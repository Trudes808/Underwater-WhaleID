"""Inventory of one deployment's video files: container times, packet counts, frame times, first / last pts.

Writes underwater_cache/tagdata/<deployment>/videos.csv. Skips files whose size changed in the last 60 s.

    .venv/bin/python experiments/01_metadata/probe_videos.py "2025-2-3 Mini-mother 11-96"
"""
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import tagdata  # noqa: E402

VIDEO_ROOT = Path("/home/sat3737/NOAA_whales/underwater_video")


def stable(p: Path, wait=60.0) -> bool:
    s0, m0 = p.stat().st_size, p.stat().st_mtime
    return (time.time() - m0) > wait and p.stat().st_size == s0


def probe(p: Path) -> dict:
    c = av.open(str(p))
    row = {"file": p.name, "size_bytes": p.stat().st_size, "format_duration_s": (c.duration or 0) / 1e6,
           "encoder": c.metadata.get("encoder", ""), "creation_time": c.metadata.get("creation_time", "")}
    v = c.streams.video[0]
    row.update(codec=v.codec_context.name, width=v.codec_context.width, height=v.codec_context.height,
               avg_rate=float(v.average_rate) if v.average_rate else np.nan)
    a = c.streams.audio[0] if c.streams.audio else None
    if a is not None:
        row.update(audio_codec=a.codec_context.name, audio_rate=a.codec_context.sample_rate,
                   audio_channels=a.codec_context.channels)
    vp, ap = [], []
    streams = [v] + ([a] if a is not None else [])
    for pkt in c.demux(streams):
        if pkt.pts is None:
            continue
        if pkt.stream.type == "video":
            vp.append(float(pkt.pts * pkt.time_base))
        else:
            ap.append((float(pkt.pts * pkt.time_base), float(pkt.duration * pkt.time_base) if pkt.duration else 0))
    vp = np.sort(np.array(vp))
    d = np.diff(vp)
    row.update(n_video_frames=len(vp), first_pts=vp[0], last_pts=vp[-1], video_span_s=vp[-1] - vp[0],
               median_frame_dt=float(np.median(d)), max_frame_dt=float(d.max()), n_dt_gt_1p5=int((d > 1.5 * np.median(d)).sum()),
               fps_from_pts=(len(vp) - 1) / (vp[-1] - vp[0]))
    if ap:
        ap = np.array(ap)
        row.update(audio_first_pts=ap[0, 0], audio_end=ap[-1, 0] + ap[-1, 1])
    c.close()
    start, seq = tagdata.video_start_from_name(p.name)
    row.update(name_start_utc=str(start), seq=seq)
    return row


if __name__ == "__main__":
    dep = sys.argv[1] if len(sys.argv) > 1 else "2025-2-3 Mini-mother 11-96"
    files = sorted(f for f in (VIDEO_ROOT / dep / "Video").iterdir()
                   if f.suffix.lower() in (".mov", ".m4v") and not f.name.startswith("._"))
    rows = []
    for f in files:
        if not stable(f):
            print("skip (still copying)", f.name)
            continue
        t = time.time()
        rows.append(probe(f))
        print(f.name, f"{time.time() - t:.0f}s", rows[-1]["n_video_frames"], rows[-1]["video_span_s"])
    out = tagdata.cache_dir(dep) / "videos.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).T.to_string())
