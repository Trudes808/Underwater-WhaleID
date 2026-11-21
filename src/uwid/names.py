"""File name conventions of the footage.

Tag recordings: ``CC-11-96-20250203-101236-000-00004.mov`` = tag serial ``CC-11-96``, local date and start time
``2025-02-03 10:12:36``, a field ``000`` and the sequence number ``00004``. Sensor files: ``20250203-164014-CC-11-96.csv``.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

TAG_VIDEO_RE = re.compile(
    r"(?P<tag>[A-Za-z]{2}-\d+-\d+)-(?P<date>\d{8})-(?P<time>\d{6})-(?P<field>\d{3})-(?P<seq>\d{3,6})")
DATE_RE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})")


def parse_tag_video_name(path) -> dict | None:
    """Fields of a tag recording's file name, or None if the name does not follow the convention."""
    m = TAG_VIDEO_RE.search(Path(path).stem)
    if not m:
        return None
    try:
        start = _dt.datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return {"tag": m["tag"], "start_local": start, "field": m["field"], "seq": int(m["seq"])}


def date_hint(path) -> _dt.date | None:
    """The recording date the file name or its folder names carry (tag name first, then ``YYYY-M-D`` patterns)."""
    info = parse_tag_video_name(path)
    if info:
        return info["start_local"].date()
    for part in [Path(path).name] + [p.name for p in Path(path).parents]:
        m = DATE_RE.search(part)
        if m:
            try:
                return _dt.date(int(m["y"]), int(m["m"]), int(m["d"]))
            except ValueError:
                continue
    return None
