"""File name conventions of the footage.

Tag recordings carry the tag serial, an optional extra part, the local date and start time, a field and the sequence
number:

- ``CC-11-96-20250203-101236-000-00004.mov``: tag ``CC-11-96``, 2025-02-03 10:12:36, field ``000``, sequence 4;
- ``CC-09-85-HRA-20250102-111224-000-00005.mov``: tag ``CC-09-85``, extra part ``HRA``, 2025-01-02 11:12:24, sequence 5.

The start time is whole seconds (the burned-in clock of CC-09-85 runs 0.4-0.9 s after it). Sequence numbers can
have gaps: a camera that splits a recording session into 4 GB files (CC-09-85, 4K) numbers files, not sessions.
Sensor files: ``20250203-164014-CC-11-96.csv``, ``20250102-225123-CC-09-85-HRA.csv`` (see
:func:`parse_sensor_name`).
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

_TAG = r"(?P<tag>[A-Za-z]{2}-\d+-\d+)(?:-(?P<extra>[A-Za-z][A-Za-z0-9]*))?"
TAG_VIDEO_RE = re.compile(_TAG + r"-(?P<date>\d{8})-(?P<time>\d{6})-(?P<field>\d{3})-(?P<seq>\d{3,6})")
SENSOR_RE = re.compile(r"(?P<date>\d{8})-(?P<time>\d{6})-" + _TAG + r"(?:$|[^0-9A-Za-z])")
DATE_RE = re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})")


def _when(date: str, time: str) -> _dt.datetime | None:
    try:
        return _dt.datetime.strptime(date + time, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def parse_tag_video_name(path) -> dict | None:
    """Fields of a tag recording's file name (``tag``, ``extra`` or None, ``start_local``, ``field``, ``seq``), or
    None if the name does not follow the convention."""
    m = TAG_VIDEO_RE.search(Path(path).stem)
    if not m:
        return None
    start = _when(m["date"], m["time"])
    if start is None:
        return None
    return {"tag": m["tag"], "extra": m["extra"], "start_local": start, "field": m["field"], "seq": int(m["seq"])}


def parse_sensor_name(path) -> dict | None:
    """Fields of a tag sensor file name (``20250102-225123-CC-09-85-HRA.csv``): ``tag``, ``extra``, ``written``."""
    m = SENSOR_RE.search(Path(path).name)
    if not m:
        return None
    when = _when(m["date"], m["time"])
    if when is None:
        return None
    return {"tag": m["tag"], "extra": m["extra"], "written": when}


def start_seconds_of_day(path) -> float | None:
    """Start time in the file name as seconds of the local day (None when the name does not parse)."""
    info = parse_tag_video_name(path)
    if not info:
        return None
    s = info["start_local"]
    return float(s.hour * 3600 + s.minute * 60 + s.second)


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
