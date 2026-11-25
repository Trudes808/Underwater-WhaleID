"""WP19: what the media really are (codec, codec tag, size, frame rate, container), measured from the files' headers
with PyAV, for EXPORT_FORMAT.md "Media facts". Prints a markdown table per group.

    .venv/bin/python experiments/12_export/media_facts.py
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import paths  # noqa: E402

GROUPS = {
    # "._*" files are macOS resource forks (AppleDouble) copied along, not videos
    "source recordings": sorted(p for p in paths.footage_dir().rglob("*") if p.suffix.lower() in (".mov", ".m4v", ".mp4")
                                and not p.name.startswith("._")),
    "cut clips (underwater_cache/clips)": sorted(p for p in paths.cache_dir("clips").rglob("*")
                                                 if p.suffix.lower() in (".mov", ".m4v", ".mp4")),
    "review videos (review_sheets/underwater)": sorted(paths.sheets_dir().glob("*.mp4")),
    "export clips (export/web_v1)": sorted((paths.cache_dir("export") / "web_v1").rglob("*.mp4")),
}


def facts(p: Path):
    try:
        with av.open(str(p)) as c:
            v = c.streams.video[0]
            cc = v.codec_context
            return (cc.name, str(getattr(cc, "codec_tag", "") or ""), f"{cc.width}x{cc.height}", cc.pix_fmt,
                    round(float(v.average_rate or 0), 3), c.format.name.split(",")[0])
    except Exception as e:  # noqa: BLE001
        return ("error", type(e).__name__, "", "", 0, "")


def browser(codec: str, tag: str, pix: str) -> str:
    if codec == "h264" and pix in ("yuv420p", "yuvj420p"):
        return "yes (H.264 8-bit 4:2:0)"
    if codec == "h264":
        return f"doubtful (H.264 {pix})"
    if codec == "hevc" and tag == "hvc1":
        return "Safari only (HEVC hvc1)"
    if codec == "hevc":
        return f"no (HEVC {tag}: even Safari wants hvc1; needs a re-tag or re-encode)"
    return "no / unknown"


for g, files in GROUPS.items():
    rows = collections.Counter()
    where = collections.defaultdict(set)
    for p in files:
        f = facts(p)
        key = (p.suffix.lower(),) + f
        rows[key] += 1
        top = p.relative_to(paths.footage_dir()).parts[0] if g.startswith("source") else ""
        where[key].add(top)
    print(f"\n### {g}: {len(files)} files\n")
    print("| ext | codec | tag | size | pixel format | fps | container | files | browser plays as is | where |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, n in sorted(rows.items(), key=lambda x: -x[1]):
        ext, codec, tag, size, pix, fps, cont = k
        print(f"| {ext} | {codec} | {tag} | {size} | {pix} | {fps} | {cont} | {n} | {browser(codec, tag, pix)} | "
              f"{'; '.join(sorted(where[k]))} |")
