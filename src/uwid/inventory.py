"""Inventory of a footage folder.

Walks the folder, pairs the ``.mov`` / ``.m4v`` forms of the same recording, lists the sensor files (sidecars) of
each deployment folder, probes every video, decodes it once (frames really decodable, contact-sheet frames, clock) and
writes ``<cache>/inventory/inventory.json`` and ``inventory.md``, plus one contact sheet per video to
``<sheets>/inventory_<stem>_<ext>.jpg``.

Files modified less than ``min_age_s`` seconds ago (still being copied) are skipped and listed as skipped. Entries of
an earlier run are reused when the file's size and modification time are unchanged.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from pathlib import Path

import numpy as np

from . import names, paths, sheets, timecode, video

DEBRIS = {".DS_Store", "Thumbs.db", "desktop.ini"}
VIDEO_DIR_NAMES = {"video", "videos"}


def is_debris(p: Path) -> bool:
    return p.name.startswith("._") or p.name in DEBRIS


def _deployment_dir(video_path: Path, root: Path) -> Path:
    parent = video_path.parent
    if parent.name.lower() in VIDEO_DIR_NAMES and parent != root:
        return parent.parent
    return parent


def _sidecars(dep: Path, root: Path) -> list[dict]:
    files = []
    it = dep.iterdir() if dep == root else dep.rglob("*")
    for p in sorted(it):
        if p.is_file() and not is_debris(p) and not video.is_video_file(p):
            st = p.stat()
            files.append({"path": str(p), "name": p.name, "size_bytes": st.st_size,
                          "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
    return files


def sheet_path(path: Path) -> Path:
    stem = path.stem.replace(" ", "_")
    return paths.sheets_dir() / f"inventory_{stem}_{path.suffix.lstrip('.').lower()}.jpg"


def describe_video(path: Path, n_tiles: int = 40, make_sheet: bool = True, decode: bool = True) -> dict:
    """Probe + frame table + one full decode (count, sheet frames, clock) of one video."""
    t0 = time.time()
    st = path.stat()
    pr = video.probe(path)
    entry: dict = {"path": str(path), "name": path.name, "form": path.suffix.lower().lstrip("."),
                   "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "probe": pr}
    vs = next((s for s in pr["streams"] if s["type"] == "video"), None)
    au = next((s for s in pr["streams"] if s["type"] == "audio"), None)
    entry["video"] = ({"codec": vs["codec"], "width": vs["width"], "height": vs["height"], "pix_fmt": vs["pix_fmt"],
                       "frames_header": vs["frames_header"], "duration_header_s": vs["duration_s"]} if vs else None)
    entry["audio"] = ({"codec": au["codec"], "sample_rate": au["sample_rate"], "channels": au["channels"],
                       "start_s": au["start_s"], "duration_header_s": au["duration_s"]} if au else None)
    entry["creation_time"] = pr.get("creation_time")
    entry["fragmented"] = pr.get("mp4", {}).get("fragmented")
    entry["edit_lists"] = pr.get("edit_lists")
    entry["name_fields"] = None
    info = names.parse_tag_video_name(path)
    if info:
        entry["name_fields"] = {**info, "start_local": info["start_local"].isoformat()}
    if vs is None:
        entry["status"] = "no video track"
        return entry
    tab = video.frame_table(path)
    entry["demux"] = {"n_packets": tab.n_packets, "n_hidden_by_edit_list": tab.n_hidden, "n_no_pts": tab.n_no_pts,
                      "n_frames_shown": tab.n_frames, "first_t": float(tab.times[0]) if tab.n_frames else None,
                      "last_t": float(tab.times[-1]) if tab.n_frames else None,
                      "duration_s": tab.duration, "errors": tab.extra.get("demux_errors")}
    entry["rate"] = tab.rate_summary()
    entry["keyframes"] = tab.keyframe_summary()
    grab = list(np.linspace(float(tab.times[0]), float(tab.times[-1]), n_tiles)) if (make_sheet and tab.n_frames) else []
    if decode:
        dc = video.decode_check(path, grab_times=grab)
        frames = dc.pop("frames")
        entry["decode"] = dc
    else:
        with video.VideoReader(path) as r:
            frames = [f for f in r.frames_at(grab) if f is not None]
    # burned-in clock on the sheet frames
    try:
        gs = timecode.load_glyphs()
    except FileNotFoundError:
        gs = None
    clock = []
    if gs is not None:
        hint = names.date_hint(path)
        for f in frames:
            rd = timecode.read_clock(f.image, gs, hint)
            clock.append({"t": f.t, "text": rd.text if rd else None,
                          "clock_s": rd.seconds_of_day if rd else None})
        ok = [c for c in clock if c["clock_s"] is not None]
        entry["clock"] = {"n_sampled": len(clock), "n_read": len(ok),
                          "first": ok[0] if ok else None, "last": ok[-1] if ok else None,
                          "samples": [[round(c["t"], 4), c["clock_s"]] for c in clock]}
        if len(ok) >= 2:
            t = np.array([c["t"] for c in ok])
            cs = np.array([c["clock_s"] for c in ok])
            fit = timecode.robust_line_fit(t, cs)
            fit.pop("residuals"), fit.pop("inliers")
            entry["clock"]["fit"] = fit
            entry["clock"]["clock_at_t0_s"] = fit["intercept"]   # seconds of the day
            entry["clock"]["monotonic"] = bool((np.diff(cs) >= 0).all())
            if info:
                name_sod = info["start_local"].hour * 3600 + info["start_local"].minute * 60 + info["start_local"].second
                entry["clock"]["clock_at_t0_minus_name_start_s"] = float(fit["intercept"] - name_sod)
    if make_sheet and frames:
        out = sheet_path(path)
        caps = [c["text"] or "no clock" for c in clock] if clock else None
        small = [f.image[::2, ::2] for f in frames]
        title = f"{path.name}  {len(frames)} frames over {tab.duration:.1f} s"
        sheets.make_sheet(small, [f"{sheets.format_time(f.t)}  #{f.index}" for f in frames], caps, cols=8,
                          tile_width=320, title=title, out=out)
        entry["sheet"] = str(out)
    entry["status"] = "ok"
    entry["seconds_to_describe"] = round(time.time() - t0, 1)
    return entry


def build_inventory(root: str | os.PathLike | None = None, out_dir: str | os.PathLike | None = None,
                    n_tiles: int = 40, make_sheets: bool = True, decode: bool = True, min_age_s: float = 60.0,
                    force: bool = False, log=print) -> dict:
    root = Path(root) if root else paths.footage_dir()
    out_dir = Path(out_dir) if out_dir else paths.cache_dir("inventory")
    paths.check_writable_target(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    old: dict = {}
    old_file = out_dir / "inventory.json"
    if old_file.is_file() and not force:
        try:
            for d in json.loads(old_file.read_text()).get("deployments", []):
                for rec in d.get("recordings", []):
                    for e in rec.get("forms", {}).values():
                        old[e["path"]] = e
        except (json.JSONDecodeError, KeyError):
            old = {}

    videos, skipped, debris = [], [], []
    now = time.time()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if is_debris(p):
                debris.append(str(p))
                continue
            if not video.is_video_file(p):
                continue
            age = now - p.stat().st_mtime
            if age < min_age_s:
                skipped.append({"path": str(p), "reason": f"modified {age:.0f} s ago (still being copied?)"})
                continue
            videos.append(p)

    deployments: dict[str, dict] = {}
    for p in videos:
        dep = _deployment_dir(p, root)
        d = deployments.setdefault(str(dep), {"folder": str(dep), "recordings": {}, "sidecars": _sidecars(dep, root)})
        rec = d["recordings"].setdefault(str(p.with_suffix("")), {"recording": p.stem, "forms": {}})
        st = p.stat()
        prev = old.get(str(p))
        if prev and prev.get("size_bytes") == st.st_size and prev.get("mtime_ns") == st.st_mtime_ns:
            entry = prev
            log(f"reuse  {p.name}")
        else:
            log(f"read   {p.name} ...")
            try:
                entry = describe_video(p, n_tiles=n_tiles, make_sheet=make_sheets, decode=decode)
            except Exception as exc:  # one broken file must not stop the inventory
                entry = {"path": str(p), "name": p.name, "form": p.suffix.lower().lstrip("."),
                         "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "status": f"error: {exc!r}"}
            log(f"       {entry.get('status')} in {entry.get('seconds_to_describe', '?')} s")
        # a file that changed while it was read is not trusted
        st2 = p.stat()
        if st2.st_size != st.st_size or st2.st_mtime_ns != st.st_mtime_ns:
            skipped.append({"path": str(p), "reason": "changed while being read"})
            del rec["forms"]
            rec["forms"] = {}
            continue
        rec["forms"][entry["form"]] = entry

    inv = {"root": str(root), "created": _dt.datetime.now().isoformat(timespec="seconds"),
           "deployments": [], "skipped": skipped, "debris_ignored": len(debris)}
    for dkey in sorted(deployments):
        d = deployments[dkey]
        recs = [r for r in d["recordings"].values() if r["forms"]]
        for r in recs:
            r["comparison"] = compare_forms(r["forms"])
        inv["deployments"].append({"folder": d["folder"], "sidecars": d["sidecars"],
                                   "recordings": sorted(recs, key=lambda r: r["recording"])})
    (out_dir / "inventory.json").write_text(json.dumps(inv, indent=1, default=str))
    (out_dir / "inventory.md").write_text(markdown(inv))
    return inv


def compare_forms(forms: dict) -> dict | None:
    """Same recording in two forms (.mov from the tag, .m4v conversion): length, frame count and clock offset."""
    if len(forms) < 2 or "mov" not in forms:
        return None
    a = forms["mov"]
    other = next(k for k in forms if k != "mov")
    b = forms[other]
    try:
        out = {"forms": ["mov", other],
               "frames_shown": [a["demux"]["n_frames_shown"], b["demux"]["n_frames_shown"]],
               "duration_s": [a["demux"]["duration_s"], b["demux"]["duration_s"]]}
        out["frame_difference"] = out["frames_shown"][0] - out["frames_shown"][1]
        ca, cb = a.get("clock", {}), b.get("clock", {})
        fa, fb = ca.get("fit") or {}, cb.get("fit") or {}
        if fa.get("intercept") is not None and fb.get("intercept") is not None:
            out["clock_at_t0_difference_s"] = fa["intercept"] - fb["intercept"]
            out["clock_slope"] = [fa["slope"], fb["slope"]]
            # how far apart the two forms' timelines are at the end of the recording (same clock -> different t)
            t_end = out["duration_s"][0]
            out["timeline_gap_at_end_s"] = (fa["intercept"] + fa["slope"] * t_end - fb["intercept"]) / fb["slope"] - t_end
        return out
    except (KeyError, TypeError):
        return None


def _fmt_s(x) -> str:
    return "" if x is None else f"{x:.2f}"


def markdown(inv: dict) -> str:
    lines = [f"# Footage inventory", "", f"Root: `{inv['root']}`, made {inv['created']}.", ""]
    hdr = ("| recording | form | codec | size | shown s (header s) | frames decodable / shown / packets | fps | "
           "sound | keyframe gap med / max s | created | first clock | clock at t=0, slope |")
    for d in inv["deployments"]:
        lines += [f"## {Path(d['folder']).name or d['folder']}", ""]
        if d["sidecars"]:
            lines.append("Sidecar files: " + ", ".join(f"`{s['name']}` ({s['size_bytes'] / 1e6:.1f} MB)"
                                                        for s in d["sidecars"]))
            lines.append("")
        lines += [hdr, "|" + "---|" * 12]
        for r in d["recordings"]:
            for form, e in sorted(r["forms"].items()):
                if e.get("status") != "ok":
                    lines.append(f"| {r['recording']} | {form} | {e.get('status')} |" + " |" * 10)
                    continue
                v, a, dm, rt, kf = e["video"], e.get("audio"), e["demux"], e["rate"], e["keyframes"]
                dec = e.get("decode", {})
                fps = f"{rt['mean_fps']:.3f} {'const' if rt['constant'] else 'variable'}" if rt.get("mean_fps") else ""
                snd = f"{a['codec']} {a['sample_rate'] / 1000:g} kHz x{a['channels']}" if a else "none"
                clk = e.get("clock") or {}
                first = (clk.get("first") or {}).get("text") or ("no clock" if clk else "")
                lines.append(
                    f"| {r['recording']} | {form} | {v['codec']} | {v['width']}x{v['height']} | "
                    f"{_fmt_s(dm['duration_s'])} ({_fmt_s(e['probe'].get('duration_header_s'))}) | "
                    f"{dec.get('n_decoded', '')} / {dm['n_frames_shown']} / {dm['n_packets']} | {fps} | {snd} | "
                    f"{_fmt_s(kf.get('spacing_median_s'))} / {_fmt_s(kf.get('spacing_max_s'))} | "
                    f"{e.get('creation_time') or ''} | {first} | {timecode._fmt_sod((clk.get('fit') or {}).get('intercept')) or ''}, {((clk.get('fit') or {}).get('slope') or 0):.5f} |")
        lines.append("")
        for r in d["recordings"]:
            c = r.get("comparison")
            if c:
                lines.append(f"- {r['recording']}: mov - {c['forms'][1]} frames {c['frame_difference']:+d}, "
                             f"clock at t=0 differs by {_fmt_s(c.get('clock_at_t0_difference_s'))} s, "
                             f"{c['forms'][1]} timeline off the clock by {_fmt_s(c.get('timeline_gap_at_end_s'))} s "
                             f"at the end")
        lines.append("")
    if inv["skipped"]:
        lines += ["## Skipped", ""] + [f"- `{s['path']}`: {s['reason']}" for s in inv["skipped"]] + [""]
    lines.append(f"macOS debris ignored: {inv['debris_ignored']} files.")
    return "\n".join(lines) + "\n"
