"""Command line: ``uwid probe | inventory | timecode | sheet | clips | scan``."""

from __future__ import annotations

import os

# shared machine: at most 8 threads per job (CLAUDE.md); set before numpy / cv2 load
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import argparse
import csv
import json
import sys
from pathlib import Path


def _json(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


def cmd_probe(a) -> int:
    from . import video
    pr = video.probe(a.file)
    tab = video.frame_table(a.file)
    pr["frames"] = {"n_packets": tab.n_packets, "n_hidden_by_edit_list": tab.n_hidden,
                    "n_frames_shown": tab.n_frames, "duration_shown_s": tab.duration,
                    "first_t": float(tab.times[0]) if tab.n_frames else None,
                    "last_t": float(tab.times[-1]) if tab.n_frames else None}
    pr["rate"] = tab.rate_summary()
    pr["keyframes"] = tab.keyframe_summary()
    if a.keyframes:
        pr["keyframe_times"] = video.keyframes(a.file, include_hidden=True)
    if a.decode:
        dc = video.decode_check(a.file)
        dc.pop("frames")
        pr["decode"] = dc
    if not a.full:
        pr.pop("mp4", None)
    print(_json(pr))
    return 0


def cmd_inventory(a) -> int:
    from . import inventory
    inv = inventory.build_inventory(a.dir, out_dir=a.out, n_tiles=a.tiles, make_sheets=not a.no_sheets,
                                    decode=not a.no_decode, min_age_s=a.min_age, force=a.force,
                                    log=lambda *m: print(*m, flush=True))
    out = Path(a.out) if a.out else None
    from . import paths
    out = out or paths.cache_dir("inventory")
    print(f"\nwrote {out / 'inventory.json'} and {out / 'inventory.md'}")
    if inv["skipped"]:
        print("skipped:")
        for s in inv["skipped"]:
            print(f"  {s['path']}: {s['reason']}")
    return 0


def cmd_timecode(a) -> int:
    from . import paths, timecode
    if a.bootstrap:
        gs = timecode.bootstrap_glyphs(a.file, a.text or timecode.DEV_CLIP_TEXT0, frame_index=a.frame)
        dest = Path(a.glyphs) if a.glyphs else timecode.default_glyph_path()
        gs.save(dest)
        print(_json(gs.meta))
        print(f"glyphs: {''.join(gs.chars)} -> {dest}")
        return 0
    glyphs = timecode.load_glyphs(Path(a.glyphs) if a.glyphs else None)
    res = timecode.read_video_clock(a.file, fps=a.fps, glyphs=glyphs, start=a.start, end=a.end,
                                    progress=not a.quiet)
    out = Path(a.out) if a.out else paths.cache_dir("timecode") / (Path(a.file).stem.replace(" ", "_") + "_clock.csv")
    paths.check_writable_target(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "t", "text", "clock_s", "datetime", "month_source", "score",
                                           "residual_s", "flag"])
        w.writeheader()
        for r in res["rows"]:
            w.writerow(r)
    out.with_suffix(".json").write_text(_json(res["summary"]))
    print(_json(res["summary"]))
    print(f"rows: {out}")
    return 0


def cmd_sheet(a) -> int:
    from . import paths, sheets
    out = Path(a.out) if a.out else paths.sheets_dir() / f"sheet_{Path(a.file).stem.replace(' ', '_')}.jpg"
    _, frames = sheets.video_sheet(a.file, n=a.n, out=out, cols=a.cols, tile_width=a.tile_width, start=a.start,
                                   end=a.end, with_clock=not a.no_clock)
    print(f"{len(frames)} frames -> {out}")
    return 0


def cmd_clips(a) -> int:
    from . import clips
    res = clips.run(a.intervals, out_dir=a.out, mode=a.mode, max_lead=a.max_lead, pad=a.pad, merge_gap=a.merge_gap,
                    dry_run=a.dry_run, overwrite=a.overwrite, read_clock=not a.no_clock, crf=a.crf, preset=a.preset)
    n_ok = sum(1 for r in res["clips"] if r["status"] in ("ok", "dry run") or r["status"].startswith("exists"))
    print(f"\n{len(res['clips'])} clips ({n_ok} ok), {len(res['dropped'])} dropped rows"
          + ("" if a.dry_run else f" -> {res['out_dir']}/clips_index.csv, dropped.csv"))
    return 0 if n_ok == len(res["clips"]) else 1


def cmd_scan(a) -> int:
    from . import pipeline
    off = a.clock_offset if a.clock_offset == "auto" else float(a.clock_offset)
    cfg = pipeline.ScanConfig(preset=a.preset, pad_s=a.pad, merge_gap_s=a.merge_gap, clock_offset_s=off,
                              near_whale_score=None if a.no_near_whale else a.near_whale_score,
                              min_whale_px=a.min_whale_px, hq_threshold=a.hq_threshold,
                              fish_veto=a.fish_veto, scan_all=a.scan_all,
                              min_age_s=a.min_age, gpu_lock=None if a.no_gpu_lock else a.gpu_lock,
                              lock_budget_s=a.lock_minutes * 60, picture_check=not a.no_picture_check)
    sums = pipeline.scan(a.dir, cfg, cut=a.cut, mode=a.mode, log=lambda *m: print(*m, flush=True),
                         out_root=Path(a.out) if a.out else None, clip_root=Path(a.clips_out) if a.clips_out else None)
    for s in sums:
        print(f"\n{s['deployment']} [{s['label']}]: kept {s['kept_s'] / 60:.1f} of {s['seconds'] / 60:.1f} min, "
              f"hq_whale windows {s['hq_windows'] / 60:.1f} min")
        for k, v in s["dropped_s"].items():
            if v > 0:
                print(f"  dropped {v / 60:7.1f} min  {k}")
        if s.get("offset_check"):
            oc = s["offset_check"]
            print(f"  clock offset check: {oc.get('verdict')} (picture minus depth "
                  f"{oc.get('median_picture_minus_depth_s', float('nan')):+.2f} s, {oc.get('n_pairs')} events)")
        if s.get("clips"):
            print(f"  {s['clips']['n']} clips, {s['clips']['bytes'] / 1e9:.2f} GB -> {s['clips']['out_dir']}")
        print(f"  tables: {s['outputs']['windows']}, {s['outputs']['intervals']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uwid", description="Underwater whale video tools (see README.md).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="metadata, real frame count, frame rate, keyframes of one video")
    s.add_argument("file")
    s.add_argument("--decode", action="store_true", help="also decode every frame and count what decodes")
    s.add_argument("--keyframes", action="store_true", help="list every keyframe time")
    s.add_argument("--full", action="store_true", help="include the MP4 box summary")
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("inventory", help="inventory of a footage folder (json + markdown + contact sheets)")
    s.add_argument("dir", nargs="?", default=None, help="footage folder (default: footage_dir of configs/paths.toml)")
    s.add_argument("--out", default=None, help="output folder (default: <cache>/inventory)")
    s.add_argument("--tiles", type=int, default=40, help="frames per contact sheet")
    s.add_argument("--no-sheets", action="store_true")
    s.add_argument("--no-decode", action="store_true", help="skip the full decode (frame count from demuxing only)")
    s.add_argument("--min-age", type=float, default=60.0, help="skip files modified less than this many seconds ago")
    s.add_argument("--force", action="store_true", help="re-read files already in the inventory")
    s.set_defaults(func=cmd_inventory)

    s = sub.add_parser("timecode", help="read the burned-in clock of a video, fit clock against time")
    s.add_argument("file")
    s.add_argument("--fps", type=float, default=None, help="sample this many frames per second (default: all)")
    s.add_argument("--start", type=float, default=0.0)
    s.add_argument("--end", type=float, default=None)
    s.add_argument("--glyphs", default=None, help="glyph file (default: <cache>/timecode/glyphs.json)")
    s.add_argument("--out", default=None, help="CSV of per-frame readings (default: <cache>/timecode/<stem>_clock.csv)")
    s.add_argument("--bootstrap", action="store_true", help="learn the glyphs from FILE instead of reading it")
    s.add_argument("--text", default=None, help="with --bootstrap: the clock text shown on frame --frame")
    s.add_argument("--frame", type=int, default=0, help="with --bootstrap: index of the frame showing --text")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_timecode)

    s = sub.add_parser("sheet", help="contact sheet of evenly spaced frames of one video")
    s.add_argument("file")
    s.add_argument("-n", type=int, default=40)
    s.add_argument("--cols", type=int, default=8)
    s.add_argument("--tile-width", type=int, default=320)
    s.add_argument("--start", type=float, default=0.0)
    s.add_argument("--end", type=float, default=None)
    s.add_argument("--no-clock", action="store_true")
    s.add_argument("--out", default=None, help="JPEG path (default: <sheets>/sheet_<stem>.jpg)")
    s.set_defaults(func=cmd_sheet)

    s = sub.add_parser("clips", help="cut keep-intervals into clips (see uwid.clips for the intervals format)")
    s.add_argument("--intervals", required=True, help="intervals JSON file")
    s.add_argument("--out", default=None, help="output folder (default: <cache>/clips)")
    s.add_argument("--mode", choices=["copy", "reencode", "auto"], default="auto")
    s.add_argument("--max-lead", type=float, default=2.0,
                   help="auto: copy when the keyframe before the start is at most this many seconds early")
    s.add_argument("--pad", type=float, default=0.0, help="seconds added before and after each keep interval")
    s.add_argument("--merge-gap", type=float, default=0.0, help="merge keep intervals closer than this (s)")
    s.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    s.add_argument("--overwrite", action="store_true", help="re-cut clips that already exist")
    s.add_argument("--no-clock", action="store_true", help="do not read the burned-in clock at clip ends")
    s.add_argument("--crf", type=int, default=17, help="reencode: x264 quality (lower = better)")
    s.add_argument("--preset", default="medium", help="reencode: x264 preset")
    s.set_defaults(func=cmd_clips)

    s = sub.add_parser("scan", help="folder of tag video -> per-second table, keep / drop intervals, clips (WP7)")
    s.add_argument("dir", help="deployment folder, a folder of deployments, or one video")
    s.add_argument("--preset", choices=["sensitive", "strict"], default="sensitive",
                   help="sensitive: either of the 2 frames of a 1 s window has a whale; strict: both")
    s.add_argument("--cut", action="store_true", help="also cut the kept stretches into clips")
    s.add_argument("--mode", choices=["copy", "reencode", "auto"], default="copy", help="clip cutting mode")
    s.add_argument("--pad", type=float, default=2.0, help="seconds added before and after each whale stretch")
    s.add_argument("--merge-gap", type=float, default=5.0, help="merge kept stretches closer than this (s, after padding)")
    s.add_argument("--clock-offset", default="2.03",
                   help="burned-in clock minus sensor time (s), or 'auto' to take the offset check's measurement")
    s.add_argument("--near-whale-score", type=float, default=0.6,
                   help="boxes covering more than 60 %% of the frame are kept when their score is at least this")
    s.add_argument("--no-near-whale", action="store_true", help="drop every box covering most of the frame (WP5a)")
    s.add_argument("--min-whale-px", type=float, default=300.0,
                   help="whale boxes shorter than this (longest side, 1920 px frame) are small: fish or far whale")
    s.add_argument("--hq-threshold", type=float, default=0.6, help="quality score for the hq_whale tier")
    s.add_argument("--fish-veto", action="store_true", help="option: a whale box must beat its own fish score")
    s.add_argument("--scan-all", action="store_true", help="run the detector on every window (cost comparison)")
    s.add_argument("--no-picture-check", action="store_true",
                   help="with a sensor log, skip the picture water model (and with it the clock offset check)")
    s.add_argument("--min-age", type=float, default=60.0, help="skip files modified less than this many seconds ago")
    s.add_argument("--gpu-lock", default="/var/tmp/sat3737-parts-cache/gpu_train.lock")
    s.add_argument("--no-gpu-lock", action="store_true")
    s.add_argument("--lock-minutes", type=float, default=12.0, help="hold the GPU lock at most this long at a time")
    s.add_argument("--out", default=None, help="results folder (default: <cache>/scan)")
    s.add_argument("--clips-out", default=None, help="clips folder (default: <cache>/clips)")
    s.set_defaults(func=cmd_scan)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
