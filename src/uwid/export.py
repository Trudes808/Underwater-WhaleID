"""WP19: export the scan and tracking results as plain files for a web page (JSON, CSV, H.264 MP4, JPEG).

The format is documented in ``docs/EXPORT_FORMAT.md`` (copied into the bundle as ``README.md``). Outlines come from
the cleaned tracks (:func:`uwid.trackstage.load_online` + :func:`uwid.trackstage.link`), the same view the review
sheets and videos draw. Media (clips, posters) are kept when they exist unless ``force``; JSON / CSV are always
rewritten, so a run after a change of the clean-up is cheap.

    uwid export                      # all scanned deployments
    uwid export "<deployment>" --force
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import paths

FORMAT = "uwid-web-export"
VERSION = 1
PRESET = "sensitive"
CLIP_W, CLIP_H = 1280, 720
SIMPLIFY_PX = 1.5            # polygon simplification tolerance (Douglas-Peucker) in pixels of the 1280 x 720 masks
BACK_EVERY = 5               # the tagged whale's back: one polygon set every 5th track frame
NDIGITS = 4                  # coordinates rounded to 4 decimals of the frame size (0.13 px at 1280 wide)
GROUPS = ("hq", "extra")

# windows.csv: exported column -> (source column, kind); kind: "sensor", "model", "rule", "clock", "time"
WINDOW_COLS = [
    ("time_s", "t0", "time"), ("end_s", "t1", "time"), ("clock", "clock", "clock"),
    ("depth_m", "depth_m", "sensor"), ("tag_state", "tag_state", "sensor"),
    ("water_state", "water", "rule"), ("water_source", "water_source", "rule"),
    ("water_picture", "water_picture", "model"), ("p_air_picture", "p_air_picture", "model"),
    ("scanned", "scanned", "rule"),
    ("whale", "whale", "model"), ("whale_score", "whale_score", "model"), ("n_whale", "n_whale", "model"),
    ("whale_px", "whale_px", "model"),
    ("small", "small", "model"), ("small_score", "small_score", "model"),
    ("fish", "fish", "model"), ("fish_score", "fish_score", "model"), ("n_fish", "n_fish", "model"),
    ("active", "active", "rule"), ("tier", "tier", "rule"), ("kept", "kept", "rule"), ("why", "why", "rule"),
    ("seafloor_share", "seafloor_share_model", "model"), ("sand", "sand", "model"), ("dark", "dark", "model"),
    ("silhouette", "silhouette", "model"), ("bubbles", "bubbles", "model"), ("hq_score", "hq_score", "model"),
]
BOX_COLS = ("whale_box_x0", "whale_box_y0", "whale_box_x1", "whale_box_y1")


def out_root() -> Path:
    return paths.cache_dir("export") / "web_v1"


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _js(v):
    """JSON-safe plain values (numpy scalars, NaN -> None)."""
    if isinstance(v, dict):
        return {str(k): _js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_js(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    return v


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_js(obj), separators=(",", ":")))
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------------------------ deployments

def deployments() -> list[str]:
    """Deployments with a scan in the ``sensitive`` preset."""
    root = paths.cache_dir("scan")
    return sorted(p.parent.parent.name for p in root.glob(f"*/{PRESET}/intervals.json"))


def dep_id(name: str, recs: list[str]) -> str:
    """Short folder id: <date>_<tag serial> from the recording names (e.g. 2025-02-03_CC-11-96), else a slug."""
    for r in recs:
        m = re.match(r"(CC-\d\d-\d\d)(?:-HRA)?-(\d{4})(\d\d)(\d\d)-", r)
        if m:
            return f"{m.group(2)}-{m.group(3)}-{m.group(4)}_{m.group(1)}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _codec_tag(path: str) -> dict:
    import av
    with av.open(path) as c:
        v = c.streams.video[0]
        cc = v.codec_context
        tag = getattr(cc, "codec_tag", None) or ""
        return dict(codec=cc.name, codec_tag=str(tag), width=cc.width, height=cc.height, container=c.format.name)


def recording_info(source: str) -> dict:
    from . import video as uv
    tab = uv.frame_table(source)
    info = _codec_tag(source)
    fps = 1.0 / tab.frame_step
    rel = os.path.relpath(source, paths.footage_dir()) if paths.is_inside_footage(source) else Path(source).name
    return dict(file=Path(source).name, source=rel, duration_s=round(float(tab.duration), 3),
                n_frames=int(tab.n_frames), fps=round(float(fps), 6), width=info["width"], height=info["height"],
                codec=info["codec"], codec_tag=info["codec_tag"], container=info["container"])


# ------------------------------------------------------------------------------------------------ scan tables

def windows_csv(w: pd.DataFrame, det_size: tuple[int, int]) -> pd.DataFrame:
    out = pd.DataFrame()
    for name, src, _ in WINDOW_COLS:
        col = w[src] if src in w else pd.Series([np.nan] * len(w))
        if col.dtype == object and col.dropna().isin([True, False]).all() and len(col.dropna()):
            col = col.map(lambda v: "" if pd.isna(v) else int(bool(v)))
        elif col.dtype == bool:
            col = col.astype(int)
        out[name] = col.values
    dw, dh = det_size
    for name, src, scale in zip(BOX_COLS, ("bx0", "by0", "bx1", "by1"), (dw, dh, dw, dh)):
        out[name] = (w[src] / scale).round(NDIGITS).values if src in w else np.nan
    for c in ("time_s", "end_s"):
        out[c] = out[c].round(3)
    for c in out.columns:
        if out[c].dtype.kind == "f" and c not in ("time_s", "end_s"):
            out[c] = out[c].round(NDIGITS)
    return out


def _det_size(dep: str, name: str) -> tuple[int, int]:
    m = paths.cache_dir("scan") / dep / "cache" / "det" / name / "meta.json"
    if m.exists():
        s = json.loads(m.read_text()).get("decode_size")
        if s:
            return int(s[0]), int(s[1])
    return 1920, 1080


def stretches(iv: dict) -> dict:
    recs = {}
    for v in iv["videos"]:
        name = Path(v["source"]).stem
        keep = []
        for k in v.get("keep", []):
            keep.append({kk: k.get(kk) for kk in ("start", "end", "reason", "tier", "advanced_ok", "hq_whale",
                                                  "hq_spans", "description", "active_s", "quiet_s", "whale_windows",
                                                  "hq_windows", "tier_windows", "silhouette", "silhouette_windows",
                                                  "sand_s", "dark_s", "max_whale_score", "max_fish_score")})
        drop = [{kk: d.get(kk) for kk in ("start", "end", "reason")} for d in v.get("drop", [])]
        recs[name] = dict(duration_s=v.get("duration"), kept=keep, dropped=drop,
                          hq_clips=[{kk: h.get(kk) for kk in ("start", "end")} for h in v.get("hq", [])])
    return recs


# ------------------------------------------------------------------------------------------------ tracks

def polygons(mask: np.ndarray, tol: float = SIMPLIFY_PX) -> list[list[list[float]]]:
    """Outer outlines of all pieces of a mask (holes are not kept), simplified by ``tol`` px, as [[x, y], ...] in
    fractions of the mask's width / height (pixel corners: x = column / width)."""
    import cv2
    h, w = mask.shape
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 3:
            continue
        a = cv2.approxPolyDP(c, tol, True).reshape(-1, 2).astype(float)
        if len(a) < 3:
            continue
        a[:, 0] = (a[:, 0] + 0.5) / w
        a[:, 1] = (a[:, 1] + 0.5) / h
        out.append(np.round(np.clip(a, 0, 1), NDIGITS).tolist())
    return out


def rasterize(polys: Iterable, w: int = CLIP_W, h: int = CLIP_H) -> np.ndarray:
    """Inverse of :func:`polygons` (for checks): fill the polygons at w x h."""
    import cv2
    m = np.zeros((h, w), np.uint8)
    pts = [np.round(np.array(p, float) * [w, h] - 0.5).astype(np.int32) for p in polys if len(p) >= 3]
    if pts:
        cv2.fillPoly(m, pts, 1)
    return m.astype(bool)


def _bbox(m: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    if not len(ys):
        return None
    h, w = m.shape
    return [round(xs[0] / w, NDIGITS), round(ys[0] / h, NDIGITS), round((xs[-1] + 1) / w, NDIGITS),
            round((ys[-1] + 1) / h, NDIGITS)]


def track_json(tr, group: str, deployment: str, rec: str, span: tuple[float, float], clip_rel: str,
               poster_rel: str, note: str | None = None) -> dict:
    """The export document of one tracked span (``tr`` cleaned and linked)."""
    from . import tracks
    meta = tr.meta
    f0, n, fps = int(meta["frames"][0]), int(meta["n_frames"]), float(meta["fps"])
    t = tr.table
    times = {}
    for i, tt in zip(t.i.to_numpy(), t.t.to_numpy()):
        times.setdefault(int(i), float(tt))
    t0 = float(meta["t0"])
    tlist = [round(times.get(j, t0 + j / fps), 4) for j in range(n)]

    def shapes(objs: list[int], kind: str, label) -> dict | None:
        frames = []
        for j in range(n):
            vis = [k for k in objs if tr.visible(k, j)]
            if not vis:
                continue
            m = np.zeros(tuple(meta["store_size"][::-1]), bool)
            for k in vis:
                m |= tr.mask(k, j)
            if not m.any():
                continue
            frames.append(dict(j=j, box=_bbox(m), polygons=polygons(m)))
        if not frames:
            return None
        return dict(number=label, kind=kind, objects=objs, first_j=frames[0]["j"], last_j=frames[-1]["j"],
                    first_t=tlist[frames[0]["j"]], last_t=tlist[frames[-1]["j"]], n_frames=len(frames),
                    frames=frames)

    numbers = []
    for num in sorted(set(tr.number.values())):
        s = shapes(sorted(k for k, v in tr.number.items() if v == num), "whale", num)
        if s:
            numbers.append(s)
    smalls = []
    for k, sn in sorted(tr.small_number.items(), key=lambda x: x[1]):
        s = shapes([k], "small", f"s{sn}")
        if s:
            smalls.append(s)
    back = []
    if tr.back is not None:
        for j in range(0, n, BACK_EVERY):
            if tr.visible(tracks.BACK_ID, j):
                back.append(dict(j=j, polygons=polygons(tr.mask(tracks.BACK_ID, j))))
    blocked = [dict(j0=int(a - f0), j1=int(b - f0), t0=tlist[max(0, min(n - 1, a - f0))],
                    t1=tlist[max(0, min(n - 1, b - f0 - 1))]) for a, b in meta.get("blocked", [])]
    endings = [dict(j=int(i), t=tlist[min(int(i), n - 1)], number=lab, reason=reason)
               for i, lab, reason, _ in tr.endings]
    links = []
    for L in getattr(tr, "links", []) or []:
        det = L.get("detector") or {}
        links.append(dict(made=bool(L.get("made")), why=L.get("why"), number_a=tr.number.get(L.get("a")),
                          number_b=tr.number.get(L.get("b")), object_a=L.get("a"), object_b=L.get("b"),
                          t_end=L.get("t_end"), gap_s=L.get("gap_s"), breath=L.get("breath"),
                          limit_s=L.get("limit_s"), others=L.get("others"),
                          others_numbers=[tr.number.get(o) for o in (L.get("others") or [])], lone=L.get("lone"),
                          detector_before=det.get("before"), detector_during=det.get("during"),
                          detector_after=det.get("after"), size_ratio=L.get("size_ratio"),
                          appearance_distance=L.get("app_dist"), checks_ok=L.get("checks_ok")))
    refused = []
    for r in meta.get("refused", []):
        b = r.get("box")
        refused.append(dict(j=int(r["i"]), t=round(float(r["t"]), 4), reason=r["reason"],
                            box=[round(b[0] / 1920, NDIGITS), round(b[1] / 1080, NDIGITS), round(b[2] / 1920, NDIGITS),
                                 round(b[3] / 1080, NDIGITS)] if b else None))
    return dict(
        format=FORMAT, version=VERSION, kind="track", group=group, note=note, deployment=deployment, recording=rec,
        span=dict(t0=span[0], t1=span[1]), tracked=dict(t0=round(t0, 4), t1=round(float(meta["t1"]), 4)),
        fps=round(fps, 6), n_frames=n,
        frames=dict(index=[f0 + j for j in range(n)], t=tlist),
        clip=dict(path=clip_rel, poster=poster_rel, width=CLIP_W, height=CLIP_H, fps=round(fps, 6), n_frames=n,
                  correspondence="frame j of the clip (0-based, decode order) is frame j of this file: source frame "
                                 "frames.index[j] shown at frames.t[j] s of the source recording"),
        coordinates="fractions of the frame width / height, origin top left, x right, y down; same for any "
                    "display size",
        simplify_px=SIMPLIFY_PX, mask_size=list(meta["store_size"]), back_every=BACK_EVERY,
        numbers=numbers, small=smalls, back=back, blocked=blocked, endings=endings, links=links,
        refused_starts=refused,
        purity=meta.get("purity"), method=meta.get("method"), start_source=meta.get("start_source"),
        blocked_rule=(meta.get("blocked_info") or {}).get("lens_source"),
        own_back_template=bool(meta.get("own_back")),
    )


def write_clip(tr, mp4: Path, poster: Path, best_j: int = 0, log=print) -> dict:
    """Plain clip of the track's frames: 1280 x 720 H.264 yuv420p, constant rate = the track's, +faststart; frame j
    of the file = track frame j (decoded exactly as the tracker decoded them)."""
    import cv2
    import imageio_ffmpeg
    from . import trackstage as T
    meta = tr.meta
    video, f0, f1, n = meta["video"], int(meta["frames"][0]), int(meta["frames"][1]), int(meta["n_frames"])
    rate = Fraction(float(meta["fps"])).limit_denominator(1001)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp4.with_suffix(".part.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{CLIP_W}x{CLIP_H}", "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "8",
           "-r", f"{rate.numerator}/{rate.denominator}", "-movflags", "+faststart", str(tmp)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    k, last = 0, None
    try:
        for fr in T.span_frames(video, f0, f1):
            if k >= n:
                break
            img = fr.image
            if img.shape[:2] != (CLIP_H, CLIP_W):
                img = cv2.resize(img, (CLIP_W, CLIP_H), interpolation=cv2.INTER_AREA)
            if k == best_j:
                poster.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(poster), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            p.stdin.write(np.ascontiguousarray(img).tobytes())
            last = img
            k += 1
        padded = 0
        while k < n and last is not None:          # decode came up short: repeat the last frame (reported)
            p.stdin.write(last.tobytes())
            k += 1
            padded += 1
    finally:
        p.stdin.close()
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed ({rc}) for {mp4}")
    os.replace(tmp, mp4)
    return dict(frames_written=k, padded=padded)


def count_frames(mp4: Path) -> tuple[int, tuple[int, int]]:
    import av
    with av.open(str(mp4)) as c:
        v = c.streams.video[0]
        k = 0
        for _ in c.decode(v):
            k += 1
        return k, (v.codec_context.width, v.codec_context.height)


# ------------------------------------------------------------------------------------------------ span lists

def hq_folders(dep: str) -> list[Path]:
    """The run folders of exactly what `uwid tracks <dep>` runs (trackstage.run_deployment without --only): every hq
    span of the scan, capped at 30 s around its centre."""
    from . import trackstage as T
    out = []
    for r in T.hq_spans(dep, PRESET).itertuples():
        a, b = r.start, r.end
        if b - a > 30.0:
            c = (a + b) / 2
            a, b = c - 15.0, c + 15.0
        out.append(T.span_dir(dep, Path(r.source).name, a, b))
    return out


def span_lists(dep: str) -> dict[str, list[tuple[Path, str | None]]]:
    """(run folder, note) per group: hq = what `uwid tracks <dep>` runs (finished runs only); extra =
    tracks/extra_stretches.csv (columns deployment, recording, t0, t1, folder, note) or, until it exists, every
    ``*_online_ownback`` folder."""
    root = paths.cache_dir("tracks")
    out = {"hq": [(d, None) for d in hq_folders(dep)], "extra": []}
    e = root / "extra_stretches.csv"
    if e.exists():
        s = pd.read_csv(e)
        s = s[s.deployment == dep] if "deployment" in s else s
        notes = s.note if "note" in s else [None] * len(s)
        if "folder" in s:
            out["extra"] = [(Path(x), None if pd.isna(n) else str(n)) for x, n in zip(s.folder, notes)]
        else:
            from . import trackstage as T
            out["extra"] = [(Path(str(T.span_dir(dep, r.recording, float(r.t0), float(r.t1))) + "_ownback"),
                             None if pd.isna(n) else str(n)) for r, n in zip(s.itertuples(), notes)]
    else:
        out["extra"] = [(d, None) for d in sorted((root / dep).glob("*/*_online_ownback"))]

    def done(d):
        m = d / "meta.json"
        return m.exists() and json.loads(m.read_text()).get("done", False)
    return {g: [(d, n) for d, n in v if done(d)] for g, v in out.items()}


def _span_of(d: Path) -> tuple[float, float]:
    a, b = d.name.split("_")[0].split("-")
    return float(a), float(b)


# ------------------------------------------------------------------------------------------------ main

def export(deps: Iterable[str] | None = None, root: Path | None = None, force: bool = False, clips: bool = True,
           link_fn: Callable | None = None, log=print) -> Path:
    from . import trackstage as T
    root = Path(root or out_root())
    root.mkdir(parents=True, exist_ok=True)
    deps = list(deps or deployments())
    idx_path = root / "index.json"
    old = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    entries = {d["name"]: d for d in old.get("deployments", [])}
    for dep in deps:
        t_dep = time.time()
        cfg, iv = T.scan_config(dep, PRESET)
        sd = T.scan_dir(dep, PRESET)
        summ = json.loads((sd / "summary.json").read_text()) if (sd / "summary.json").exists() else {}
        clock_of = {r["recording"]: r.get("clock", {}) for r in summ.get("recordings", [])}
        names = [Path(v["source"]).stem for v in iv["videos"]]
        did = dep_id(dep, names)
        ddir = root / did
        recs = []
        for v in iv["videos"]:
            src, stem = v["source"], Path(v["source"]).stem
            info = recording_info(src)
            w = T.windows_table(dep, Path(src).name, PRESET)
            if w is None:                       # the dev clip's table is named with "_" for the space in its name
                alt = T.scan_dir(dep, PRESET) / "windows" / (Path(src).name.replace(" ", "_") + ".csv")
                w = pd.read_csv(alt) if alt.exists() else None
            first_clock = None
            if w is not None:
                wc = windows_csv(w, _det_size(dep, Path(src).name))
                (ddir / "windows").mkdir(parents=True, exist_ok=True)
                wc.to_csv(ddir / "windows" / f"{stem}.csv", index=False)
                c0 = w.loc[w.t0 == 0, "clock"] if "clock" in w else []
                first_clock = (str(c0.iloc[0]) or None) if len(c0) and isinstance(c0.iloc[0], str) else None
            other = [p.name for p in Path(src).parent.glob(stem + ".*") if p.suffix.lower() != Path(src).suffix.lower()]
            cl = clock_of.get(stem) or {}
            recs.append(dict(info, recording=stem, first_frame_clock=first_clock,
                             clock_fit=dict(ok=cl.get("ok"), kind=cl.get("kind"), rms_s=cl.get("rms_s")) if cl else None,
                             other_forms=sorted(other),
                             windows=f"{did}/windows/{stem}.csv" if w is not None else None))
        write_json(ddir / "stretches.json", dict(format=FORMAT, version=VERSION, kind="stretches", deployment=dep,
                                                 preset=PRESET, scan_created=iv.get("created"),
                                                 recordings=stretches(iv)))
        spans = []
        for group, dirs in span_lists(dep).items():
            for d, note in dirs:
                tr = T.load_online(d)
                if link_fn is None:
                    T.link(tr, dep, tr.meta.get("start_source", "scan"))
                else:
                    link_fn(tr)
                rec = Path(tr.meta["video"]).stem
                a, b = _span_of(d)
                stem = f"{a:07.2f}-{b:07.2f}"
                jrel = f"{did}/tracks/{group}/{rec}/{stem}.json"
                crel = f"{did}/clips/{group}/{rec}/{stem}.mp4"
                prel = f"{did}/clips/{group}/{rec}/{stem}.jpg"
                doc = track_json(tr, group, dep, rec, (a, b), crel, prel, note=note)
                mp4, jpg = root / crel, root / prel
                if clips and (force or not mp4.exists() or not jpg.exists()):
                    area = {}
                    for nb in doc["numbers"]:
                        for fr in nb["frames"]:
                            bx = fr["box"]
                            area[fr["j"]] = area.get(fr["j"], 0) + (bx[2] - bx[0]) * (bx[3] - bx[1])
                    best = max(area, key=area.get) if area else doc["n_frames"] // 2
                    res = write_clip(tr, mp4, jpg, best_j=best, log=log)
                    nf, size = count_frames(mp4)
                    doc["clip"]["check"] = dict(decoded_frames=nf, size=list(size), padded=res["padded"],
                                                ok=nf == doc["n_frames"] and tuple(size) == (CLIP_W, CLIP_H))
                    write_json(mp4.with_suffix(".check.json"), doc["clip"]["check"])
                elif (mp4.with_suffix(".check.json")).exists():
                    doc["clip"]["check"] = json.loads(mp4.with_suffix(".check.json").read_text())
                if not clips:
                    doc["clip"] = None
                write_json(root / jrel, doc)
                spans.append(dict(group=group, note=note, recording=rec, t0=a, t1=b, track=jrel,
                                  clip=crel if clips else None, poster=prel if clips else None,
                                  n_frames=doc["n_frames"], numbers=len(doc["numbers"]),
                                  links_made=sum(L["made"] for L in doc["links"]),
                                  first_t=doc["tracked"]["t0"], last_t=doc["tracked"]["t1"]))
                log(f"{did} {group} {rec[-5:]} {stem}: {len(doc['numbers'])} numbers"
                    + (f", clip {doc['clip'].get('check', {}).get('decoded_frames')}/{doc['n_frames']}"
                       if clips and doc.get("clip") else ""))
        kept = sum(len(v.get("keep", [])) for v in iv["videos"])
        entries[dep] = dict(
            id=did, name=dep, tag=(re.search(r"CC-\d\d-\d\d", names[0]).group(0) if names and
                                   re.search(r"CC-\d\d-\d\d", names[0]) else None),
            date=did.split("_")[0] if re.match(r"\d{4}-\d\d-\d\d_", did) else None,
            preset=PRESET, sensor_log=bool(summ.get("sensor_log")), recordings=recs,
            counts=dict(recordings=len(recs), kept_stretches=kept,
                        hq_spans=sum(len(k.get("hq_spans", [])) for v in iv["videos"] for k in v.get("keep", [])),
                        tracked_spans={g: sum(s["group"] == g for s in spans) for g in GROUPS},
                        numbers=sum(s["numbers"] for s in spans)),
            stretches=f"{did}/stretches.json", spans=spans)
        log(f"{dep}: {len(recs)} recordings, {len(spans)} spans in {time.time() - t_dep:.0f} s")
    index = dict(format=FORMAT, version=VERSION, kind="index", created=_now(),
                 command="uwid export " + " ".join(sys.argv[2:]) if sys.argv[1:2] == ["export"] else "uwid.export.export",
                 readme="README.md", coordinates="fractions of the frame width / height, origin top left",
                 deployments=[entries[k] for k in sorted(entries)])
    write_json(idx_path, index)
    doc = Path(__file__).resolve().parents[2] / "docs" / "EXPORT_FORMAT.md"
    if doc.exists():
        (root / "README.md").write_text(doc.read_text())
    return root
