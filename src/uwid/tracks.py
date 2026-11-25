"""Stage 2 of the two-stage chain: precise masks and individual tracks through a short stretch.

Stage 1 (cheap): OWLv2, prompt "whale", every 15th frame (2 per 1 s window); a window has a whale when both its
frames carry a whale box >= 0.25 (WP5a's operating point, ``experiments/04_whale_fish_detection``). Its boxes are
read from the WP5a cache (``underwater_cache/detect/<stem>/owl_owl_all_s15`` or ``..._s5``) or computed on the fly.

Stage 2 (this module): on a stretch stage 1 marked, a video segmenter follows every other whale frame by frame and
keeps its identity:

- the tagged whale's own back is object 0, started from a fixed box over the bottom of the frame; it is drawn as
  ``back`` and never counts as a whale;
- whale tracks start from stage 1's boxes: on the first stage-1 frame of the stretch and again on every later
  stage-1 frame where a whale box is not covered by an existing track (``start="owl"``), or from SAM 3's own text
  detector (``start="text"``, :class:`Sam3VideoModel`);
- a track that lies mostly inside the back is dropped; a track that stays fish-sized (``SMALL_TRACK``) is drawn thin
  grey as ``sN small`` with no whale number; a whale track that is fish-sized on a frame is drawn ``N small`` there;
  where one whale outline lies inside an older one the younger is not drawn (:func:`load_tracks`).

Usage (e.g. on ``uwid scan`` intervals)::

    d = run_tracks(video, t0, t1, method="owl_sam3")          # GPU (take the lock); skips finished stretches
    tr = load_tracks(d); track_summary(tr); render_tracks(tr, "x.mp4", "x.jpg")   # CPU

Output (:func:`run_tracks`): ``underwater_cache/tracks/<stem>/<t0>-<t1>_<method>/`` with ``masks.npz`` (per object
and frame a bit-packed mask at ``STORE_SIZE``), ``tracks.parquet`` (one row per object and frame: frame index, time,
track id, area, box, score, flags) and ``meta.json``. :func:`render_tracks` draws outlines onto the video and writes
an H.264 ``.mp4`` (yuv420p, no sound) and a still sheet.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from . import ownback, paths

DEV_VIDEO = "/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
TEST_DIR = "/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96/Video"

STAGE1_QUERY, STAGE1_THR, STAGE1_STRIDE = "whale", 0.25, 15
STAGE1_RUNS = ("owl_owl_all_s15", "owl_owl_all_s5")
W, H = 1920, 1080
STORE_SIZE = (1280, 720)          # masks are kept at this size (w, h)
# WP18 speck rule (outlines shown to readers through Tracks.mask when Tracks.despeck is on, i.e. runs loaded with
# trackstage.load_online): per frame and whale object, keep the largest 8-connected piece; keep any other piece that
# is large (>= SPECK_MAX_SHARE of the object's area in that frame) OR close (gap to the largest piece <=
# SPECK_MIN_GAP_PX at store size); drop the others (far, small pieces). Values set before looking at the test
# deployments (the orchestrator's starting values, kept unchanged after the dev clip's stored runs, tables WP18).
# Stored masks stay raw; roles, numbers and links are computed on the raw masks.
SPECK_MAX_SHARE = 0.05
SPECK_MIN_GAP_PX = 40.0
BACK_BOX = (0, int(0.75 * H), W, H)  # fixed prompt for the tagged whale's back (WP5a: bottom 25 %)
BACK_ID = 0


# ------------------------------------------------------------------------------------------------ stage 1

def stem_of(video: str | os.PathLike) -> str:
    return Path(video).stem.replace(" ", "_")


def has_stage1(video) -> bool:
    from . import detect
    return any((detect.detect_cache_dir(video, r) / "meta.json").exists() for r in STAGE1_RUNS)


def stage1_boxes(video, thr: float = STAGE1_THR, stride: int = STAGE1_STRIDE) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Stage-1 whale boxes (score >= ``thr``, not in the clock, not > 60 % of the frame) on every ``stride``-th frame,
    the processed frame indices, and the frame rate the cache used."""
    from . import detect
    for r in STAGE1_RUNS:
        d = detect.detect_cache_dir(video, r)
        if (d / "meta.json").exists():
            meta = json.loads((d / "meta.json").read_text())
            b, frames = detect.load_cached(video, r)
            break
    else:
        raise FileNotFoundError(f"no stage-1 cache for {video}")
    frames = frames[frames % stride == 0]
    b = b[(b["query"] == STAGE1_QUERY) & (b.score >= thr) & b.frame.isin(frames)].copy()
    area = (b.x1 - b.x0).clip(lower=0) * (b.y1 - b.y0).clip(lower=0)
    b = b[~detect.in_clock(b) & (area <= 0.6 * W * H)]
    return b.reset_index(drop=True), frames, float(meta["fps"])


def stage1_windows(video) -> pd.DataFrame:
    """One row per 1 s window: ``whale`` (both frames >= 0.25), number of whale boxes on its frames (max), the largest
    whale box's longest side / frame width (max)."""
    b, frames, fps = stage1_boxes(video, thr=0.0)
    f = pd.DataFrame(dict(frame=frames))
    f["window"] = (f.frame / fps).astype(int)
    top = b.groupby("frame").score.max()
    f["score"] = top.reindex(f.frame).fillna(0).to_numpy()
    strong = b[b.score >= STAGE1_THR].copy()
    strong["size"] = np.maximum(strong.x1 - strong.x0, strong.y1 - strong.y0) / W
    f["n"] = strong.groupby("frame").size().reindex(f.frame).fillna(0).to_numpy()
    f["size"] = strong.groupby("frame")["size"].max().reindex(f.frame).fillna(0).to_numpy()
    g = f.groupby("window")
    out = pd.DataFrame(dict(score2=g.score.apply(lambda s: np.sort(s.to_numpy())[::-1][min(1, len(s) - 1)]),
                            n=g.n.max(), size=g["size"].max(), n_frames=g.size()))
    out["whale"] = out.score2 >= STAGE1_THR
    full = pd.RangeIndex(int(out.index.min()), int(out.index.max()) + 1, name="window")
    out = out.reindex(full)
    out["whale"] = out["whale"].fillna(False).astype(bool)
    return out.fillna(0).reset_index()


def runs_of(win: pd.DataFrame, min_s: float = 8.0, gap_s: int = 1) -> list[dict]:
    """Runs of whale windows (gaps of up to ``gap_s`` whale-free windows bridged) lasting >= ``min_s``."""
    w = win.set_index("window")
    idx = w.index[w.whale].to_numpy()
    runs, out = [], []
    for i in idx:
        if runs and i - runs[-1][1] <= gap_s + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    for a, b in runs:
        if b + 1 - a < min_s:
            continue
        seg = w.loc[a:b]
        out.append(dict(t0=int(a), t1=int(b + 1), dur=int(b + 1 - a), whale_windows=int(seg.whale.sum()),
                        max_n=int(seg.n.max()), med_n=float(seg.n.median()), max_size=round(float(seg["size"].max()), 3),
                        med_size=round(float(seg["size"].median()), 3)))
    return out


# ------------------------------------------------------------------------------------------------ stage 2: run

METHODS = {
    # name: (model class, processor class, checkpoint, how tracks start)
    "owl_sam3": ("Sam3TrackerVideoModel", "Sam3TrackerVideoProcessor", "facebook/sam3", "owl"),
    "owl_sam2": ("Sam2VideoModel", "Sam2VideoProcessor", "facebook/sam2.1-hiera-large", "owl"),
    "text_sam3": ("Sam3VideoModel", "Sam3VideoProcessor", "facebook/sam3", "text"),
}
METHOD_LABEL = {"owl_sam3": "OWLv2 boxes -> SAM 3 tracker", "owl_sam2": "OWLv2 boxes -> SAM 2.1 video",
                "text_sam3": 'SAM 3 video, text "whale"'}
KEEP_HISTORY = 24      # frames of tracker memory kept (SAM 2 / 3 read 6 memory frames and up to 16 object pointers)
START_MIN_SIZE = 0.10  # a stage-1 box starts a track only if its longest side >= this share of the frame width
SMALL_SIZE = 0.10      # a track is drawn as "small" on frames where its outline's longest side < this share
SMALL_TRACK = 0.15     # a track whose outline stays below this share (90th percentile over its frames) is fish-sized:
                       # drawn grey, labelled "small", no whale number


def tracks_dir(video, t0: float, t1: float, method: str) -> Path:
    return paths.cache_dir("tracks") / stem_of(video) / f"{t0:07.2f}-{t1:07.2f}_{method}"


def is_done(video, t0: float, t1: float, method: str) -> bool:
    p = tracks_dir(video, t0, t1, method) / "meta.json"
    return p.exists() and bool(json.loads(p.read_text()).get("done"))


def _scale_box(b, sx, sy):
    return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]


def despeckle(m: np.ndarray, max_share: float = SPECK_MAX_SHARE, min_gap: float = SPECK_MIN_GAP_PX
              ) -> tuple[np.ndarray, dict]:
    """Drop far, small pieces of one object's outline (WP18 speck rule, see SPECK_MAX_SHARE). Returns (mask, info)
    with info = pieces, dropped (number of pieces), dropped_px, area_px."""
    import cv2
    m = np.asarray(m, bool)
    area = int(m.sum())
    info = dict(pieces=0, dropped=0, dropped_px=0, area_px=area)
    if area == 0:
        return m, info
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    info["pieces"] = n - 1
    if n <= 2:
        return m, info
    sizes = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(sizes))
    dist = cv2.distanceTransform((lab != main).astype(np.uint8), cv2.DIST_L2, 5)
    out = m.copy()
    for k in range(1, n):
        if k == main:
            continue
        a = int(sizes[k - 1])
        if a >= max_share * area:
            continue
        gap = float(dist[lab == k].min())
        if gap <= min_gap:
            continue
        out[lab == k] = False
        info["dropped"] += 1
        info["dropped_px"] += a
    return out, info


def box_covered(box_s: Sequence[float], masks: dict[int, np.ndarray], back: np.ndarray | None,
                own_frac: float = 0.5, fill_frac: float = 0.25, back_frac: float = 0.6) -> bool:
    """Is a stage-1 box (in mask pixel coordinates) already explained by a track? Covered when some track mask lies
    mostly inside it (>= ``own_frac`` of the mask), or fills >= ``fill_frac`` of it, or the box lies mostly
    (>= ``back_frac``) on the back."""
    x0, y0, x1, y1 = [int(round(v)) for v in box_s]
    x0, y0 = max(x0, 0), max(y0, 0)
    area = max((x1 - x0) * (y1 - y0), 1)
    if back is not None and back[y0:y1, x0:x1].sum() >= back_frac * area:
        return True
    for m in masks.values():
        tot = int(m.sum())
        if tot == 0:
            continue
        inter = int(m[y0:y1, x0:x1].sum())
        if inter >= own_frac * tot or inter >= fill_frac * area:
            return True
    return False


def _gpu_cap():
    from .detect import _torch_setup
    return _torch_setup()


class _Writer:
    """Collects per-object, per-frame masks (bit-packed) and table rows."""

    def __init__(self, n_frames: int, size=STORE_SIZE):
        self.n, (self.w, self.h) = n_frames, size
        self.packed: dict[int, np.ndarray] = {}
        self.rows: list[dict] = []

    def add(self, i: int, frame: int, t: float, obj: int, mask: np.ndarray, score: float):
        if obj not in self.packed:
            self.packed[obj] = np.zeros((self.n, self.h, (self.w + 7) // 8), np.uint8)
        area = int(mask.sum())
        row = dict(i=i, frame=frame, t=t, obj=obj, score=float(score), area=area, x0=np.nan, y0=np.nan, x1=np.nan,
                   y1=np.nan)
        if area:
            self.packed[obj][i] = np.packbits(mask, axis=1)
            ys, xs = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
            sx, sy = W / self.w, H / self.h
            row.update(x0=xs[0] * sx, y0=ys[0] * sy, x1=(xs[-1] + 1) * sx, y1=(ys[-1] + 1) * sy)
        self.rows.append(row)

    def save(self, out: Path, meta: dict):
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "masks.npz", **{f"obj_{k}": v for k, v in self.packed.items()})
        pd.DataFrame(self.rows).to_parquet(out / "tracks.parquet", index=False)
        (out / "meta.json").write_text(json.dumps(meta, indent=1))


_LOADED: dict = {}


def _load_model(method: str, dev: str, dtype):
    """Processor and model of a method, loaded once per process (many short runs in one GPU job)."""
    import importlib
    key = (method, dev, str(dtype))
    if key not in _LOADED:
        model_cls, proc_cls, ckpt, _ = METHODS[method]
        tf = importlib.import_module("transformers")
        proc = getattr(tf, proc_cls).from_pretrained(ckpt)
        model = getattr(tf, model_cls).from_pretrained(ckpt, torch_dtype=dtype).to(dev).eval()
        _LOADED.clear()
        _LOADED[key] = (proc, model)
    return _LOADED[key]


def _in_ranges(f: int, ranges) -> bool:
    return any(a <= f < b for a, b in ranges or ())


def run_tracks(video, t0: float, t1: float, method: str = "owl_sam3", prompt: str = "whale", force: bool = False,
               max_frames: int | None = None, hide: Sequence[tuple[int, int]] = (),
               gate: Sequence[tuple[int, int]] = (), frames: tuple[int, int] | None = None,
               name: str | None = None, reverse: bool = False, log=print) -> Path:
    """Track every other whale (and the back) from ``t0`` to ``t1`` s. Returns the output folder (see module doc).
    A finished stretch is not run again (``meta.json`` with ``done``).

    Blocked views (WP8b), as absolute frame-index ranges [a, b):

    - ``hide``: these frames are not shown to the tracker at all; it meets a jump (its own frame counter stays
      consecutive, so the last frames before the gap are its most recent memory). No masks are stored for them.
    - ``gate``: these frames are shown and get masks, but nothing from them enters the tracker's memory (their memory
      entries and object pointers are deleted after each step).
    - In both, no new track starts inside the ranges.

    ``frames`` = (first, end) frame indices instead of ``t0``/``t1`` (first is still moved to a stage-1 frame);
    ``name`` = output folder name under ``tracks/<stem>/``. ``reverse``: run the stretch backwards (last frame first;
    frames are held in memory at the stored mask size, so keep such stretches short); outputs are stored by forward
    frame position as usual (WP8c forward / backward check)."""
    out = tracks_dir(video, t0, t1, method) if name is None else tracks_dir(video, 0, 0, method).with_name(name)
    if max_frames is not None:          # short trial runs never stand in for the full stretch
        out = out.with_name(out.name + f"_first{max_frames}")
    if (out / "meta.json").exists() and json.loads((out / "meta.json").read_text()).get("done") and not force:
        log(f"{out.name}: done already")
        return out
    from . import video as uv
    torch, dev = _gpu_cap()
    dev = os.environ.get("UWID_DEVICE", dev)       # "cpu" for a slow check without the GPU
    model_cls, proc_cls, ckpt, start = METHODS[method]
    if start != "owl" and (hide or gate):
        raise ValueError("hide / gate are implemented for the box-started trackers only")
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    proc, model = _load_model(method, dev, dtype)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    s1_boxes, s1_frames, s1_fps = stage1_boxes(video) if start == "owl" else (None, np.zeros(0, int), None)
    tab = uv.frame_table(video)
    i0, i1 = (tab.index_at(t0), tab.index_at(t1)) if frames is None else (int(frames[0]), int(frames[1]))
    i0 = int(np.ceil(i0 / STAGE1_STRIDE) * STAGE1_STRIDE)   # start on a stage-1 frame (same for every method)
    n = i1 - i0 if max_frames is None else min(max_frames, i1 - i0)
    wr = _Writer(n)
    sw, sh = STORE_SIZE
    sx, sy = sw / W, sh / H
    sess = proc.init_video_session(inference_device=dev, dtype=dtype)
    if start == "text":
        sess = proc.add_text_prompt(sess, prompt)
    s1_by_frame = {int(f): g for f, g in s1_boxes.groupby("frame")} if s1_boxes is not None else {}
    s1_set = set(int(f) for f in s1_frames)
    prev: dict[int, np.ndarray] = {}
    next_id = 1
    events = []
    t_model = 0.0
    t_start = time.time()
    k = -1                                   # the tracker's own frame counter
    n_fed = 0
    with torch.inference_mode():
        def _source():
            gen = uv.iter_frames(video, start=float(tab.times[i0]) - 1e-3, end=float(tab.times[i0 + n - 1]) + 1e-3)
            if not reverse:
                yield from enumerate(gen)
                return
            import cv2
            buf = []
            for j, f_ in enumerate(gen):
                if j >= n:
                    break
                buf.append((j, uv.Frame(f_.index, f_.t, cv2.resize(f_.image, (sw, sh), interpolation=cv2.INTER_AREA))))
            yield from reversed(buf)

        psc = sw / W if reverse else 1.0        # prompt scale: reversed frames are held at the stored size
        for i, fr in _source():
            if i >= n:
                break
            hidden = _in_ranges(fr.index, hide)
            gated = _in_ranges(fr.index, gate)
            if hidden:
                continue
            k = i if not (hide or reverse) else k + 1     # hidden / reversed: consecutive counter; else the offset
            n_fed += 1
            img = fr.image
            tm = time.time()
            inputs = proc(images=img, device=dev, return_tensors="pt")
            pix = inputs.pixel_values[0].to(dtype)
            if start == "owl":
                # every object prompted on this frame goes in ONE call: the session keeps only the last call's list
                # of objects with new inputs
                new = [(BACK_ID, [float(v) for v in BACK_BOX])] if n_fed == 1 else []
                if fr.index in s1_set and not gated:
                    back = prev.get(BACK_ID)
                    whales = {kk: v for kk, v in prev.items() if kk != BACK_ID}
                    for _, b in s1_by_frame.get(fr.index, s1_boxes.iloc[0:0]).sort_values("score", ascending=False).iterrows():
                        box = [b.x0, b.y0, b.x1, b.y1]
                        if max(b.x1 - b.x0, b.y1 - b.y0) < START_MIN_SIZE * W:
                            continue
                        if box_covered(_scale_box(box, sx, sy), whales, back):
                            continue
                        # also not covered by a box already started on this frame
                        if any(_box_iou(box, nb) > 0.3 for kk, nb in new if kk != BACK_ID):
                            continue
                        new.append((next_id, box))
                        events.append(dict(i=i, frame=fr.index, t=fr.t, obj=next_id, event="start", score=float(b.score),
                                           box=[round(float(v), 1) for v in box]))
                        next_id += 1
                if new:
                    proc.add_inputs_to_inference_session(sess, frame_idx=k, obj_ids=[kk for kk, _ in new],
                                                         input_boxes=[[[v * psc for v in nb] for _, nb in new]],
                                                         original_size=(int(round(H * psc)), int(round(W * psc))))
                if sess.get_obj_num() == 0:
                    t_model += time.time() - tm
                    continue
                o = model(inference_session=sess, frame_idx=k, frame=pix)
                ms = proc.post_process_masks([o.pred_masks], original_sizes=[[sh, sw]], binarize=False)[0][:, 0] > 0
                ms = ms.cpu().numpy()
                sc = o.object_score_logits.float().view(-1).cpu().numpy()
                ids = list(o.object_ids)
                if gated:                    # nothing from a blocked frame stays in memory
                    for d in sess.output_dict_per_obj.values():
                        d["non_cond_frame_outputs"].pop(k, None)
            else:
                o = model(inference_session=sess, frame_idx=k, frame=pix)
                pp = proc.postprocess_outputs(sess, o, original_sizes=[[sh, sw]])
                ids = [int(x) for x in pp["object_ids"].tolist()]
                ms = pp["masks"].cpu().numpy().astype(bool)
                sc = pp["scores"].float().cpu().numpy()
                ids = [x + 1 for x in ids]       # SAM 3 ids start at 0; 0 is the back here
            t_model += time.time() - tm
            prev = {}
            for kk, m, s_ in zip(ids, ms, sc):
                wr.add(i, fr.index, fr.t, int(kk), m, s_)
                prev[int(kk)] = m
            _prune(sess, k)
            if i % 60 == 0:
                log(f"  {method} frame {i}/{n} objs {len(ids)} gpu peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                    f"{(i + 1) / max(time.time() - t_start, 1e-6):.2f} fr/s")
    elapsed = time.time() - t_start
    meta = dict(video=str(video), t0=t0, t1=t1, method=method, label=METHOD_LABEL[method], prompt=prompt, ckpt=ckpt,
                frames=[int(i0), int(i0 + n)], n_frames=n, fps=float(1 / tab.frame_step), store_size=list(STORE_SIZE),
                back_box=list(BACK_BOX), start_min_size=START_MIN_SIZE, events=events,
                hide=[list(map(int, r)) for r in hide], gate=[list(map(int, r)) for r in gate], n_fed=n_fed,
                reverse=bool(reverse),
                seconds_total=round(elapsed, 1), seconds_model=round(t_model, 1),
                gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", done=True)
    wr.save(out, meta)
    log(f"{out.name}: {n} frames ({n_fed} shown) in {elapsed:.0f} s, peak {meta['gpu_peak_gb']} GB")
    del sess
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _prune(sess, i: int, keep: int = KEEP_HISTORY):
    """Drop tracker history older than ``keep`` frames and decoded frames older than the last one, so memory stays
    flat over a long stretch (streaming)."""
    old = i - keep
    for d in sess.output_dict_per_obj.values():
        nc = d.get("non_cond_frame_outputs", {})
        for f in [f for f in nc if f < old]:
            del nc[f]
    pf = sess.processed_frames
    if pf:
        # keep the keys: the session counts frames with len(processed_frames) (object-pointer reach depends on it)
        for f in [f for f in pf if f < i and pf[f] is not None]:
            pf[f] = None


# ------------------------------------------------------------------------------------------------ load + clean

@dataclass
class Tracks:
    """A finished stretch, cleaned for display: ``roles`` per raw object id ('whale', 'back', 'on_back' = dropped
    because it lies mostly on the back, 'flicker' = present on < 3 frames, 'small' = fish-sized throughout, 'duplicate'
    = hidden as a copy of an older track on nearly all its frames), ``number`` = display number of each whale
    (1.. in order of first appearance), ``hidden`` = set of (i, obj) frames where a whale outline duplicates an older
    track and is not drawn."""
    dir: Path
    meta: dict
    table: pd.DataFrame
    packed: dict[int, np.ndarray]
    back: np.ndarray | None                   # packed back masks (n, h, w/8) or None
    roles: dict[int, str] = field(default_factory=dict)
    number: dict[int, int] = field(default_factory=dict)
    small_number: dict[int, int] = field(default_factory=dict)   # fish-sized tracks: s1, s2, .. (first seen order)
    hidden: set = field(default_factory=set)
    back_share: dict[int, float] = field(default_factory=dict)
    screened: dict[int, str] = field(default_factory=dict)       # objects given a fixed role by a purity screen
    endings: list = field(default_factory=list)                   # WP8c: (i, number-before, reason, (x, y)) per cut
    links: list = field(default_factory=list)                     # WP11a: every lone-link candidate with evidence
    link_marks: list = field(default_factory=list)                # WP11a: (i, number, text, (x, y)) for drawing
    own_back: np.ndarray | None = None        # WP17: own-back template at store size (flag own_back_template), or None
    despeck: bool = False                     # WP18: Tracks.mask drops far specks of whale outlines (load_online sets it)
    _speck_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return int(self.meta["n_frames"])

    def mask(self, obj: int, i: int) -> np.ndarray:
        w = int(self.meta["store_size"][0])
        src = self.back if obj == BACK_ID else self.packed[obj]
        m = np.unpackbits(src[i], axis=1)[:, :w].astype(bool)
        if self.despeck and obj != BACK_ID and self.roles.get(obj) == "whale":
            key = (obj, i)
            if key not in self._speck_cache:
                if len(self._speck_cache) > 4096:
                    self._speck_cache.clear()
                self._speck_cache[key] = despeckle(m)[0]
            return self._speck_cache[key]
        return m

    def raw_mask(self, obj: int, i: int) -> np.ndarray:
        """The stored outline, without the WP18 speck rule."""
        w = int(self.meta["store_size"][0])
        src = self.back if obj == BACK_ID else self.packed[obj]
        return np.unpackbits(src[i], axis=1)[:, :w].astype(bool)

    def is_small(self, obj: int, i: int) -> bool:
        m = self.mask(obj, i)
        ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        if not len(ys):
            return True
        return max(xs[-1] - xs[0] + 1, ys[-1] - ys[0] + 1) / m.shape[1] < SMALL_SIZE

    def visible(self, obj: int, i: int) -> bool:
        if self.roles.get(obj) not in ("whale", "back", "small") or (i, obj) in self.hidden:
            return False
        src = self.back if obj == BACK_ID else self.packed.get(obj)
        return src is not None and bool(src[i].any())


def read_run(run_dir: str | os.PathLike, back_from: str | os.PathLike | None = None) -> Tracks:
    """A run as stored, without cleaning (roles empty). ``back_from``: see :func:`load_tracks`."""
    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text())
    tab = pd.read_parquet(d / "tracks.parquet")
    z = np.load(d / "masks.npz")
    packed = {int(k.split("_")[1]): z[k] for k in z.files}
    back = packed.pop(BACK_ID, None) if meta["method"] != "text_sam3" else None
    if meta["method"] == "text_sam3" and back_from is not None:
        zb = np.load(Path(back_from) / "masks.npz")
        back = zb[f"obj_{BACK_ID}"] if f"obj_{BACK_ID}" in zb.files else None
    tr = Tracks(d, meta, tab, packed, back)
    tpl = ownback.load_run_template(d)
    if tpl is not None and back is not None:          # WP17 flag was on for this run: own back = back inside template
        w, h = (int(v) for v in meta["store_size"])
        tr.own_back = ownback.upsample(tpl[0], (h, w))
    return tr


def load_tracks(run_dir: str | os.PathLike, back_from: str | os.PathLike | None = None, on_back: float = 0.6,
                dup: float = 0.7, min_frames: int = 3) -> Tracks:
    """Load a run and clean it: drop whale tracks whose outline lies >= ``on_back`` on the back (median over the
    frames where it is seen), drop tracks seen on fewer than ``min_frames`` frames, hide a whale outline on a frame
    where >= ``dup`` of the smaller of two outlines lies inside the other (the younger track is hidden). ``back_from``
    = another run of the same stretch to take the back from (for the text-prompt method, which has no back)."""
    return clean(read_run(run_dir, back_from), on_back=on_back, dup=dup, min_frames=min_frames)


def clean(tr: Tracks, on_back: float = 0.6, dup: float = 0.7, min_frames: int = 3) -> Tracks:
    """Roles, numbers and duplicate hiding (see :func:`load_tracks`); objects listed in ``tr.screened`` keep the
    role they were given there."""
    meta, tab, packed, back = tr.meta, tr.table, tr.packed, tr.back
    tr.roles, tr.number, tr.hidden, tr.back_share, tr.small_number = {}, {}, set(), {}, {}
    if back is not None:
        tr.roles[BACK_ID] = "back"
    w = int(meta["store_size"][0])
    unpack = lambda p: np.unpackbits(p, axis=1)[:, :w].astype(bool)  # noqa: E731
    first = {}
    for k, p in packed.items():
        seen = np.nonzero(p.reshape(len(p), -1).any(1))[0]
        if len(seen) and k in tr.screened:
            tr.roles[k] = tr.screened[k]
            first[k] = int(seen[0])
            continue
        if len(seen) < min_frames:
            tr.roles[k] = "flicker"
            continue
        first[k] = int(seen[0])
        if back is not None:
            sh = []
            for i in seen[:: max(1, len(seen) // 40)]:
                m = unpack(p[i])
                b = unpack(back[i]) if tr.own_back is None else unpack(back[i]) & tr.own_back
                sh.append((m & b).sum() / max(m.sum(), 1))
            tr.back_share[k] = float(np.median(sh))
            if tr.back_share[k] >= on_back:
                tr.roles[k] = "on_back"
                continue
        tb = tab[(tab.obj == k) & (tab.area > 0)]
        size = np.maximum(tb.x1 - tb.x0, tb.y1 - tb.y0) / W
        tr.roles[k] = "whale" if len(size) and np.percentile(size, 90) >= SMALL_TRACK else "small"
    whales = sorted([k for k, r in tr.roles.items() if r == "whale"], key=lambda k: (first[k], k))
    # per-frame duplicates: younger track hidden where it sits inside an older one (or the older inside it)
    for i in range(tr.n):
        on = [k for k in whales if packed[k][i].any()]
        if len(on) < 2:
            continue
        ms = {k: unpack(packed[k][i]) for k in on}
        for a_ in range(len(on)):
            for b_ in range(a_ + 1, len(on)):
                a, b = on[a_], on[b_]
                inter = (ms[a] & ms[b]).sum()
                if inter >= dup * min(ms[a].sum(), ms[b].sum()):
                    tr.hidden.add((i, b if first[b] >= first[a] else a))
    # a track hidden as a duplicate on (nearly) all its frames gets no number of its own
    for k in whales:
        seen = sum(1 for i in range(tr.n) if packed[k][i].any() and (i, k) not in tr.hidden)
        if seen < min_frames:
            tr.roles[k] = "duplicate"
    whales = [k for k in whales if tr.roles[k] == "whale"]
    tr.number = {k: j + 1 for j, k in enumerate(whales)}
    smalls = sorted([k for k, r in tr.roles.items() if r == "small"], key=lambda k: (first[k], k))
    tr.small_number = {k: j + 1 for j, k in enumerate(smalls)}
    return tr


def track_summary(tr: Tracks, min_gap: int = 5) -> pd.DataFrame:
    """One row per displayed whale: first / last time seen, frames seen, gaps (runs of >= ``min_gap`` frames unseen
    between first and last), frames drawn as small, frames hidden as duplicate. Times are the frames' times in the
    recording."""
    fps = tr.meta["fps"]
    tt = tr.table.groupby("i").t.first()
    ts = lambda i: float(tt.get(i, tr.meta["t0"] + i / fps))  # noqa: E731
    rows = []
    for k, num in sorted(tr.number.items(), key=lambda x: x[1]):
        vis = np.array([tr.visible(k, i) for i in range(tr.n)])
        idx = np.nonzero(vis)[0]
        if not len(idx):
            rows.append(dict(number=num, obj=k, frames_seen=0))
            continue
        gaps = []
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a - 1 >= min_gap:
                gaps.append(f"{ts(a + 1):.1f}-{ts(b):.1f}")
        small = sum(tr.is_small(k, i) for i in idx)
        rows.append(dict(number=num, obj=k, first_s=round(ts(idx[0]), 2), last_s=round(ts(idx[-1]), 2),
                         frames_seen=int(len(idx)), gaps_s=";".join(gaps), small_frames=int(small),
                         dup_hidden=int(sum(1 for (i, o) in tr.hidden if o == k)),
                         back_share=round(tr.back_share.get(k, np.nan), 3)))
    return pd.DataFrame(rows)


def _t_of(tr: Tracks, i: int) -> float:
    t = tr.table.loc[tr.table.i == i, "t"]
    return round(float(t.iloc[0]), 2) if len(t) else round(tr.meta["t0"] + i / tr.meta["fps"], 2)


def reference_overlap(tr: Tracks, ref_masks: str | os.PathLike) -> pd.DataFrame:
    """IoU of the drawn outlines with the reference masks (WP3 AI by-eye labels) on the reference frames inside the
    stretch. Whale instances are matched one-to-one to visible whale tracks (Hungarian on IoU); the back is compared
    with the back class. Fish instances are ignored."""
    from PIL import Image
    from scipy.optimize import linear_sum_assignment
    ref = Path(ref_masks)
    f0, f1 = tr.meta["frames"]
    sw, sh = tr.meta["store_size"]
    rows = []
    for mp in sorted(ref.glob("*_meta.json")):
        f = int(mp.name.split("_")[0])
        if not (f0 <= f < f1):
            continue
        i = f - f0
        meta = json.loads(mp.read_text())
        inst = np.array(Image.open(ref / f"{f:06d}_inst.png").resize((sw, sh), Image.NEAREST))
        cls = np.array(Image.open(ref / f"{f:06d}_class.png").resize((sw, sh), Image.NEAREST))
        wh = [(int(k), v) for k, v in meta["instances"].items() if v["class"] == "whale"]
        tracks_on = [k for k in tr.number if tr.visible(k, i)]
        tm = {k: tr.mask(k, i) for k in tracks_on}
        R = np.zeros((len(wh), len(tracks_on)))
        for a, (iid, _) in enumerate(wh):
            g = inst == iid
            for b, k in enumerate(tracks_on):
                u = (g | tm[k]).sum()
                R[a, b] = (g & tm[k]).sum() / u if u else 0
        match = {}
        if R.size:
            ra, cb = linear_sum_assignment(-R)
            match = {a: (tracks_on[b], R[a, b]) for a, b in zip(ra, cb) if R[a, b] > 0}
        for a, (iid, v) in enumerate(wh):
            k, iou = match.get(a, (None, 0.0))
            rows.append(dict(frame=f, t=_t_of(tr, i), what="whale", ref_id=iid,
                             grade=v.get("grade", ""), note=v.get("note", ""), track=tr.number.get(k) if k else None,
                             iou=round(float(iou), 3), ref_px=int((inst == iid).sum())))
        matched = {m[0] for m in match.values()}
        for k in tracks_on:
            if k not in matched:
                rows.append(dict(frame=f, t=_t_of(tr, i), what="unmatched track",
                                 track=tr.number[k], iou=np.nan, ref_px=0))
        if tr.back is not None and (cls == 2).sum() > 1000:
            b, g = tr.mask(BACK_ID, i), cls == 2
            rows.append(dict(frame=f, t=_t_of(tr, i), what="back",
                             iou=round(float((b & g).sum() / max((b | g).sum(), 1)), 3), ref_px=int(g.sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ render

# whale colours (RGB): warm and bright so they stand out on blue water; the back and "small" have their own
PALETTE = [(255, 59, 48), (255, 214, 10), (255, 55, 255), (124, 252, 0), (255, 149, 0), (255, 255, 255),
           (191, 90, 242), (255, 111, 145), (0, 255, 170), (255, 190, 150), (170, 255, 90), (255, 0, 120)]
BACK_COLOUR = (120, 200, 255)      # thin light blue
SMALL_COLOUR = (160, 160, 160)


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _contours(m: np.ndarray, min_area: float = 30.0):
    import cv2
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) >= min_area] or list(cs)
    return cs


def _label(img, text, anchor, colour, scale=0.6, thick=1):
    import cv2
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick + 1)
    x = int(np.clip(anchor[0] - tw // 2, 2, img.shape[1] - tw - 6))
    y = int(np.clip(anchor[1] - 10, th + 46, img.shape[0] - 6))
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), colour, -1)
    dark = (0, 0, 0) if sum(colour) > 300 else (255, 255, 255)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, dark, thick + 1, cv2.LINE_AA)


def draw_frame(img: np.ndarray, tr: Tracks, i: int, header: str, lost: dict[int, float]) -> np.ndarray:
    """Outlines of frame ``i`` onto ``img`` (RGB at the stored mask size)."""
    import cv2
    out = img.copy()
    labels = []
    if tr.back is not None and tr.visible(BACK_ID, i):
        cs = _contours(tr.mask(BACK_ID, i), 200)
        cv2.polylines(out, cs, True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(out, cs, True, BACK_COLOUR, 1, cv2.LINE_AA)
        big = max(cs, key=cv2.contourArea)
        top = big[big[:, 0, 1].argmin(), 0]
        labels.append(("back", (int(top[0]), int(top[1])), BACK_COLOUR, 0.5))
    for k, r in tr.roles.items():
        if r == "small" and tr.visible(k, i):
            cs = _contours(tr.mask(k, i), 5)
            cv2.polylines(out, cs, True, SMALL_COLOUR, 1, cv2.LINE_AA)
            big = max(cs, key=cv2.contourArea)
            top = big[big[:, 0, 1].argmin(), 0]
            labels.append((f"s{tr.small_number.get(k, 0)} small", (int(top[0]), int(top[1])), SMALL_COLOUR, 0.4))
    in_view = []
    shown_nums = set()
    for k, num in sorted(tr.number.items(), key=lambda x: x[1]):
        if not tr.visible(k, i):
            continue
        m = tr.mask(k, i)
        ys, xs = np.nonzero(m)
        size = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / m.shape[1]
        cs = _contours(m)
        small = size < SMALL_SIZE
        col = SMALL_COLOUR if small else PALETTE[(num - 1) % len(PALETTE)]
        if small:
            cv2.polylines(out, cs, True, col, 1, cv2.LINE_AA)
        else:
            cv2.polylines(out, cs, True, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.polylines(out, cs, True, col, 3, cv2.LINE_AA)
        big = max(cs, key=cv2.contourArea)
        top = big[big[:, 0, 1].argmin(), 0]
        labels.append((f"{num} small" if small else f"{num}", (int(top[0]), int(top[1])), col, 0.5 if small else 0.7))
        if num not in shown_nums:
            in_view.append((num, small))
            shown_nums.add(num)
    for text, anchor, col, sc in labels:
        _label(out, text, anchor, col, scale=sc)
    # WP8c: where a number was ended by a purity guard, say so for 1.5 s at its last position
    fps = tr.meta.get("fps", 30.0)
    marks = [(e_i, f"{num} ended: {reason}", pos) for e_i, num, reason, pos in tr.endings] + \
            [(e_i, f"{num} linked: {text}", pos) for e_i, num, text, pos in tr.link_marks]
    for e_i, txt, pos in marks:
        if pos is not None and 0 <= i - e_i < 1.5 * fps:
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x = int(np.clip(pos[0] - tw // 2, 2, out.shape[1] - tw - 4))
            y = int(np.clip(pos[1] + 24, th + 50, out.shape[0] - 8))
            cv2.rectangle(out, (x - 3, y - th - 4), (x + tw + 3, y + 4), (20, 20, 20), -1)
            cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # header: two lines on a dark band
    band = out[:44].astype(np.float32) * 0.35
    out[:44] = band.astype(np.uint8)
    cv2.putText(out, header, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x = 8
    cv2.putText(out, "in view:", (x, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x += 70
    for num, small in in_view:
        col = SMALL_COLOUR if small else PALETTE[(num - 1) % len(PALETTE)]
        cv2.putText(out, f"{num}", (x, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        x += 14 + 12 * len(str(num))
    for k, sn in sorted(tr.small_number.items(), key=lambda x: x[1]):
        if tr.visible(k, i):
            cv2.putText(out, f"s{sn}", (x, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, SMALL_COLOUR, 1, cv2.LINE_AA)
            x += 14 + 10 * len(str(sn))
    if lost:
        x += 16
        txt = "lost: " + "  ".join(f"{n} (since {t:.1f} s)" for n, t in sorted(lost.items()))
        cv2.putText(out, txt, (x, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out


def render_tracks(tr: Tracks, out_mp4: str | os.PathLike, out_sheet: str | os.PathLike | None = None,
                  crf: int = 21, n_sheet: int = 8, log=print) -> dict:
    """Draw the outlines of a cleaned run onto its stretch of video: H.264 .mp4 (yuv420p, no sound, full frame rate,
    at the stored mask size) and a sheet of ``n_sheet`` frames across the stretch."""
    import cv2
    from . import timecode, video as uv
    meta = tr.meta
    vid = meta["video"]
    sw, sh = meta["store_size"]
    f0, f1 = meta["frames"]
    tab = uv.frame_table(vid)
    fps_num = meta["fps"]
    glyphs = timecode.load_glyphs()
    import datetime as _dt
    import re
    m = re.search(r"-(20\d{2})(\d{2})(\d{2})-", Path(vid).name)
    date_hint = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sw}x{sh}",
           "-r", f"{fps_num:.5f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sheet_at = set(np.linspace(0, tr.n - 1, n_sheet).round().astype(int).tolist())
    sheet = []
    last_seen: dict[int, float] = {}
    last_small: dict[int, bool] = {}
    name = Path(vid).name
    for i, fr in enumerate(uv.iter_frames(vid, start=float(tab.times[f0]) - 1e-3, end=float(tab.times[f1 - 1]) + 1e-3)):
        if i >= tr.n:
            break
        rd = timecode.read_clock(fr.image, glyphs, date_hint=date_hint)
        clock = rd.text if rd is not None else "--"
        img = cv2.resize(fr.image, (sw, sh), interpolation=cv2.INTER_AREA)
        t = fr.t
        lost = {}
        seen_now = {num for k, num in tr.number.items() if tr.visible(k, i)}   # linked fragments share a number
        for k, num in tr.number.items():
            if tr.visible(k, i):
                last_seen[num] = t
                last_small[num] = tr.is_small(k, i)
            elif num in last_seen and not last_small.get(num) and num not in seen_now:
                lost[num] = last_seen[num]        # only tracks last seen large are listed as lost
        header = f"{name}   t = {t:7.2f} s   clock {clock}   {meta['label']}"
        out = draw_frame(img, tr, i, header, lost)
        pr.stdin.write(out.tobytes())
        if i in sheet_at:
            sheet.append(out)
    pr.stdin.close()
    pr.wait()
    res = dict(mp4=str(out_mp4), mb=round(out_mp4.stat().st_size / 1e6, 1))
    if out_sheet is not None and sheet:
        tiles = [cv2.resize(s, (sw // 2, sh // 2), interpolation=cv2.INTER_AREA) for s in sheet]
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)]
        cv2.imwrite(str(out_sheet), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        res["sheet"] = str(out_sheet)
    log(f"rendered {out_mp4.name}: {res['mb']} MB")
    return res


# ------------------------------------------------------------------------------------------------ WP8c: purity

@dataclass
class PurityConfig:
    """Guards that END a track (its number) instead of letting it continue, so that a number never moves from one
    animal to another; a whale the detector still sees gets a new number (a fragment). ``None`` switches a guard off.
    Applied to a finished run (:func:`purify`): cutting a track at frame i and numbering the rest anew is what the
    online policy would do, except that online the new number would start from the next stage-1 box.
    Defaults = the "strict" setting of WP8c (0 numbers moved between animals on the 50 gap runs, 1.95 tracks per
    whale; experiments/07_tracks_demo/tables.md)."""
    max_block_s: float | None = 1.0        # a number does not cross a hidden / blocked view longer than this ...
    lone_block_s: float | None = 4.0       # ... unless it was the only whale-sized track alive (then up to this)
    max_absent_s: float | None = 0.5       # a track unseen this long and then seen again gets a new number
    min_iou_prev: float | None = 0.45       # outline overlap with its own previous frame below this -> end
    max_area_ratio: float | None = 2.0     # outline area changes by more than this factor between frames -> end
    swallow: float | None = 0.35           # outline newly covers >= this share of another live track's outline -> end
    max_drift: float | None = 0.10         # cosine distance of the masked DINOv3 feature to the track's running mean
    grow_from_small: float | None = 1.5    # started fish-sized (< 0.15 of the width in its first second), grows > x
    start_clutter: float | None = 0.3     # numbered only from a sampled frame whose region is < this share bubbles,
                                           # surface or air patches (a start on clutter is not a whale)
    fish_speed: float | None = 1.5         # first second faster than this many body lengths / s and smaller than
    fish_size: float = 0.3                 # this share of the width -> fish-like: drawn `small`, no whale number
    fish_settle: float | None = None       # a track fish-like at its start ends where a 1 s window moves slower than
                                           # this (body lengths / s): it has settled on something that is not a fish
    blocked_no_outline: bool = True        # no outline is drawn on a blocked frame
    whale_size: float = 0.10               # "whale-sized" for the lone test


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def purity_cuts(tr: Tracks, cfg: PurityConfig, blocked: Sequence[tuple[int, int]] = (),
                feats: dict | None = None) -> tuple[dict, dict]:
    """Where each raw track must end, and which frame ranges must not be drawn.

    ``blocked``: frame-index ranges [a, b) of blocked views (absolute frames). The run's own hidden ranges (meta
    ``hide``) count as blocked too. ``feats``: the run's ``pure_feats.npz`` (keys i, obj, emb, cls) for the drift
    guard and the start screen. Returns (cuts, drops): cuts[obj] = [(i, reason)], drops[obj] = [(i0, i1, reason)]
    (positions in the run)."""
    f0 = tr.meta["frames"][0]
    fps = tr.meta["fps"]
    rng = [(a - f0, b - f0) for a, b in list(blocked) + [tuple(r) for r in tr.meta.get("hide", [])]]
    blk = np.zeros(tr.n, bool)
    for a, b in rng:
        blk[max(a, 0):max(min(b, tr.n), 0)] = True
    hid = np.zeros(tr.n, bool)
    for a, b in tr.meta.get("hide", []):
        hid[max(a - f0, 0):max(min(b - f0, tr.n), 0)] = True
    tab = tr.table[tr.table.area > 0].copy()
    tab["size"] = np.maximum(tab.x1 - tab.x0, tab.y1 - tab.y0) / W
    size = {(int(r.i), int(r.obj)): float(r.size) for r in tab.itertuples()}
    objs = sorted(tr.packed)
    vis = {k: tr.packed[k].reshape(tr.n, -1).any(1) & ~hid for k in objs}
    cuts: dict[int, list] = {k: [] for k in objs}
    drops: dict[int, list] = {k: [] for k in objs}
    cache: dict = {}

    def M(k, i):
        if (k, i) not in cache:
            if len(cache) > 400:
                cache.clear()
            cache[(k, i)] = tr.mask(k, i)
        return cache[(k, i)]

    # blocked spans: lengths, and which tracks were alive (whale-sized) on the last open frame before each
    spans = []
    i = 0
    while i < tr.n:
        if blk[i]:
            j = i
            while j < tr.n and blk[j]:
                j += 1
            alive = [k for k in objs if i > 0 and vis[k][i - 1] and size.get((i - 1, k), 0) >= cfg.whale_size]
            spans.append((i, j, (j - i) / fps, alive))
            i = j
        else:
            i += 1
    for k in objs:
        v = vis[k]
        seen = np.nonzero(v)[0]
        if not len(seen):
            continue
        # blocked views
        if cfg.max_block_s is not None:
            for a, b, dur, alive in spans:
                lone = len(alive) == 1 and alive[0] == k
                limit = cfg.lone_block_s if (lone and cfg.lone_block_s is not None) else cfg.max_block_s
                if dur > limit and seen[0] < a:
                    after = seen[seen >= b]
                    if len(after):
                        cuts[k].append((int(after[0]), f"blocked view {dur:.1f} s"))
        if cfg.blocked_no_outline:
            for a, b, dur, alive in spans:
                if v[a:b].any():
                    drops[k].append((a, b, "view blocked"))
        # absence, jumps, swallowing
        prev = None
        for i in seen:
            if blk[i]:
                continue
            if prev is not None:
                gap = (i - prev - int(blk[prev + 1:i].sum())) / fps     # unseen time outside blocked views
                if gap > 0 and cfg.max_absent_s is not None and gap > cfg.max_absent_s:
                    cuts[k].append((int(i), f"re-found after {gap:.1f} s"))
                elif i == prev + 1 and max(size.get((int(prev), k), 0), size.get((int(i), k), 0)) >= cfg.whale_size:
                    # fish-sized outlines jitter from frame to frame; the jump guards only watch whale-sized ones
                    if cfg.min_iou_prev is not None or cfg.max_area_ratio is not None:
                        m0, m1 = M(k, prev), M(k, i)
                        s0, s1 = m0.sum(), m1.sum()
                        if cfg.max_area_ratio is not None and max(s0, s1) > cfg.max_area_ratio * max(min(s0, s1), 1):
                            cuts[k].append((int(i), "area jump"))
                        elif cfg.min_iou_prev is not None and _mask_iou(m0, m1) < cfg.min_iou_prev:
                            cuts[k].append((int(i), "shape jump"))
                    if cfg.swallow is not None:
                        for j in objs:
                            if j == k or not vis[j][prev]:
                                continue
                            if size.get((int(prev), j), 0) < cfg.whale_size:
                                continue        # swallowing a fish-sized track is not a merge of two whales
                            mj = M(j, prev)
                            sj = mj.sum()
                            if (M(k, i) & mj).sum() >= cfg.swallow * sj and (M(k, prev) & mj).sum() < cfg.swallow * sj:
                                cuts[k].append((int(i), "swallowed another track"))
                                break
            prev = i
        # grows from fish-sized
        if cfg.grow_from_small is not None:
            first_s = [size.get((int(i), k), 0) for i in seen if i < seen[0] + fps]
            s0 = max(first_s) if first_s else 0
            if 0 < s0 < 0.15:
                for i in seen[seen >= seen[0] + fps]:
                    if size.get((int(i), k), 0) > cfg.grow_from_small * s0:
                        cuts[k].append((int(i), "grew from fish-sized"))
                        break
    # a fish-like start that settles: the outline has moved onto something slow (WP8c, clip 07: fish -> calf)
    if cfg.fish_settle is not None and cfg.fish_speed is not None:
        for k in objs:
            if not fish_like(tr.table, k, fps, cfg.fish_speed, cfg.fish_size):
                continue
            g = tr.table[(tr.table.obj == k) & (tr.table.area > 0)].sort_values("i")
            cx, cy = ((g.x0 + g.x1) / 2).to_numpy(), ((g.y0 + g.y1) / 2).to_numpy()
            sz = np.maximum(g.x1 - g.x0, g.y1 - g.y0).to_numpy()
            ii = g.i.to_numpy()
            win = int(round(fps))
            for st in range(win, len(g) - win, max(1, win // 4)):
                path = np.hypot(np.diff(cx[st:st + win]), np.diff(cy[st:st + win])).sum()
                if path / max(float(np.median(sz[st:st + win])), 1) < cfg.fish_settle:
                    cuts[k].append((int(ii[st]), "fish-like start settled"))
                    break
    # appearance drift
    if feats is not None and cfg.max_drift is not None and len(feats["i"]):
        for k in objs:
            sel = feats["obj"] == k
            ii, ee = feats["i"][sel], feats["emb"][sel].astype(np.float32)
            order = np.argsort(ii)
            ii, ee = ii[order], ee[order]
            ee /= np.maximum(np.linalg.norm(ee, axis=1, keepdims=True), 1e-6)
            cut_at = sorted(c for c, _ in cuts[k])
            mean = None
            for i, e in zip(ii, ee):
                if blk[i]:
                    continue
                if mean is not None and any(prev_i < c <= i for c in cut_at):
                    mean = None
                if mean is not None:
                    d = 1 - float(e @ (mean / np.linalg.norm(mean)))
                    if d > cfg.max_drift:
                        cuts[k].append((int(i), f"appearance drift {d:.2f}"))
                        cut_at.append(int(i))
                        mean = None
                mean = e.copy() if mean is None else 0.7 * mean + 0.3 * e
                prev_i = i
    for k in objs:
        cuts[k] = sorted({c[0]: c for c in sorted(cuts[k])}.values())
    # start screen: each segment is numbered only from its first sampled frame with a whale-like region
    if feats is not None and cfg.start_clutter is not None and len(feats["i"]):
        for k in objs:
            sel = feats["obj"] == k
            ii, cc = feats["i"][sel], feats["cls"][sel].astype(np.float32)
            bounds = [0] + [c for c, _ in cuts[k]] + [tr.n]
            for a, b in zip(bounds[:-1], bounds[1:]):
                inside = (ii >= a) & (ii < b) & ~blk[np.clip(ii, 0, tr.n - 1)]
                if not inside.any():
                    continue
                clutter = cc[inside][:, [4, 5, 6]].sum(1)                     # bubbles, surface, air
                ok = ii[inside][clutter < cfg.start_clutter]
                first_ok = int(ok.min()) if len(ok) else b
                if first_ok > a:
                    drops[k].append((a, first_ok, "starts on bubbles / surface / air"))
    return cuts, drops


def fish_like(tab: pd.DataFrame, k: int, fps: float, speed: float, max_size: float) -> bool:
    """First second of track ``k``: moves faster than ``speed`` body lengths per second while smaller than
    ``max_size`` of the frame width (fish dart; whales near the tag drift slowly relative to their size)."""
    g = tab[(tab.obj == k) & (tab.area > 0)].sort_values("i")
    if len(g) < 5:
        return False
    g = g[g.i < g.i.iloc[0] + fps]
    sz = np.maximum(g.x1 - g.x0, g.y1 - g.y0)
    if len(g) < 5 or sz.max() / W >= max_size:
        return False
    path = np.hypot(np.diff((g.x0 + g.x1) / 2), np.diff((g.y0 + g.y1) / 2)).sum()
    return path / max(float(sz.median()), 1) * fps / len(g) > speed


def apply_cuts(tr: Tracks, cuts: dict, drops: dict, cfg: "PurityConfig | None" = None, **clean_kw) -> Tracks:
    """A new Tracks where every raw track is split at its cuts into segments (objects ``k * 1000 + s``) and dropped
    frame ranges are not drawn; then cleaned and numbered as usual. ``endings`` lists each cut with the number the
    track had before it and the reason (for the renderer)."""
    packed, rows = {}, []
    w = int(tr.meta["store_size"][0])
    seg_of = {}
    for k, p in tr.packed.items():
        bounds = [0] + [c for c, _ in cuts.get(k, [])] + [tr.n]
        for s_, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            q = np.zeros_like(p)
            q[a:b] = p[a:b]
            for a2, b2, _ in drops.get(k, []):
                lo, hi = max(a, a2), min(b, b2)
                if hi > lo:
                    q[lo:hi] = 0
            if not q.any():
                continue
            nk = k * 1000 + s_
            packed[nk] = q
            seg_of[(k, s_)] = nk
            r = tr.table[(tr.table.obj == k) & (tr.table.i >= a) & (tr.table.i < b)].copy()
            r["obj"] = nk
            rows.append(r)
    tab = pd.concat(rows, ignore_index=True) if rows else tr.table.iloc[0:0]
    if len(tab):                # dropped frames have no outline: area 0 in the table
        empty = {nk: ~q.reshape(len(q), -1).any(1) for nk, q in packed.items()}
        gone = np.array([bool(empty[o][i]) for o, i in zip(tab.obj.to_numpy(), tab.i.to_numpy())])
        tab.loc[gone, "area"] = 0
    nt = Tracks(tr.dir, dict(tr.meta), tab, packed, tr.back)
    if cfg is not None and cfg.fish_speed is not None:
        for nk in packed:
            if fish_like(tab, nk, tr.meta["fps"], cfg.fish_speed, cfg.fish_size):
                nt.screened[nk] = "small"
    clean(nt, **clean_kw)
    for k, lst in cuts.items():
        for s_, (i, reason) in enumerate(lst):
            before = seg_of.get((k, s_))
            num = nt.number.get(before)
            if num is None:
                continue
            last = np.nonzero(nt.packed[before].reshape(nt.n, -1).any(1))[0]
            pos = None
            if len(last):
                m = nt.mask(before, int(last[-1]))
                ys, xs = np.nonzero(m)
                pos = (int(xs.mean()), int(ys.min()))
            nt.endings.append((int(i), num, reason, pos))
    return nt


def purify(tr: Tracks, cfg: PurityConfig, blocked=(), feats=None) -> Tracks:
    """The WP8c policy on a finished run: :func:`purity_cuts` then :func:`apply_cuts`."""
    cuts, drops = purity_cuts(tr, cfg, blocked, feats)
    return apply_cuts(tr, cuts, drops, cfg)


# ------------------------------------------------------------------------------------------------ WP11a: lone links

@dataclass
class LinkConfig:
    """A separate layer on top of the purity guards: where the scene holds only one whale-sized animal around a break,
    the fragment that ends and the next one that starts are the same animal and get one displayed number (their own
    ids stay). Set on the WP8 stretches (dev clip, 00006, 00007) and frozen before the held-out test (WP11a)."""
    window_s: float = 2.0            # no second whale-sized object this long before the break and after it
    max_gap_s: float = 8.0           # longest break without a breath (lone-whale gap runs: 6/6 kept at 8 s)
    max_gap_breath_s: float = 12.0   # with a sensor breath inside the break (longest lens-out of 164 breaths on two
    #                                  tags 9.7 s, + froth after the entry)
    min_size: float = 300 / 1920     # whale-sized for counting other animals (the scan's size rule)
    size_ratio: float = 1.5          # apparent size after / before at most this ...
    size_ratio_per_s: float = 0.25   # ... plus this per second of break
    max_app_dist: float = 0.30       # cosine distance of masked DINOv3 features, end of one fragment / start of the next
    promote_slow_small: bool = True  # a slow, steady object under the size limit is a whale (the calf), fish stay small
    promote_min_s: float = 1.0       # ... if it stays in view at least this long
    merge_duplicates: bool = False   # join two co-visible outlines lying one inside the other: OFF (on the gap runs 1
    #                                  of its 3 joins was two different whales)


def _frag_info(tr: Tracks, k: int) -> dict:
    vis = np.nonzero([tr.visible(k, i) for i in range(tr.n)])[0]
    t = tr.table[(tr.table.obj == k) & (tr.table.area > 0)]
    t = t.assign(size=np.maximum(t.x1 - t.x0, t.y1 - t.y0) / W)
    return dict(first=int(vis[0]), last=int(vis[-1]), n=len(vis), sizes=t.set_index("i")["size"].to_dict()) if len(vis) \
        else None


def link_lone(tr: Tracks, detector: pd.DataFrame | None = None, emb: dict | None = None,
              breaths: Sequence[tuple[int, int]] = (), cfg: LinkConfig | None = None) -> Tracks:
    """Link fragments across breaks in lone scenes (in place; returns ``tr``).

    ``detector``: whale objects of the start-box source (frame, x0, y0, x1, y1 in 1920 x 1080), used to see a second
    animal the tracker did not follow. ``emb``: dict(i, obj, emb) masked features per object and sampled frame.
    ``breaths``: frame ranges (absolute) where the depth sensor says the lens was out of the water.
    Adds ``tr.links`` (rows with every link's evidence and whether it was made) and ``tr.link_marks`` (for drawing);
    every fragment of a chain shows the chain's number."""
    cfg = cfg or LinkConfig()
    fps = tr.meta["fps"]
    f0 = tr.meta["frames"][0]
    # 1. the calf rule: slow, steady objects under the size limit are whales; fish-like ones stay small
    if cfg.promote_slow_small:
        for k, r in list(tr.roles.items()):
            if r == "small" and tr.screened.get(k) != "small":
                fi = _frag_info(tr, k)
                if fi and fi["n"] >= cfg.promote_min_s * fps and not fish_like(tr.table, k, fps, 1.5, 0.3):
                    tr.roles[k] = "whale"
    info = {k: _frag_info(tr, k) for k, r in tr.roles.items() if r == "whale"}
    info = {k: v for k, v in info.items() if v}
    order = sorted(info, key=lambda k: (info[k]["first"], k))
    # detector objects not explained by any track (fish tracks included): a second animal the tracker did not follow
    det_count = {}
    if detector is not None and len(detector):
        d = detector.copy()
        d = d[np.maximum(d.x1 - d.x0, d.y1 - d.y0) / W >= cfg.min_size]
        tb = tr.table[tr.table.area > 0]
        tb_by_i = {int(i): g[["x0", "y0", "x1", "y1"]].to_numpy() for i, g in tb.groupby("i")}
        for f, g in d.groupby("frame"):
            i = int(f) - f0
            boxes = g[["x0", "y0", "x1", "y1"]].to_numpy()
            kept = []
            for b in boxes:
                if all(_box_iou(b, q) <= 0.3 for q in kept):
                    kept.append(b)
            tbs = tb_by_i.get(i, np.zeros((0, 4)))
            unexplained = [b for b in kept if not any(_box_iou(b, q) >= 0.3 for q in tbs)]
            det_count[i] = len(unexplained)
    hidden = set()
    for a, b in tr.meta.get("blocked", []) + tr.meta.get("hide", []):
        hidden.update(range(a - f0, b - f0))

    def e_of(k, lo, hi, last=True):
        if emb is None or not len(emb["i"]):
            return None
        sel = (emb["obj"] == k) & (emb["i"] >= lo) & (emb["i"] <= hi)
        if not sel.any():
            return None
        ii = emb["i"][sel]
        e = emb["emb"][sel][np.argsort(ii)].astype(np.float32)
        e = e[-3:].mean(0) if last else e[:3].mean(0)
        return e / max(np.linalg.norm(e), 1e-6)

    succ, pred, rows = {}, {}, []
    tbx = tr.table[tr.table.area > 0].set_index(["i", "obj"])[["x0", "y0", "x1", "y1"]]
    tbx_d = {key: v for key, v in zip(tbx.index, tbx.to_numpy())}

    def box_of(k, i):
        return tbx_d.get((i, k))

    def judge(a, b):
        A, B = info[a], info[b]
        gap = (B["first"] - A["last"]) / fps
        breath = any(x - f0 < B["first"] and y - f0 > A["last"] for x, y in breaths)
        limit = cfg.max_gap_breath_s if breath else cfg.max_gap_s
        lo, hi = A["last"] - int(cfg.window_s * fps), B["first"] + int(cfg.window_s * fps)

        # a second whale-sized animal at the same time as A or B, or during the break (later fragments of the same
        # animal do not count; an outline mostly on A / B is a part of it, not another animal)
        def other_at(c, i):
            if info[c]["sizes"].get(i, 0) < cfg.min_size:
                return False
            ref = a if i <= A["last"] else b if i >= B["first"] else None
            if ref is None:
                return True
            if not tr.visible(ref, i):
                return False
            # an outline whose box touches A's / B's box is taken as a part of the same animal (a flipper tracked on
            # its own, a duplicate); another animal is somewhere else in the frame
            bc, br = box_of(c, i), box_of(ref, i)
            return bc is not None and br is not None and _box_iou(bc, br) == 0.0
        others = sorted({c for c in info if c not in (a, b) and any(
            lo <= i <= hi and other_at(c, i) for i in info[c]["sizes"])})
        # unexplained detector objects: a second animal needs >= 2 detector frames (1 s) in the window
        # detector frames the tracker did not see (blocked) cannot be explained by a track: left out (WP11a fix)
        det = {"before": sum(det_count.get(i, 0) > 0 for i in range(lo, A["last"] + 1) if i not in hidden),
               "during": sum(det_count.get(i, 0) > 1 for i in range(A["last"] + 1, B["first"]) if i not in hidden),
               "after": sum(det_count.get(i, 0) > 0 for i in range(B["first"], hi + 1) if i not in hidden)}
        lone = not others and det["before"] < 2 and det["after"] < 2 and det["during"] < 2
        sa = np.median([s_ for i, s_ in A["sizes"].items() if i >= A["last"] - fps / 2] or [0])
        sb = np.median([s_ for i, s_ in B["sizes"].items() if i <= B["first"] + fps / 2] or [0])
        ratio = max(sa, sb) / max(min(sa, sb), 1e-6)
        ea, eb = e_of(a, A["last"] - fps, A["last"]), e_of(b, B["first"], B["first"] + fps, last=False)
        dist = None if ea is None or eb is None else round(1 - float(ea @ eb), 3)
        checks_ok = ratio <= cfg.size_ratio + cfg.size_ratio_per_s * gap and (dist is None or dist <= cfg.max_app_dist)
        made = bool(lone and gap <= limit and checks_ok)
        return dict(a=a, b=b, t_end=round(float(tr.meta["t0"] + A["last"] / fps), 2), gap_s=round(gap, 2),
                    breath=breath, limit_s=limit, others=others, detector=det, lone=lone,
                    size_ratio=round(float(ratio), 2), app_dist=dist, checks_ok=bool(checks_ok), made=made,
                    why=("linked" if made else "not lone" if not lone else "gap too long" if gap > limit
                         else "checks contradict"))

    # duplicates: two fragments outlining the same animal at the same time (one inside the other on most shared
    # frames) are one animal; they share a number and do not count as "another animal" for each other
    same = {}
    keys = list(info) if cfg.merge_duplicates else []
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            p_, q_ = keys[x], keys[y]
            lo_, hi_ = max(info[p_]["first"], info[q_]["first"]), min(info[p_]["last"], info[q_]["last"])
            if hi_ - lo_ < 5:
                continue
            fr_ = [i for i in range(lo_, hi_ + 1, 3) if tr.visible(p_, i) and tr.visible(q_, i)]
            if len(fr_) < 2:
                continue
            cont = []
            for i in fr_:
                mp, mq = tr.mask(p_, i), tr.mask(q_, i)
                cont.append((mp & mq).sum() / max(min(mp.sum(), mq.sum()), 1))
            if np.median(cont) >= 0.5:
                same.setdefault(p_, set()).add(q_)
                same.setdefault(q_, set()).add(p_)
                rows.append(dict(a=p_, b=q_, why="same animal (outlines overlap)", made=True, gap_s=0.0,
                                 containment=round(float(np.median(cont)), 2)))
    _judge = judge

    def judge(a, b):                                       # noqa: F811
        r = _judge(a, b)
        if r["others"]:
            r["others"] = [c for c in r["others"] if c not in same.get(a, set()) | same.get(b, set())]
            if not r["others"] and r["why"] == "not lone":
                det = r["detector"]
                r["lone"] = det["before"] < 2 and det["after"] < 2 and det["during"] < 2
                lim = r["limit_s"]
                r["made"] = bool(r["lone"] and r["gap_s"] <= lim and r["checks_ok"])
                r["why"] = ("linked" if r["made"] else "not lone" if not r["lone"] else "gap too long"
                            if r["gap_s"] > lim else "checks contradict")
        return r

    for a in sorted(info, key=lambda k: info[k]["last"]):
        A = info[a]
        cands = sorted([b for b in order if b != a and b not in pred and info[b]["first"] > A["last"]
                        and (info[b]["first"] - A["last"]) / fps <= cfg.max_gap_breath_s], key=lambda x: info[x]["first"])
        for b in cands:                      # every judged candidate is kept (made or not), for later stages
            r = judge(a, b)
            rows.append(r)
            if r["made"]:
                succ[a], pred[b] = b, a
                break
    # chains -> displayed numbers (duplicates join the chain of the fragment they sit on)
    for k, others_ in same.items():
        for o in others_:
            if o not in pred and o not in succ and info[o]["first"] > info[k]["first"]:
                pred[o] = ("same", k)
    heads = [k for k in order if k not in pred]
    tr.link_marks = []
    num = {}
    dup_of = {o: v[1] for o, v in pred.items() if isinstance(v, tuple)}
    for j, h in enumerate(sorted(heads, key=lambda k: info[k]["first"])):
        k = h
        while True:
            num[k] = j + 1
            for o, host in dup_of.items():
                if host == k:
                    num[o] = j + 1
            if k not in succ:
                break
            nk = succ[k]
            m = tr.mask(nk, info[nk]["first"])
            ys, xs = np.nonzero(m)
            gap = (info[nk]["first"] - info[k]["last"]) / fps
            tr.link_marks.append((info[nk]["first"], j + 1, f"lone, gap {gap:.1f} s",
                                  (int(xs.mean()), int(ys.min())) if len(ys) else None))
            k = nk
    tr.number = num
    tr.links = rows
    tr.endings = [e for e in tr.endings]          # endings keep the number the fragment had before linking
    return tr
