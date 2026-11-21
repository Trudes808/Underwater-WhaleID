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
- a track that lies mostly inside the back is dropped, a track whose outline is fish-sized on a frame is drawn as
  ``small`` on that frame.

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

from . import paths

DEV_VIDEO = "/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
TEST_DIR = "/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96/Video"

STAGE1_QUERY, STAGE1_THR, STAGE1_STRIDE = "whale", 0.25, 15
STAGE1_RUNS = ("owl_owl_all_s15", "owl_owl_all_s5")
W, H = 1920, 1080
STORE_SIZE = (1280, 720)          # masks are kept at this size (w, h)
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


def tracks_dir(video, t0: float, t1: float, method: str) -> Path:
    return paths.cache_dir("tracks") / stem_of(video) / f"{t0:07.2f}-{t1:07.2f}_{method}"


def _scale_box(b, sx, sy):
    return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]


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


def run_tracks(video, t0: float, t1: float, method: str = "owl_sam3", prompt: str = "whale", force: bool = False,
               max_frames: int | None = None, log=print) -> Path:
    """Track every other whale (and the back) from ``t0`` to ``t1`` s. Returns the output folder (see module doc).
    Resumable in the sense that a finished stretch is not run again (``meta.json`` with ``done``)."""
    out = tracks_dir(video, t0, t1, method)
    if max_frames is not None:          # short trial runs never stand in for the full stretch
        out = out.with_name(out.name + f"_first{max_frames}")
    if (out / "meta.json").exists() and json.loads((out / "meta.json").read_text()).get("done") and not force:
        log(f"{out.name}: done already")
        return out
    import importlib
    from . import video as uv
    torch, dev = _gpu_cap()
    model_cls, proc_cls, ckpt, start = METHODS[method]
    tf = importlib.import_module("transformers")
    dtype = torch.bfloat16
    proc = getattr(tf, proc_cls).from_pretrained(ckpt)
    model = getattr(tf, model_cls).from_pretrained(ckpt, torch_dtype=dtype).to(dev).eval()
    torch.cuda.reset_peak_memory_stats()

    s1_boxes, s1_frames, s1_fps = stage1_boxes(video) if start == "owl" else (None, np.zeros(0, int), None)
    tab = uv.frame_table(video)
    i0, i1 = tab.index_at(t0), tab.index_at(t1)
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
    with torch.inference_mode():
        for i, fr in enumerate(uv.iter_frames(video, start=float(tab.times[i0]) - 1e-3, end=float(tab.times[i0 + n - 1]) + 1e-3)):
            if i >= n:
                break
            img = fr.image
            tm = time.time()
            inputs = proc(images=img, device=dev, return_tensors="pt")
            pix = inputs.pixel_values[0].to(dtype)
            if start == "owl":
                if i == 0:
                    proc.add_inputs_to_inference_session(sess, frame_idx=i, obj_ids=[BACK_ID], input_boxes=[[list(BACK_BOX)]],
                                                         original_size=(H, W))
                if fr.index in s1_set:
                    new = []
                    back = prev.get(BACK_ID)
                    whales = {k: v for k, v in prev.items() if k != BACK_ID}
                    for _, b in s1_by_frame.get(fr.index, pd.DataFrame()).sort_values("score", ascending=False).iterrows():
                        box = [b.x0, b.y0, b.x1, b.y1]
                        if max(b.x1 - b.x0, b.y1 - b.y0) < START_MIN_SIZE * W:
                            continue
                        if box_covered(_scale_box(box, sx, sy), whales, back):
                            continue
                        # also not covered by a box already started on this frame
                        if any(_box_iou(box, nb) > 0.3 for _, nb in new):
                            continue
                        new.append((next_id, box))
                        events.append(dict(i=i, frame=fr.index, t=fr.t, obj=next_id, event="start", score=float(b.score),
                                           box=[round(float(v), 1) for v in box]))
                        next_id += 1
                    if new:
                        proc.add_inputs_to_inference_session(sess, frame_idx=i, obj_ids=[k for k, _ in new],
                                                             input_boxes=[[nb for _, nb in new]], original_size=(H, W))
                if sess.get_obj_num() == 0:
                    t_model += time.time() - tm
                    continue
                o = model(inference_session=sess, frame_idx=i, frame=pix)
                ms = proc.post_process_masks([o.pred_masks], original_sizes=[[sh, sw]], binarize=False)[0][:, 0] > 0
                ms = ms.cpu().numpy()
                sc = o.object_score_logits.float().view(-1).cpu().numpy()
                ids = list(o.object_ids)
            else:
                o = model(inference_session=sess, frame_idx=i, frame=pix)
                pp = proc.postprocess_outputs(sess, o, original_sizes=[[sh, sw]])
                ids = [int(k) for k in pp["object_ids"].tolist()]
                ms = pp["masks"].cpu().numpy().astype(bool)
                sc = pp["scores"].float().cpu().numpy()
                ids = [k + 1 for k in ids]       # SAM 3 ids start at 0; 0 is the back here
            t_model += time.time() - tm
            prev = {}
            for k, m, s in zip(ids, ms, sc):
                wr.add(i, fr.index, fr.t, int(k), m, s)
                prev[int(k)] = m
            _prune(sess, i)
            if i % 60 == 0:
                log(f"  {method} frame {i}/{n} objs {len(ids)} gpu peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                    f"{(i + 1) / max(time.time() - t_start, 1e-6):.2f} fr/s")
    elapsed = time.time() - t_start
    meta = dict(video=str(video), t0=t0, t1=t1, method=method, label=METHOD_LABEL[method], prompt=prompt, ckpt=ckpt,
                frames=[int(i0), int(i0 + n)], n_frames=n, fps=float(1 / tab.frame_step), store_size=list(STORE_SIZE),
                back_box=list(BACK_BOX), start_min_size=START_MIN_SIZE, events=events,
                seconds_total=round(elapsed, 1), seconds_model=round(t_model, 1),
                gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                gpu_name=torch.cuda.get_device_name(0), done=True)
    wr.save(out, meta)
    log(f"{out.name}: {n} frames in {elapsed:.0f} s ({n / elapsed:.2f} fr/s), peak {meta['gpu_peak_gb']} GB")
    del model, sess
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
    because it lies mostly on the back, 'flicker' = present on < 3 frames), ``number`` = display number of each whale
    (1.. in order of first appearance), ``hidden`` = set of (i, obj) frames where a whale outline duplicates an older
    track and is not drawn."""
    dir: Path
    meta: dict
    table: pd.DataFrame
    packed: dict[int, np.ndarray]
    back: np.ndarray | None                   # packed back masks (n, h, w/8) or None
    roles: dict[int, str] = field(default_factory=dict)
    number: dict[int, int] = field(default_factory=dict)
    hidden: set = field(default_factory=set)
    back_share: dict[int, float] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.meta["n_frames"])

    def mask(self, obj: int, i: int) -> np.ndarray:
        w = int(self.meta["store_size"][0])
        src = self.back if obj == BACK_ID else self.packed[obj]
        return np.unpackbits(src[i], axis=1)[:, :w].astype(bool)

    def visible(self, obj: int, i: int) -> bool:
        if self.roles.get(obj) not in ("whale", "back") or (i, obj) in self.hidden:
            return False
        src = self.back if obj == BACK_ID else self.packed.get(obj)
        return src is not None and bool(src[i].any())


def load_tracks(run_dir: str | os.PathLike, back_from: str | os.PathLike | None = None, on_back: float = 0.6,
                dup: float = 0.7, min_frames: int = 3) -> Tracks:
    """Load a run and clean it: drop whale tracks whose outline lies >= ``on_back`` on the back (median over the
    frames where it is seen), drop tracks seen on fewer than ``min_frames`` frames, hide a whale outline on a frame
    where >= ``dup`` of the smaller of two outlines lies inside the other (the younger track is hidden). ``back_from``
    = another run of the same stretch to take the back from (for the text-prompt method, which has no back)."""
    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text())
    tab = pd.read_parquet(d / "tracks.parquet")
    z = np.load(d / "masks.npz")
    packed = {int(k.split("_")[1]): z[k] for k in z.files}
    back = packed.pop(BACK_ID, None) if meta["method"] != "text_sam3" else None
    if meta["method"] == "text_sam3":
        packed = {k: v for k, v in packed.items()}
        if back_from is not None:
            zb = np.load(Path(back_from) / "masks.npz")
            back = zb[f"obj_{BACK_ID}"] if f"obj_{BACK_ID}" in zb.files else None
    tr = Tracks(d, meta, tab, packed, back)
    if back is not None:
        tr.roles[BACK_ID] = "back"
    w = int(meta["store_size"][0])
    unpack = lambda p: np.unpackbits(p, axis=1)[:, :w].astype(bool)  # noqa: E731
    first = {}
    for k, p in packed.items():
        seen = np.nonzero(p.reshape(len(p), -1).any(1))[0]
        if len(seen) < min_frames:
            tr.roles[k] = "flicker"
            continue
        first[k] = int(seen[0])
        if back is not None:
            sh = []
            for i in seen[:: max(1, len(seen) // 40)]:
                m = unpack(p[i])
                sh.append((m & unpack(back[i])).sum() / max(m.sum(), 1))
            tr.back_share[k] = float(np.median(sh))
            if tr.back_share[k] >= on_back:
                tr.roles[k] = "on_back"
                continue
        tr.roles[k] = "whale"
    whales = sorted([k for k, r in tr.roles.items() if r == "whale"], key=lambda k: (first[k], k))
    tr.number = {k: j + 1 for j, k in enumerate(whales)}
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
    return tr


def track_summary(tr: Tracks, min_gap: int = 5) -> pd.DataFrame:
    """One row per displayed whale: first / last time seen, frames seen, gaps (runs of >= ``min_gap`` frames unseen
    between first and last), frames drawn as small, frames hidden as duplicate."""
    fps = tr.meta["fps"]
    t0 = tr.meta["t0"]
    rows = []
    w = int(tr.meta["store_size"][0])
    for k, num in sorted(tr.number.items(), key=lambda x: x[1]):
        vis = np.array([tr.visible(k, i) for i in range(tr.n)])
        idx = np.nonzero(vis)[0]
        if not len(idx):
            rows.append(dict(number=num, obj=k, frames_seen=0))
            continue
        gaps = []
        j = idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a - 1 >= min_gap:
                gaps.append(f"{t0 + (a + 1) / fps:.1f}-{t0 + b / fps:.1f}")
        tb = tr.table[(tr.table.obj == k) & (tr.table.area > 0)]
        size = np.maximum(tb.x1 - tb.x0, tb.y1 - tb.y0) / W
        rows.append(dict(number=num, obj=k, first_s=round(t0 + idx[0] / fps, 2), last_s=round(t0 + idx[-1] / fps, 2),
                         frames_seen=int(len(idx)), gaps_s=";".join(gaps), small_frames=int((size < SMALL_SIZE).sum()),
                         dup_hidden=int(sum(1 for (i, o) in tr.hidden if o == k)),
                         back_share=round(tr.back_share.get(k, np.nan), 3)))
    return pd.DataFrame(rows)


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
            rows.append(dict(frame=f, t=round(tr.meta["t0"] + i / tr.meta["fps"], 2), what="whale", ref_id=iid,
                             grade=v.get("grade", ""), note=v.get("note", ""), track=tr.number.get(k) if k else None,
                             iou=round(float(iou), 3), ref_px=int((inst == iid).sum())))
        matched = {m[0] for m in match.values()}
        for k in tracks_on:
            if k not in matched:
                rows.append(dict(frame=f, t=round(tr.meta["t0"] + i / tr.meta["fps"], 2), what="unmatched track",
                                 track=tr.number[k], iou=np.nan, ref_px=0))
        if tr.back is not None and (cls == 2).sum() > 1000:
            b, g = tr.mask(BACK_ID, i), cls == 2
            rows.append(dict(frame=f, t=round(tr.meta["t0"] + i / tr.meta["fps"], 2), what="back",
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
    in_view = []
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
        in_view.append(num)
    for text, anchor, col, sc in labels:
        _label(out, text, anchor, col, scale=sc)
    # header: two lines on a dark band
    band = out[:44].astype(np.float32) * 0.35
    out[:44] = band.astype(np.uint8)
    cv2.putText(out, header, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x = 8
    cv2.putText(out, "in view:", (x, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    x += 70
    for num in in_view:
        col = PALETTE[(num - 1) % len(PALETTE)]
        cv2.putText(out, f"{num}", (x, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        x += 14 + 12 * len(str(num))
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
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sw}x{sh}",
           "-r", f"{fps_num:.5f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sheet_at = set(np.linspace(0, tr.n - 1, n_sheet).round().astype(int).tolist())
    sheet = []
    last_seen: dict[int, float] = {}
    name = Path(vid).name
    for i, fr in enumerate(uv.iter_frames(vid, start=float(tab.times[f0]) - 1e-3, end=float(tab.times[f1 - 1]) + 1e-3)):
        if i >= tr.n:
            break
        rd = timecode.read_clock(fr.image, glyphs)
        clock = rd.text if rd is not None else "--"
        img = cv2.resize(fr.image, (sw, sh), interpolation=cv2.INTER_AREA)
        t = fr.t
        lost = {}
        for k, num in tr.number.items():
            if tr.visible(k, i):
                last_seen[num] = t
            elif num in last_seen:
                lost[num] = last_seen[num]
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
