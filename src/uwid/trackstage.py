"""The tracking stage (WP11a): masks and individual numbers on the scan's high-quality spans, with the purity policy
running ONLINE.

Input = the scan's output for a deployment (``underwater_cache/scan/<deployment>/<preset>/intervals.json``, window
tables beside it, detector cache under ``<deployment>/cache/det/``). For each hq span of a kept stretch with
``advanced_ok``:

1. patch pass (DINOv3 ViT-B/16 + the WP5b patch classifier, every 3rd frame): bubble / surface / air shares;
2. blocked frames (WP8b rule): lens out of water from the depth sensor when the deployment has a log (surfacings,
   mapped to video time through the window table's ``sensor_t0``), else >= 25 % ``air`` patches; or >= 50 % of the
   non-back patches ``bubbles``; runs < 0.2 s dropped, open gaps < 0.3 s filled. Blocked frames are CUT OUT of the
   tracker's view (it meets a jump);
3. tracking: OWLv2-box-started SAM 3 tracker; starts = the scan's decision objects labelled ``whale`` (after the back
   filter, the whole-frame rule and the size rule) on the scan's detector frames, if the box is not already
   explained by a live track and passes the start screen (< 30 % bubbles / surface / air patches inside);
4. the purity guards of :class:`uwid.tracks.PurityConfig` (strict defaults, frozen) act while tracking: when one
   fires, that tracker object is REMOVED from the session (its memory goes with it) and the ending is recorded; a new
   number can only come from a later start box. A track whose first second is fish-like gets no whale number.

Output per span: ``underwater_cache/tracks/<deployment>/<recording>/<t0>-<t1>_online/`` (the usual masks.npz,
tracks.parquet, meta.json with endings / refused starts / blocked ranges, summary.json); one ``index.csv`` per
deployment. ``uwid tracks`` is the command.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import ownback, paths, tracks

BUBBLES, SURFACE, AIR, BACK = 4, 5, 6, 2
AIR_SHARE, BUBBLE_SHARE, LENS_PAD_S, MIN_BLOCK_S, MIN_OPEN_S = 0.25, 0.50, 0.2, 0.2, 0.3
LENS_OUT_LAG_S, LENS_IN_LEAD_S = 0.3, 0.4            # as the scan (ScanConfig)
PATCH_STEP = 3
TRACK_W, TRACK_H = 1920, 1080                        # tracker box coordinates (any video size is scaled to these)


# ------------------------------------------------------------------------------------------------ scan input

def scan_dir(deployment: str, preset: str = "sensitive") -> Path:
    return paths.cache_dir("scan") / deployment / preset


def scan_config(deployment: str, preset: str = "sensitive"):
    from . import pipeline
    iv = json.loads((scan_dir(deployment, preset) / "intervals.json").read_text())
    cfg = pipeline.ScanConfig()
    for k, v in iv.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    return cfg, iv


def scan_key(name: str) -> str:
    """The scan's file key of a recording (``pipeline.Recording.key``: spaces in the name become ``_``; WP19: the
    dev clip "2025-12-20 CG.m4v" is stored as "2025-12-20_CG.m4v"). Names without spaces are unchanged."""
    return name.replace(" ", "_")


def windows_table(deployment: str, name: str, preset: str = "sensitive") -> pd.DataFrame | None:
    p = scan_dir(deployment, preset) / "windows" / f"{scan_key(name)}.csv"
    return pd.read_csv(p) if p.exists() else None


def sensor_offset(w: pd.DataFrame | None) -> float | None:
    """video time - sensor time for a recording (median over its windows), or None without a log."""
    if w is None or "sensor_t0" not in w or not w.sensor_t0.notna().any():
        return None
    return float(np.nanmedian(w.t0 - w.sensor_t0))


def lens_out_spans(deployment: str, name: str, preset: str = "sensitive") -> list[tuple[float, float]]:
    """Lens-out-of-water intervals in video seconds from the depth sensor (empty without a log)."""
    w = windows_table(deployment, name, preset)
    off = sensor_offset(w)
    f = paths.cache_dir("tagdata") / deployment / "surfacings.csv"
    if off is None or not f.exists():
        return []
    s = pd.read_csv(f)
    out = []
    for r in s.itertuples():
        a, b = float(r.start_t) + LENS_OUT_LAG_S + off, float(r.end_t) - LENS_IN_LEAD_S + off
        if b > a:
            out.append((a, b))
    return out


def hq_spans(deployment: str, preset: str = "sensitive") -> pd.DataFrame:
    """Every hq span of a kept stretch with advanced_ok: recording, source, span and stretch times, median depth of
    its windows, whether a sensor lens-out overlaps it."""
    cfg, iv = scan_config(deployment, preset)
    rows = []
    for v in iv["videos"]:
        name = Path(v["source"]).name
        w = windows_table(deployment, name, preset)
        lens = lens_out_spans(deployment, name, preset)
        for k in v.get("keep", []):
            if not k.get("advanced_ok"):
                continue
            for a, b in k.get("hq_spans", []):
                d = np.nan
                if w is not None:
                    s = w[(w.t0 >= a - 1e-6) & (w.t1 <= b + 1e-6)]
                    d = float(s.depth_m.median()) if len(s) and s.depth_m.notna().any() else np.nan
                rows.append(dict(recording=name, source=v["source"], start=float(a), end=float(b),
                                 dur=float(b - a), stretch_start=float(k["start"]), stretch_end=float(k["end"]),
                                 depth_med=d, lens_out=any(x < b and y > a for x, y in lens)))
    return pd.DataFrame(rows)


def scan_starts(deployment: str, name: str, f0: int, f1: int, preset: str = "sensitive") -> pd.DataFrame:
    """The scan's decision objects labelled whale on its detector frames in [f0, f1): columns frame, x0, y0, x1, y1
    (tracker coordinates, 1920 x 1080), score."""
    from . import pipeline
    cfg, _ = scan_config(deployment, preset)
    det = paths.cache_dir("scan") / deployment / "cache" / "det" / scan_key(name)
    meta = json.loads((det / "meta.json").read_text()) if (det / "meta.json").exists() else {}
    width, height = meta.get("decode_size") or [1920, 1080]
    parts = [pd.read_parquet(p) for p in sorted(det.glob("part_*.parquet"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=["frame", "x0", "y0", "x1", "y1", "score"])
    b = pd.concat(parts, ignore_index=True)
    b = b[(b.frame >= f0) & (b.frame < f1)]
    w = windows_table(deployment, name, preset)
    shallow_of = {}
    if w is not None:
        for r in w.itertuples():
            sh = bool(np.isfinite(r.depth_m) and r.depth_m < cfg.near_whale_min_depth_m)
            for c in ("frame_0", "frame_1"):
                fv = getattr(r, c)
                if np.isfinite(fv):
                    shallow_of[int(fv)] = sh
    rows = []
    sx, sy = TRACK_W / width, TRACK_H / height
    for f, g in b.groupby("frame"):
        obj = pipeline.decision_objects(g, cfg, width, height, shallow=shallow_of.get(int(f), False))
        for r in obj[obj.label == "whale"].itertuples():
            rows.append(dict(frame=int(f), x0=r.x0 * sx, y0=r.y0 * sy, x1=r.x1 * sx, y1=r.y1 * sy, score=float(r.score)))
    return pd.DataFrame(rows, columns=["frame", "x0", "y0", "x1", "y1", "score"])


def scan_rows(deployment: str, name: str, f0: int, f1: int) -> tuple[dict, tuple[int, int]]:
    """Raw detector rows per frame in [f0, f1) (scan cache format, detector pixels) and the detector frame size: the
    input of the WP10 box classifier."""
    det = paths.cache_dir("scan") / deployment / "cache" / "det" / scan_key(name)
    meta = json.loads((det / "meta.json").read_text()) if (det / "meta.json").exists() else {}
    size = tuple(meta.get("decode_size") or [1920, 1080])
    parts = [pd.read_parquet(p) for p in sorted(det.glob("part_*.parquet"))]
    parts = [p for p in parts if len(p)]
    if not parts:
        return {}, size
    b = pd.concat(parts, ignore_index=True)
    b = b[(b.frame >= f0) & (b.frame < f1)]
    return {int(f): g for f, g in b.groupby("frame")}, size


def box_classes(rows: pd.DataFrame, rgb: np.ndarray, size: tuple[int, int], boxes: list) -> list[str]:
    """WP10 box classifier (whale / fish / other) for start boxes (tracker coordinates) of one detector frame:
    each start box is the decision object of one whale row; its class is that row's class."""
    import cv2
    from . import boxclf
    img = rgb if (rgb.shape[1], rgb.shape[0]) == tuple(size) else cv2.resize(rgb, tuple(size), interpolation=cv2.INTER_AREA)
    cls = boxclf.classify(rows, img)
    sx, sy = size[0] / TRACK_W, size[1] / TRACK_H
    out = []
    w = rows[rows["query"] == "whale"]
    for b in boxes:
        d = (np.abs(w.x0 - b[0] * sx) + np.abs(w.y0 - b[1] * sy) + np.abs(w.x1 - b[2] * sx) + np.abs(w.y1 - b[3] * sy))
        out.append(str(cls.loc[d.idxmin(), "boxclf_class"]) if len(d) and d.min() < 4 else "")
    return out


def wp5a_starts(video: str, f0: int, f1: int) -> pd.DataFrame:
    """WP5a stage-1 boxes (the WP8 demos' start source), for comparing with the after-the-fact runs."""
    b, frames, _ = tracks.stage1_boxes(video)
    b = b[(b.frame >= f0) & (b.frame < f1)]
    return b[["frame", "x0", "y0", "x1", "y1", "score"]].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ blocked view

def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    out, a = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and a is None:
            a = i
        elif not m and a is not None:
            out.append((a, i))
            a = None
    return out


def clean_blocked(mask: np.ndarray, fps: float) -> np.ndarray:
    m = np.asarray(mask, bool).copy()
    for a, b in runs_of(m):
        if b - a < MIN_BLOCK_S * fps:
            m[a:b] = False
    r = runs_of(m)
    for (a0, b0), (a1, b1) in zip(r[:-1], r[1:]):
        if a1 - b0 < MIN_OPEN_S * fps:
            m[b0:a1] = True
    return m


def patch_pass(video: str, f0: int, f1: int, step: int = PATCH_STEP, model=None, clf=None,
               frame_store: str | os.PathLike | None = None) -> dict:
    """Patch classes every ``step``-th frame in [f0, f1): frames, labels (N, 45, 80) uint8."""
    import cv2
    import torch
    from . import patchseg as ps
    from . import video as uv
    model = model or ps.load_backbone()
    clf = clf or ps.load_classifier(ps.CACHE / "clf_w1280_h0_all.pt")[0]
    want = list(range(f0 - f0 % step + (step if f0 % step else 0), f1, step))
    fr_i, labs, batch = [], [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            f = ps.extract(model, [b[1] for b in batch], 1280)
            for j, (idx, _) in enumerate(batch):
                labs.append(ps.predict_probs(clf, f[j]).argmax(-1).cpu().numpy().astype(np.uint8))
                fr_i.append(idx)
        batch.clear()

    d = stored(frame_store, video, f0, f1)
    if d is not None:                                   # WP13 store: the same shown frames, read instead of decoded
        sm = json.loads((d / "index.json").read_text())
        arr = np.load(d / "frames.npy", mmap_mode="r")
        pos = {ix: j for j, ix in enumerate(sm["index"])}
        it = (uv.Frame(index=w, t=0.0, image=np.array(arr[pos[w]])) for w in want if w in pos)
    else:
        it = uv.iter_frames_at(video, want)
    for fr in it:
        img = fr.image
        if img.shape[1] != 1920:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        batch.append((fr.index, img))
        if len(batch) == 8:
            flush()
    flush()
    return dict(frames=np.array(fr_i, np.int64), labels=np.stack(labs) if labs else np.zeros((0, 45, 80), np.uint8))


def picture_out(deployment: str | None, name: str, times: np.ndarray, preset: str = "sensitive") -> np.ndarray | None:
    """Per frame: the scan's picture water model (1 s windows) does NOT call the window under water (air or
    crossing). None when the deployment has no window table (then the sensor decides alone)."""
    w = windows_table(deployment, name, preset) if deployment else None
    if w is None or "water_picture" not in w:
        return None
    wp = dict(zip(np.floor(w.t0).astype(int), w.water_picture.astype(str)))
    # a window the picture model never saw keeps the sensor decision (counts as "agrees")
    return np.array([wp.get(int(np.floor(t)), "nan") != "underwater" for t in times])


def blocked_frames(frames: np.ndarray, labels: np.ndarray, f0: int, f1: int, fps: float, times: np.ndarray,
                   lens_out: Sequence[tuple[float, float]] | None,
                   picture_not_under: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Per frame of [f0, f1): blocked (bool) by the WP8b rule. ``lens_out`` = sensor intervals in video seconds, or
    None without a log (then the air share decides). ``times`` = video time of each frame of [f0, f1)."""
    n = f1 - f0
    idx = np.arange(f0, f1)
    if len(frames):
        lab = labels.reshape(len(labels), -1)
        nb = np.maximum((lab != BACK).sum(1), 1)
        bub = (lab == BUBBLES).sum(1) / nb
        air = (lab == AIR).mean(1)
        near = np.abs(idx[:, None] - frames[None, :]).argmin(1)
        bub_f, air_f = bub[near] >= BUBBLE_SHARE, air[near] >= AIR_SHARE
    else:
        bub_f = air_f = np.zeros(n, bool)
    if lens_out is None:
        lens = air_f
        src = "air share (no sensor log)"
    else:
        lens = np.zeros(n, bool)
        for a, b in lens_out:
            lens |= (times >= a - LENS_PAD_S) & (times <= b + LENS_PAD_S)
        src = "depth sensor"
        if picture_not_under is not None:
            # WP11a follow-up: on CC-09-85 the depth / video time link is good to ~0.3 s and the whale rests at the
            # surface: a depth lens-out blocks a frame only where the picture water model agrees (air / crossing)
            n_sensor = int(lens.sum())
            lens &= picture_not_under
            src = f"depth sensor where the picture agrees ({n_sensor} -> {int(lens.sum())} frames)"
    m = clean_blocked(lens | bub_f, fps)
    return m, dict(lens_source=src, n_lens=int(lens.sum()), n_bubbles=int(bub_f.sum()), n_blocked=int(m.sum()))


def clutter_share(labels: np.ndarray, box: Sequence[float]) -> float:
    """Share of bubbles / surface / air patches inside a box (tracker coordinates) on a patch-label map."""
    gh, gw = labels.shape
    x0, y0, x1, y1 = box
    a0, a1 = int(np.floor(x0 / TRACK_W * gw)), int(np.ceil(x1 / TRACK_W * gw))
    b0, b1 = int(np.floor(y0 / TRACK_H * gh)), int(np.ceil(y1 / TRACK_H * gh))
    sub = labels[max(b0, 0):max(b1, b0 + 1), max(a0, 0):max(a1, a0 + 1)]
    return float(np.isin(sub, [BUBBLES, SURFACE, AIR]).mean()) if sub.size else 0.0


# ------------------------------------------------------------------------------------------------ online tracker

def remove_objects(sess, obj_ids: Sequence[int]) -> None:
    """Drop objects from a SAM 2 / SAM 3 tracker session (their memory goes with them); remaining objects are
    re-indexed the way SAM 3 video's own ``remove_object`` does it."""
    for obj_id in obj_ids:
        old = sess._obj_id_to_idx.get(obj_id)
        if old is None:
            continue
        old_ids = list(sess.obj_ids)
        keep_old = [i for i in range(len(old_ids)) if i != old]
        new_ids = [old_ids[i] for i in keep_old]
        remap = {o: n for n, o in enumerate(keep_old)}
        from collections import OrderedDict
        sess._obj_id_to_idx = OrderedDict((oid, n) for n, oid in enumerate(new_ids))
        sess._obj_idx_to_id = OrderedDict((n, oid) for n, oid in enumerate(new_ids))
        sess.obj_ids = new_ids
        for name in ("point_inputs_per_obj", "mask_inputs_per_obj", "output_dict_per_obj", "frames_tracked_per_obj"):
            c = getattr(sess, name)
            items = [(remap[k], v) for k, v in c.items() if k in remap]
            c.clear()
            c.update(items)
        if obj_id in sess.obj_with_new_inputs:
            sess.obj_with_new_inputs.remove(obj_id)


@dataclass
class OnlineResult:
    out: Path
    endings: list
    refused: list
    blocked: list
    seconds: float


# ------------------------------------------------------------------------------------------------ frame store
# WP13 (flag, off by default): a span's frames decoded ahead of time (CPU, no GPU lock held) into a raw uint8 store,
# so the locked tracker reads them instead of decoding. The store holds exactly what ``video.iter_frames`` yields
# (same images, indices, times); ``frame_store=None`` everywhere means the old path.

def store_dir(root: str | os.PathLike, video: str, f0: int, f1: int) -> Path:
    return Path(root) / f"{Path(video).stem}_{int(f0)}_{int(f1)}"


def store_frames(video: str, f0: int, f1: int, root: str | os.PathLike, log=print) -> Path:
    """Decode frames [f0, f1) exactly as :func:`track_online` does and keep them as one raw array (``frames.npy``,
    N x H x W x 3) + ``index.json`` (indices, times); ``index.json`` gets ``complete`` only at the end."""
    from . import video as uv
    tab = uv.frame_table(video)
    d = store_dir(root, video, f0, f1)
    d.mkdir(parents=True, exist_ok=True)
    n = f1 - f0
    arr, idx, ts = None, [], []
    for i, fr in enumerate(uv.iter_frames(video, start=float(tab.times[f0]) - 1e-3, end=float(tab.times[f1 - 1]) + 1e-3)):
        if i >= n:
            break
        if arr is None:
            arr = np.lib.format.open_memmap(d / "frames.npy", mode="w+", dtype=np.uint8, shape=(n,) + fr.image.shape)
        arr[i] = fr.image
        idx.append(int(fr.index))
        ts.append(float(fr.t))
    if arr is not None:
        arr.flush()
        del arr
    (d / "index.json").write_text(json.dumps(dict(video=str(video), f0=int(f0), f1=int(f1), index=idx, t=ts,
                                                  complete=True)))
    return d


def stored(root: str | os.PathLike | None, video: str, f0: int, f1: int) -> Path | None:
    if root is None:
        return None
    d = store_dir(root, video, f0, f1)
    j = d / "index.json"
    return d if j.exists() and json.loads(j.read_text()).get("complete") else None


def span_frames(video: str, f0: int, f1: int, frame_store: str | os.PathLike | None = None):
    """Frames [f0, f1) in order: from a complete store when ``frame_store`` has one, else decoded."""
    from . import video as uv
    d = stored(frame_store, video, f0, f1)
    if d is None:
        tab = uv.frame_table(video)
        yield from uv.iter_frames(video, start=float(tab.times[f0]) - 1e-3, end=float(tab.times[f1 - 1]) + 1e-3)
        return
    meta = json.loads((d / "index.json").read_text())
    arr = np.load(d / "frames.npy", mmap_mode="r")
    for j, (ix, t) in enumerate(zip(meta["index"], meta["t"])):
        yield uv.Frame(index=int(ix), t=float(t), image=np.array(arr[j]))


def span_start(deployment: str, source: str, t0: float, t1: float, preset: str = "sensitive",
               start_source: str = "scan") -> tuple[int, int, pd.DataFrame]:
    """The frames a span runs on: [first start-box frame at or after t0, frame of t1), and its start boxes."""
    from . import video as uv
    name = Path(source).name
    tab = uv.frame_table(source)
    f0, f1 = tab.index_at(t0), tab.index_at(t1)
    starts = scan_starts(deployment, name, f0, f1, preset) if start_source == "scan" else wp5a_starts(source, f0, f1)
    if len(starts):
        later = starts[starts.frame >= f0]
        f0 = int(later.frame.min()) if len(later) else f0       # begin on the first start-box frame
    return f0, f1, starts


def track_online(video: str, f0: int, f1: int, starts: pd.DataFrame, blocked: np.ndarray, patch: dict,
                 out: Path, cfg: tracks.PurityConfig | None = None, log=print, extra_meta: dict | None = None,
                 start_rows: tuple[dict, tuple[int, int]] | None = None,
                 frame_store: str | os.PathLike | None = None, own_back: np.ndarray | None = None) -> OnlineResult:
    """The online policy over frames [f0, f1) (f0 should be a start-box frame). ``blocked`` (bool per frame) frames
    are not shown to the tracker. Writes the usual run folder to ``out``. ``own_back`` (WP17 flag
    ``own_back_template``): a template at store size; the start check then counts only the back object inside it
    (None = the back object, as before)."""
    import torch
    from . import patchseg as ps
    from . import video as uv
    cfg = cfg or tracks.PurityConfig()
    torch_, dev = tracks._gpu_cap()
    dev = os.environ.get("UWID_DEVICE", dev)
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    proc, model = tracks._load_model("owl_sam3", dev, dtype)
    backbone = ps.load_backbone(device=dev, dtype=torch.float16 if dev == "cuda" else torch.float32)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tab = uv.frame_table(video)
    fps = float(1 / tab.frame_step)
    n = f1 - f0
    wr = tracks._Writer(n)
    sw, sh = tracks.STORE_SIZE
    sx, sy = sw / TRACK_W, sh / TRACK_H
    sess = proc.init_video_session(inference_device=dev, dtype=dtype)
    by_frame = {int(f): g for f, g in starts.groupby("frame")} if len(starts) else {}
    pf, plab = patch["frames"], patch["labels"]
    W1 = tracks.W
    live: dict[int, dict] = {}            # obj -> state: first, first_size, last_vis, prev_mask, ema
    endings, refused = [], []
    next_id = 1
    k = -1
    last_fed = None
    hidden_run = 0
    t_start = time.time()
    lone_before = None
    wr_last: dict = {}
    emb_rows: list = []                   # (i, obj, masked DINOv3 feature) for the link layer
    clf_cache: dict = {}

    def size_of(m):
        ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
        return (max(xs[-1] - xs[0] + 1, ys[-1] - ys[0] + 1) / m.shape[1]) if len(ys) else 0.0

    def end(obj, i, reason):
        st = live.pop(obj)
        pos = None
        if st.get("prev_mask") is not None and st["prev_mask"].any():
            ys, xs = np.nonzero(st["prev_mask"])
            pos = (int(xs.mean()), int(ys.min()))
        endings.append(dict(i=int(i), frame=int(f0 + i), t=float(tab.times[f0 + i]), obj=int(obj), reason=reason, pos=pos))
        remove_objects(sess, [obj])

    with torch.inference_mode():
        for i, fr in enumerate(span_frames(video, f0, f1, frame_store)):
            if i >= n:
                break
            if blocked[i]:
                if hidden_run == 0:
                    whales = [o for o, s_ in live.items() if s_.get("size", 0) >= cfg.whale_size]
                    lone_before = whales[0] if len(whales) == 1 else None
                hidden_run += 1
                continue
            k += 1
            img = fr.image
            # a blocked view just ended: numbers do not cross a long one
            if hidden_run:
                dur = hidden_run / fps
                for obj in list(live):
                    limit = cfg.lone_block_s if (obj == lone_before and cfg.lone_block_s is not None) else cfg.max_block_s
                    if limit is not None and dur > limit:
                        end(obj, i, f"blocked view {dur:.1f} s")
                hidden_run = 0
            inputs = proc(images=img, device=dev, return_tensors="pt")
            pix = inputs.pixel_values[0].to(dtype)
            new = [(tracks.BACK_ID, [float(v) for v in tracks.BACK_BOX])] if k == 0 else []
            if fr.index in by_frame:
                labels = plab[np.abs(pf - fr.index).argmin()] if len(pf) else None
                back_m = ownback.own_back(wr_last.get(tracks.BACK_ID), own_back)
                cover = {o: s_["prev_mask"] for o, s_ in live.items() if s_.get("prev_mask") is not None}
                for _, b in by_frame[fr.index].sort_values("score", ascending=False).iterrows():
                    box = [float(b.x0), float(b.y0), float(b.x1), float(b.y1)]
                    if max(box[2] - box[0], box[3] - box[1]) < tracks.START_MIN_SIZE * W1:
                        continue
                    if tracks.box_covered(tracks._scale_box(box, sx, sy), cover, back_m):
                        continue
                    if any(tracks._box_iou(box, nb) > 0.3 for o, nb in new if o != tracks.BACK_ID):
                        continue
                    if cfg.start_clutter is not None and labels is not None and clutter_share(labels, box) >= cfg.start_clutter:
                        refused.append(dict(i=i, frame=int(fr.index), t=float(fr.t), reason="start on bubbles / surface / air",
                                            box=[round(v, 1) for v in box]))
                        continue
                    if start_rows is not None and fr.index in start_rows[0]:
                        if fr.index not in clf_cache:
                            fb = by_frame[fr.index]
                            clf_cache[fr.index] = dict(zip(range(len(fb)), box_classes(
                                start_rows[0][fr.index], img, start_rows[1],
                                [[r.x0, r.y0, r.x1, r.y1] for r in fb.itertuples()])))
                            clf_cache[fr.index + 0.5] = [[r.x0, r.y0, r.x1, r.y1] for r in fb.itertuples()]
                        allb = clf_cache[fr.index + 0.5]
                        j = int(np.argmin([sum(abs(p_ - q_) for p_, q_ in zip(bb, box)) for bb in allb]))
                        c = clf_cache[fr.index][j]
                        if c != "whale":
                            refused.append(dict(i=i, frame=int(fr.index), t=float(fr.t), reason=f"box classifier: {c or 'no class'}",
                                                box=[round(v, 1) for v in box]))
                            continue
                    new.append((next_id, box))
                    live[next_id] = dict(first=i, sizes=[], prev_mask=None, last_vis=i, ema=None, size=0.0, first_frame_i=i)
                    next_id += 1
            if new:
                proc.add_inputs_to_inference_session(sess, frame_idx=k, obj_ids=[o for o, _ in new],
                                                     input_boxes=[[nb for _, nb in new]], original_size=(TRACK_H, TRACK_W))
            o_ = model(inference_session=sess, frame_idx=k, frame=pix)
            ms = proc.post_process_masks([o_.pred_masks], original_sizes=[[sh, sw]], binarize=False)[0][:, 0] > 0
            ms = ms.cpu().numpy()
            sc = o_.object_score_logits.float().view(-1).cpu().numpy()
            ids = [int(x) for x in o_.object_ids]
            cur = dict(zip(ids, ms))
            feats = None
            if i % PATCH_STEP == 0 and cfg.max_drift is not None:
                import cv2
                im2 = img if img.shape[1] == 1920 else cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
                feats = ps.extract(backbone, [im2], 1280)[0].float()
            prev_masks = {o: s_["prev_mask"] for o, s_ in live.items() if s_.get("prev_mask") is not None}
            for obj in [o for o in ids if o in live]:
                st = live[obj]
                m = cur[obj]
                if not m.any():
                    continue
                sz = size_of(m)
                reason = None
                gap = ((i - st["last_vis"] - int(np.sum(blocked[st["last_vis"] + 1:i]))) / fps
                       if st["prev_mask"] is not None else 0.0)      # unseen time outside blocked views
                if st["prev_mask"] is not None and cfg.max_absent_s is not None and gap > cfg.max_absent_s:
                    reason = f"re-found after {gap:.1f} s"
                elif st["prev_mask"] is not None and st["last_vis"] == last_fed and max(sz, st["size"]) >= cfg.whale_size:
                    a0, a1 = st["prev_mask"].sum(), m.sum()
                    if cfg.max_area_ratio is not None and max(a0, a1) > cfg.max_area_ratio * max(min(a0, a1), 1):
                        reason = "area jump"
                    elif cfg.min_iou_prev is not None and tracks._mask_iou(st["prev_mask"], m) < cfg.min_iou_prev:
                        reason = "shape jump"
                    elif cfg.swallow is not None:
                        for oj, mj in prev_masks.items():
                            if oj == obj or live.get(oj, {}).get("size", 0) < cfg.whale_size:
                                continue
                            sj = mj.sum()
                            if sj and (m & mj).sum() >= cfg.swallow * sj and (st["prev_mask"] & mj).sum() < cfg.swallow * sj:
                                reason = "swallowed another track"
                                break
                if reason is None and cfg.grow_from_small is not None:
                    if i < st["first"] + fps:
                        st["first_max"] = max(st.get("first_max", 0), sz)
                    elif 0 < st.get("first_max", 1) < 0.15 and sz > cfg.grow_from_small * st["first_max"]:
                        reason = "grew from fish-sized"
                if reason is None and feats is not None:
                    gh, gw = feats.shape[:2]
                    import cv2
                    wgt = cv2.resize(m.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
                    if wgt.sum() < 0.5:
                        wgt = (wgt > 0).astype(np.float32)
                    wt = torch.from_numpy(wgt).to(feats.device)
                    e = ((feats * wt[..., None]).sum((0, 1)) / wt.sum()).cpu().numpy()
                    e /= max(np.linalg.norm(e), 1e-6)
                    emb_rows.append((i, obj, e.astype(np.float16)))
                    if st["ema"] is not None:
                        d = 1 - float(e @ (st["ema"] / np.linalg.norm(st["ema"])))
                        if d > cfg.max_drift:
                            reason = f"appearance drift {d:.2f}"
                    st["ema"] = e if st["ema"] is None else 0.7 * st["ema"] + 0.3 * e
                if reason is not None:
                    end(obj, i, reason)
                    cur.pop(obj, None)
                    continue
                st["prev_mask"], st["last_vis"], st["size"] = m, i, sz
            wr_last = {}
            for obj, m, s_ in zip(ids, ms, sc):
                if obj in cur:
                    wr.add(i, fr.index, fr.t, int(obj), cur[obj], s_)
                    wr_last[obj] = cur[obj]
            last_fed = i
            tracks._prune(sess, k)
            if i % 150 == 0:
                log(f"  online frame {i}/{n} live {len(live)} ended {len(endings)}")
    elapsed = time.time() - t_start
    meta = dict(video=str(video), t0=float(tab.times[f0]), t1=float(tab.times[f1 - 1]), method="online_sam3",
                label="OWLv2 / scan boxes -> SAM 3 tracker, purity guards online (strict)", frames=[int(f0), int(f1)],
                n_frames=n, fps=fps, store_size=list(tracks.STORE_SIZE), back_box=list(tracks.BACK_BOX),
                purity=asdict(cfg), endings=endings, refused=refused,
                blocked=[[int(f0 + a), int(f0 + b)] for a, b in runs_of(blocked)], n_blocked=int(np.sum(blocked)),
                seconds_total=round(elapsed, 1), gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)
                if dev == "cuda" else 0.0, done=True, **(extra_meta or {}))
    wr.save(out, meta)
    np.savez_compressed(out / "emb.npz", i=np.array([r[0] for r in emb_rows], np.int32),
                        obj=np.array([r[1] for r in emb_rows], np.int32),
                        emb=np.stack([r[2] for r in emb_rows]) if emb_rows else np.zeros((0, 768), np.float16))
    if dev == "cuda":
        torch.cuda.empty_cache()
    return OnlineResult(out, endings, refused, meta["blocked"], elapsed)


def load_online(d: str | os.PathLike) -> tracks.Tracks:
    """A finished online run, cleaned and numbered; fish-like starts get no whale number; endings carry numbers.
    Whale outlines read through ``tr.mask`` have far specks removed (WP18 speck rule, :func:`uwid.tracks.despeckle`);
    roles, numbers and links are computed on the stored outlines."""
    tr = finish(tracks.read_run(d))
    tr.despeck = True
    return tr


def run_emb(d: str | os.PathLike) -> dict | None:
    p = Path(d) / "emb.npz"
    return dict(np.load(p)) if p.exists() else None


def link(tr: tracks.Tracks, deployment: str | None, start_source: str = "scan",
         cfg: tracks.LinkConfig | None = None, emb: dict | None = None) -> tracks.Tracks:
    """The lone-link layer on a finished online run or stitched encounter (in place)."""
    from . import video as uv
    v = tr.meta["video"]
    name = Path(v).name
    f0 = tr.meta["frames"][0]
    if start_source == "scan" and deployment:
        det = scan_starts(deployment, name, f0, f0 + tr.n)
        tab = uv.frame_table(v)
        breaths = [(tab.index_at(a), tab.index_at(b)) for a, b in lens_out_spans(deployment, name)]
    else:
        det = wp5a_starts(v, f0, f0 + tr.n)
        breaths = []
        if "CC-11-96" in name:
            tab = uv.frame_table(v)
            s = pd.read_csv(paths.cache_dir("tagdata") / "2025-2-3 Mini-mother 11-96" / "surfacings.csv")
            breaths = [(tab.index_at(r.pts_exit_pred), tab.index_at(r.pts_entry_pred))
                       for r in s[s.in_video.astype(bool)].itertuples() if f"-{int(r.piece):05d}" in name]
    was = tr.despeck
    tr.despeck = False                    # link rules see the stored outlines (WP18: unchanged by the speck rule)
    try:
        return tracks.link_lone(tr, det, emb if emb is not None else run_emb(tr.dir), breaths, cfg)
    finally:
        tr.despeck = was


def summarize(d: Path) -> dict:
    tr = load_online(d)
    m = tr.meta
    outlined = sum(1 for i in range(tr.n) if any(tr.visible(k, i) for k in tr.number))
    reasons = pd.Series([e["reason"].split(" ")[0] + (" " + e["reason"].split(" ")[1]
                         if e["reason"].split(" ")[0] in ("blocked", "re-found", "appearance", "area", "shape", "grew",
                                                          "swallowed", "start") else "")
                         for e in m.get("endings", [])]).value_counts().to_dict()
    s = dict(video=Path(m["video"]).name, t0=round(m["t0"], 2), t1=round(m["t1"], 2), frames=m["n_frames"],
             whale_numbers=len(tr.number), small_numbers=len(tr.small_number),
             endings=len(m.get("endings", [])), endings_by_reason=reasons,
             endings_list=[(round(e["t"], 2), e["reason"]) for e in m.get("endings", [])],
             refused_starts=len(m.get("refused", [])), blocked_frames=m.get("n_blocked", 0),
             frames_with_whale_outline=outlined, seconds=m["seconds_total"],
             gpu_s_per_video_s=round(m["seconds_total"] / (m["n_frames"] / m["fps"]), 2), gpu_peak_gb=m.get("gpu_peak_gb"))
    (Path(d) / "summary.json").write_text(json.dumps(s, indent=1, default=str))
    return s


LINK_COLS = ["a", "b", "t_end", "gap_s", "breath", "limit_s", "others", "det_before", "det_during", "det_after",
             "lone", "size_ratio", "app_dist", "checks_ok", "made", "why", "num_a", "num_b"]


def links_table(tr: tracks.Tracks) -> pd.DataFrame:
    """Every lone-link candidate the link layer judged (made or not) with its evidence, one row each."""
    rows = []
    for r in getattr(tr, "links", []) or []:
        d = {k: r.get(k) for k in LINK_COLS}
        det = r.get("detector") or {}
        d.update(det_before=det.get("before"), det_during=det.get("during"), det_after=det.get("after"),
                 others=";".join(str(o) for o in r.get("others") or []),
                 num_a=tr.number.get(r["a"]), num_b=tr.number.get(r["b"]))
        rows.append(d)
    return pd.DataFrame(rows, columns=LINK_COLS)


def write_links(tr: tracks.Tracks, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    links_table(tr).to_csv(path, index=False)
    return path


def encounter_dir(deployment: str, name: str, t0: float, t1: float) -> Path:
    """Folder for results of a stitched encounter (its pieces stay in their own span folders)."""
    return paths.cache_dir("tracks") / deployment / Path(name).stem / f"{t0:07.2f}-{t1:07.2f}_encounter"


# ------------------------------------------------------------------------------------------------ stage

def span_dir(deployment: str, name: str, t0: float, t1: float) -> Path:
    return paths.cache_dir("tracks") / deployment / Path(name).stem / f"{t0:07.2f}-{t1:07.2f}_online"


def run_span(deployment: str, source: str, t0: float, t1: float, preset: str = "sensitive",
             start_source: str = "scan", force: bool = False, log=print,
             frame_store: str | os.PathLike | None = None, own_back_template: bool = True,
             suffix: str = "") -> Path:
    """One span: patch pass, blocked frames, online tracking, summary (skipped when done). ``frame_store``: read
    the span's frames from a complete :func:`store_frames` store there (WP13 flag; None = decode as before).
    ``own_back_template`` (WP17 flag, ON by default since WP18, user 2026-11-24): the tagged whale's own back = the
    SAM back object inside the
    300 s block template of :mod:`uwid.ownback`, both for refusing start boxes and for the ``on_back`` role; the
    template is stored with the run (``own_back.npz``). ``suffix``: appended to the run folder's name."""
    from . import video as uv
    name = Path(source).name
    out = span_dir(deployment, name, t0, t1)
    if suffix:
        out = out.with_name(out.name + suffix)
    if (out / "meta.json").exists() and json.loads((out / "meta.json").read_text()).get("done") and not force:
        log(f"{out.name}: done already")
        return out
    tab = uv.frame_table(source)
    fps = float(1 / tab.frame_step)
    f0, f1, starts = span_start(deployment, source, t0, t1, preset, start_source)
    out.mkdir(parents=True, exist_ok=True)
    pp = out / "patch.npz"
    if pp.exists():
        z = np.load(pp)
        patch = dict(frames=z["frames"], labels=z["labels"])
    else:
        patch = patch_pass(source, f0, f1, frame_store=frame_store)
        np.savez_compressed(pp, **patch)
    lens = lens_out_spans(deployment, name, preset) if start_source == "scan" else None
    if start_source != "scan" and "CC-11-96" in name:
        lens = []                                 # 2025-02-03 has a log: handled by the gap/ rule via surfacings
        s = pd.read_csv(paths.cache_dir("tagdata") / "2025-2-3 Mini-mother 11-96" / "surfacings.csv")
        for r in s[s.in_video.astype(bool)].itertuples():
            if f"-{int(r.piece):05d}" in name:
                lens.append((float(r.pts_exit_pred), float(r.pts_entry_pred)))
    has_log = bool(lens) or (start_source == "scan" and sensor_offset(windows_table(deployment, name, preset)) is not None)
    blocked, binfo = blocked_frames(patch["frames"], patch["labels"], f0, f1, fps, tab.times[f0:f1],
                                    lens if has_log else None,
                                    picture_out(deployment, name, tab.times[f0:f1], preset) if has_log else None)
    rows = scan_rows(deployment, name, f0, f1) if start_source == "scan" else None
    extra = dict(deployment=deployment, preset=preset, start_source=start_source,
                 n_start_boxes=int(len(starts)), blocked_info=binfo,
                 frames_from="store" if stored(frame_store, source, f0, f1) else "decoded")
    ob_px = None
    if own_back_template:
        tm, info = ownback.block_template(source, windows_table(deployment, name, preset), float(tab.times[f0]),
                                          paths.cache_dir("tracks") / deployment / Path(name).stem / "ownback")
        extra["own_back"] = info
        if tm is not None:
            ownback.save_run_template(out, tm, info)
            ob_px = ownback.upsample(tm, (tracks.STORE_SIZE[1], tracks.STORE_SIZE[0]))
        log(f"  own-back template: {info}")
    track_online(source, f0, f1, starts, blocked, patch, out, log=log, start_rows=rows, frame_store=frame_store,
                 extra_meta=extra, own_back=ob_px)
    s = summarize(out)
    tr = link(load_online(out), deployment, start_source)
    write_links(tr, out / "links.csv")                          # every link candidate, made or not
    log(f"{name} {t0:.1f}-{t1:.1f}: {s['whale_numbers']} whale numbers, {s['endings']} endings, "
        f"{s['gpu_s_per_video_s']} GPU s / s")
    return out


def pieces(t0: float, t1: float, piece_s: float = 26.0, overlap_s: float = 1.0) -> list[tuple[float, float]]:
    """A long stretch as equal pieces of at most ``piece_s`` that overlap by ``overlap_s`` (each piece is one
    resumable GPU step under the lock's time limit; :func:`stitch` joins them)."""
    n = max(1, int(np.ceil((t1 - t0 - overlap_s) / (piece_s - overlap_s))))
    step = (t1 - t0 - overlap_s) / n
    return [(round(t0 + j * step, 2), round(t0 + (j + 1) * step + overlap_s, 2)) for j in range(n)]


def finish(tr: tracks.Tracks) -> tracks.Tracks:
    """Fish screen, cleaning and numbering, endings with numbers (shared by single runs and stitched encounters)."""
    cfg = tracks.PurityConfig(**{k: v for k, v in tr.meta.get("purity", {}).items()
                                 if k in tracks.PurityConfig.__dataclass_fields__})
    for o in tr.packed:
        if cfg.fish_speed is not None and tracks.fish_like(tr.table, o, tr.meta["fps"], cfg.fish_speed, cfg.fish_size):
            tr.screened[o] = "small"
    tracks.clean(tr)
    tr.endings = []
    for e in tr.meta.get("endings", []):
        num = tr.number.get(e["obj"])
        lab = num if num is not None else (f"s{tr.small_number[e['obj']]}" if e["obj"] in tr.small_number else None)
        if lab is not None:
            tr.endings.append((e["i"], lab, e["reason"], tuple(e["pos"]) if e.get("pos") else None))
    return tr


def stitch(dirs: Sequence[Path], min_iou: float = 0.5) -> tuple[tracks.Tracks, dict]:
    """Join consecutive overlapping piece runs into one raw Tracks: in each overlap the earlier piece's outlines are
    kept, and an object of the later piece that overlaps an earlier object (mean IoU >= ``min_iou`` over the overlap)
    continues that object's id. Returns (raw tracks, embeddings re-keyed)."""
    runs = [tracks.read_run(d) for d in dirs]
    F0 = runs[0].meta["frames"][0]
    F1 = max(r.meta["frames"][0] + r.n for r in runs)
    n = F1 - F0
    sw, sh = runs[0].meta["store_size"]
    packed, rows, endings, blocked, emb = {}, [], [], [], dict(i=[], obj=[], emb=[])
    back = np.zeros((n, sh, (sw + 7) // 8), np.uint8)
    prev_map: dict = {}
    for p, r in enumerate(runs):
        rf0 = r.meta["frames"][0]
        start = 0 if p == 0 else max(0, (runs[p - 1].meta["frames"][0] + runs[p - 1].n) - rf0)   # first frame not in p-1
        idmap = {}
        if p > 0:
            pr = runs[p - 1]
            pf0 = pr.meta["frames"][0]
            ov = range(rf0, pf0 + pr.n)
            for k in r.packed:
                best, bi = None, 0.0
                for k0 in pr.packed:
                    ious = [tracks._mask_iou(r.mask(k, f - rf0), pr.mask(k0, f - pf0)) for f in ov
                            if r.packed[k][f - rf0].any() and pr.packed[k0][f - pf0].any()]
                    if ious and np.mean(ious) > bi:
                        best, bi = k0, float(np.mean(ious))
                if best is not None and bi >= min_iou and best in prev_map:
                    idmap[k] = prev_map[best]
        cur_map = {}
        for k, q in r.packed.items():
            nid = idmap.get(k, p * 100000 + k)
            cur_map[k] = nid
            if nid not in packed:
                packed[nid] = np.zeros((n, sh, (sw + 7) // 8), np.uint8)
            packed[nid][rf0 - F0 + start: rf0 - F0 + r.n] = q[start:]
            t = r.table[(r.table.obj == k) & (r.table.i >= start)].copy()
            t["obj"], t["i"] = nid, t.i + (rf0 - F0)
            rows.append(t)
        if r.back is not None:
            back[rf0 - F0 + start: rf0 - F0 + r.n] = r.back[start:]
        for e in r.meta.get("endings", []):
            if e["i"] >= start:
                endings.append(dict(e, i=e["i"] + rf0 - F0, obj=cur_map.get(e["obj"], p * 100000 + e["obj"])))
        blocked += r.meta.get("blocked", [])
        ep = Path(r.dir) / "emb.npz"
        if ep.exists():
            z = np.load(ep)
            for i, o, e in zip(z["i"], z["obj"], z["emb"]):
                if i >= start:
                    emb["i"].append(i + rf0 - F0), emb["obj"].append(cur_map.get(int(o), p * 100000 + int(o))), emb["emb"].append(e)
        prev_map = cur_map
    meta = dict(runs[0].meta)
    meta.update(frames=[F0, F1], n_frames=n, t1=runs[-1].meta["t1"], endings=endings, blocked=blocked,
                pieces=[str(d) for d in dirs], seconds_total=sum(r.meta["seconds_total"] for r in runs))
    tr = tracks.Tracks(Path(dirs[0]), meta, pd.concat(rows, ignore_index=True), packed, back)
    e = dict(i=np.array(emb["i"]), obj=np.array(emb["obj"]), emb=np.array(emb["emb"])) if emb["i"] else None
    return tr, e


def write_index(deployment: str) -> Path:
    root = paths.cache_dir("tracks") / deployment
    rows = []
    for f in sorted(root.glob("*/*_online/summary.json")):
        s = json.loads(f.read_text())
        s["folder"] = str(f.parent)
        s.pop("endings_list", None)
        s["endings_by_reason"] = json.dumps(s.get("endings_by_reason", {}))
        rows.append(s)
    root.mkdir(parents=True, exist_ok=True)
    p = root / "index.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    links = []                                    # all span and encounter link candidates, beside index.csv
    for f in sorted(list(root.glob("*/*_online/links.csv")) + list(root.glob("*/*_encounter/links.csv"))):
        t = pd.read_csv(f)
        if len(t):
            t.insert(0, "folder", str(f.parent))
            links.append(t)
    (pd.concat(links, ignore_index=True) if links else pd.DataFrame(columns=["folder"] + LINK_COLS)).to_csv(
        root / "links.csv", index=False)
    return p


def run_deployment(deployment: str, preset: str = "sensitive", only: Sequence[str] = (), max_seconds: float = 30.0,
                   render: bool = False, budget_s: float | None = 720.0, log=print,
                   own_back_template: bool = True) -> list[Path]:
    """All hq spans (or ``only`` = ["<recording>:<t0>-<t1>", ...]) of a deployment, each capped at ``max_seconds``
    around its centre; a new span starts only while less than ``budget_s`` has passed (re-run to continue).
    ``own_back_template``: see :func:`run_span` (on by default)."""
    spans = hq_spans(deployment, preset)
    items = []
    if only:
        for o in only:
            rec, _, rng = o.partition(":")
            a, b = (float(x) for x in rng.split("-"))
            src = spans[spans.recording.str.contains(rec)].source
            if not len(src):
                raise SystemExit(f"no scanned recording matches {rec}")
            items.append((src.iloc[0], a, b))
    else:
        for r in spans.itertuples():
            a, b = r.start, r.end
            if max_seconds and b - a > max_seconds:
                c = (a + b) / 2
                a, b = c - max_seconds / 2, c + max_seconds / 2
            items.append((r.source, a, b))
    outs = []
    t_start = time.time()
    for src, a, b in items:
        d = span_dir(deployment, Path(src).name, a, b)
        done = (d / "meta.json").exists() and json.loads((d / "meta.json").read_text()).get("done")
        if not done and budget_s is not None and time.time() - t_start > budget_s:
            log("budget reached; run again to continue")
            break
        outs.append(run_span(deployment, src, a, b, preset, log=log, own_back_template=own_back_template))
        if render:
            render_span(outs[-1])
    write_index(deployment)
    return outs


def render_span(d: Path, name: str | None = None) -> dict:
    tr = load_online(d)
    sheets = paths.sheets_dir(create=True)
    name = name or f"tracks_{Path(tr.meta['video']).stem}_{tr.meta['t0']:.0f}"
    return tracks.render_tracks(tr, sheets / f"{name}.mp4", sheets / f"{name}.jpg", crf=20)
