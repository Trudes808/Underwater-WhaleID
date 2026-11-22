"""Text-prompted detectors with one output format, and a resumable per-video cache.

Output format (one row per box, a pandas DataFrame): ``frame`` (decode-order index, as in the WP3 labels), ``t``
(frame / fps), ``model``, ``query`` (the prompt text that produced the box), ``score``, ``x0 y0 x1 y1`` (pixels of
the full frame). Frames that were processed but gave no box are listed in the cache's ``frames`` file, so "no box"
and "not run" are kept apart.

Runners (all ``detect(images) -> list[DataFrame]`` on RGB uint8 frames):

- :class:`OwlRunner` - OWLv2 (``google/owlv2-base-patch16-ensemble``). Each query is scored on its own (sigmoid per
  query), so one pass with many queries equals one pass per query.
- :class:`GroundingRunner` - LLMDet / MM-Grounding-DINO through transformers zero-shot object detection. Queries in
  one pass compete with each other, so the query set is part of the result.
- :class:`SamMasker` - SAM 2.1 masks from boxes.
- :class:`Mask2FormerSeg` - Mask2Former semantic (ADE20K) or panoptic (COCO) label maps.

Cache: ``underwater_cache/detect/<video stem>/<run name>/`` with ``chunk_<first frame>.parquet`` (boxes),
``chunk_<first frame>.frames.npy`` (frames processed) and ``meta.json``. :func:`run_cached` skips chunks already
done and stops at a time budget, so a long job is the same command run again until it reports done.

Frames are decoded with OpenCV in order (no seeking), like the WP3 label tool, so frame indices agree with the labels.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from . import paths

COLUMNS = ["frame", "t", "model", "query", "score", "x0", "y0", "x1", "y1"]

# Burned-in clock, bottom right (ClockLayout in timecode.py: x 1756-1916, y 1062-1075 at 1920x1080), with margin.
CLOCK_BOX = (1740, 1050, 1920, 1080)


def detect_cache_dir(video: str | os.PathLike, run_name: str) -> Path:
    stem = Path(video).stem.replace(" ", "_")
    return paths.cache_dir("detect") / stem / run_name


def in_clock(df: pd.DataFrame, width: int = 1920, height: int = 1080, min_frac: float = 0.5) -> pd.Series:
    """Boxes lying mostly (``min_frac`` of their area) inside the burned-in clock region."""
    sx, sy = width / 1920, height / 1080
    cx0, cy0, cx1, cy1 = CLOCK_BOX[0] * sx, CLOCK_BOX[1] * sy, CLOCK_BOX[2] * sx, CLOCK_BOX[3] * sy
    ix = (np.minimum(df.x1, cx1) - np.maximum(df.x0, cx0)).clip(lower=0)
    iy = (np.minimum(df.y1, cy1) - np.maximum(df.y0, cy0)).clip(lower=0)
    area = ((df.x1 - df.x0) * (df.y1 - df.y0)).clip(lower=1)
    return (ix * iy / area) >= min_frac


# ----------------------------------------------------------------------------------------------- frames

def video_info(video: str | os.PathLike) -> dict:
    import cv2
    v = cv2.VideoCapture(str(video))
    out = dict(fps=float(v.get(cv2.CAP_PROP_FPS)), n_listed=int(v.get(cv2.CAP_PROP_FRAME_COUNT)),
               width=int(v.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    v.release()
    return out


def iter_frames_cv(video: str | os.PathLike, frames: Iterable[int] | None = None, stride: int | None = None,
                   start: int = 0, stop: int | None = None, prefetch: int = 8) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, RGB uint8) in decode order. ``frames`` = explicit sorted indices, or every ``stride``-th frame
    from ``start`` (frames before ``start`` are grabbed, not converted). Decoding runs in a thread."""
    import cv2
    want = None if frames is None else sorted(set(int(f) for f in frames))
    if want is not None:
        if not want:
            return
        stop = want[-1] + 1 if stop is None else min(stop, want[-1] + 1)
        wset = set(want)
    q: queue.Queue = queue.Queue(maxsize=prefetch)
    halt = threading.Event()

    def worker():
        v = cv2.VideoCapture(str(video))
        i = 0
        try:
            while not halt.is_set():
                if stop is not None and i >= stop:
                    break
                take = (i in wset) if want is not None else (i >= start and (stride is None or (i - start) % stride == 0))
                if take:
                    ok, bgr = v.read()
                    if not ok:
                        break
                    q.put((i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                else:
                    if not v.grab():
                        break
                i += 1
        finally:
            v.release()
            q.put(None)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
    finally:
        halt.set()
        while th.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                time.sleep(0.01)


# ----------------------------------------------------------------------------------------------- runners

def _torch_setup():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        # shared GPU: never take more than UWID_GPU_GB (default 8.5 GB); exceeding it raises out-of-memory instead
        total = torch.cuda.get_device_properties(0).total_memory
        cap = float(os.environ.get("UWID_GPU_GB", "8.5")) * 1e9
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total))
    return torch, dev


def _nms_per_query(df: pd.DataFrame, iou: float = 0.5, per_query: int = 30) -> pd.DataFrame:
    if df.empty:
        return df
    import torch
    from torchvision.ops import nms
    keep = []
    for _, g in df.groupby("query", sort=False):
        b = torch.tensor(g[["x0", "y0", "x1", "y1"]].to_numpy(np.float32))
        s = torch.tensor(g["score"].to_numpy(np.float32))
        k = nms(b, s, iou)[:per_query].numpy()
        keep.append(g.iloc[k])
    return pd.concat(keep, ignore_index=True)


def patch_score_column(query: str) -> str:
    """Column name of the per-patch score of ``query`` in OwlRunner(patch_scores=True) output."""
    return "score__" + "_".join(query.lower().split())


class OwlRunner:
    """OWLv2 boxes for a list of text queries; keeps boxes with score >= ``min_score`` after per-query NMS."""

    def __init__(self, queries: Sequence[str], model_id: str = "google/owlv2-base-patch16-ensemble",
                 min_score: float = 0.03, batch: int = 4, name: str = "owlv2-base", patch_scores: bool = False,
                 gpu_preprocess: bool = False):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch, self.dev = _torch_setup()
        self.queries, self.min_score, self.batch, self.name = list(queries), min_score, batch, name
        # patch_scores: every box also carries the score of every query at its patch (columns score__<query>),
        # so a "whale" box can be compared with the "fish" score of the same box (WP7 fish rule)
        self.patch_scores = patch_scores
        # gpu_preprocess: the processor's own torch ops (pad, anti-aliased resize, normalise) run on the GPU instead of
        # the CPU (same operations; ~0.2 s per 1080p frame saved on a loaded CPU)
        self.pre_device = self.dev if gpu_preprocess else None
        self.proc = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id, torch_dtype=self.torch.float16).to(self.dev).eval()
        tok = self.proc(text=[self.queries], return_tensors="pt")
        self.input_ids = tok["input_ids"].to(self.dev)
        self.attn = tok["attention_mask"].to(self.dev)

    def detect(self, images: Sequence[np.ndarray]) -> list[pd.DataFrame]:
        torch = self.torch
        out_all = []
        for i in range(0, len(images), self.batch):
            ims = images[i:i + self.batch]
            kw = {"device": self.pre_device} if self.pre_device else {}
            pix = self.proc(images=list(ims), return_tensors="pt", **kw)["pixel_values"].to(self.dev, torch.float16)
            n = len(ims)
            with torch.no_grad():
                out = self.model(input_ids=self.input_ids.repeat(n, 1), attention_mask=self.attn.repeat(n, 1),
                                 pixel_values=pix)
            logits = out.logits.float().sigmoid().cpu().numpy()          # (n, P, Q)
            boxes = out.pred_boxes.float().cpu().numpy()                   # (n, P, 4) cx cy w h, of the padded square
            for j, im in enumerate(ims):
                h, w = im.shape[:2]
                side = max(h, w)
                cx, cy, bw, bh = (boxes[j] * side).T
                xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
                p, q = np.nonzero(logits[j] >= self.min_score)
                cols = dict(query=[self.queries[k] for k in q], score=logits[j][p, q],
                            x0=xyxy[p, 0], y0=xyxy[p, 1], x1=xyxy[p, 2], y1=xyxy[p, 3])
                if self.patch_scores:
                    for k, qq in enumerate(self.queries):
                        cols[patch_score_column(qq)] = logits[j][p, k]
                df = pd.DataFrame(cols)
                df = df[(df.x1 - df.x0 > 1) & (df.y1 - df.y0 > 1)]
                out_all.append(_nms_per_query(df))
        return out_all


class GroundingRunner:
    """LLMDet / Grounding-DINO style detector (transformers zero-shot object detection). All queries go in one
    prompt; each box gets the query whose tokens it matches best."""

    def __init__(self, queries: Sequence[str], model_id: str = "iSEE-Laboratory/llmdet_large",
                 min_score: float = 0.05, batch: int = 2, name: str = "llmdet-large"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch, self.dev = _torch_setup()
        self.queries, self.min_score, self.batch, self.name = list(queries), min_score, batch, name
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, torch_dtype=self.torch.float16) \
            .to(self.dev).eval()

    def detect(self, images: Sequence[np.ndarray]) -> list[pd.DataFrame]:
        torch = self.torch
        res = []
        for i in range(0, len(images), self.batch):
            ims = list(images[i:i + self.batch])
            inp = self.proc(images=ims, text=[self.queries] * len(ims), return_tensors="pt")
            inp = {k: (v.to(self.dev, torch.float16) if v.dtype.is_floating_point else v.to(self.dev))
                   for k, v in inp.items()}
            with torch.no_grad():
                out = self.model(**inp)
            out.logits = out.logits.float()
            out.pred_boxes = out.pred_boxes.float()
            post = self.proc.post_process_grounded_object_detection(
                out, inp["input_ids"], threshold=self.min_score, text_threshold=self.min_score,
                target_sizes=[im.shape[:2] for im in ims], text_labels=[self.queries] * len(ims))
            for r in post:
                b = r["boxes"].cpu().numpy().reshape(-1, 4)
                labels = r.get("text_labels", r.get("labels"))
                df = pd.DataFrame(dict(query=[str(x) for x in labels], score=r["scores"].cpu().numpy(),
                                       x0=b[:, 0], y0=b[:, 1], x1=b[:, 2], y1=b[:, 3]))
                df = df[df["query"].isin(self.queries)]
                res.append(_nms_per_query(df))
        return res


class TiledRunner:
    """Run ``inner`` on a grid of overlapping tiles (for small objects such as 15-20 px fish) and map the boxes back
    to the full frame. Boxes cut by an inner tile edge are kept; per-query NMS merges duplicates."""

    def __init__(self, inner, grid: tuple[int, int] = (2, 2), overlap: float = 0.15):
        self.inner, self.grid, self.overlap = inner, grid, overlap
        self.queries, self.name = inner.queries, inner.name + f"-tiles{grid[0]}x{grid[1]}"
        self.batch = 1

    def _tiles(self, h, w):
        nx, ny = self.grid
        tw, th = w / (nx - (nx - 1) * self.overlap), h / (ny - (ny - 1) * self.overlap)
        for iy in range(ny):
            for ix in range(nx):
                x0 = int(round(ix * tw * (1 - self.overlap)))
                y0 = int(round(iy * th * (1 - self.overlap)))
                yield x0, y0, min(w, int(round(x0 + tw))), min(h, int(round(y0 + th)))

    def detect(self, images):
        res = []
        for im in images:
            h, w = im.shape[:2]
            tiles = list(self._tiles(h, w))
            dets = self.inner.detect([im[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles])
            parts = []
            for (x0, y0, _, _), d in zip(tiles, dets):
                d = d.copy()
                d[["x0", "x1"]] += x0
                d[["y0", "y1"]] += y0
                parts.append(d)
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS[3:])
            res.append(_nms_per_query(df))
        return res


class Sam3Runner:
    """SAM 3 (transformers ``Sam3Model``): text prompt in, instance masks + boxes + scores out. The image is encoded
    once per frame and reused for every prompt. Adds column ``area_frac`` (mask pixels / frame pixels) to the common
    format. With ``keep_masks`` the boolean masks of the last call are in ``self.last_masks`` (list per image of
    lists aligned with the rows)."""

    def __init__(self, queries: Sequence[str], model_id: str = "facebook/sam3", min_score: float = 0.1,
                 keep_masks: bool = False, name: str = "sam3", dtype: str = "float16"):
        from transformers import Sam3Model, Sam3Processor
        self.torch, self.dev = _torch_setup()
        self.queries, self.min_score, self.keep_masks, self.name = list(queries), min_score, keep_masks, name
        self.batch = 1
        self.dtype = getattr(self.torch, dtype)
        self.proc = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, torch_dtype=self.dtype).to(self.dev).eval()
        self.text = {}
        for q in self.queries:
            t = self.proc(text=q, return_tensors="pt")
            self.text[q] = {k: v.to(self.dev) for k, v in t.items() if k in ("input_ids", "attention_mask")}
        self.last_masks = []

    def detect(self, images: Sequence[np.ndarray]) -> list[pd.DataFrame]:
        torch = self.torch
        res, self.last_masks = [], []
        for im in images:
            h, w = im.shape[:2]
            pix = self.proc(images=im, return_tensors="pt")["pixel_values"].to(self.dev, self.dtype)
            rows, masks = [], []
            with torch.no_grad():
                ve = self.model.get_vision_features(pixel_values=pix)
                for q in self.queries:
                    out = self.model(vision_embeds=ve, **self.text[q])
                    for k in ("pred_logits", "pred_boxes", "pred_masks", "presence_logits"):
                        if getattr(out, k, None) is not None:
                            setattr(out, k, getattr(out, k).float())
                    r = self.proc.post_process_instance_segmentation(out, threshold=self.min_score, mask_threshold=0.5,
                                                                     target_sizes=[(h, w)])[0]
                    b = r["boxes"].cpu().numpy().reshape(-1, 4)
                    ms = r["masks"].cpu().numpy().astype(bool) if len(b) else np.zeros((0, h, w), bool)
                    for i in range(len(b)):
                        rows.append(dict(query=q, score=float(r["scores"][i]), x0=b[i, 0], y0=b[i, 1], x1=b[i, 2],
                                         y1=b[i, 3], area_frac=float(ms[i].mean())))
                        if self.keep_masks:
                            masks.append(ms[i])
            res.append(pd.DataFrame(rows, columns=["query", "score", "x0", "y0", "x1", "y1", "area_frac"]))
            self.last_masks.append(masks)
        return res


class YoloeRunner:
    """YOLOE-26 (ultralytics, AGPL-3.0) with text prompts: boxes (+ masks, not kept). Weights and the MobileCLIP text
    encoder live in ``underwater_cache/detect/weights`` (the loader downloads into the working directory, so it is
    switched there while loading)."""

    def __init__(self, queries: Sequence[str], weights: str = "yoloe-26l-seg.pt", min_score: float = 0.05,
                 imgsz: int = 1280, batch: int = 8, name: str | None = None):
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLOE
        self.torch, self.dev = _torch_setup()
        self.queries, self.min_score, self.imgsz, self.batch = list(queries), min_score, imgsz, batch
        self.name = name or weights.replace(".pt", "")
        wdir = paths.cache_dir("detect") / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        here = os.getcwd()
        try:
            os.chdir(wdir)
            self.model = YOLOE(str(wdir / weights))
            self.model.set_classes(self.queries, self.model.get_text_pe(self.queries))
        finally:
            os.chdir(here)

    def detect(self, images: Sequence[np.ndarray]) -> list[pd.DataFrame]:
        ims = [np.ascontiguousarray(im[:, :, ::-1]) for im in images]      # ultralytics expects BGR arrays
        rs = self.model.predict(ims, conf=self.min_score, imgsz=self.imgsz, half=self.dev == "cuda",
                                device=0 if self.dev == "cuda" else "cpu", verbose=False)
        out = []
        for r in rs:
            b = r.boxes.xyxy.cpu().numpy().reshape(-1, 4)
            out.append(pd.DataFrame(dict(query=[self.queries[int(c)] for c in r.boxes.cls.cpu().numpy()],
                                         score=r.boxes.conf.cpu().numpy(), x0=b[:, 0], y0=b[:, 1], x1=b[:, 2],
                                         y1=b[:, 3])))
        return out


class SamMasker:
    """SAM 2.1 (transformers ``Sam2Model``) masks from boxes on one frame."""

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large"):
        from transformers import Sam2Model, Sam2Processor
        self.torch, self.dev = _torch_setup()
        self.model = Sam2Model.from_pretrained(model_id).to(self.dev).eval()
        self.proc = Sam2Processor.from_pretrained(model_id)

    def masks(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Boolean masks (n, H, W) for boxes (n, 4) xyxy; the best of SAM's three proposals by its own IoU score."""
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0,) + image.shape[:2], bool)
        torch = self.torch
        inp = self.proc(images=image, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.dev)
        with torch.no_grad():
            out = self.model(**inp, multimask_output=True)
        ms = self.proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]   # (n, 3, H, W)
        best = out.iou_scores[0].float().cpu().numpy().argmax(-1)                        # (n,)
        return np.stack([ms[k, best[k]].numpy() > 0 for k in range(len(boxes))])


class Mask2FormerSeg:
    """Mask2Former semantic label map (ADE20K semantic, or the semantic view of COCO panoptic)."""

    def __init__(self, model_id: str = "facebook/mask2former-swin-large-ade-semantic", short_side: int = 540):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.torch, self.dev = _torch_setup()
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, torch_dtype=self.torch.float16) \
            .to(self.dev).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.short_side = short_side
        self.panoptic = "panoptic" in model_id

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Class-id map at (H, W) of ``image`` (int16)."""
        import cv2
        torch = self.torch
        h, w = image.shape[:2]
        s = self.short_side / min(h, w)
        small = cv2.resize(image, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        inp = self.proc(images=small, return_tensors="pt", do_resize=False)
        inp = {k: (v.to(self.dev, torch.float16) if v.dtype.is_floating_point else v.to(self.dev)) for k, v in inp.items()}
        with torch.no_grad():
            out = self.model(pixel_values=inp["pixel_values"], pixel_mask=inp.get("pixel_mask"))
        out.class_queries_logits = out.class_queries_logits.float()
        out.masks_queries_logits = out.masks_queries_logits.float()
        seg = self.proc.post_process_semantic_segmentation(out, target_sizes=[(h, w)])[0]
        return seg.cpu().numpy().astype(np.int16)


# ----------------------------------------------------------------------------------------------- cache

def run_cached(video: str | os.PathLike, runner, run_name: str, frames: Sequence[int] | None = None,
               stride: int | None = None, stop: int | None = None, chunk: int = 300,
               time_budget_s: float = 780.0, log=print) -> bool:
    """Run ``runner`` over ``video`` and cache the boxes in chunks of ``chunk`` frame indices.

    Chooses frames by explicit list or by ``stride`` (up to ``stop``). Returns True when every chunk is done,
    False when the time budget ran out (run the same call again to continue)."""
    info = video_info(video)
    fps = info["fps"]
    out_dir = detect_cache_dir(video, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(video=str(video), run_name=run_name, model=runner.name, queries=list(runner.queries),
                stride=stride, chunk=chunk, fps=fps)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    if frames is None:
        last = info["n_listed"] if stop is None else stop
        frames = list(range(0, last, stride or 1))
    frames = sorted(set(int(f) for f in frames))
    by_chunk: dict[int, list[int]] = {}
    for f in frames:
        by_chunk.setdefault((f // chunk) * chunk, []).append(f)
    todo = [c for c in sorted(by_chunk) if not (out_dir / f"chunk_{c:07d}.frames.npy").exists()]
    if not todo:
        return True
    t_start = time.time()
    need = [f for c in todo for f in by_chunk[c]]
    pending: dict[int, list] = {}
    n_done = 0
    batch_frames, batch_imgs = [], []

    def flush():
        nonlocal batch_frames, batch_imgs
        if not batch_frames:
            return
        dets = runner.detect(batch_imgs)
        for f, d in zip(batch_frames, dets):
            d = d.copy()
            d.insert(0, "frame", f)
            d.insert(1, "t", f / fps)
            d.insert(2, "model", runner.name)
            pending.setdefault((f // chunk) * chunk, []).append((f, d))
        batch_frames, batch_imgs = [], []

    def write_ready(upto_frame: int | None):
        nonlocal n_done
        for c in sorted(list(pending)):
            if upto_frame is not None and c + chunk > upto_frame and len(pending[c]) < len(by_chunk[c]):
                continue
            items = pending.pop(c)
            got = sorted(f for f, _ in items)
            if len(got) < len(by_chunk[c]):          # end of stream: some frames never decoded
                log(f"chunk {c}: {len(got)} of {len(by_chunk[c])} frames decoded")
            dfs = [d for _, d in items if len(d)]
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=COLUMNS)
            df = df.reindex(columns=COLUMNS + [c for c in df.columns if c not in COLUMNS])
            df.to_parquet(out_dir / f"chunk_{c:07d}.parquet", index=False)
            np.save(out_dir / f"chunk_{c:07d}.frames.npy", np.asarray(got, np.int64))
            n_done += 1

    bs = getattr(runner, "batch", 4)
    for f, img in iter_frames_cv(video, frames=need):
        batch_frames.append(f)
        batch_imgs.append(img)
        if len(batch_frames) >= bs:
            flush()
            write_ready(f)
        if time.time() - t_start > time_budget_s:
            flush()
            write_ready(f)
            # drop partial chunks: they are redone next time
            log(f"time budget reached after {n_done} chunks")
            return False
    flush()
    write_ready(None)
    elapsed = time.time() - t_start
    log(f"{run_name}: {len(need)} frames in {elapsed:.0f} s ({len(need) / max(elapsed, 1e-6):.1f} frames/s)")
    return True


def load_cached(video: str | os.PathLike, run_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """(boxes, processed frame indices) of a cached run."""
    d = detect_cache_dir(video, run_name)
    parts = sorted(d.glob("chunk_*.parquet"))
    fr = sorted(d.glob("chunk_*.frames.npy"))
    dfs = [pd.read_parquet(p) for p in parts]
    dfs = [x for x in dfs if len(x)]
    boxes = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=COLUMNS)
    frames = np.concatenate([np.load(p) for p in fr]) if fr else np.zeros(0, np.int64)
    return boxes, np.sort(frames)
