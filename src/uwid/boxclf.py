"""Whale / fish / other for OWLv2 detector boxes: a linear classifier on frozen DINOv3 crop features (WP10).

Why: the scan separates whales from fish by box size alone (a "whale" box under 300 px is ``small``). Near fish can
be larger than far whales (2025-01-02: jacks of 120-560 px very near the lens), so the size rule passes fish as
whales in the 300-800 px band. This module looks at the box itself.

Input: the detector's rows in the scan's cache format (``uwid.detect.OwlRunner(..., patch_scores=True)``: one row per
prompt and box, columns ``frame, query, score, x0, y0, x1, y1, score__whale, score__fish, score__school_of_fish,
score__small_fish, score__air_bubbles``) and the RGB frame they were found on (boxes in its pixels).

Steps (:func:`classify`):
1. :func:`unique_boxes`: rows with the same coordinates are one OWLv2 patch box (each patch box is scored for every
   prompt); keep those whose best whale / fish prompt score is >= ``MIN_SCORE``.
2. :func:`group_objects`: boxes of one frame overlapping at IoU >= 0.5 are one object (greedy, best score first);
   the best box of each object is classified, the other boxes of the object get its answer.
3. Features (:func:`box_features`): DINOv3 ViT-B/16 embedding (class token + mean patch token, last block) of a
   square crop around the box (``tight``: side = longest box side; ``ctx``: twice that), the box geometry and the
   detector's five prompt scores.
4. A linear softmax model (:class:`LinearBoxModel`, standardised features) gives p(whale), p(fish), p(other).

Trained weights live in ``underwater_cache/boxclf/`` (never in the repo); experiments/08_whale_fish_boxes says on
what they were fitted and how they were tested.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROMPTS = ("whale", "fish", "school of fish", "small fish", "air bubbles")
WHALE_Q = "whale"
FISH_QS = ("fish", "school of fish", "small fish")
CLASSES = ("whale", "fish", "other")
MIN_SCORE = 0.2
GROUP_IOU = 0.5
CROP_SIZE = 224
CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache/boxclf")
DEFAULT_WEIGHTS = CACHE / "boxclf_v1.npz"


def _score_col(q: str) -> str:
    return "score__" + "_".join(q.lower().split())


def _s(q: str) -> str:
    """Column of a prompt's score in unique_boxes() output (s_whale, s_fish_prompt, s_school_of_fish, ...)."""
    return "s_fish_prompt" if q == "fish" else "s_" + "_".join(q.split())


# ------------------------------------------------------------------------------------------------ boxes and objects

def unique_boxes(boxes: pd.DataFrame, min_score: float = MIN_SCORE) -> pd.DataFrame:
    """One row per OWLv2 patch box (rows of all prompts with identical coordinates merged). Columns: frame (if
    given), x0, y0, x1, y1, s_<prompt> for the five prompts, s_whale, s_fish (best fish prompt), rows (list of input
    row labels). Only boxes with max(s_whale, s_fish) >= ``min_score``."""
    cols = ["frame", "x0", "y0", "x1", "y1"] + [_s(q) for q in PROMPTS] + ["s_fish", "rows", "queries"]
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=cols)
    b = boxes.copy()
    if "frame" not in b:
        b["frame"] = 0
    for q in PROMPTS:
        c = _score_col(q)
        if c not in b:   # no patch scores: fall back to the row's own score for its prompt
            b[c] = np.where(b["query"] == q, b["score"], np.nan)
    key = ["frame", "x0", "y0", "x1", "y1"]
    b["_k"] = list(zip(b.frame, b.x0.round(2), b.y0.round(2), b.x1.round(2), b.y1.round(2)))
    agg = {_score_col(q): "max" for q in PROMPTS}
    g = b.groupby("_k", sort=False)
    u = g[key].first().join(g.agg(agg)).reset_index(drop=True)
    u["rows"] = list(g.apply(lambda x: list(x.index), include_groups=False))
    u["queries"] = list(g["query"].agg(lambda x: sorted(set(x))))
    u = u.rename(columns={_score_col(q): _s(q) for q in PROMPTS})
    u["s_fish"] = u[[_s(q) for q in FISH_QS]].max(axis=1)      # best fish prompt
    best = np.fmax(u.s_whale.to_numpy(float), u.s_fish.to_numpy(float))
    return u[best >= min_score].reset_index(drop=True)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4); b = np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-6, None)


def group_objects(u: pd.DataFrame, iou: float = GROUP_IOU) -> pd.DataFrame:
    """Adds ``obj`` (object number within the frame) and ``rep`` (True for the object's best box)."""
    u = u.copy()
    u["obj"] = -1; u["rep"] = False
    for _, idx in u.groupby("frame").groups.items():
        idx = np.asarray(list(idx))
        sc = np.fmax(u.loc[idx, "s_whale"].to_numpy(float), u.loc[idx, "s_fish"].to_numpy(float))
        order = idx[np.argsort(-sc)]
        xy = u.loc[order, ["x0", "y0", "x1", "y1"]].to_numpy(float)
        M = iou_matrix(xy, xy)
        obj = -np.ones(len(order), int); k = 0
        for i in range(len(order)):
            if obj[i] >= 0:
                continue
            obj[i] = k
            u.loc[order[i], "rep"] = True
            for j in range(i + 1, len(order)):
                if obj[j] < 0 and M[i, j] >= iou:
                    obj[j] = k
            k += 1
        u.loc[order, "obj"] = obj
    return u


# ------------------------------------------------------------------------------------------------ features

GEOM_NAMES = ("log_px", "log_w", "log_h", "aspect", "cx", "cy", "y1", "x_edge", "y_edge", "log_area")


def geometry(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Box geometry in 1920 px units (the scan's size scale)."""
    s = 1920.0 / width
    b = np.asarray(xyxy, float).reshape(-1, 4)
    w = np.clip(b[:, 2] - b[:, 0], 1, None) * s; h = np.clip(b[:, 3] - b[:, 1], 1, None) * s
    px = np.maximum(w, h)
    cx = (b[:, 0] + b[:, 2]) / 2 / width; cy = (b[:, 1] + b[:, 3]) / 2 / height
    x_edge = ((b[:, 0] <= 2) | (b[:, 2] >= width - 2)).astype(float)
    y_edge = ((b[:, 1] <= 2) | (b[:, 3] >= height - 2)).astype(float)
    area = w * h / (1920.0 * 1920.0 * height / width)
    return np.stack([np.log(px), np.log(w), np.log(h), np.log(w / h), cx, cy, b[:, 3] / height, x_edge, y_edge,
                     np.log(area)], 1)


def score_features(u: pd.DataFrame) -> np.ndarray:
    """Logits of the five prompt scores and whale minus best fish."""
    cols = [_s(q) for q in PROMPTS]
    s = np.clip(u[cols].to_numpy(float), 1e-4, 1 - 1e-4)
    s = np.nan_to_num(s, nan=1e-4)
    lg = np.log(s / (1 - s))
    fish = lg[:, 1:4].max(1)
    return np.concatenate([lg, (lg[:, 0] - fish)[:, None]], 1)


SCORE_NAMES = tuple("logit_" + "_".join(q.split()) for q in PROMPTS) + ("whale_minus_fish",)


def crop_squares(xyxy: np.ndarray, scale: float, min_side: float = 32.0) -> np.ndarray:
    b = np.asarray(xyxy, float).reshape(-1, 4)
    cx = (b[:, 0] + b[:, 2]) / 2; cy = (b[:, 1] + b[:, 3]) / 2
    side = np.maximum(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) * scale, min_side * scale)
    return np.stack([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], 1)


def _crop(image: np.ndarray, sq, size: int) -> np.ndarray:
    import cv2
    H, W = image.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in sq]
    side = max(1, x1 - x0)
    canvas = np.empty((side, side, 3), np.uint8); canvas[:] = (124, 116, 104)  # ImageNet mean colour
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)
    if ax1 > ax0 and ay1 > ay0:
        canvas[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = image[ay0:ay1, ax0:ax1]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
    return cv2.resize(canvas, (size, size), interpolation=interp)


def crop_embeddings(model, image: np.ndarray, xyxy: np.ndarray, scales=(1.0, 2.0), size: int = CROP_SIZE,
                    batch: int = 64) -> dict[float, np.ndarray]:
    """DINOv3 class token + mean patch token (last block, normed) of square crops around each box: {scale: (n, 2C)}."""
    import torch
    from .patchseg import IMAGENET_MEAN, IMAGENET_STD
    p = next(model.parameters())
    m = torch.tensor(IMAGENET_MEAN, device=p.device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=p.device).view(1, 3, 1, 1)
    out = {}
    for sc in scales:
        sq = crop_squares(xyxy, sc)
        crops = [_crop(image, q, size) for q in sq]
        embs = []
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(np.stack(crops[i:i + batch])).to(p.device).permute(0, 3, 1, 2).float() / 255.0
            x = ((x - m) / s).to(p.dtype)
            with torch.no_grad():
                (pt, cls), = model.get_intermediate_layers(x, n=1, return_class_token=True, norm=True)
            embs.append(torch.cat([cls, pt.mean(1)], 1).float().cpu().numpy())
        out[sc] = np.concatenate(embs) if embs else np.zeros((0, 2 * int(getattr(model, "embed_dim", 768))), np.float32)
    return out


@dataclass
class FeatureSpec:
    """Which feature blocks go into the model, in this order: any of ``tight``, ``ctx`` (crop embeddings), ``geom``,
    ``scores``."""
    blocks: tuple = ("tight", "ctx", "geom", "scores")

    def build(self, parts: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(parts[b], np.float32).reshape(len(parts[b]), -1) for b in self.blocks], 1)


def box_features(model, image: np.ndarray, u: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    H, W = image.shape[:2]
    xy = u[["x0", "y0", "x1", "y1"]].to_numpy(float)
    parts = {"geom": geometry(xy, W, H), "scores": score_features(u)}
    scales = [s for s, n in ((1.0, "tight"), (2.0, "ctx")) if n in spec.blocks]
    if scales:
        e = crop_embeddings(model, image, xy, tuple(scales))
        if 1.0 in e:
            parts["tight"] = e[1.0]
        if 2.0 in e:
            parts["ctx"] = e[2.0]
    return spec.build(parts)


# ------------------------------------------------------------------------------------------------ model

@dataclass
class LinearBoxModel:
    """Standardised features -> softmax over CLASSES. ``W`` (K, D), ``b`` (K,)."""
    mu: np.ndarray
    sd: np.ndarray
    W: np.ndarray
    b: np.ndarray
    spec: FeatureSpec = field(default_factory=FeatureSpec)
    classes: tuple = CLASSES
    meta: dict = field(default_factory=dict)

    def proba(self, X: np.ndarray) -> np.ndarray:
        z = ((np.asarray(X, np.float64) - self.mu) / self.sd) @ self.W.T + self.b
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    @classmethod
    def from_sklearn(cls, scaler, lr, spec: FeatureSpec, meta: dict | None = None) -> "LinearBoxModel":
        classes = tuple(CLASSES[int(c)] for c in lr.classes_)
        W, b = lr.coef_, lr.intercept_
        if W.shape[0] == 1:  # two classes
            W = np.concatenate([-W / 2, W / 2]); b = np.array([-b[0] / 2, b[0] / 2])
        return cls(scaler.mean_, scaler.scale_, W, b, spec, classes, meta or {})

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mu=self.mu, sd=self.sd, W=self.W, b=self.b,
                 info=json.dumps({"blocks": list(self.spec.blocks), "classes": list(self.classes), "meta": self.meta}))

    @classmethod
    def load(cls, path: Path) -> "LinearBoxModel":
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["info"]))
        return cls(d["mu"], d["sd"], d["W"], d["b"], FeatureSpec(tuple(info["blocks"])), tuple(info["classes"]),
                   info.get("meta", {}))


_LOADED: dict = {}


def _backbone():
    if "model" not in _LOADED:
        import torch
        from .patchseg import load_backbone
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _LOADED["model"] = load_backbone(dev, torch.float16 if dev == "cuda" else torch.float32)
    return _LOADED["model"]


def classify(boxes: pd.DataFrame, frame: np.ndarray, model=None, clf: LinearBoxModel | str | Path | None = None,
             min_score: float = MIN_SCORE) -> pd.DataFrame:
    """Class and probability for every detector row of one frame.

    ``boxes``: the detector rows of ONE frame (scan cache format, pixels of ``frame``); ``frame``: RGB uint8.
    Returns a DataFrame with the index of ``boxes`` and columns ``boxclf_class`` (whale / fish / other, or "" for rows
    whose patch box scores below ``min_score`` on every whale / fish prompt), ``boxclf_p`` (probability of that class),
    ``p_whale``, ``p_fish``, ``p_other``, ``boxclf_obj`` (object number in the frame; rows of one object share the
    answer of its best box). ``model``: a loaded DINOv3 backbone (loaded once and kept otherwise); ``clf``: a
    :class:`LinearBoxModel` or the path of its weights (default ``DEFAULT_WEIGHTS``)."""
    out = pd.DataFrame(index=boxes.index if boxes is not None else [],
                       columns=["boxclf_class", "boxclf_p", "p_whale", "p_fish", "p_other", "boxclf_obj"])
    out["boxclf_class"] = ""
    if boxes is None or len(boxes) == 0:
        return out
    if clf is None or isinstance(clf, (str, Path)):
        path = Path(clf or DEFAULT_WEIGHTS)
        if str(path) not in _LOADED:
            _LOADED[str(path)] = LinearBoxModel.load(path)
        clf = _LOADED[str(path)]
    b = boxes.copy()
    b["frame"] = 0
    u = unique_boxes(b, min_score)
    if not len(u):
        return out
    u = group_objects(u)
    reps = u[u.rep].reset_index(drop=True)
    need_img = any(x in clf.spec.blocks for x in ("tight", "ctx"))
    X = box_features(model or (_backbone() if need_img else None), frame, reps, clf.spec)
    P = clf.proba(X)
    full = np.zeros((len(reps), len(CLASSES)))
    for k, c in enumerate(clf.classes):
        full[:, CLASSES.index(c)] = P[:, k]
    by_obj = {int(o): full[i] for i, o in enumerate(reps.obj)}
    for _, r in u.iterrows():
        p = by_obj[int(r.obj)]
        k = int(np.argmax(p))
        for ix in r.rows:
            out.loc[ix, ["p_whale", "p_fish", "p_other"]] = p
            out.loc[ix, "boxclf_class"] = CLASSES[k]
            out.loc[ix, "boxclf_p"] = float(p[k])
            out.loc[ix, "boxclf_obj"] = int(r.obj)
    for c in ("boxclf_p", "p_whale", "p_fish", "p_other"):
        out[c] = out[c].astype(float)
    return out
