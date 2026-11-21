"""Shared helpers of 04b_patch_segmentation: folds over time, pixel-level scoring against the reference masks."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from uwid import patchseg as ps

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REF = Path("/home/sat3737/NOAA_whales/underwater_cache/reference")
DEV_MASKS = REF / "2025-12-20_CG" / "masks"
SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
DEV_VIDEO = "/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
TEST_DIR = Path("/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96/Video")
TEST_STEMS = sorted(p.name for p in REF.glob("CC-11-96-20250203-*") if (p / "index.json").exists())
DEV_FPS = 29.9742

# time blocks of the dev clip (frame index ranges); the 31 mask frames fall 8 / 7 / 8 / 8 into them
BLOCKS = [(0, 700), (700, 1420), (1420, 2085), (2085, 2857)]
BUFFER = 100  # frames (3.3 s): training frames this close to a held-out block are dropped too
K = len(ps.CLASSES)  # classes of the reference masks (seafloor, fish, gear never in the dev clip)


def block_of(frame: int) -> int:
    for i, (a, b) in enumerate(BLOCKS):
        if a <= frame < b:
            return i
    raise ValueError(frame)


def train_frames_for(block: int, frames) -> list[int]:
    a, b = BLOCKS[block]
    return [f for f in frames if not (a - BUFFER <= f < b + BUFFER)]


# Label folders. The dev clip's "fish" (f1480-f1515) are not fish (user, 2026-11-20): code 3 is ignored there.
DEV = ps.LabelFolder("dev", DEV_MASKS, Path(DEV_VIDEO), ignore_codes=(ps.FISH,))
CONFIG_ARGS = {"w960": (960, (11,), 1), "w1280": (1280, (11,), 1), "w1920": (1920, (11,), 1),
               "w1280_L4": (1280, (2, 5, 8, 11), 1), "w1920_t2": (1920, (11,), 2)}


def test_folder(stem: str) -> ps.LabelFolder:
    return ps.LabelFolder(stem, REF / stem / "masks", REF / stem)


def load_folder(folder: ps.LabelFolder, config: str = "w1280", model=None) -> dict:
    w, layers, tiles = CONFIG_ARGS[config]
    return ps.cached_features(folder, w, layers, tiles, model=model, tag=config)


def load_dev(config: str):
    d = load_folder(DEV, config)
    return d["feats"], d["targets"], d["frames"]


def xy_from(feats, targets, idx, purity=0.75):
    X, Y = [], []
    for i in idx:
        y = ps.hard_labels(targets[i], purity)
        ok = y != ps.IGNORE
        X.append(torch.from_numpy(feats[i][ok])); Y.append(torch.from_numpy(y[ok]))
    return torch.cat(X), torch.cat(Y)


def read_mask(folder: ps.LabelFolder, frame: int):
    """Class mask (ignored codes -> 255), instance map and meta (without ignored classes)."""
    return (folder.class_mask(frame), *folder.instances(frame))


def border_band(cm: np.ndarray, width: int) -> np.ndarray:
    """True within ``width`` px of a boundary between two different codes (including unsure)."""
    k = np.ones((3, 3), np.uint8)
    edge = np.zeros(cm.shape, bool)
    edge[:, 1:] |= cm[:, 1:] != cm[:, :-1]
    edge[1:, :] |= cm[1:, :] != cm[:-1, :]
    if width <= 0:
        return np.zeros(cm.shape, bool)
    return cv2.dilate(edge.astype(np.uint8), k, iterations=width).astype(bool)


class Scorer:
    """Accumulates pixel confusion (strict and with a border band ignored), whale recall by grade, fish."""

    def __init__(self, band: int = 12):
        self.band = band
        self.conf = np.zeros((K, K + 1), np.int64)       # rows truth, cols prediction (+unknown)
        self.conf_band = np.zeros((K, K + 1), np.int64)
        self.grade = {}  # grade -> [pixels, pixels predicted whale, instances, instances found]
        self.faint_water = [0, 0]
        self.fish = {"instances": 0, "found": 0, "pred_components": 0, "pred_components_on_fish": 0}
        self.frames = 0

    def add(self, pred: np.ndarray, cm: np.ndarray, inst: np.ndarray, meta: dict):
        """pred: (H, W) predicted codes (0..7, 8 = unknown) at the mask's size."""
        self.frames += 1
        lab = cm != ps.IGNORE
        t, p = cm[lab].astype(np.int64), pred[lab].astype(np.int64)
        self.conf += np.bincount(t * (K + 1) + p, minlength=K * (K + 1)).reshape(K, K + 1)
        lab2 = lab & ~border_band(cm, self.band)
        t, p = cm[lab2].astype(np.int64), pred[lab2].astype(np.int64)
        self.conf_band += np.bincount(t * (K + 1) + p, minlength=K * (K + 1)).reshape(K, K + 1)
        for iid, m in meta.get("instances", {}).items():
            px = inst == int(iid)
            if not px.any():
                continue
            if m["class"] == "whale":
                g = self.grade.setdefault(m.get("grade", "?"), [0, 0, 0, 0])
                hit = int((pred[px] == ps.WHALE).sum())
                g[0] += int(px.sum()); g[1] += hit; g[2] += 1; g[3] += int(hit >= 0.25 * px.sum())
                if m.get("grade") == "faint":
                    self.faint_water[0] += int(px.sum()); self.faint_water[1] += int((pred[px] == ps.WATER).sum())
            elif m["class"] == "fish":
                near = cv2.dilate(px.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=8).astype(bool)
                self.fish["instances"] += 1; self.fish["found"] += int((pred[near] == ps.FISH).any())
        fishpred = (pred == ps.FISH).astype(np.uint8)
        if fishpred.any():
            n, lab_cc = cv2.connectedComponents(fishpred)
            fishtruth = cv2.dilate((cm == ps.FISH).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=8)
            self.fish["pred_components"] += n - 1
            self.fish["pred_components_on_fish"] += len(set(np.unique(lab_cc[fishtruth > 0])) - {0})

    @staticmethod
    def iou(conf):
        out = {}
        for c in range(K):
            tp = conf[c, c]; fn = conf[c].sum() - tp; fp = conf[:, c].sum() - tp
            out[ps.CLASSES[c]] = None if conf[c].sum() == 0 else float(tp / max(tp + fn + fp, 1))
        return out

    def summary(self) -> dict:
        c = self.conf

        def share(t, p):
            return None if c[t].sum() == 0 else float(c[t, p] / c[t].sum())
        s = {
            "frames": self.frames,
            "iou": self.iou(self.conf), "iou_band": self.iou(self.conf_band),
            "pixel_acc": float(np.trace(c[:, :K]) / c.sum()),
            "whale_recall_by_grade": {g: {"pixel_recall": v[1] / max(v[0], 1), "instances": v[2],
                                          "found_25pct": v[3]} for g, v in sorted(self.grade.items())},
            "confusions": {
                "back_as_whale": share(ps.BACK, ps.WHALE), "whale_as_back": share(ps.WHALE, ps.BACK),
                "bubbles_as_whale": share(ps.BUBBLES, ps.WHALE), "surface_as_whale": share(ps.SURFACE, ps.WHALE),
                "whale_as_water": share(ps.WHALE, ps.WATER), "water_as_whale": share(ps.WATER, ps.WHALE),
                "faint_whale_as_water": self.faint_water[1] / max(self.faint_water[0], 1),
                "back_as_water": share(ps.BACK, ps.WATER),
            },
            "fish": self.fish,
            "conf": self.conf.tolist(),
        }
        tot = sum(v[0] for v in self.grade.values()); hit = sum(v[1] for v in self.grade.values())
        s["whale_pixel_recall"] = hit / max(tot, 1)
        return s


def pred_codes(probs: torch.Tensor, size=(1080, 1920)) -> np.ndarray:
    """(h, w, K) probabilities -> (H, W) argmax codes at full size."""
    return ps.upsample_probs(probs, size).argmax(-1).to(torch.uint8).cpu().numpy()


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"
