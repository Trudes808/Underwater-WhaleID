"""Semantic segmentation with little training: a small classifier on frozen DINOv3 patch features (WP5b).

Pipeline: an RGB frame is resized to a multiple of 16 px, the frozen DINOv3 ViT-B/16 gives one 768-d feature per
16 x 16 px patch, a linear (or small two-layer) classifier names each patch, and the class probabilities are
upsampled to the frame. Labels come from the reference masks (``<frame>_class.png``, codes as in
``experiments/02_reference_labels/README.md``): each patch gets the share of its pixels in every class, unsure pixels
(255) are ignored, and a patch whose majority class covers less than ``purity`` of its labelled pixels is ignored in
training (region borders, coarse polygons).

The decision unit is a ~1 s window: :func:`window_probs` averages the class maps of the window's frames and
:func:`whale_presence_scores` turns class maps into whale-presence scores (largest whale region, persistence).

Features are cached by the experiment scripts under ``underwater_cache/patchseg/``; nothing here writes files except
:func:`save_classifier`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV3_REPO = Path("/home/sat3737/dinov3")
DINOV3_WEIGHTS = DINOV3_REPO / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache/patchseg")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 16

# codes of the reference masks (README of 02_reference_labels)
CLASSES = ["water", "whale", "back", "fish", "bubbles", "surface", "air", "seafloor", "gear"]  # gear: drone parts (test masks)
IGNORE = 255
WATER, WHALE, BACK, FISH, BUBBLES, SURFACE, AIR, SEAFLOOR, GEAR = range(9)
UNKNOWN = 9  # output code for "unlike anything in training"
COLORS = np.array([
    [20, 60, 110],    # water: dark blue
    [255, 60, 40],    # whale: red
    [170, 80, 220],   # back: purple
    [255, 230, 0],    # fish: yellow
    [240, 240, 240],  # bubbles: white
    [0, 230, 230],    # surface: cyan
    [120, 170, 255],  # air: light blue
    [160, 100, 40],   # seafloor: brown
    [90, 90, 90],     # gear: grey
    [255, 0, 200],    # unknown: magenta
], dtype=np.uint8)


# ------------------------------------------------------------------------------------------------------------------
# backbone and features
# ------------------------------------------------------------------------------------------------------------------

def load_backbone(device: str = "cuda", dtype=torch.float16):
    if str(DINOV3_REPO) not in sys.path:
        sys.path.insert(0, str(DINOV3_REPO))
    # torch.hub.load(..., source="local") imports hubconf.py, which needs torchmetrics (segmentors); the backbone
    # builder itself does not, so import it directly (as Dinov3-WhaleID does)
    from dinov3.hub import backbones
    model = backbones.dinov3_vitb16(pretrained=True, weights=str(DINOV3_WEIGHTS))
    return model.eval().to(device=device, dtype=dtype)


def input_size(width: int, src_w: int = 1920, src_h: int = 1080) -> tuple[int, int]:
    """(W, H) both multiples of 16 for a target width, keeping the aspect ratio as near as the grid allows."""
    w = int(round(width / PATCH)) * PATCH
    h = int(round(width * src_h / src_w / PATCH)) * PATCH
    return w, h


def _prep(images: list[np.ndarray], size: tuple[int, int], device, dtype) -> torch.Tensor:
    x = torch.from_numpy(np.stack(images)).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(size[1], size[0]), mode="bilinear", antialias=True, align_corners=False)
    m = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return ((x - m) / s).to(dtype)


@torch.no_grad()
def extract(model, images: list[np.ndarray], width: int = 1920, layers: tuple[int, ...] = (11,),
            tiles: int = 1) -> torch.Tensor:
    """Patch features for RGB uint8 frames (all the same size). Returns (B, h, w, C) in the model's dtype.

    ``tiles`` > 1 cuts the frame into tiles x tiles pieces and runs each at ``width`` (so ``tiles=2, width=1920``
    looks at a 1080p frame at 2x: 8 px of the frame per patch); the feature maps are stitched back together.
    """
    p = next(model.parameters())
    dev, dt = p.device, p.dtype
    if tiles == 1:
        x = _prep(images, input_size(width, images[0].shape[1], images[0].shape[0]), dev, dt)
        outs = model.get_intermediate_layers(x, n=list(layers), reshape=True, norm=True)
        f = torch.cat(list(outs), 1)  # B, C, h, w
        return f.permute(0, 2, 3, 1).contiguous()
    H, W = images[0].shape[:2]
    rows = []
    for ti in range(tiles):
        cols = []
        for tj in range(tiles):
            y0, y1 = H * ti // tiles, H * (ti + 1) // tiles
            x0, x1 = W * tj // tiles, W * (tj + 1) // tiles
            crops = [np.ascontiguousarray(im[y0:y1, x0:x1]) for im in images]
            cols.append(extract(model, crops, width, layers, 1))
        rows.append(torch.cat(cols, 2))
    return torch.cat(rows, 1)


# ------------------------------------------------------------------------------------------------------------------
# label folders: training data = any number of folders of reference masks
# ------------------------------------------------------------------------------------------------------------------

@dataclass
class LabelFolder:
    """One folder of reference masks (``<frame>_class.png``, ``<frame>_inst.png``, ``<frame>_meta.json``) and where its
    frames come from: a folder of ``<frame>.jpg`` (full size) or the video itself (decoded in order, OpenCV frame
    index, as WP3 labels it). ``ignore_codes``: classes to treat as unsure in this folder (for example fish where the
    labeller's fish turned out not to be fish). Adding a deployment or a new class = adding a folder."""

    name: str
    masks: Path
    images: Path
    ignore_codes: tuple[int, ...] = ()
    exclude_frames: tuple[int, ...] = ()

    def frames(self) -> list[int]:
        fr = sorted(int(p.name.split("_")[0]) for p in Path(self.masks).glob("*_class.png"))
        return [f for f in fr if f not in self.exclude_frames]

    def class_mask(self, frame: int) -> np.ndarray:
        import cv2
        cm = cv2.imread(str(Path(self.masks) / f"{frame:06d}_class.png"), cv2.IMREAD_UNCHANGED)
        if self.ignore_codes:
            cm = cm.copy(); cm[np.isin(cm, self.ignore_codes)] = IGNORE
        return cm

    def instances(self, frame: int) -> tuple[np.ndarray, dict]:
        """Instance map and meta, without instances of ignored classes."""
        import cv2
        import json
        inst = cv2.imread(str(Path(self.masks) / f"{frame:06d}_inst.png"), cv2.IMREAD_UNCHANGED)
        meta = json.loads((Path(self.masks) / f"{frame:06d}_meta.json").read_text())
        ign = {CLASSES[c] for c in self.ignore_codes}
        meta = {**meta, "instances": {k: v for k, v in meta.get("instances", {}).items() if v.get("class") not in ign}}
        return inst, meta

    def images_for(self, frames: list[int]) -> dict[int, np.ndarray]:
        import cv2
        p = Path(self.images)
        if p.is_dir():
            out = {}
            for f in frames:
                for sub in ("", "sample", "frames"):
                    jp = p / sub / f"{f:06d}.jpg"
                    if jp.exists():
                        out[f] = cv2.cvtColor(cv2.imread(str(jp)), cv2.COLOR_BGR2RGB); break
            return out
        want = set(frames); out = {}
        cap = cv2.VideoCapture(str(p)); i = 0
        while len(out) < len(want) and i <= max(want):
            if i in want:
                ok, fr = cap.read()
                if not ok:
                    break
                out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            elif not cap.grab():
                break
            i += 1
        return out


def cached_features(folder: LabelFolder, width: int = 1280, layers: tuple[int, ...] = (11,), tiles: int = 1,
                    model=None, tag: str | None = None) -> dict:
    """Patch features of a label folder's mask frames, cached in CACHE/labels_<name>_<tag>.npz (features only;
    targets are rebuilt from the masks on every load so that relabelled masks and ``ignore_codes`` take effect).
    Frames added to the folder later are extracted and appended."""
    tag = tag or f"w{width}" + (f"_t{tiles}" if tiles > 1 else "") + ("" if layers == (11,) else "_L" + "-".join(map(str, layers)))
    path = CACHE / f"labels_{folder.name}_{tag}.npz"
    have = {}
    if path.exists():
        d = np.load(path)
        have = {int(f): d["feats"][i] for i, f in enumerate(d["frames"])}
    need = [f for f in folder.frames() if f not in have]
    if need:
        model = model or load_backbone()
        imgs = folder.images_for(need)
        for f in need:
            if f in imgs:
                have[f] = extract(model, [imgs[f]], width, layers, tiles)[0].cpu().numpy()
        fr = sorted(have)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, feats=np.stack([have[f] for f in fr]), frames=np.array(fr))
    fr = [f for f in folder.frames() if f in have]
    feats = np.stack([have[f] for f in fr])
    targets = np.stack([patch_targets(folder.class_mask(f), feats.shape[1:3]) for f in fr])
    return {"feats": feats, "targets": targets, "frames": fr, "folder": folder}


def training_set(data: list[dict], idx: list[list[int]] | None = None, purity: float = 0.75):
    """(X, Y) from several cached_features() results; ``idx[j]`` selects frames (positions) of ``data[j]``."""
    X, Y = [], []
    for j, d in enumerate(data):
        for i in (range(len(d["frames"])) if idx is None else idx[j]):
            y = hard_labels(d["targets"][i], purity)
            ok = y != IGNORE
            X.append(torch.from_numpy(d["feats"][i][ok])); Y.append(torch.from_numpy(y[ok]))
    return torch.cat(X), torch.cat(Y)


# ------------------------------------------------------------------------------------------------------------------
# labels on the patch grid
# ------------------------------------------------------------------------------------------------------------------

def patch_targets(class_mask: np.ndarray, grid: tuple[int, int], n_classes: int = len(CLASSES)) -> np.ndarray:
    """Share of each class among the labelled pixels of every patch: (h, w, n_classes + 1); the last channel is the
    share of ignored (255) pixels among all pixels. ``grid`` = (h, w) of the feature map; the mask is resampled
    to it by area."""
    h, w = grid
    m = torch.from_numpy(class_mask.astype(np.int64))
    onehot = torch.zeros(n_classes + 1, *m.shape)
    mm = m.clone(); mm[mm == IGNORE] = n_classes
    onehot.scatter_(0, mm[None], 1.0)
    pooled = F.adaptive_avg_pool2d(onehot[None], (h, w))[0]  # (K+1, h, w): share of all pixels
    lab = pooled[:n_classes]
    tot = lab.sum(0, keepdim=True).clamp_min(1e-6)
    out = torch.cat([lab / tot, pooled[n_classes:]], 0)
    return out.permute(1, 2, 0).numpy().astype(np.float32)


SMALL_CLASSES = {FISH: 0.3}  # small objects rarely fill a patch: a patch is a fish patch when >= 30 % of it is fish


def hard_labels(targets: np.ndarray, purity: float = 0.75, max_ignored: float = 0.25) -> np.ndarray:
    """Majority class per patch, IGNORE where the majority is below ``purity`` or too many pixels are unsure.
    Small classes (``SMALL_CLASSES``) win a patch when they cover at least their own share of it."""
    lab = targets[..., :-1]
    y = lab.argmax(-1).astype(np.int64)
    ok = (lab.max(-1) >= purity) & (targets[..., -1] <= max_ignored)
    for c, share in SMALL_CLASSES.items():
        if c < lab.shape[-1]:
            small = lab[..., c] >= share
            y[small] = c; ok |= small & (targets[..., -1] <= max_ignored)
    y[~ok] = IGNORE
    return y


# ------------------------------------------------------------------------------------------------------------------
# classifier
# ------------------------------------------------------------------------------------------------------------------

class PatchClassifier(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden: int = 0, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sd", torch.ones(dim))
        if hidden:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, n_classes))
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))
        self.hidden = hidden
        self.n_classes = n_classes

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net((f.float() - self.mu) / self.sd)


@dataclass
class TrainConfig:
    hidden: int = 0
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch: int = 4096
    class_weight_power: float = 0.5  # weight = (1 / freq) ** power, normalised
    dropout: float = 0.1
    seed: int = 0
    n_classes: int = len(CLASSES)
    extra: dict = field(default_factory=dict)


def train_classifier(X: torch.Tensor, y: torch.Tensor, cfg: TrainConfig, device: str = "cuda") -> PatchClassifier:
    """X (N, C) features, y (N,) class codes (no IGNORE). Classes absent from y get weight 0 and a low bias."""
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    X = X.to(device); y = y.to(device)
    clf = PatchClassifier(X.shape[1], cfg.n_classes, cfg.hidden, cfg.dropout).to(device)
    Xf = X.float()
    clf.mu.copy_(Xf.mean(0)); clf.sd.copy_(Xf.std(0).clamp_min(1e-3))
    counts = torch.bincount(y, minlength=cfg.n_classes).float()
    w = torch.where(counts > 0, (counts.sum() / counts.clamp_min(1)) ** cfg.class_weight_power, torch.zeros_like(counts))
    w = w / (w[counts > 0].mean())
    opt = torch.optim.AdamW(clf.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = len(y)
    steps = cfg.epochs * max(1, n // cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    clf.train()
    step = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n - cfg.batch + 1 if n >= cfg.batch else 1, cfg.batch):
            idx = perm[i:i + cfg.batch]
            loss = F.cross_entropy(clf(X[idx]), y[idx], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
    clf.eval()
    with torch.no_grad():  # classes never seen: push their logit down so they are never predicted
        last = clf.net[-1]
        for c in range(cfg.n_classes):
            if counts[c] == 0:
                last.weight[c].zero_(); last.bias[c].fill_(-1e4)
    return clf


@torch.no_grad()
def predict_probs(clf: PatchClassifier, feats: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """feats (..., C) -> class probabilities (..., K) float32 on the features' device."""
    shp = feats.shape[:-1]
    f = feats.reshape(-1, feats.shape[-1])
    out = torch.cat([F.softmax(clf(f[i:i + chunk].to(clf.mu.device)), -1) for i in range(0, len(f), chunk)])
    return out.reshape(*shp, -1)


def upsample_probs(probs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """(h, w, K) -> (H, W, K) bilinear; ``size`` = (H, W)."""
    p = probs.permute(2, 0, 1)[None].float()
    return F.interpolate(p, size=size, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)


def save_classifier(clf: PatchClassifier, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": clf.state_dict(), "dim": clf.mu.shape[0], "n_classes": clf.n_classes,
                "hidden": clf.hidden, "meta": meta}, path)


def load_classifier(path: Path, device: str = "cuda") -> tuple[PatchClassifier, dict]:
    d = torch.load(path, map_location=device, weights_only=False)
    clf = PatchClassifier(d["dim"], d["n_classes"], d["hidden"]).to(device)
    clf.load_state_dict(d["state"]); clf.eval()
    return clf, d["meta"]


# ------------------------------------------------------------------------------------------------------------------
# "unknown": distance to the nearest training feature
# ------------------------------------------------------------------------------------------------------------------

class FeatureBank:
    """Cosine distance from a patch to its nearest training patch (L2-normalised features)."""

    def __init__(self, X: torch.Tensor, max_n: int = 60000, seed: int = 0, device: str = "cuda"):
        g = torch.Generator().manual_seed(seed)
        if len(X) > max_n:
            X = X[torch.randperm(len(X), generator=g)[:max_n]]
        self.B = F.normalize(X.to(device).half(), dim=-1)

    @torch.no_grad()
    def distance(self, feats: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        shp = feats.shape[:-1]
        f = F.normalize(feats.reshape(-1, feats.shape[-1]).to(self.B.device).half(), dim=-1)
        d = torch.cat([1 - (f[i:i + chunk] @ self.B.T).max(1).values.float() for i in range(0, len(f), chunk)])
        return d.reshape(shp)


# ------------------------------------------------------------------------------------------------------------------
# windows
# ------------------------------------------------------------------------------------------------------------------

def window_probs(prob_maps: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Average of the class maps of a window's frames ((n, h, w, K) or a list) -> (h, w, K)."""
    if isinstance(prob_maps, list):
        prob_maps = torch.stack(prob_maps)
    return prob_maps.float().mean(0)


def _largest_component(mask: np.ndarray) -> int:
    import cv2
    if not mask.any():
        return 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


def whale_region_size(probs: torch.Tensor | np.ndarray, thr: float = 0.5) -> int:
    """Largest 8-connected region (in patches) where p(whale) > thr on one class map (h, w, K)."""
    p = probs[..., WHALE]
    p = p.cpu().numpy() if torch.is_tensor(p) else p
    return _largest_component(p > thr)


def whale_presence_scores(frame_probs: torch.Tensor, center: int | None = None, thr: float = 0.5,
                          min_region: int = 4) -> dict:
    """Scores for one window from its frames' class maps (n, h, w, K):
    ``single``: largest whale region on the centre frame; ``mean``: largest whale region on the averaged map;
    ``persist``: share of the window's frames with a whale region >= ``min_region`` patches;
    ``median``: median over frames of the largest whale region."""
    n = frame_probs.shape[0]
    c = n // 2 if center is None else center
    sizes = [whale_region_size(frame_probs[i], thr) for i in range(n)]
    return {
        "single": sizes[c],
        "mean": whale_region_size(window_probs(frame_probs), thr),
        "persist": float(np.mean([s >= min_region for s in sizes])),
        "median": float(np.median(sizes)),
        "n_frames": n,
    }


def colorize(labels: np.ndarray) -> np.ndarray:
    return COLORS[np.clip(labels, 0, len(COLORS) - 1)]
