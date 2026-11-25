"""WP15 mesh helpers (CPU, headless): point-splat rendering, silhouette match to the input outline, normalisation,
rigid alignment (PCA starts + ICP), chamfer distance, simple proportions. Needs numpy, scipy, trimesh, opencv
(the triposg env has them)."""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_points(path, n: int = 400_000, seed: int = 0, largest_only: bool = True):
    """Surface samples + normals. ``largest_only`` keeps the largest connected piece (by area) and reports the
    share of the area it holds."""
    m = trimesh.load(path, process=False)
    share = 1.0
    if largest_only:
        parts = m.split(only_watertight=False)
        if len(parts) > 1:
            areas = np.array([p.area for p in parts])
            share = float(areas.max() / areas.sum())
            m = parts[int(areas.argmax())]
    rng = np.random.default_rng(seed)
    pts, fi = trimesh.sample.sample_surface(m, n, seed=int(rng.integers(1 << 30)))
    return np.asarray(pts, np.float64), np.asarray(m.face_normals[fi], np.float64), share, m


def rot(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def proper_rotations() -> list[np.ndarray]:
    """The 24 rotations that map coordinate axes onto coordinate axes (det +1)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                R[r, c] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def render(pts, nrm, R, size: int = 400, scale: float | None = None, center=None, colour=(205, 205, 210),
           bg=(30, 30, 34)) -> np.ndarray:
    """Orthographic view: camera looks along -z of the rotated frame (x right, y up). Point splat with a z-buffer,
    headlight + key light shading. ``scale``: pixels per unit (shared across views of one mesh)."""
    p = pts @ R.T
    n = nrm @ R.T
    c = p.mean(0) if center is None else center @ R.T
    p = p - c
    if scale is None:
        scale = 0.45 * size / np.abs(p[:, :2]).max()
    u = np.round(size / 2 + p[:, 0] * scale).astype(int)
    v = np.round(size / 2 - p[:, 1] * scale).astype(int)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u, v, z, n = u[ok], v[ok], p[ok, 2], n[ok]
    light = np.array([-0.45, 0.55, 0.70])
    light /= np.linalg.norm(light)
    shade = 0.18 + 0.82 * np.abs(n @ light) ** 1.2
    order = np.argsort(-z)                       # nearest (largest z) first
    idx = v[order] * size + u[order]
    first = np.unique(idx, return_index=True)[1]
    img = np.empty((size * size, 3), np.float32)
    img[:] = bg
    sel = order[first]
    img[v[sel] * size + u[sel]] = np.clip(np.outer(shade[sel], colour), 0, 255)
    img = img.reshape(size, size, 3).astype(np.uint8)
    # close the small holes between splats
    fg = np.zeros((size, size), np.uint8)
    fg[v[sel], u[sel]] = 255
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    holes = (closed > 0) & (fg == 0)
    if holes.any():
        blur = cv2.blur(img, (3, 3))
        img[holes] = blur[holes]
    return img


def silhouette(pts, R, size: int = 128) -> np.ndarray:
    """Binary silhouette of the points seen along -z after rotation R, fitted into a size x size square with the
    same padding rule as TripoSG's input crop (longest side = size / 1.2)."""
    p = (pts @ R.T)[:, :2]
    lo, hi = p.min(0), p.max(0)
    s = (size / 1.2) / (hi - lo).max()
    off = (size - (hi - lo) * s) / 2
    u = np.clip(((p[:, 0] - lo[0]) * s + off[0]).astype(int), 0, size - 1)
    v = np.clip(((hi[1] - p[:, 1]) * s + off[1]).astype(int), 0, size - 1)
    m = np.zeros((size, size), np.uint8)
    m[v, u] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out.astype(bool)


def input_silhouette(alpha: np.ndarray, size: int = 128) -> np.ndarray:
    """The outline cropped to its box and padded to a square, as TripoSG's prepare_image does (padding 0.1)."""
    ys, xs = np.nonzero(alpha)
    a = alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    h, w = a.shape
    L = int(max(h, w) * 1.2)
    sq = np.zeros((L, L), np.uint8)
    sq[(L - h) // 2:(L - h) // 2 + h, (L - w) // 2:(L - w) // 2 + w] = a
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_NEAREST) > 0


def iou(a, b) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def camera_rotation(pts, alpha):
    """Among the 24 axis-aligned rotations, the one whose silhouette best matches the input outline (the model's
    camera convention is found per mesh, not assumed). Returns (rotation, IoU). In orthographic views a mirrored
    outline is matched by the opposite view, so a mirror check is not possible this way."""
    tgt = input_silhouette(alpha)
    best = (-1.0, None)
    for R in proper_rotations():
        s = iou(silhouette(pts, R), tgt)
        if s > best[0]:
            best = (s, R)
    return best[1], best[0]


def normalise(pts):
    """Centre on the centroid, scale to unit RMS radius."""
    c = pts - pts.mean(0)
    return c / np.sqrt((c ** 2).sum(1).mean())


def pca_frame(pts):
    """Rows = principal axes, longest first; right-handed."""
    c = pts - pts.mean(0)
    w, V = np.linalg.eigh(np.cov(c.T))
    V = V[:, ::-1].T
    if np.linalg.det(V) < 0:
        V[2] *= -1
    return V


def chamfer(a, b) -> float:
    """Symmetric mean nearest-neighbour distance."""
    da = cKDTree(b).query(a)[0]
    db = cKDTree(a).query(b)[0]
    return float((da.mean() + db.mean()) / 2)


def icp(src, dst, iters: int = 40):
    """Rigid point-to-point ICP (rotation + translation, no scale)."""
    tree = cKDTree(dst)
    R, t = np.eye(3), np.zeros(3)
    cur = src.copy()
    for _ in range(iters):
        _, j = tree.query(cur)
        q = dst[j]
        mc, mq = cur.mean(0), q.mean(0)
        U, _, Vt = np.linalg.svd((cur - mc).T @ (q - mq))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        dR = Vt.T @ D @ U.T
        cur = (cur - mc) @ dR.T + mq
        R, t = dR @ R, dR @ (t - mc) + mq
    return cur, R, t


def aligned_distance(a, b, n: int = 6000, seed: int = 0) -> float:
    """Chamfer distance after scale normalisation and the best of 24 PCA-start rigid ICP alignments, in units of
    the RMS radius. A mirrored match is not allowed."""
    rng = np.random.default_rng(seed)
    a = normalise(a[rng.choice(len(a), min(n, len(a)), replace=False)])
    b = normalise(b[rng.choice(len(b), min(n, len(b)), replace=False)])
    Va, Vb = pca_frame(a), pca_frame(b)
    best = np.inf
    for S in proper_rotations():           # all 24 axis matchings between the two PCA frames
        start = a @ (Vb.T @ S @ Va).T
        cur, _, _ = icp(start, b, iters=30)
        best = min(best, chamfer(cur, b))
    return best


def proportions(pts, slabs: int = 40) -> dict:
    """In the PCA frame: length L (axis 1, 0.5-99.5 % range), lateral span (axis 2), depth (axis 3). Per slab along
    the length: depth and span. Max body depth D = largest slab depth; flipper estimate = (largest slab span - D) / 2
    (assumes a round body cross-section). All in units of L."""
    V = pca_frame(pts)
    p = (pts - pts.mean(0)) @ V.T
    lo, hi = np.percentile(p[:, 0], [0.5, 99.5])
    L = hi - lo
    edges = np.linspace(lo, hi, slabs + 1)
    depth, span = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        q = p[(p[:, 0] >= a) & (p[:, 0] < b)]
        if len(q) < 50:
            depth.append(0.0)
            span.append(0.0)
            continue
        d2 = np.percentile(q[:, 1], [1, 99])
        d3 = np.percentile(q[:, 2], [1, 99])
        span.append(d2[1] - d2[0])
        depth.append(d3[1] - d3[0])
    depth, span = np.array(depth), np.array(span)
    D = depth.max()
    S = span.max()
    return dict(length_to_depth=float(L / D), span_to_length=float(S / L), depth_to_length=float(D / L),
                flipper_est_to_length=float(max(0.0, (S - D) / 2) / L),
                axis_ratio_2_1=float(np.ptp(p[:, 1]) / np.ptp(p[:, 0])),
                axis_ratio_3_1=float(np.ptp(p[:, 2]) / np.ptp(p[:, 0])))
