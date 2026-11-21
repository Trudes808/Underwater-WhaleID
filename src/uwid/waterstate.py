"""Water state of a camera-tag video from the picture alone: underwater / air per 1 s window, exits and entries.

Work package WP4 (experiments/03_water_state). This module holds the frame features and the window step shared by
the experiment and by the command line later (WP7).
"""
from __future__ import annotations

import numpy as np

DECODE_SIZE = (448, 256)   # width, height the frames are decoded at
BOTTOM_CROP = 24           # rows left out at the bottom (burned-in clock box; ~2 % of the frame)

COLOR_NAMES = [
    "s_mean", "v_mean", "v_std", "v_p95", "r_mean", "g_mean", "b_mean", "b_minus_r", "frac_white", "frac_blue",
    "s_top", "v_top", "s_bottom", "v_bottom", "hue_cos", "hue_sin", "log_lap_var", "frac_dark",
]


def color_features(img: np.ndarray) -> np.ndarray:
    """A handful of colour and brightness numbers of one RGB uint8 frame (any size), all scaled to about 0-1.

    Saturation and value (HSV), mean R, G, B, blue minus red, share of white pixels (S < 40, V > 200 of 255),
    share of saturated blue-green pixels, S and V of the top and bottom thirds, saturation-weighted mean hue
    (cos, sin), log variance of the Laplacian of the grey image (sharpness / texture) and share of dark pixels.
    """
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32) * (np.pi / 90.0)   # OpenCV hue 0-180 -> 0-2pi
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    rgb = img.astype(np.float32) / 255.0
    H = img.shape[0]
    top, bot = slice(0, H // 3), slice(2 * H // 3, H)
    sw = s.sum() + 1e-6
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(grey, cv2.CV_32F)
    hue_u8 = hsv[..., 0]
    return np.array([
        s.mean(), v.mean(), v.std(), np.percentile(v, 95),
        rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean(), (rgb[..., 2] - rgb[..., 0]).mean(),
        ((s < 40 / 255) & (v > 200 / 255)).mean(),
        ((s > 80 / 255) & (hue_u8 >= 80) & (hue_u8 <= 130)).mean(),
        s[top].mean(), v[top].mean(), s[bot].mean(), v[bot].mean(),
        float((np.cos(h) * s).sum() / sw), float((np.sin(h) * s).sum() / sw),
        np.log(lap.var() + 1e-3) / 10.0,
        (v < 40 / 255).mean(),
    ], dtype=np.float32)


# ------------------------------------------------------------------ window step (frames -> windows and events)

def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving mean over k frames (edges padded with the edge value)."""
    x = np.asarray(x, float)
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    c = np.cumsum(np.r_[0.0, xp])
    return (c[k:] - c[:-k]) / k


def viterbi2(llr: np.ndarray, penalty: float) -> np.ndarray:
    """Two-state chain: the 0/1 sequence maximising sum(llr[i] * s[i]) - penalty * (number of switches)."""
    n = len(llr)
    if n == 0:
        return np.zeros(0, bool)
    back = np.zeros((n, 2), np.int8)
    v0, v1 = 0.0, float(llr[0])
    for i in range(1, n):
        # to state 0
        a, b = v0, v1 - penalty
        back[i, 0] = 0 if a >= b else 1
        n0 = max(a, b)
        a, b = v0 - penalty, v1
        back[i, 1] = 0 if a >= b else 1
        n1 = max(a, b) + float(llr[i])
        v0, v1 = n0, n1
    s = np.zeros(n, np.int8)
    s[-1] = 1 if v1 > v0 else 0
    for i in range(n - 1, 0, -1):
        s[i - 1] = back[i, s[i]]
    return s.astype(bool)


def hysteresis(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.zeros(len(x), bool)
    st = bool(x[0] > (lo + hi) / 2) if len(x) else False
    for i, v in enumerate(x):
        if st and v < lo:
            st = False
        elif not st and v > hi:
            st = True
        out[i] = st
    return out


def binarize(score: np.ndarray, step: dict) -> np.ndarray:
    """Frame scores (logit of 'air') -> air (True) / underwater per frame, after the window step ``step``:
    kind 'raw' (threshold per frame), 'pool' (mean over k frames, then threshold), 'hyst' (pool, then hysteresis
    +-h around the threshold), 'chain' (pool, then a two-state chain with a switch penalty)."""
    kind = step["kind"]
    x = np.asarray(score, float) - step.get("bias", 0.0)
    if kind == "raw":
        return x > 0
    x = smooth(x, int(step.get("k", 5)))
    if kind == "pool":
        return x > 0
    if kind == "hyst":
        return hysteresis(x, -step["h"], step["h"])
    if kind == "chain":
        return viterbi2(x, step["penalty"])
    raise ValueError(kind)


def events_from_binary(t: np.ndarray, air: np.ndarray):
    """Exits (underwater -> air) and entries at the midpoints between the two frames where the state flips."""
    import pandas as pd
    t = np.asarray(t, float)
    d = np.diff(air.astype(np.int8))
    i = np.nonzero(d)[0]
    return pd.DataFrame({"t": (t[i] + t[i + 1]) / 2, "kind": np.where(d[i] > 0, "exit", "entry")})


def windows_from_binary(t: np.ndarray, air: np.ndarray, p_air: np.ndarray, duration: float, win: float = 1.0):
    """One row per 1 s window: state ('crossing' when an exit / entry falls in it, else the frames' state),
    score (mean probability of 'air' of the window's frames) and the events inside."""
    import pandas as pd
    t = np.asarray(t, float)
    ev = events_from_binary(t, air)
    n = int(np.floor(duration / win + 1e-9))
    k_fr = np.floor(t / win).astype(int)
    ok = (k_fr >= 0) & (k_fr < n)
    cnt = np.bincount(k_fr[ok], minlength=n)[:n]
    n_air = np.bincount(k_fr[ok], weights=air[ok].astype(float), minlength=n)[:n]
    p_sum = np.bincount(k_fr[ok], weights=np.asarray(p_air, float)[ok], minlength=n)[:n]
    k_ev = np.floor(ev.t.values / win).astype(int)
    n_ev = np.bincount(k_ev[(k_ev >= 0) & (k_ev < n)], minlength=n)[:n]
    state = np.where(cnt == 0, "unknown", np.where(n_air / np.maximum(cnt, 1) > 0.5, "air", "underwater")).astype(object)
    state[n_ev > 0] = "crossing"
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(cnt > 0, p_sum / np.maximum(cnt, 1), np.nan)
    evtxt = np.full(n, "", dtype=object)
    for r in ev.itertuples():
        k = int(np.floor(r.t / win))
        if 0 <= k < n:
            evtxt[k] = (evtxt[k] + ";" if evtxt[k] else "") + f"{r.kind}@{r.t:.2f}"
    k = np.arange(n)
    return pd.DataFrame({"k": k, "t0": k * win, "t1": (k + 1) * win, "state": state, "score": score,
                         "events": evtxt}), ev


# ------------------------------------------------------------------ predictor (picture only)

PARAMS_PATH = __import__("pathlib").Path(__file__).with_name("waterstate_params.json")
DINO_REPO = "/home/sat3737/dinov3"
DINO_WEIGHTS = "/home/sat3737/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


def load_params(path=None) -> dict:
    import json
    return json.loads(__import__("pathlib").Path(path or PARAMS_PATH).read_text())


def _frame_features(path, feature: str, fps: float, device: str | None, batch: int = 32, log=None):
    """Decode at ``fps`` (448x256, bottom strip with the burned-in clock blanked) and compute ``feature`` per frame:
    'color' (18 numbers, CPU) or 'dino' (DINOv3 ViT-B/16 CLS + mean patch token, GPU)."""
    from . import video as uv
    T, F, buf = [], [], []
    model = None
    if feature == "dino":
        import sys
        import torch
        if DINO_REPO not in sys.path:
            sys.path.insert(0, DINO_REPO)
        from dinov3.hub import backbones
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dt = torch.float16 if device.startswith("cuda") else torch.float32
        model = backbones.dinov3_vitb16(pretrained=True, weights=DINO_WEIGHTS).to(device).to(dt).eval()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        def flush():
            if not buf:
                return
            x = torch.from_numpy(np.stack(buf)).to(device).permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                o = model.forward_features(((x - mean) / std).to(dt))
            F.append(torch.cat([o["x_norm_clstoken"].float(), o["x_norm_patchtokens"].float().mean(1)], 1).cpu().numpy())
            buf.clear()
    for fr in uv.iter_frames(path, fps=fps, size=DECODE_SIZE):
        T.append(fr.t)
        if feature == "color":
            F.append(color_features(fr.image[: DECODE_SIZE[1] - BOTTOM_CROP])[None])
        else:
            im = fr.image.copy()
            im[DECODE_SIZE[1] - BOTTOM_CROP:] = 0
            buf.append(im)
            if len(buf) >= batch:
                flush()
    if feature == "dino":
        flush()
    return np.asarray(T, float), np.concatenate(F).astype(np.float32)


def frame_scores(X: np.ndarray, p: dict) -> np.ndarray:
    """Logit of 'air' per frame from the stored scaler + logistic regression, clipped to +-clip."""
    z = (X - np.asarray(p["mean"], np.float32)) / np.asarray(p["scale"], np.float32)
    s = z @ np.asarray(p["coef"], np.float32) + float(p["intercept"])
    return np.clip(s, -p.get("clip", 10.0), p.get("clip", 10.0))


def predict_video(path, params: dict | str | None = None, device: str | None = None, method: str | None = None,
                  step: dict | str | None = None):
    """Water state of one video from the picture alone.

    Returns ``(windows, events, frames)``: ``windows`` has one row per 1 s window of presentation time (k, t0, t1,
    state in underwater / crossing / air, score = mean probability of air of the window's frames, events inside);
    ``events`` lists exits (underwater -> air) and entries (time in s, kind); ``frames`` has the sampled frame
    times, per-frame score (logit of air) and the per-frame state after the window step.
    ``method`` picks one of the fitted models stored in the parameter file (default: its ``default``); ``step``
    overrides its window step (a dict, or the name of a stored one such as ``"step_v1_chain"``).
    The DINOv3 model needs a GPU for speed (about 270 frames/s on the shared card; run it under the GPU lock).
    """
    import pandas as pd
    from . import video as uv
    P = params if isinstance(params, dict) else load_params(params)
    m = P["models"][method or P["default"]]
    fps = float(P.get("fps", 5.0))
    t, X = _frame_features(path, m["feature"], fps, device)
    if m.get("columns"):
        X = X[:, m["columns"]]
    s = frame_scores(X, m)
    st = m[step] if isinstance(step, str) else (step or m["step"])
    air = binarize(s, st)
    p_air = 1 / (1 + np.exp(-(s - st.get("bias", 0.0))))
    duration = float(uv.frame_table(path).duration)
    windows, events = windows_from_binary(t, air, p_air, duration)
    frames = pd.DataFrame({"t": t, "score": s, "air": air})
    return windows, events, frames
