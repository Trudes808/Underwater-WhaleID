"""Per-frame features for the water-state experiment (WP4), one pass over a video.

Decodes the video at ``--fps`` frames per second (default 5) at 448x256, and stores per sampled frame:
- ``color``: a handful of colour / brightness numbers (``uwid.waterstate.color_features``), CPU;
- ``clip``: CLIP ViT-L/14 image embedding (unit norm, 768, fp16), whole frame squashed to 224x224;
- ``dino``: DINOv3 ViT-B/16 CLS token + mean patch token (2 x 768, fp16) at 448x256.

Output: ``underwater_cache/waterstate/feats/<video stem>.npz`` (t, idx, color, clip, dino) and a JSON with the
timings of each stage (decode, colour, CLIP, DINOv3), so cost per method can be reported.

Run GPU parts through the lock:
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock nice -n 10 .venv/bin/python experiments/03_water_state/extract.py VIDEO
Use ``--color-only`` for a CPU-only pass. Untrusted inputs (public downloads): run with ``python -I``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import video as uv  # noqa: E402
from uwid.waterstate import color_features, DECODE_SIZE, BOTTOM_CROP  # noqa: E402

CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache/waterstate/feats")
DINO_REPO = "/home/sat3737/dinov3"
DINO_W = "/home/sat3737/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
CLIP_ID = "openai/clip-vit-large-patch14"


def load_models(device):
    import torch
    from transformers import CLIPModel
    clip = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16).to(device).eval()
    # torch.hub.load imports the checkout's eval code (needs torchmetrics, not installed): import the backbone directly
    if DINO_REPO not in sys.path:
        sys.path.insert(0, DINO_REPO)
    from dinov3.hub import backbones
    dino = backbones.dinov3_vitb16(pretrained=True, weights=DINO_W).to(device).half().eval()
    return clip, dino


def clip_embed(clip, x):
    """x: (B,3,H,W) float in [0,1] on device -> unit-norm (B,768)."""
    import torch
    import torch.nn.functional as F
    m = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
    s = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
    y = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
    y = ((y - m) / s).half()
    out = clip.get_image_features(pixel_values=y)
    if not torch.is_tensor(out):
        out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
    return F.normalize(out.float(), dim=-1)


def dino_embed(dino, x):
    import torch
    m = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    y = ((x - m) / s).half()
    o = dino.forward_features(y)
    cls = o["x_norm_clstoken"].float()
    pat = o["x_norm_patchtokens"].float().mean(1)
    return torch.cat([cls, pat], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out", default=str(CACHE))
    ap.add_argument("--color-only", action="store_true")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-s", type=float, default=None, help="only the first N seconds (tests)")
    a = ap.parse_args()

    vid = Path(a.video)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{vid.stem}{'_color' if a.color_only else ''}.npz"
    if dst.exists():
        print("exists", dst)
        return
    # refuse files still being copied
    s0 = vid.stat().st_size
    if time.time() - vid.stat().st_mtime < 60:
        raise SystemExit(f"{vid} changed in the last 60 s: still being copied?")

    gpu = not a.color_only
    if gpu:
        import torch
        torch.set_num_threads(8)
        dev = "cuda"
        clip, dino = load_models(dev)
    T, I, C, E_clip, E_dino = [], [], [], [], []
    tm = {"decode": 0.0, "color": 0.0, "clip": 0.0, "dino": 0.0, "n": 0}
    batch = []

    def flush():
        if not batch:
            return
        import torch
        x = torch.from_numpy(np.stack(batch)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            torch.cuda.synchronize(); t0 = time.perf_counter()
            E_clip.append(clip_embed(clip, x).half().cpu().numpy())
            torch.cuda.synchronize(); t1 = time.perf_counter()
            E_dino.append(dino_embed(dino, x).half().cpu().numpy())
            torch.cuda.synchronize(); t2 = time.perf_counter()
        tm["clip"] += t1 - t0
        tm["dino"] += t2 - t1
        batch.clear()

    t_dec = time.perf_counter()
    for fr in uv.iter_frames(vid, fps=a.fps, size=DECODE_SIZE, end=a.max_s):
        t_got = time.perf_counter()
        tm["decode"] += t_got - t_dec
        img = fr.image[: DECODE_SIZE[1] - BOTTOM_CROP]
        c0 = time.perf_counter()
        C.append(color_features(img))
        tm["color"] += time.perf_counter() - c0
        T.append(fr.t)
        I.append(fr.index)
        if gpu:
            # model input: the full 448x256 frame with the bottom strip (burned-in clock) blacked out
            im = fr.image.copy()
            im[DECODE_SIZE[1] - BOTTOM_CROP:] = 0
            batch.append(im)
            if len(batch) >= a.batch:
                flush()
        tm["n"] += 1
        t_dec = time.perf_counter()
    if gpu:
        flush()
    if vid.stat().st_size != s0:
        raise SystemExit("file size changed while reading")
    res = {"t": np.array(T), "idx": np.array(I), "color": np.array(C, np.float32)}
    if gpu:
        res["clip"] = np.concatenate(E_clip)
        res["dino"] = np.concatenate(E_dino)
    np.savez(dst, **res)
    tm.update(video=str(vid), fps=a.fps, decode_size=DECODE_SIZE, gpu=gpu)
    (out / f"{dst.stem}_timing.json").write_text(json.dumps(tm, indent=1))
    print(json.dumps(tm))


if __name__ == "__main__":
    main()
