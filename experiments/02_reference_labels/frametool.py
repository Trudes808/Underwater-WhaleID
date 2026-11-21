"""Small frame tool for the by-eye reference labels (WP3). Decodes in order from the start (seeking is not trusted on
these files), saves chosen frames as full-size JPEG to the cache, and makes viewing sheets for labelling.

    PY=.venv/bin/python; T=experiments/02_reference_labels/frametool.py
    $PY $T dump  VIDEO OUTDIR --step 15                 # every 15th frame (2 per second) + index.json (frames decoded, fps)
    $PY $T dump  VIDEO OUTDIR --idx 100 101 102         # chosen frames
    $PY $T dump  VIDEO OUTDIR --times 12.5 900.0        # frames nearest to these times (s), decoded in order
    $PY $T sheet OUTDIR --out a.jpg 0 15 30 45 60 75     # <= 6 frames, 2 columns, 960 px wide each
    $PY $T crop  OUTDIR 450 x0 y0 x1 y1 --out c.jpg      # full-size crop (upscaled to >= 1400 px) with a coordinate grid
    $PY $T strip VIDEO a b --out s.jpg [--tw 480 --cols 4]   # every frame a..b, for stepping through a transition
    $PY $T grid  DIR --out g [--tw 128 --cols 15 --per 180]   # all JPEGs of a folder as small thumbnails, g_00.jpg, g_01.jpg ...

Frame index = position in decode order from 0. Time = index / fps, fps from the container (29.974 on the dev clip).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def decode(video: str, want=None, stop=None):
    """Yield (index, fps, BGR frame) in decode order; want = set of indices or None for all; stop = last index."""
    v = cv2.VideoCapture(str(video))
    fps = v.get(cv2.CAP_PROP_FPS)
    i = 0
    while True:
        if stop is not None and i > stop:
            break
        if want is not None and i not in want:
            ok = v.grab()
        else:
            ok, f = v.read()
        if not ok:
            break
        if want is None or i in want:
            yield i, fps, f
        i += 1
    v.release()
    yield i, fps, None   # sentinel: i = number of frames decoded (or stop + 1)


def gridded(im: Image.Image, step=100, scale=1.0, origin=(0, 0)) -> Image.Image:
    d = ImageDraw.Draw(im); f = font(14)
    w, h = im.size; ox, oy = origin
    for gx in range(((ox + step - 1) // step) * step, ox + int(w * scale) + 1, step):
        x = (gx - ox) / scale; d.line([(x, 0), (x, h)], fill=(255, 255, 0), width=1); d.text((x + 2, 2), str(gx), fill=(255, 255, 0), font=f)
    for gy in range(((oy + step - 1) // step) * step, oy + int(h * scale) + 1, step):
        y = (gy - oy) / scale; d.line([(0, y), (w, y)], fill=(255, 255, 0), width=1); d.text((2, y + 2), str(gy), fill=(255, 255, 0), font=f)
    return im


def tile(ims, labels, cols=2, tw=960, fs=18):
    w0, h0 = ims[0].size; th = int(tw * h0 / w0); lh = fs + 8
    rows = (len(ims) + cols - 1) // cols
    out = Image.new("RGB", (cols * tw, rows * (th + lh)), (0, 0, 0)); d = ImageDraw.Draw(out)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        x, y = (i % cols) * tw, (i // cols) * (th + lh)
        out.paste(im.resize((tw, th), Image.BILINEAR), (x, y + lh)); d.text((x + 4, y + 3), lab, fill=(255, 255, 255), font=font(fs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["dump", "sheet", "crop", "strip", "grid"]); ap.add_argument("args", nargs="*")
    ap.add_argument("--per", type=int, default=180)
    ap.add_argument("--out", type=Path); ap.add_argument("--step", type=int); ap.add_argument("--idx", type=int, nargs="*")
    ap.add_argument("--times", type=float, nargs="*"); ap.add_argument("--cols", type=int, default=2); ap.add_argument("--tw", type=int, default=960)
    ap.add_argument("--grid", type=int, default=0, help="sheet: draw a coordinate grid every N source pixels")
    a = ap.parse_intermixed_args()
    if a.cmd == "dump":
        video, outdir = a.args[0], Path(a.args[1]); outdir.mkdir(parents=True, exist_ok=True)
        fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
        if a.step:
            want, stop = None, None
        elif a.times is not None:
            want = {int(round(t * fps)) for t in a.times}; stop = max(want)
        else:
            want = set(a.idx); stop = max(want)
        n = 0
        for i, fps, f in decode(video, want, stop):
            if f is None:
                n = i; break
            if a.step and i % a.step:
                continue
            cv2.imwrite(str(outdir / f"{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 93])
        meta = outdir / "index.json"; old = json.loads(meta.read_text()) if meta.exists() else {}
        if a.step:
            old.update(frames_decoded=n, fps=fps, step=a.step)
        else:
            old.setdefault("decoded_to", 0); old["decoded_to"] = max(old["decoded_to"], n)
            old["fps"] = fps
        meta.write_text(json.dumps(old, indent=1)); print("decoded", n, "fps", fps)
    elif a.cmd == "sheet":
        d = Path(a.args[0]); idx = [int(x) for x in a.args[1:]]
        ims, labs = [], []
        idxf = d / "index.json"; fps = json.loads(idxf.read_text())["fps"] if idxf.exists() else 29.97
        for i in idx:
            im = Image.open(d / f"{i:06d}.jpg").convert("RGB")
            if a.grid:
                sc = im.size[0] / a.tw; im = gridded(im.resize((a.tw, int(im.size[1] / sc))), step=a.grid, scale=sc)
            ims.append(im); labs.append(f"f{i}  t={i / fps:.2f}s")
        a.out.parent.mkdir(parents=True, exist_ok=True); tile(ims, labs, a.cols, a.tw).save(a.out, quality=90)
    elif a.cmd == "crop":
        d = Path(a.args[0]); i, x0, y0, x1, y1 = map(int, a.args[1:6]); step = int(a.args[6]) if len(a.args) > 6 else 50
        im = Image.open(d / f"{i:06d}.jpg").convert("RGB").crop((x0, y0, x1, y1))
        sc = max(1.0, 1400 / max(im.size)); im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)), Image.LANCZOS)
        a.out.parent.mkdir(parents=True, exist_ok=True); gridded(im, step=step, scale=1 / sc, origin=(x0, y0)).save(a.out, quality=92)
    elif a.cmd == "grid":
        d = Path(a.args[0]); fs = sorted(d.glob("*.jpg")); idxf = d / "index.json"
        fps = json.loads(idxf.read_text())["fps"] if idxf.exists() else float(a.args[1]) if len(a.args) > 1 else 29.97
        th = a.tw * 9 // 16
        for s in range(0, len(fs), a.per):
            part = fs[s:s + a.per]; rows = (len(part) + a.cols - 1) // a.cols
            W = Image.new("RGB", (a.cols * a.tw, rows * (th + 15))); dr = ImageDraw.Draw(W)
            for k, f in enumerate(part):
                i = int(f.stem[:6]); x, y = (k % a.cols) * a.tw, (k // a.cols) * (th + 15)
                W.paste(Image.open(f).convert("RGB").resize((a.tw, th)), (x, y + 15)); dr.text((x + 2, y + 1), f"f{i} {i / fps:.1f}s", fill=(255, 255, 0))
            W.save(f"{a.out}_{s // a.per:02d}.jpg", quality=85)
    else:
        video, s, e = a.args[0], int(a.args[1]), int(a.args[2])
        ims, labs = [], []
        for i, fps, f in decode(video, set(range(s, e + 1)), e):
            if f is None:
                break
            ims.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))); labs.append(f"f{i} t={i / fps:.3f}")
        a.out.parent.mkdir(parents=True, exist_ok=True); tile(ims, labs, a.cols, a.tw, fs=14).save(a.out, quality=88)


if __name__ == "__main__":
    main()
