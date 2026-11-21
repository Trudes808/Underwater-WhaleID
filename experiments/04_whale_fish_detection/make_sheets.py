"""Review sheets for people (``review_sheets/underwater/detect_*.jpg``): boxes drawn on frames with model, prompt and
score printed; crops for small things (fish candidates).

    .venv/bin/python experiments/04_whale_fish_detection/make_sheets.py dev  --run owl_owl_all_s5 --thr 0.25
    .venv/bin/python experiments/04_whale_fish_detection/make_sheets.py fish --video <path> --run owl_owl_all_s15 --thr 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from common import (DEV_LABELS, DEV_VIDEO, FISH_Q, SHEETS, detect, ep, fps_of, frame_scores, load_masks,
                    mask_frames, tag_boxes)

COL = {"whale": (255, 60, 60), "humpback whale": (255, 140, 0), "fish": (60, 255, 60), "school of fish": (0, 200, 255),
       "small fish": (180, 255, 120)}


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def draw(img: np.ndarray, boxes: pd.DataFrame, title: str, width: int = 640, mask=None) -> Image.Image:
    im = Image.fromarray(img)
    if mask is not None:
        arr = np.asarray(im).astype(np.float32)
        arr[mask] = 0.55 * arr[mask] + 0.45 * np.array([255, 0, 255])
        im = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(im)
    f = font(28)
    for _, r in boxes.iterrows():
        c = COL.get(r["query"], (255, 255, 0))
        d.rectangle([r.x0, r.y0, r.x1, r.y1], outline=c, width=5)
        d.text((r.x0 + 6, max(r.y0 + 4, 0)), f"{r['query']} {r.score:.2f}", fill=c, font=f,
               stroke_width=2, stroke_fill=(0, 0, 0))
    s = width / im.width
    im = im.resize((width, int(im.height * s)), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, width, 22], fill=(0, 0, 0))
    d.text((4, 3), title, fill=(255, 255, 255), font=font(15))
    return im


def crop(img: np.ndarray, r, title: str, size: int = 240) -> Image.Image:
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    half = max(80, 2 * max(r.x1 - r.x0, r.y1 - r.y0))
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
    im = Image.fromarray(img[y0:y1, x0:x1].copy())
    d = ImageDraw.Draw(im)
    d.rectangle([r.x0 - x0, r.y0 - y0, r.x1 - x0, r.y1 - y0], outline=(60, 255, 60), width=2)
    im = im.resize((size, int(size * im.height / max(im.width, 1))), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size + 34), (0, 0, 0))
    canvas.paste(im, (0, 34))
    d = ImageDraw.Draw(canvas)
    for i, line in enumerate(title.split("\n")[:2]):
        d.text((3, 2 + 16 * i), line, fill=(255, 255, 255), font=font(13))
    return canvas


def grid(tiles, cols, out, header):
    if not tiles:
        print("nothing for", out)
        return
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h + 30), (20, 20, 20))
    ImageDraw.Draw(sheet).text((6, 6), header, fill=(255, 255, 255), font=font(18))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, 30 + (i // cols) * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print("wrote", out, len(tiles), "tiles")


def dev_sheets(run, thr, query="whale"):
    fps = fps_of(DEV_VIDEO)
    labels = ep.load_frame_labels(DEV_LABELS)
    b, frames = detect.load_cached(DEV_VIDEO, run)
    b = tag_boxes(b)
    wb = b[(b["query"] == query) & ~b.clock & ~b.whole]
    truth = ep.window_truth(labels, fps)
    win = ep.window_of(frames, fps)
    wf = set(truth.window[truth.whale_free])
    # 1) false alarms: whale boxes >= thr on processed frames of whale-free windows
    fa_frames = sorted(set(wb[(wb.score >= thr)].frame) & set(frames[np.isin(win, list(wf))]))
    # 2) misses: frames of whale windows that did not fire (both-frames rule) - read from op table
    opw = pd.read_csv(detect.detect_cache_dir(DEV_VIDEO, run) / "op_windows_dev.csv")
    miss_w = set(opw.window[opw.whale & (opw.score.fillna(0) < thr)])
    miss_frames = [f for f in frames if ep.window_of(f, fps) in miss_w and f % 15 == 0]
    # 3) bottom-zone boxes
    bz_frames = sorted(set(wb[(wb.score >= thr) & wb.bottom].frame))
    # 4) successes: mask frames with whales, SAM masks if cached
    succ = [f for f in mask_frames() if f in (570, 900, 1065, 1290, 1545, 2175, 2250, 2340, 2565, 1890, 1485, 720)]
    want = sorted(set(fa_frames) | set(miss_frames) | set(bz_frames) | set(succ))
    imgs = dict(detect.iter_frames_cv(DEV_VIDEO, frames=want))
    lab_by_frame = labels.set_index("frame")

    def title(f, extra=""):
        w = ep.window_of(f, fps)
        t = f"f{f} t={f / fps:.1f}s win {w} | OWLv2-base '{query}' boxes>={thr if not extra else 0.1}"
        if f in lab_by_frame.index:
            r = lab_by_frame.loc[f]
            t += f" | label: {r.state}, {r.view}, whales {r.other_whales}+{r.faint_whales} faint"
        return t + extra
    tiles = [draw(imgs[f], wb[(wb.frame == f) & (wb.score >= thr)], title(f)) for f in fa_frames if f in imgs]
    grid(tiles, 4, SHEETS / "detect_dev_false_alarms.jpg",
         f"Dev clip: OWLv2 'whale' boxes >= {thr} in WHALE-FREE windows (single frames; the 2-frame window rule removes all of them)")
    tiles = [draw(imgs[f], wb[(wb.frame == f) & (wb.score >= 0.1)], title(f, " (boxes >= 0.1)")) for f in miss_frames if f in imgs]
    grid(tiles, 4, SHEETS / "detect_dev_misses.jpg", f"Dev clip: missed whale windows at the operating point (OWLv2 'whale', both frames >= {thr})")
    tiles = [draw(imgs[f], wb[(wb.frame == f) & (wb.score >= thr) & wb.bottom], title(f)) for f in bz_frames if f in imgs]
    grid(tiles, 4, SHEETS / "detect_dev_bottom_zone.jpg",
         f"Dev clip: 'whale' boxes >= {thr} in the bottom zone (touch the bottom edge, top below 75 % height): back or low whale?")
    store_p = detect.detect_cache_dir(DEV_VIDEO, run) / "sam_maskframes.npz"
    store = np.load(store_p) if store_p.exists() else None
    tiles = []
    for f in succ:
        if f not in imgs:
            continue
        sel = wb[(wb.frame == f) & (wb.score >= thr)]
        m = None
        if store is not None:
            ms = [np.unpackbits(store[f"{f}_{i}"])[:1920 * 1080].reshape(1080, 1920).astype(bool)
                  for i in sel.index if f"{f}_{i}" in store]
            m = np.any(ms, 0) if ms else None
        tiles.append(draw(imgs[f], sel, title(f) + (" + SAM 2.1 masks" if m is not None else ""), mask=m))
    grid(tiles, 3, SHEETS / "detect_dev_successes.jpg", f"Dev clip mask frames: OWLv2 'whale' boxes >= {thr}" +
         (" with SAM 2.1 masks from the boxes (magenta)" if store is not None else ""))


def fish_sheet(video, run, thr, out_name, max_tiles=96, queries=None, stride=None):
    """Every fish box >= thr (all fish prompts, merged by IoU per frame), as crops with context, highest first."""
    fps = fps_of(video)
    b, frames = detect.load_cached(video, run)
    b = tag_boxes(b)
    fb = b[b["query"].isin(queries or FISH_Q) & ~b.clock & ~b.whole & (b.score >= thr)].sort_values("score", ascending=False)
    if stride:
        fb = fb[fb.frame % stride == 0]
    # one box per place per frame
    keep = []
    for f, g in fb.groupby("frame", sort=False):
        taken = []
        for i, r in g.iterrows():
            a = np.array([r.x0, r.y0, r.x1, r.y1])
            if any(_iou(a, t) > 0.3 for t in taken):
                continue
            taken.append(a)
            keep.append(i)
    fb = fb.loc[keep].sort_values("score", ascending=False)
    n_all = len(fb)
    fb["window"] = ep.window_of(fb.frame, fps)
    n_win = fb.window.nunique()
    fb = fb.drop_duplicates("window")          # best box per 1 s window, so the sheet spreads over time
    fb = fb.head(max_tiles)
    imgs = dict(detect.iter_frames_cv(video, frames=sorted(set(fb.frame))))
    tiles = [crop(imgs[r.frame], r, f"f{r.frame} t={r.frame / fps:.1f}s\n{r['query']} {r.score:.2f}") for _, r in fb.iterrows()
             if r.frame in imgs]
    grid(tiles, 8, SHEETS / out_name,
         f"{Path(video).stem}: OWLv2 fish-prompt boxes >= {thr}: {n_all} places in {n_win} windows; best per window, "
         f"top {len(tiles)} windows shown. Fish or not? (green box)")
    return fb


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(u, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "fish"])
    ap.add_argument("--run", default="owl_owl_all_s5")
    ap.add_argument("--video", default=DEV_VIDEO)
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--out")
    ap.add_argument("--stride", type=int)
    a = ap.parse_args()
    if a.cmd == "dev":
        dev_sheets(a.run, a.thr)
    else:
        fish_sheet(a.video, a.run, a.thr, a.out or f"detect_fish_{Path(a.video).stem.replace(' ', '_')}.jpg",
                   stride=a.stride)


if __name__ == "__main__":
    main()
