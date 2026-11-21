"""Reference masks from my own prompts (WP3). Prompts are boxes / points / polygons written by eye; SAM 2.1
(facebook/sam2.1-hiera-large, public weights) turns box / point prompts into masks on a crop around the prompt.
Adapted from Dinov3-WhaleID/experiments/test_day/pixtruth_draw.py (copied, not imported).

    PY=.venv/bin/python; T=experiments/02_reference_labels/masktool.py; F=<frames dir>; L=<labels dir>
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock $PY $T run   --frames $F --prompts prompts.json --out $L --cache <samcache> [KEY ...]
    $PY $T check --frames $F --out $L --sheet a.jpg KEY[:x0,y0,x1,y1] ...   # overlay with outlines, coordinate grid
    $PY $T review --frames $F --out $L --sheet a.jpg KEY ...              # overlay for humans: one colour per instance, legend

prompts.json: {"meta": {...}, "<key>": [op, ...]}; key = 6-digit frame index (the frame JPEG name). Ops are painted in
order, later ops overwrite earlier ones. op:
  {"c": cls, "box": [x0,y0,x1,y1], "p": [[x,y],...], "n": [[x,y],...], "mm": "best"|0|1|2}       SAM (box and/or points)
  {"c": cls, "poly": [[x,y],...]} | {"c": cls, "rect": [x0,y0,x1,y1]} | {"c": cls, "line": [[x,y],...], "w": px}   drawn directly
  optional: "id": instance id (whale / fish; unique per frame), "grade": clear|hazy|faint (whales), "note": text,
            "over": [classes] (paint only over these classes), "sub": true (erase: paint class 0 / instance 0 where the
            op's mask is, restricted to "over" if given), "clip": true (SAM mask cut to the box), "crop": [x0,y0,x1,y1].
Classes (class PNG values): 0 water / background, 1 whale (other whales), 2 back (the tagged whale's own body),
3 fish, 4 bubbles (coarse region), 5 surface (sea surface seen from below, coarse), 6 air (above-water scene, air frames),
7 seafloor, 8 gear (drone, tag parts, anything man-made), 255 unsure. Instance PNG: whale / fish instance id (0 = none). <key>_meta.json lists instances with class and grade.
Coordinates are full-resolution pixels (1920 x 1080).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
CLS = {"water": 0, "whale": 1, "back": 2, "fish": 3, "bubbles": 4, "surface": 5, "air": 6, "seafloor": 7, "gear": 8, "unsure": 255}
NAMES = {v: k for k, v in CLS.items()}
PAL = {1: (255, 140, 0), 2: (150, 75, 200), 3: (255, 0, 0), 4: (255, 255, 255), 5: (0, 255, 255), 6: (120, 160, 255), 7: (190, 140, 60), 8: (255, 0, 255), 255: (128, 128, 128)}
INST_PAL = [(255, 140, 0), (255, 50, 50), (60, 220, 60), (255, 230, 0), (255, 0, 255), (0, 160, 255), (255, 128, 192), (160, 255, 120)]


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def gridded(im, step=100, scale=1.0, origin=(0, 0)):
    d = ImageDraw.Draw(im); f = font(13); w, h = im.size; ox, oy = origin
    for gx in range(((ox + step - 1) // step) * step, ox + int(w * scale) + 1, step):
        x = (gx - ox) / scale; d.line([(x, 0), (x, h)], fill=(255, 255, 0), width=1); d.text((x + 2, 2), str(gx), fill=(255, 255, 0), font=f)
    for gy in range(((oy + step - 1) // step) * step, oy + int(h * scale) + 1, step):
        y = (gy - oy) / scale; d.line([(0, y), (w, y)], fill=(255, 255, 0), width=1); d.text((2, y + 2), str(gy), fill=(255, 255, 0), font=f)
    return im


def tile(ims, labels, cols=2, tw=960):
    th = int(tw * ims[0].size[1] / ims[0].size[0]); lh = 26
    out = Image.new("RGB", (cols * tw, ((len(ims) + cols - 1) // cols) * (th + lh)), (0, 0, 0)); d = ImageDraw.Draw(out)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        x, y = (i % cols) * tw, (i // cols) * (th + lh)
        out.paste(im.resize((tw, th), Image.BILINEAR), (x, y + lh)); d.text((x + 4, y + 4), lab, fill=(255, 255, 255), font=font(16))
    return out


class Sam:
    def __init__(self, cache: Path):
        import torch
        from transformers import Sam2Model, Sam2Processor
        torch.set_num_threads(8)
        self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(self.dev).eval()
        self.p = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large"); self.cache = cache; cache.mkdir(parents=True, exist_ok=True)

    def mask(self, img, key, op):
        k = hashlib.sha1(json.dumps([key, {q: op[q] for q in ("box", "p", "n", "mm", "crop") if q in op}], sort_keys=True).encode()).hexdigest()[:16]
        f = self.cache / f"{key}_{k}.npz"
        if f.exists():
            return np.unpackbits(np.load(f)["m"])[: H * W].reshape(H, W).astype(bool)
        pts = (op.get("p") or []) + (op.get("n") or []); lbl = [1] * len(op.get("p") or []) + [0] * len(op.get("n") or [])
        if "crop" in op:
            x0, y0, x1, y1 = op["crop"]
        else:
            xs = ([op["box"][0], op["box"][2]] if "box" in op else []) + [q[0] for q in pts]
            ys = ([op["box"][1], op["box"][3]] if "box" in op else []) + [q[1] for q in pts]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys); s = max(x1 - x0, y1 - y0)
            mg = max(40, 0.2 * s) if "box" in op else max(120, 0.4 * s)
            half = max(s / 2 + mg, 128); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            x0, x1 = cx - max(half, (x1 - x0) / 2 + mg), cx + max(half, (x1 - x0) / 2 + mg)
            y0, y1 = cy - max(half, (y1 - y0) / 2 + mg), cy + max(half, (y1 - y0) / 2 + mg)
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
        crop = Image.fromarray(img[y0:y1, x0:x1]); kw = {}
        if "box" in op:
            b = op["box"]; kw["input_boxes"] = [[[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0]]]
        if pts:
            kw["input_points"] = [[[[q[0] - x0, q[1] - y0] for q in pts]]]; kw["input_labels"] = [[lbl]]
        inp = self.p(images=crop, return_tensors="pt", **kw).to(self.dev)
        mm = op.get("mm", 0 if "box" in op else "best")
        with self.torch.no_grad():
            out = self.m(**inp, multimask_output=(mm != 0 or "box" not in op))
        ms = self.p.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0][0]
        sc = out.iou_scores.cpu()[0, 0]; kk = int(sc.argmax()) if mm == "best" else min(int(mm), ms.shape[0] - 1)
        full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = ms[kk].numpy().astype(bool)
        np.savez_compressed(f, m=np.packbits(full.ravel()))
        return full


def paint(key, ops, img, sam):
    lab = np.zeros((H, W), np.uint8); inst = np.zeros((H, W), np.uint8); meta = {}
    for op in ops:
        c = CLS[op["c"]]
        if "poly" in op or "rect" in op or "line" in op:
            m = np.zeros((H, W), np.uint8)
            if "line" in op:
                cv2.polylines(m, [np.array(op["line"], np.int32)], False, 1, int(op.get("w", 6)))
            elif "poly" in op:
                cv2.fillPoly(m, [np.array(op["poly"], np.int32)], 1)
            else:
                x0, y0, x1, y1 = op["rect"]; m[y0:y1, x0:x1] = 1
            m = m.astype(bool)
        else:
            m = sam.mask(img, key, op)
            if op.get("clip") and "box" in op:
                b = op["box"]; cm = np.zeros_like(m); cm[b[1]:b[3], b[0]:b[2]] = True; m &= cm
        if "over" in op:
            m &= np.isin(lab, [CLS[x] for x in op["over"]])
        if op.get("sub"):
            lab[m] = 0; inst[m] = 0; continue
        lab[m] = c
        if "id" in op:
            inst[m] = op["id"]; meta[str(op["id"])] = {"class": op["c"], "grade": op.get("grade", ""), "note": op.get("note", "")}
        else:
            inst[m & (inst > 0)] = 0
    present = {k: v for k, v in meta.items() if (inst == int(k)).any()}
    return lab, inst, present


def overlay(img, lab, inst, alpha=0.4, by_instance=False):
    o = img.astype(np.float32).copy()
    layers = [(lab == c, PAL[c]) for c in PAL if c != 1 and c != 3]
    for i in np.unique(inst):
        if i:
            layers.append((inst == i, INST_PAL[(int(i) - 1) % len(INST_PAL)] if by_instance else PAL[int(lab[inst == i][0])]))
    for m, col in layers:
        if m.any():
            o[m] = o[m] * (1 - alpha) + np.array(col) * alpha
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE); cv2.drawContours(o, cs, -1, col, 2)
    return o.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "check", "review"]); ap.add_argument("keys", nargs="*")
    ap.add_argument("--frames", type=Path, required=True); ap.add_argument("--prompts", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path); ap.add_argument("--sheet", type=Path); ap.add_argument("--cols", type=int, default=2); ap.add_argument("--tw", type=int, default=960)
    ap.add_argument("--title", default="")
    a = ap.parse_intermixed_args()
    load = lambda k: np.asarray(Image.open(a.frames / f"{k}.jpg").convert("RGB"))
    if a.cmd == "run":
        pr = json.loads(a.prompts.read_text()); keys = a.keys or sorted(k for k in pr if k != "meta"); sam = Sam(a.cache); a.out.mkdir(parents=True, exist_ok=True)
        for k in keys:
            lab, inst, meta = paint(k, pr[k], load(k), sam)
            Image.fromarray(lab).save(a.out / f"{k}_class.png"); Image.fromarray(inst).save(a.out / f"{k}_inst.png")
            (a.out / f"{k}_meta.json").write_text(json.dumps({"frame": int(k), "instances": meta}, indent=1))
            u, n = np.unique(lab, return_counts=True); print(k, {NAMES[int(x)]: int(y) for x, y in zip(u, n)}, "inst", sorted(meta), flush=True)
        return
    ims, labs = [], []
    for s in a.keys:
        bb = None
        if ":" in s:
            s, b = s.split(":"); bb = tuple(map(int, b.split(",")))
        img = load(s); lab = np.asarray(Image.open(a.out / f"{s}_class.png")); inst = np.asarray(Image.open(a.out / f"{s}_inst.png"))
        meta = json.loads((a.out / f"{s}_meta.json").read_text())["instances"]
        o = Image.fromarray(overlay(img, lab, inst, by_instance=(a.cmd == "review")))
        d = ImageDraw.Draw(o)
        for i, mm in meta.items():
            ys, xs = np.nonzero(inst == int(i))
            if len(xs):
                d.text((int(xs.min()), max(0, int(ys.min()) - 22)), f"{mm['class']} {i} {mm['grade']}", fill=(255, 255, 255), font=font(20), stroke_width=2, stroke_fill=(0, 0, 0))
        if bb:
            x0, y0, x1, y1 = bb; o = o.crop(bb); o = gridded(o.resize((a.tw, int(a.tw * (y1 - y0) / (x1 - x0)))), 50, (x1 - x0) / a.tw, (x0, y0))
        elif a.cmd == "check":
            o = gridded(o.resize((a.tw, int(a.tw * H / W))), 100, W / a.tw)
        ims.append(o); labs.append(f"f{int(s)}  " + " ".join(sorted({NAMES[int(k)] for k in np.unique(lab) if k})))
    a.sheet.parent.mkdir(parents=True, exist_ok=True)
    out = tile(ims, labs, a.cols, a.tw)
    if a.title:
        t = Image.new("RGB", (out.size[0], out.size[1] + 30)); t.paste(out, (0, 30)); ImageDraw.Draw(t).text((6, 5), a.title, fill=(255, 255, 255), font=font(18)); out = t
    out.save(a.sheet, quality=88)


if __name__ == "__main__":
    main()
