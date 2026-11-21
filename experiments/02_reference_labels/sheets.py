"""Check sheets for humans (WP3): frames with their by-eye labels printed on them, and mask overlays.

    PY=.venv/bin/python; D=experiments/02_reference_labels
    $PY $D/sheets.py frames --labels $D/labels/<video>/frames.csv --frames <cache frames dir> --out <review dir>/reference_dev_frames --per 6
    $PY $D/sheets.py masks  --masks <cache masks dir> --frames <cache frames dir> --out <review dir>/reference_dev_masks --per 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from masktool import NAMES, font, overlay, tile  # noqa: E402


def label_text(r):
    s = f"f{r['frame']} t={float(r['time_s']):.1f}s  {r['state']} / {r['view']}  whales {r['other_whales']} (+{r['faint_whales']} faint)  back {r['back']}"
    s2 = f"seafloor {r.get('seafloor', '?')}  fish {r['fish']}" + (f"  other: {r['other']}" if r["other"] else "")
    return s, s2, r["note"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["frames", "masks"])
    ap.add_argument("--labels", type=Path); ap.add_argument("--frames", type=Path, required=True); ap.add_argument("--masks", type=Path)
    ap.add_argument("--out", required=True); ap.add_argument("--per", type=int, default=6); ap.add_argument("--tw", type=int, default=960)
    ap.add_argument("--title", default="AI by-eye labels (WP3)")
    a = ap.parse_args()
    if a.cmd == "frames":
        rows = list(csv.DictReader(open(a.labels)))
        for k in range(0, len(rows), a.per):
            ims = []
            for r in rows[k:k + a.per]:
                im = Image.open(a.frames / f"{int(r['frame']):06d}.jpg").convert("RGB").resize((a.tw, a.tw * 9 // 16))
                d = ImageDraw.Draw(im); f = font(17)
                lines = [x for x in label_text(r) if x]
                wrapped = []
                for ln in lines:
                    while len(ln) > 95:
                        cut = ln.rfind(" ", 0, 95); cut = cut if cut > 0 else 95; wrapped.append(ln[:cut]); ln = ln[cut:].lstrip()
                    wrapped.append(ln)
                d.rectangle([0, 0, a.tw, 8 + 21 * len(wrapped)], fill=(0, 0, 0))
                for i, ln in enumerate(wrapped):
                    d.text((6, 4 + 21 * i), ln, fill=(255, 255, 0) if i == 0 else (255, 255, 255), font=f)
                ims.append(im)
            out = tile(ims, [""] * len(ims), 2, a.tw)
            out.save(f"{a.out}_{k // a.per + 1:02d}.jpg", quality=85)
        print("frame sheets", (len(rows) + a.per - 1) // a.per)
    else:
        keys = sorted(p.name[:6] for p in a.masks.glob("*_class.png"))
        for k in range(0, len(keys), a.per):
            ims, labs = [], []
            for key in keys[k:k + a.per]:
                img = np.asarray(Image.open(a.frames / f"{key}.jpg").convert("RGB"))
                lab = np.asarray(Image.open(a.masks / f"{key}_class.png")); inst = np.asarray(Image.open(a.masks / f"{key}_inst.png"))
                meta = json.loads((a.masks / f"{key}_meta.json").read_text())["instances"]
                o = Image.fromarray(overlay(img, lab, inst, by_instance=True)); d = ImageDraw.Draw(o)
                for i, mm in meta.items():
                    ys, xs = np.nonzero(inst == int(i))
                    if len(xs):
                        d.text((int(xs.min()), max(0, int(ys.min()) - 26)), f"{mm['class']} {i} {mm['grade']}", fill=(255, 255, 255), font=font(24), stroke_width=3, stroke_fill=(0, 0, 0))
                ims.append(o)
                labs.append(f"f{int(key)}  regions: " + ", ".join(NAMES[int(c)] for c in np.unique(lab) if c and int(c) not in (1, 3)))
            out = tile(ims, labs, 2, a.tw)
            hdr = Image.new("RGB", (out.size[0], out.size[1] + 30)); hdr.paste(out, (0, 30))
            ImageDraw.Draw(hdr).text((6, 5), a.title + " / SAM 2.1 masks from my prompts.  back purple, bubbles white, surface cyan, air blue, "
                                     "seafloor brown, unsure grey; each whale / fish its own colour", fill=(255, 255, 255), font=font(17))
            out = hdr
            out.save(f"{a.out}_{k // a.per + 1:02d}.jpg", quality=85)
        print("mask sheets", (len(keys) + a.per - 1) // a.per)


if __name__ == "__main__":
    main()
