"""Sheets of the blocked spans (WP3b part B): per span one frame before, two inside, one after, so the user can see what
was called blocked. Frames come from the coarse frame folders the spans were found with (<cache>/<stem>/coarse_*s/,
480 px wide, time = video time from the first frame).

    PY=.venv/bin/python
    $PY experiments/02_reference_labels/blocked_sheets.py --out /home/sat3737/NOAA_whales/review_sheets/underwater/reference_d2
Writes reference_d2_<piece>_blocked_NN.jpg for every recording with a labels/<stem>/blocked_spans.csv.
"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache/reference")


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def frames_of(stem):
    out = []
    for d in sorted((CACHE / stem).glob("coarse_*s")):
        for k, t in json.loads((d / "index.json").read_text())["times"].items():
            out.append((float(t), d / f"{int(k):06d}.jpg"))
    return sorted(out)


def nearest(fr, t, lo=None, hi=None):
    c = [x for x in fr if (lo is None or x[0] >= lo) and (hi is None or x[0] <= hi)]
    return min(c, key=lambda x: abs(x[0] - t)) if c else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--per", type=int, default=6)
    a = ap.parse_args(); tw, th = 400, 225
    for f in sorted(HERE.glob("labels/CC-09-85*/blocked_spans.csv")):
        stem = f.parent.name; spans = list(csv.DictReader(open(f)))
        if not spans:
            continue
        fr = frames_of(stem); rows = []
        for s in spans:
            a0, a1 = float(s["start_s"]), float(s["end_s"]); L = a1 - a0
            picks = [("before", nearest(fr, a0 - 3, hi=a0 - 1)), ("inside", nearest(fr, a0 + L / 3, a0, a1)),
                     ("inside", nearest(fr, a0 + 2 * L / 3, a0, a1)), ("after", nearest(fr, a1 + 2, lo=a1 + 1))]
            row = Image.new("RGB", (4 * tw, th + 44)); d = ImageDraw.Draw(row)
            d.text((4, 2), f"{s['start_s']}-{s['end_s']} s  {s['kind']}   whale before {s['whale_before']} / after {s['whale_after']}"
                   + (f" / same {s['same_animal']}" if s["same_animal"] else ""), fill=(255, 255, 0), font=font(15))
            d.text((4, 21), s["note"][:150], fill=(230, 230, 230), font=font(12))
            for i, (lab, p) in enumerate(picks):
                if p is None:
                    continue
                row.paste(Image.open(p[1]).convert("RGB").resize((tw, th)), (i * tw, 44))
                d.text((i * tw + 4, 46), f"{lab} {p[0]:.0f}s", fill=(255, 255, 0), font=font(14), stroke_width=2, stroke_fill=(0, 0, 0))
            rows.append(row)
        for k in range(0, len(rows), a.per):
            part = rows[k:k + a.per]; S = Image.new("RGB", (4 * tw, 30 + sum(r.size[1] for r in part)))
            ImageDraw.Draw(S).text((4, 6), f"{stem}: blocked views (AI by-eye, WP3b). Columns: before, inside, inside, after",
                                    fill=(255, 255, 255), font=font(15)); y = 30
            for r in part:
                S.paste(r, (0, y)); y += r.size[1]
            S.save(f"{a.out}_{stem[-5:]}_blocked_{k // a.per + 1:02d}.jpg", quality=85)
        print(stem, len(rows), "spans")


if __name__ == "__main__":
    main()
