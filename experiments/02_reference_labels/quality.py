"""Size and quality of the largest other whale per labelled frame (WP3b part A, user request 2026-11-20).

    PY=.venv/bin/python; Q=experiments/02_reference_labels/quality.py
    $PY $Q view   --out <scratch>/qv [--per 6 --tw 640] [STEM ...]      # labelling sheets: whale frames with a 0.1 grid
    $PY $Q merge  LINES.txt                                            # add the columns to labels/<stem>/frames.csv
    $PY $Q review --out /home/sat3737/NOAA_whales/review_sheets/underwater/reference_quality_examples --pick PICK.txt
    $PY $Q stats                                                       # distributions per video (markdown)
    $PY $Q frames <cache>/<stem>/sample --out <scratch>/f               # labelling sheets of a new sample (0.1 ticks)

Columns added to frames.csv (empty when the frame has no clearly visible other whale):
  whale_size        longest visible extent of the LARGEST other whale / frame width, 0.1 steps (fills the frame = 1.0)
  whale_frame_share share of the frame area it covers, 0.05 steps
  whale_cut         no / yes: cut by the frame edge
  quality           hq / ok / low (rule: README.md, "Size and quality")
  quality_note      a few words on close calls
LINES.txt: `stem|frame|whale_size|whale_frame_share|whale_cut|quality|quality_note`; # lines ignored. Columns of
frames.csv that already exist are kept in place; the five are appended (or overwritten when present).
Frames: dev clip `<cache>/2025-12-20_CG/frames/`, recordings `<cache>/<stem>/sample/`.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
CACHE = Path("/home/sat3737/NOAA_whales/underwater_cache/reference")
NEW = ["whale_size", "whale_frame_share", "whale_cut", "quality", "quality_note"]
GRADES = {"hq", "ok", "low"}


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def frame_path(stem, f):
    d = CACHE / stem / ("frames" if (CACHE / stem / "frames").is_dir() and not (CACHE / stem / "sample").is_dir() else "sample")
    return d / f"{int(f):06d}.jpg"


def short(stem):
    return "dev" if stem.startswith("2025-12-20") else stem.split("-")[-1]


def stems(args):
    return args or sorted(p.name for p in (HERE / "labels").iterdir() if (p / "frames.csv").exists())


def rows_of(stem):
    return list(csv.DictReader(open(HERE / "labels" / stem / "frames.csv")))


def wrap(s, n):
    out = []
    while len(s) > n:
        c = s.rfind(" ", 0, n); c = c if c > 0 else n; out.append(s[:c]); s = s[c:].lstrip()
    return out + ([s] if s else [])


def panel(stem, r, tw, lines, grid=True, path=None):
    im = Image.open(path or frame_path(stem, r["frame"])).convert("RGB")
    th = tw * im.size[1] // im.size[0]; im = im.resize((tw, th), Image.LANCZOS); d = ImageDraw.Draw(im)
    if grid:
        # ticks every 0.1 of width / height on all edges, dotted lines at 0.5 (keeps the picture readable)
        for k in range(1, 10):
            x = k * tw / 10; y = k * th / 10; L = 16 if k == 5 else 10
            for seg in ([(x, 0), (x, L)], [(x, th - L), (x, th)], [(0, y), (L, y)], [(tw - L, y), (tw, y)]):
                d.line(seg, fill=(255, 255, 0), width=2)
            d.text((x + 2, th - 24), f".{k}", fill=(255, 255, 0), font=font(11))
            if k == 5:
                for z in range(0, max(tw, th), 12):
                    if z < th:
                        d.point([(x, z), (x, z + 1), (x, z + 2)], fill=(255, 255, 0))
                    if z < tw:
                        d.point([(z, y), (z + 1, y), (z + 2, y)], fill=(255, 255, 0))
    hdr = Image.new("RGB", (tw, 6 + 17 * len(lines)), (0, 0, 0)); dh = ImageDraw.Draw(hdr)
    for i, ln in enumerate(lines):
        dh.text((4, 3 + 17 * i), ln, fill=(255, 255, 0) if i == 0 else (230, 230, 230), font=font(13))
    out = Image.new("RGB", (tw, hdr.size[1] + th)); out.paste(hdr, (0, 0)); out.paste(im, (0, hdr.size[1]))
    return out


def sheet(panels, cols, path, title=""):
    w = panels[0].size[0]; rows = [panels[i:i + cols] for i in range(0, len(panels), cols)]
    hs = [max(p.size[1] for p in rw) for rw in rows]; top = 30 if title else 0
    S = Image.new("RGB", (cols * w, top + sum(hs)), (0, 0, 0)); y = top
    if title:
        ImageDraw.Draw(S).text((6, 6), title, fill=(255, 255, 255), font=font(17))
    for rw, h in zip(rows, hs):
        for j, p in enumerate(rw):
            S.paste(p, (j * w, y))
        y += h
    S.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["view", "frames", "merge", "review", "stats"]); ap.add_argument("args", nargs="*")
    ap.add_argument("--out"); ap.add_argument("--per", type=int, default=6); ap.add_argument("--tw", type=int, default=640)
    ap.add_argument("--cols", type=int, default=3); ap.add_argument("--pick"); ap.add_argument("--min", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "view":
        todo = [(s, r) for s in stems(a.args) for r in rows_of(s) if int(r["other_whales"]) >= a.min]
        for k in range(0, len(todo), a.per):
            ps = [panel(s, r, a.tw, [f"{short(s)} f{r['frame']}  w{r['other_whales']}+{r['faint_whales']}f  {r['view']}"]
                        + wrap(r["note"], a.tw // 8)[:3]) for s, r in todo[k:k + a.per]]
            sheet(ps, a.cols, f"{a.out}_{k // a.per:03d}.jpg")
        print(len(todo), "frames,", (len(todo) + a.per - 1) // a.per, "sheets")
    elif a.cmd == "frames":
        # labelling sheets of a sample folder (no labels yet): `frames <cache>/<stem>/sample --out prefix`
        import json
        d = Path(a.args[0]); fs = sorted(d.glob("*.jpg"))
        meta = json.loads((d.parent / "index.json").read_text()) if (d.parent / "index.json").exists() else {}
        tt = meta.get("sample", {})
        for k in range(0, len(fs), a.per):
            ps = []
            for f in fs[k:k + a.per]:
                i = int(f.stem); r = {"frame": i}
                im = panel(None, r, a.tw, [f"f{i}  t={tt.get(str(i), 0):.1f}s"], path=f)
                ps.append(im)
            sheet(ps, a.cols, f"{a.out}_{k // a.per:03d}.jpg")
        print(len(fs), "frames")
    elif a.cmd == "merge":
        lab = {}
        for ln in open(a.args[0]):
            p = [x.strip() for x in ln.rstrip("\n").split("|")]
            if ln.startswith("#") or len(p) < 2 or not p[1].isdigit():
                continue
            if len(p) != 7:
                raise SystemExit(f"bad line: {ln}")
            stem, f, sz, sh, cut, q, note = p
            assert q in GRADES and cut in ("no", "yes"), ln
            assert 0 < float(sz) <= 1.0 and 0 < float(sh) <= 1.0, ln
            lab[(stem, int(f))] = dict(whale_size=f"{float(sz):.1f}", whale_frame_share=f"{float(sh):.2f}", whale_cut=cut, quality=q, quality_note=note)
        for stem in sorted({s for s, _ in lab}):
            fn = HERE / "labels" / stem / "frames.csv"; rows = rows_of(stem); fields = list(rows[0].keys())
            fields += [c for c in NEW if c not in fields]
            n = 0
            for r in rows:
                v = lab.get((stem, int(r["frame"])))
                if int(r["other_whales"]) >= 1 and v is None:
                    raise SystemExit(f"{stem} f{r['frame']}: whale frame without a size / quality line")
                if v is not None and int(r["other_whales"]) < 1:
                    raise SystemExit(f"{stem} f{r['frame']}: size line on a frame without a clear whale")
                r.update(v or {c: "" for c in NEW}); n += v is not None
            with open(fn, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
            print(stem, n, "whale frames graded", sorted(Counter(r["quality"] for r in rows if r["quality"]).items()))
    elif a.cmd == "review":
        # PICK: `stem|frame|why` lines, in sheet order
        picks = [ln.rstrip("\n").split("|") for ln in open(a.pick) if ln.strip() and not ln.startswith("#")]
        ps = []
        for stem, f, *why in picks:
            r = next(x for x in rows_of(stem) if int(x["frame"]) == int(f))
            l1 = f"{r['quality'].upper()}   size {r['whale_size']}  area {r['whale_frame_share']}  cut {r['whale_cut']}   ({short(stem)} f{f})"
            l2 = [x for x in [r["quality_note"], (why[0] if why else "")] if x]
            ps.append(panel(stem, r, 960, [l1] + wrap("; ".join(l2), 110)[:2], grid=False))
        for k in range(0, len(ps), a.per):
            sheet(ps[k:k + a.per], 2, f"{a.out}_{k // a.per + 1:02d}.jpg",
                  "AI by-eye whale size / quality (WP3b), a first proposal for the user to correct. size = longest extent of the largest other whale / frame width")
        print(len(ps), "panels")
    else:
        for s in stems(a.args):
            rows = [r for r in rows_of(s) if r.get("quality")]
            if not rows:
                continue
            print(f"| {short(s)} | {len(rows)} | " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['quality'] for r in rows).items())) + " | "
                  + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r['whale_size'] for r in rows).items())) + " | "
                  + f"cut {sum(r['whale_cut'] == 'yes' for r in rows)} |")


if __name__ == "__main__":
    main()
