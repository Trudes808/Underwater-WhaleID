"""Fish candidate sheets for the user (every fish call goes on one): left the whole frame with the boxes, right a
full-size crop around the boxes (native pixels, shrunk only if the boxes span more than 960 x 540).

    PY=.venv/bin/python
    $PY experiments/02_reference_labels/fish_sheets.py --out /home/sat3737/NOAA_whales/review_sheets/underwater/reference_fish_candidates
Reads labels/*/fish_calls.json and the sampled frames in underwater_cache/reference/<stem>/sample/.
"""
import argparse
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


def panel(stem, call):
    im = Image.open(CACHE / stem / "sample" / f"{call['frame']:06d}.jpg").convert("RGB")
    boxes = call["boxes"]
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes); x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w = max(960, (x1 - x0) + 80); h = max(540, (y1 - y0) + 80)
    if w / h > 16 / 9:
        h = w * 9 / 16
    else:
        w = h * 16 / 9
    cx0 = int(min(max(0, cx - w / 2), 1920 - w)); cy0 = int(min(max(0, cy - h / 2), 1080 - h))
    crop = im.crop((cx0, cy0, int(cx0 + w), int(cy0 + h))).resize((960, 540), Image.LANCZOS)
    dc = ImageDraw.Draw(crop); s = 960 / w
    for b in boxes:
        dc.rectangle([(b[0] - cx0) * s - 3, (b[1] - cy0) * s - 3, (b[2] - cx0) * s + 3, (b[3] - cy0) * s + 3], outline=(255, 60, 60), width=2)
    full = im.resize((960, 540)); df = ImageDraw.Draw(full)
    for b in boxes:
        df.rectangle([b[0] / 2 - 2, b[1] / 2 - 2, b[2] / 2 + 2, b[3] / 2 + 2], outline=(255, 60, 60), width=2)
    df.rectangle([cx0 / 2, cy0 / 2, (cx0 + w) / 2, (cy0 + h) / 2], outline=(255, 255, 0), width=1)
    row = Image.new("RGB", (1920, 540 + 52)); row.paste(full, (0, 52)); row.paste(crop, (960, 52)); d = ImageDraw.Draw(row)
    d.text((6, 4), f"{stem}  f{call['frame']}  fish counted: {call['count']}   (right: crop at {s:.2f}x native)", fill=(255, 255, 0), font=font(18))
    d.text((6, 28), str(call.get("note", ""))[:170], fill=(255, 255, 255), font=font(15))
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--per", type=int, default=3)
    a = ap.parse_args()
    rows = []
    for f in sorted(HERE.glob("labels/*/fish_calls.json")):
        for call in json.loads(f.read_text()):
            rows.append(panel(f.parent.name, call))
    for k in range(0, len(rows), a.per):
        part = rows[k:k + a.per]; sheet = Image.new("RGB", (1920, sum(r.size[1] for r in part) + 30)); y = 30
        ImageDraw.Draw(sheet).text((6, 6), "AI by-eye fish calls (WP3), for the user to confirm. Red = counted fish; yellow = crop area", fill=(255, 255, 255), font=font(17))
        for r in part:
            sheet.paste(r, (0, y)); y += r.size[1]
        sheet.save(f"{a.out}_{k // a.per + 1:02d}.jpg", quality=88)
    print(len(rows), "fish calls,", (len(rows) + a.per - 1) // a.per, "sheets")


if __name__ == "__main__":
    main()
