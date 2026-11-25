"""WP16 Q5: one review sheet of the 12 boat photos chosen near tag events / surfacings (selection made by eye,
stored outside the repo in underwater_cache/photo_link/sheet_selection.csv: photo names carry people's names).
Tiles are labelled with running number, camera time and catalogue animal, never with file names. Photos showing a
person were left out before selection. Output: review_sheets/underwater/photo_link_tag_candidates_01.jpg
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT = Path("/home/sat3737/NOAA_whales/underwater_cache/photo_link")
SHEET = Path("/home/sat3737/NOAA_whales/review_sheets/underwater/photo_link_tag_candidates_01.jpg")
IMG = "/var/tmp/sat3737-whale-days/{day}/images/{key}.jpg"
PHOTOS = "/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{day}/safe/photos.json"
TW, TH, CAP = 720, 480, 58


def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    s = pd.read_csv(OUT / "sheet_selection.csv").sort_values("n")
    meta = {d: json.loads(Path(PHOTOS.format(day=d)).read_text()) for d in s.day.unique()}
    cols, rows = 4, (len(s) + 3) // 4
    sheet = Image.new("RGB", (cols * TW, 70 + rows * (TH + CAP)), (20, 20, 20))
    d0 = ImageDraw.Draw(sheet)
    d0.text((10, 8), "WP16 photo link: boat photos at tag events / tag surfacings (camera time HST; shift = tag - camera "
            "from the tag-on photo: 02-03 -173.3 s, 01-02 -170.2 s)", fill=(255, 255, 255), font=font(22))
    d0.text((10, 38), "Surfacing tiles are crops around the whale; lag = photo - tag surfacing start (s). Caption after '|' "
            "= AI by-eye reading, not ground truth.", fill=(200, 200, 200), font=font(20))
    for j, r in enumerate(s.itertuples()):
        im = Image.open(IMG.format(day=r.day, key=r.key)).convert("RGB")
        if r.kind == "surfacing":
            bb = [i["bbox"] for i in meta[r.day][r.key]["instances"]]
            x0, y0 = min(b[0] for b in bb), min(b[1] for b in bb)
            x1, y1 = max(b[2] for b in bb), max(b[3] for b in bb)
            cx, cy, w = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, (y1 - y0) * 1.5) * 1.3
            im = im.crop((int(cx - w / 2), int(cy - w / 3), int(cx + w / 2), int(cy + w / 3)))
        im.thumbnail((TW, TH))
        x, y = (j % cols) * TW, 70 + (j // cols) * (TH + CAP)
        sheet.paste(im, (x + (TW - im.size[0]) // 2, y))
        lag = "" if pd.isna(r.lag) else f", lag {r.lag:+.1f} s"
        ImageDraw.Draw(sheet).text((x + 6, y + TH + 4), f"#{r.n}  {r.local[:22]}  {r.kind}{lag}", fill=(255, 230, 0), font=font(19))
        ImageDraw.Draw(sheet).text((x + 6, y + TH + 28), f"catalogue {r.label[11:]} | {r.eye}", fill=(255, 255, 255), font=font(17))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET, quality=88)
    print(SHEET, sheet.size)


if __name__ == "__main__":
    main()
