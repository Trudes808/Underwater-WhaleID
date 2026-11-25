"""The second clock layout (``DD/MM/YY HH:MM:SS.mmm`` in a proportional font, tag CC-09-85): found in the picture,
glyphs learned from the video's own file name, read back. Synthetic video only."""
import datetime as dt
from fractions import Fraction

import av
import cv2
import numpy as np

from conftest import H, W, frame_image
from uwid import timecode as tc

FONT = cv2.FONT_HERSHEY_SIMPLEX


def dmy_text(t: dt.datetime) -> str:
    return f"{t.day:02d}/{t.month:02d}/{t.year % 100:02d} {t:%H:%M:%S}.{t.microsecond // 1000:03d}"


def draw_dmy(img, text):
    # digits at a fixed advance, separators narrower (like the tag's proportional overlay)
    x, y0 = 92, H - 22
    img[y0:H, x - 3:W] = 0
    for ch in text:
        adv = 5 if ch in ":." else (14 if ch == " " else 10 if ch != "/" else 9)
        if ch != " ":
            cv2.putText(img, ch, (x, H - 6), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        x += adv


def make_dmy_video(path, start: dt.datetime, n=600, fps=10):
    with av.open(str(path), "w") as c:
        v = c.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = W, H, "yuv420p"
        v.options = {"crf": "8", "g": str(fps), "bf": "0", "preset": "veryfast"}
        for i in range(n):
            img = frame_image(i * 7, n)
            draw_dmy(img, dmy_text(start + dt.timedelta(seconds=i / fps)))
            fr = av.VideoFrame.from_ndarray(img, format="rgb24")
            fr.pts, fr.time_base = i, Fraction(1, fps)
            for pkt in v.encode(fr):
                c.mux(pkt)
        for pkt in v.encode():
            c.mux(pkt)
    return path


def test_find_learn_and_read_dmy_clock(data_dirs):
    name_start = dt.datetime(2025, 2, 3, 22, 45, 30)
    clock_start = name_start + dt.timedelta(seconds=0.4)       # the clock runs 0.4 s after the name
    p = make_dmy_video(data_dirs["footage"] / "CC-09-85-HRA-20250203-224530-000-00007.mp4", clock_start)
    lay, info = tc.find_layout(p)
    assert lay is not None, info
    assert lay.fmt == "DD/MM/YY" and len(lay.centers) == 21
    gs = tc.bootstrap_from_name(p, lay, n_probes=40, burst=6)
    assert gs.meta["share_read"] > 0.9
    assert set("0123456789") <= set(gs.chars)
    assert abs(gs.meta["passes"][-1]["offset_s"] - 0.4) < 0.06
    res = tc.read_video_clock(p, glyphs=gs, every_s=5.0, burst=5)
    s = res["summary"]
    assert s["share_read"] > 0.95 and s["monotonic"] and s["clock_format"] == "DD/MM/YY"
    truth = {r["index"]: dmy_text(clock_start + dt.timedelta(seconds=r["index"] / 10)) for r in res["rows"]}
    assert sum(r["text"] == truth[r["index"]] for r in res["rows"]) >= 0.95 * len(res["rows"])
    rd = res["rows"][0]
    assert rd["datetime"].startswith("2025-02-03T22:45:30")
    # glyphs_for: learns once, stores, and picks the stored set next time
    tc._GLYPH_CHOICE.clear()
    g2 = tc.glyphs_for(p)
    assert g2 is not None and tc.glyph_path_for(g2.layout).exists()
    tc._GLYPH_CHOICE.clear()
    g3 = tc.glyphs_for(p, learn=False)
    assert g3 is not None and g3.layout.fmt == "DD/MM/YY"


def test_no_clock_no_layout():
    frames = [frame_image(i * 9, 10) for i in range(8)]
    lay, info = tc.find_layout_in_frames([tc.to_gray(f).astype(np.uint8) for f in frames])
    assert lay is None


def test_render_and_parse_dmy():
    d = dt.date(2025, 1, 2)
    assert tc.render(15 * 3600 + 45 * 60 + 2.475, d, "DD/MM/YY") == "02/01/25 15:45:02.475"
    sod, when, src = tc.parse_dmy("02/01/25 15:45:02.475")
    assert when == dt.datetime(2025, 1, 2, 15, 45, 2, 475000) and abs(sod - 56702.475) < 1e-9
    assert tc.parse_dmy("02/01/25 25:45:02.475") is None
    assert tc._certain(100.2, 100.6, d, "DD/MM/YY", "DMY")[:20] == "01/01/00 00:01:40.??"[:0] + tc._certain(
        100.2, 100.6, d, "DD/MM/YY", "DMY")[:20]
    c = tc._certain(3600 + 100.2, 3600 + 100.6, d, "DD/MM/YY", "DMY")
    assert c.startswith("02/01/25 01:01:40.") and c[-3:] == "???"
