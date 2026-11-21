import datetime as dt

import numpy as np

from conftest import CLOCK_LAYOUT, clock_text, frame_image, make_video
from uwid import timecode as tc


def _glyphs(tmp_path):
    start = dt.datetime(2025, 12, 20, 11, 29, 36, 600000)
    p = make_video(tmp_path / "boot.mp4", n=120, fps=30, gop=30, clock_start=start)
    layout = tc.ClockLayout(**CLOCK_LAYOUT)
    gs = tc.bootstrap_glyphs(p, clock_text(start), layout=layout)
    return gs, start


def test_bootstrap_names_all_digits(tmp_path):
    gs, _ = _glyphs(tmp_path)
    assert set("0123456789:.Dec") <= set(gs.chars)
    assert gs.meta["n_unnamed_clusters"] == 0


def test_read_every_frame(tmp_path):
    gs, _ = _glyphs(tmp_path)
    start = dt.datetime(2025, 12, 20, 23, 59, 58, 0)  # crosses midnight
    p = make_video(tmp_path / "read.mp4", n=120, fps=30, gop=30, clock_start=start)
    res = tc.read_video_clock(p, glyphs=gs)
    s = res["summary"]
    assert s["n_read"] == 120 and s["monotonic"]
    truth = [clock_text(start + dt.timedelta(seconds=i / 30)) for i in range(120)]
    got = [r["text"] for r in res["rows"]]
    assert sum(a == b for a, b in zip(got, truth)) == 120
    assert abs(s["fit"]["slope"] - 1) < 0.01 and s["fit"]["max_abs_residual_inliers_s"] < 0.12


def test_no_clock_returns_none(tmp_path):
    gs, _ = _glyphs(tmp_path)
    img = frame_image(5, 10, clock=None)
    assert tc.read_clock(img, gs) is None
    img = np.zeros_like(img)
    assert tc.read_clock(img, gs) is None


def test_unknown_month_uses_hint(tmp_path):
    gs, _ = _glyphs(tmp_path)  # knows D, e, c only
    t = dt.datetime(2025, 2, 3, 10, 12, 38, 100000)
    img = frame_image(0, 1, clock=clock_text(t))
    rd = tc.read_clock(img, gs, date_hint=dt.date(2025, 2, 3))
    assert rd is not None and rd.month_source == "hint"
    assert rd.datetime == dt.datetime(2025, 2, 3, 10, 12, 38, 100000)
    rd = tc.read_clock(img, gs)
    assert rd is not None and rd.datetime is None and abs(rd.seconds_of_day - (10 * 3600 + 12 * 60 + 38.1)) < 1e-6


def test_unreadable_frames_are_flagged(tmp_path):
    gs, _ = _glyphs(tmp_path)
    start = dt.datetime(2025, 12, 20, 8, 0, 0)
    p = make_video(tmp_path / "gaps.mp4", n=60, fps=30, gop=30, clock_start=start, clock_every=3)
    res = tc.read_video_clock(p, glyphs=gs)
    flags = [r["flag"] for r in res["rows"]]
    assert flags.count("unreadable") == 40 and res["summary"]["n_read"] == 20


def test_glyph_save_load(tmp_path):
    gs, start = _glyphs(tmp_path)
    gs.save(tmp_path / "g.json")
    g2 = tc.GlyphSet.load(tmp_path / "g.json")
    img = frame_image(0, 1, clock=clock_text(start))
    assert tc.read_clock(img, g2).text == clock_text(start)


def test_robust_fit_ignores_outliers():
    x = np.arange(100) / 2.0
    y = 1000 + x + np.random.default_rng(0).uniform(-0.05, 0.05, 100)
    y[[10, 50]] += 30
    f = tc.robust_line_fit(x, y)
    assert abs(f["slope"] - 1) < 1e-3 and f["n_outliers"] == 2
