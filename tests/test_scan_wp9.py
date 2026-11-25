"""WP9 additions to uwid.pipeline: floating time scanned when the picture sees water, tiers, the clock offset for a
new tag, the decision drawn on sheets. Synthetic data only."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import make_video
from uwid import pipeline as pl, tagdata


def _float_setup(data_dirs):
    src = make_video(data_dirs["footage"] / "v.mp4", n=40 * 10, fps=10, gop=10)
    segs = pd.DataFrame({"t0": [0, 20.0], "t1": [20.0, 40.0], "state": ["whale", "floating"]})
    sensors = SimpleNamespace(segments=segs, air=pd.DataFrame({"t0": [], "t1": []}),
                              depth_t=np.array([0.0, 40.0]), depth_m=np.array([5.0, 5.0]))
    # picture: under water 20-30 s, surfaced 30-40 s
    pic = pd.DataFrame({"k": np.arange(40), "state": ["underwater"] * 30 + ["surfaced"] * 10, "score": 0.1})
    return src, sensors, pic


def test_floating_scanned_where_the_picture_sees_water(data_dirs):
    src, sensors, pic = _float_setup(data_dirs)
    rec = pl.Recording(src, "mp4")
    w = pl.build_windows(rec, pl.ScanConfig(), sensors, tagdata.VideoClock(0.0), None, pic)
    assert w.loc[25, "tag_state"] == "floating" and w.loc[25, "water"] == "underwater" and w.loc[25, "scanned"]
    assert w.loc[35, "water"] == "surfaced" and not w.loc[35, "scanned"]
    assert w.loc[25, "water_source"] == "picture" and np.isfinite(w.loc[25, "depth_m"])
    old = pl.build_windows(rec, pl.ScanConfig(scan_floating=False), sensors, tagdata.VideoClock(0.0), None, pic)
    assert not old.loc[25, "scanned"] and old.loc[25, "water"] == "waterline"
    # a whale next to the floating tag is kept; a floating window without one keeps its reason
    frames = np.unique(w.loc[w.scanned, ["frame_0", "frame_1"]].to_numpy().ravel())
    fs = pd.DataFrame({"frame": frames, "whale": 0.0, "whale_part": 0.0, "small": 0.0, "fish": 0.0})
    fs.loc[fs.frame == int(w.frame_0[25]), ["whale", "whale_part"]] = 0.8
    ww, keep = pl.apply_detections(w, fs, frames, pl.ScanConfig(), 40.0)
    assert keep == [(23.0, 28.0)]
    assert ww.loc[21, "why"] == "tag floating" and ww.loc[35, "why"] == "tag floating"
    assert pl.ScanConfig(scan_floating=False).label() == "sensitive+nofloat"


def test_tiers_and_min_tier_filter():
    w = pd.DataFrame({"whale": [True, True, True, False], "hq_whale": [True, False, False, False],
                      "whale_px": [900.0, 500.0, 320.0, np.nan]})
    assert list(pl.window_tier(w, pl.ScanConfig())) == ["hq", "ok", "low", "none"]
    assert pl.best_tier(["low", "none", "ok"]) == "ok" and pl.best_tier(["none"]) == "none"
    iv = {"videos": [{"source": "a.mov", "keep": [{"start": 0, "end": 5, "tier": "low"},
                                                   {"start": 9, "end": 12, "tier": "ok"}],
                      "drop": [{"start": 5, "end": 9, "reason": "underwater with no whale"}]}]}
    out = pl.filter_intervals_by_tier(iv, "ok")
    v = out["videos"][0]
    assert v["keep"] == [{"start": 9, "end": 12, "tier": "ok"}]
    assert v["drop"][0]["start"] == 0 and "not cut" in v["drop"][0]["reason"]
    assert len(pl.filter_intervals_by_tier(iv, "low")["videos"][0]["keep"]) == 2
    assert iv["videos"][0]["keep"][0]["tier"] == "low"          # input untouched


def test_offset_modes(data_dirs):
    v = make_video(data_dirs["footage"] / "CC-11-96-20250203-101236-000-00004.mp4", n=10)
    w = make_video(data_dirs["footage"] / "CC-09-85-HRA-20250102-111224-000-00005.mp4", n=10)
    dep = lambda p: SimpleNamespace(recordings=[pl.Recording(p, "mp4")])
    assert pl.resolve_offset(pl.ScanConfig(), dep(v)) == ("tag", 2.03)
    assert pl.resolve_offset(pl.ScanConfig(), dep(w)) == ("auto", 0.0)
    assert pl.resolve_offset(pl.ScanConfig(clock_offset_s=1.5), dep(w)) == ("given", 1.5)
    assert pl.resolve_offset(pl.ScanConfig(clock_offset_s="auto"), dep(v)) == ("auto", 0.0)


def test_offset_check_midpoint_and_per_recording():
    rng = np.random.default_rng(1)
    t = np.arange(20) * 60.0 + 100
    de = pd.DataFrame({"t": np.r_[t, t + 8], "kind": ["exit"] * 20 + ["entry"] * 20,
                       "recording": (["a"] * 10 + ["b"] * 10) * 2})
    # the picture: clock offset 1.0 s, exits 0.3 s early and entries 0.3 s late (other lens lags), small noise
    pe = pd.DataFrame({"t": np.r_[t + 1.0 - 0.3, t + 8 + 1.0 + 0.3] + rng.normal(0, 0.03, 40),
                       "kind": de.kind})
    r = pl.offset_check(pe, de)
    assert r["n_pairs"] == 40
    assert r["midpoint_s"] == pytest.approx(1.0, abs=0.05) and r["half_difference_s"] == pytest.approx(0.3, abs=0.05)
    assert set(r["per_recording"]) == {"a", "b"}
    assert r["per_recording"]["a"]["midpoint_s"] == pytest.approx(1.0, abs=0.05)


def test_depth_events_skip_surface_brushes():
    surf = pd.DataFrame({"start_t": [10.0, 50.0], "end_t": [15.0, 50.3]})
    st = SimpleNamespace(surfacings=surf)
    ev = pl.SensorTimeline.depth_events(st, pl.ScanConfig())
    assert list(ev.kind) == ["exit", "entry"] and ev.t.tolist() == pytest.approx([10.3, 14.6])


def test_decision_objects_one_box_per_object():
    rows = []
    for q, sc in (("whale", 0.96), ("small fish", 0.32), ("fish", 0.23)):     # one region, three prompts
        rows.append(dict(query=q, score=sc, x0=100, y0=100, x1=1093, y1=1000, score__whale=0.96, score__fish=0.23,
                         score__school_of_fish=0.1, score__small_fish=0.32))
    rows.append(dict(query="whale", score=0.54, x0=10, y0=10, x1=297, y1=205, score__whale=0.54, score__fish=0.49,
                     score__school_of_fish=0.1, score__small_fish=0.2))
    rows.append(dict(query="fish", score=0.49, x0=10, y0=10, x1=297, y1=205, score__whale=0.54, score__fish=0.49,
                     score__school_of_fish=0.1, score__small_fish=0.2))
    rows.append(dict(query="fish", score=0.6, x0=1500, y0=500, x1=1540, y1=520, score__whale=0.01,
                     score__fish=0.6, score__school_of_fish=0.1, score__small_fish=0.2))
    o = pl.decision_objects(pd.DataFrame(rows), pl.ScanConfig(), 1920, 1080)
    assert list(o.label) == ["whale", "small", "fish"]
    assert o.score.tolist() == pytest.approx([0.96, 0.54, 0.6])
    assert o.px.iloc[1] == pytest.approx(287.0)


def test_ring_and_silhouette():
    img = np.full((540, 960, 3), 200, np.uint8)
    img[200:340, 300:660] = 30                       # a dark box on a bright surround
    ring = pl.ring_brightness(img, 300, 200, 660, 340)
    assert ring == pytest.approx(200 / 255, abs=0.01)
    w = pd.DataFrame({"whale": [True, True, True, False], "whale_px": [700.0, 700.0, 300.0, np.nan],
                      "bright": [0.12, 0.6, 0.12, np.nan], "ring": [0.7, 0.62, 0.7, np.nan]})
    assert list(pl.silhouette_flag(w, pl.ScanConfig())) == [True, False, False, False]


def test_describe_clip_traces_columns():
    w = pd.DataFrame({"t0": [0.0, 1, 2, 3], "t1": [1.0, 2, 3, 4], "clock": ["12:07:17.0", "12:07:18.0", "12:07:19.0",
                                                                          "12:07:20.0"],
                      "depth_m": [3.0, 4, 5, 6], "tier": ["hq", "hq", "low", "fish"], "whale_px": [900.0, 950, 400, np.nan],
                      "n_whale": [1.0, 1, 1, 0], "n_fish": [0.0, 0, 2, 3], "fish": [False, False, True, True],
                      "water": ["underwater", "crossing", "underwater", "underwater"], "tag_state": ["whale"] * 4,
                      "silhouette": [False, True, False, False]})
    d = pl.describe_clip(w, 0.0, 4.0)
    assert d.startswith("12:07:17 to 12:07:20, 4 s, depth 3 to 6 m.")
    assert "One whale in view, largest 0.49 of the frame width (high quality 2 s, low 1 s)." in d
    assert "silhouette" in d and "Fish: a few (up to 3 fish boxes in a frame, 2 s)." in d
    assert "Tagged whale surfaces at 12:07:18." in d


def test_back_box_is_not_a_whale():
    rows = [dict(frame=1, query="whale", score=0.5, x0=10, y0=470, x1=1915, y1=1080, score__whale=0.5),
            dict(frame=1, query="whale", score=0.8, x0=10, y0=0, x1=1900, y1=600, score__whale=0.8)]
    b = pd.DataFrame(rows)
    assert list(pl.is_back_box(b, 1920, 1080)) == [True, False]
    o = pl.decision_objects(b, pl.ScanConfig(), 1920, 1080)
    assert len(o) == 1 and o.score.iloc[0] == pytest.approx(0.8)
    fs = pl.frame_scores(b[:1], np.array([1]), pl.ScanConfig(), 1920, 1080)
    assert fs.whale.iloc[0] == 0.0
    fs = pl.frame_scores(b[:1], np.array([1]), pl.ScanConfig(back_filter=False), 1920, 1080)
    assert fs.whale.iloc[0] == pytest.approx(0.5)


def _guard_windows(gap_s, n=120):
    t0 = np.arange(float(n))
    t1 = t0 + 1
    active = np.zeros(n, bool)
    active[10:15] = True
    active[15 + gap_s:15 + gap_s + 5] = True
    return active, t0, t1


def test_clip_guard_joins_under_30_s_and_splits_over():
    a, t0, t1 = _guard_windows(29)
    hard = np.zeros(len(a), bool)
    assert pl.clip_intervals(a, t0, t1, hard, 2.0, 30.0, 120.0) == [(8.0, 51.0)]           # 29 s quiet: one clip
    a, t0, t1 = _guard_windows(31)
    assert pl.clip_intervals(a, t0, t1, hard, 2.0, 30.0, 120.0) == [(8.0, 17.0), (44.0, 53.0)]   # 31 s: two clips
    # buffer clipped at the file's ends
    a = np.zeros(120, bool); a[0] = True; a[119] = True
    assert pl.clip_intervals(a, t0, t1, hard, 2.0, 30.0, 119.5) == [(0.0, 3.0), (117.0, 119.5)]
    # a hard window (tag in air, long lens-out) ends a clip even inside the guard; the buffer stops at it
    a, t0, t1 = _guard_windows(10)
    hard[20] = True
    assert pl.clip_intervals(a, t0, t1, hard, 2.0, 30.0, 120.0) == [(8.0, 17.0), (23.0, 32.0)]


def test_hard_breaks_breath_versus_long_lens_out():
    w = pd.DataFrame({"tag_state": ["whale"] * 60 + ["air"] * 5,
                      "water": ["underwater"] * 10 + ["surfaced"] * 8 + ["underwater"] * 10 + ["surfaced"] * 25
                      + ["underwater"] * 7 + ["air"] * 5})
    h = pl.hard_breaks(w, pl.ScanConfig())
    assert not h[10:18].any() and h[28:53].all() and h[60:].all()


def test_veto_by_classifier():
    rows = [dict(frame=1, query="whale", score=0.6, x0=100, y0=100, x1=700, y1=400, boxclf_class="fish"),
            dict(frame=1, query="whale", score=0.5, x0=800, y0=50, x1=1500, y1=500, boxclf_class="other"),
            dict(frame=2, query="whale", score=0.7, x0=100, y0=100, x1=900, y1=500, boxclf_class="whale"),
            dict(frame=3, query="whale", score=0.7, x0=100, y0=100, x1=900, y1=500, boxclf_class="")]
    b = pd.DataFrame(rows)
    fs = pl.frame_scores(b, np.array([1, 2, 3]), pl.ScanConfig(), 1920, 1080).set_index("frame")
    assert fs.loc[1, "whale"] == 0.0 and fs.loc[1, "veto_fish"] == pytest.approx(0.6)
    assert fs.loc[1, "veto_other"] == pytest.approx(0.5)
    assert fs.loc[2, "whale"] == pytest.approx(0.7) and fs.loc[3, "whale"] == pytest.approx(0.7)   # unseen: rule
    fs = pl.frame_scores(b, np.array([1]), pl.ScanConfig(boxclf=False), 1920, 1080)
    assert fs.whale.iloc[0] == pytest.approx(0.6)


def test_mark_continuing_across_files():
    a = [{"start": 0.0, "end": 40.0}, {"start": 100.0, "end": 120.0}]
    b = [{"start": 0.0, "end": 10.0}]
    pl.mark_continuing([(1000.0, 1120.0, 120.0, "f1.mov", a), (1120.5, 1300.0, 180.0, "f2.mov", b)])
    assert a[-1]["continues_in"] == "f2.mov" and b[0]["continues_from"] == "f1.mov"
    c = [{"start": 0.0, "end": 10.0}]
    pl.mark_continuing([(1000.0, 1120.0, 120.0, "f1.mov", [{"start": 0.0, "end": 50.0}]),
                        (1200.0, 1300.0, 100.0, "f3.mov", c)])     # 80 s apart: not one scene
    assert "continues_from" not in c[0]


def test_window_classes_from_frames():
    w = pd.DataFrame({"frame_0": [0, 2, 4, 6], "frame_1": [1, 3, 5, 7]})
    fr = pd.DataFrame({"frame": range(6), "grey_mean": [100, 100, 10, 12, 100, 100],
                       "share_seafloor": [0.3, 0.2, 0.5, 0.5, 0.0, 0.1], "sand_p": [0.1, 0.2, 0.9, 0.9, 0.7, 0.6]})
    out = pl.fill_window_classes(w, fr)
    assert list(out.seafloor) == ["", "", "", ""]          # withdrawn 2026-11-23
    assert list(out.dark) == [False, True, False, ""]
    assert list(out.sand) == [False, False, True, ""]


def test_attach_box_classes_by_coordinates():
    boxes = pd.DataFrame({"frame": [1, 1, 2], "query": ["whale", "fish", "whale"], "score": [0.5, 0.2, 0.6],
                          "x0": [10.04, 10.04, 5.0], "y0": [20.0, 20.0, 5.0], "x1": [500.0, 500.0, 50.0],
                          "y1": [300.0, 300.0, 60.0]})
    cls = pd.DataFrame({"frame": [1, 1], "x0": [10.0, 10.0], "y0": [20.0, 20.0], "x1": [500.0, 500.0],
                        "y1": [300.0, 300.0], "boxclf_class": ["fish", "fish"], "p_whale": [0.1, 0.1],
                        "p_fish": [0.8, 0.8], "p_other": [0.1, 0.1]})
    out = pl.attach_box_classes(boxes, cls)
    assert list(out.boxclf_class) == ["fish", "fish", ""]


def test_sheet_decision_draws_only_what_decides():
    rows = [dict(frame=1, query="fish", score=0.39, x0=10, y0=10, x1=600, y1=300),          # under fish threshold
            dict(frame=1, query="whale", score=0.36, x0=1200, y0=100, x1=1900, y1=800, boxclf_class="other"),
            dict(frame=1, query="fish", score=0.30, x0=1200, y0=100, x1=1900, y1=800, boxclf_class="other"),
            dict(frame=1, query="whale", score=0.5, x0=100, y0=500, x1=700, y1=900, boxclf_class="fish"),
            dict(frame=1, query="fish", score=0.62, x0=1000, y0=900, x1=1050, y1=930)]
    o = pl.decision_objects(pd.DataFrame(rows), pl.ScanConfig(), 1920, 1080)
    got = sorted(zip(o.label, o.score.round(2), o.clf))
    assert got == [("fish", 0.5, True), ("fish", 0.62, False), ("other", 0.36, True)]


def test_description_lengths_and_fish_wording():
    w = pd.DataFrame({"t0": [0.0, 1, 2, 3, 4], "t1": [1.0, 2, 3, 4, 4.4], "clock": [f"10:00:0{i}.0" for i in range(5)],
                      "tier": ["fish", "none", "none", "none", "none"], "whale_px": np.nan, "n_whale": 0.0,
                      "n_fish": [0.0, 0, 0, 0, 0], "fish": [True, False, False, False, False],
                      "active": [True, False, False, False, False]})
    d = pl.describe_clip(w, 0.0, 4.4)
    assert "4.4 s" in d and "3.4 s of 4.4 s without anything detected." in d
    assert "Fish: a few (1 s)." in d and "0 boxes" not in d
