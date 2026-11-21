"""uwid.pipeline (WP7) on synthetic data: discovery, window states, keep rule, reasons, offset check, filters, and
one end-to-end scan with a fake detector and a fake water model (no GPU, no real footage)."""

import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import H, W, decode_index, make_video
from uwid import clips, detect, pipeline as pl, tagdata, video


def _age(p, seconds=3600):
    t = time.time() - seconds
    os.utime(p, (t, t))


def test_discover_pairs_forms_logs_and_copying(data_dirs):
    root = data_dirs["footage"]
    dep = root / "2025-1-1 Tag A"
    (dep / "Data").mkdir(parents=True)
    (dep / "Video").mkdir()
    log = dep / "Data" / "tag.csv"
    log.write_text("Date (UTC),Time (UTC),Accelerometer X [m/s2],Depth (100bar) [m]\n")
    a_mov = make_video(dep / "Video" / "CC-1-2-20250101-100000-000-00001.mov", n=30)
    a_m4v = make_video(dep / "Video" / "CC-1-2-20250101-100000-000-00001.m4v", n=30)
    fresh = make_video(dep / "Video" / "CC-1-2-20250101-103000-000-00002.mov", n=30)
    loose = make_video(root / "loose clip.mp4", n=30)
    (dep / "Video" / "._CC-1-2-20250101-100000-000-00001.mov").write_bytes(b"x")
    for p in (log, a_mov, a_m4v, loose):
        _age(p)
    deps = {d.name: d for d in pl.discover(root, min_age_s=60)}
    assert set(deps) == {"2025-1-1 Tag A", "loose_clip"}
    d = deps["2025-1-1 Tag A"]
    assert d.sensor_csv == log
    assert [r.path for r in d.recordings] == [a_mov]
    reasons = {s["path"]: s["reason"] for s in d.skipped}
    assert "other form" in reasons[str(a_m4v)] and "copying" in reasons[str(fresh)]
    assert deps["loose_clip"].sensor_csv is None
    # a single file: its own deployment folder's log is found
    one = pl.discover(a_mov, min_age_s=60)
    assert len(one) == 1 and one[0].sensor_csv == log


def test_keep_intervals_pad_merge_and_clamp():
    t0 = np.arange(60.0)
    t1 = np.minimum(t0 + 1, 59.5)
    whale = np.zeros(60, bool)
    whale[[10, 11, 20, 40, 59]] = True
    keep = pl.keep_intervals(whale, t0, t1, pad=2.0, merge_gap=5.0, duration=59.5)
    # 10-12 -> 8-14; 20-21 -> 18-23: gap 4 < 5 -> merged; 40 -> 38-43; 59 -> 57-59.5 (clamped)
    assert keep == [(8.0, 23.0), (38.0, 43.0), (57.0, 59.5)]
    # a gap of exactly 5 s after padding is not "closer than 5 s"
    whale2 = np.zeros(60, bool)
    whale2[[10, 20]] = True
    assert pl.keep_intervals(whale2, t0, t1, 2.0, 5.0, 59.5) == [(8.0, 13.0), (18.0, 23.0)]


def _sensor_windows():
    """A fake deployment on a 40 s video: deck 0-5, flight 5-10, whale 10-30 (surfaced 18.2-21.0), floating 30-40."""
    segs = pd.DataFrame({"t0": [0, 5, 10, 30], "t1": [5, 10, 30, 40.0], "state": ["air", "flight", "whale", "floating"]})
    air = pd.DataFrame({"t0": [18.2], "t1": [21.0]})
    sensors = SimpleNamespace(segments=segs, air=air)
    vclock = tagdata.VideoClock(t0=0.0)
    return sensors, vclock


def test_build_windows_sensor_states(data_dirs, monkeypatch):
    src = make_video(data_dirs["footage"] / "v.mp4", n=40 * 10, fps=10, gop=10)
    sensors, vclock = _sensor_windows()
    cfg = pl.ScanConfig()
    w = pl.build_windows(pl.Recording(src, "mp4"), cfg, sensors, vclock, None, None)
    assert len(w) == 40
    st = dict(zip(w.k, zip(w.tag_state, w.water, w.scanned)))
    assert st[2] == ("air", "air", False) and st[7] == ("flight", "air", False)
    assert st[12] == ("whale", "underwater", True)
    assert st[18] == ("whale", "crossing", True)       # lens out at 18.2: part of the window under water
    assert st[19] == ("whale", "surfaced", False) and st[20] == ("whale", "surfaced", False)
    assert st[21] == ("whale", "underwater", True)
    assert st[35] == ("floating", "waterline", False)
    assert set(w.water_source[w.tag_state == "whale"]) == {"depth"}
    # two frames per window, 0.5 s apart
    assert (w.frame_1 - w.frame_0 == 5).all()


def test_apply_detections_reasons_and_presets(data_dirs):
    src = make_video(data_dirs["footage"] / "v.mp4", n=40 * 10, fps=10, gop=10)
    sensors, vclock = _sensor_windows()
    w = pl.build_windows(pl.Recording(src, "mp4"), pl.ScanConfig(), sensors, vclock, None, None)
    frames = np.unique(w.loc[w.scanned, ["frame_0", "frame_1"]].to_numpy().ravel())
    # whale seen in both frames of window 12, in one frame of window 25
    sc = {f: 0.0 for f in frames}
    sc[int(w.frame_0[12])] = sc[int(w.frame_1[12])] = 0.5
    sc[int(w.frame_1[25])] = 0.4
    fs = pd.DataFrame({"frame": list(sc), "whale": list(sc.values()), "small": 0.0, "fish": 0.1})
    out = {}
    for preset in ("strict", "sensitive"):
        cfg = pl.ScanConfig(preset=preset)
        ww, keep = pl.apply_detections(w, fs, frames, cfg, 40.0)
        out[preset] = (ww, keep)
    ws, ks = out["strict"]
    assert ks == [(10.0, 15.0)]
    assert ws.loc[12, "why"] == "whale" and ws.loc[10, "why"] == "padding" and ws.loc[25, "why"] == "underwater with no whale"
    wv, kv = out["sensitive"]
    assert kv == [(10.0, 15.0), (23.0, 28.0)]
    reasons = dict(zip(wv.k, wv.why))
    assert reasons[2] == "tag in air" and reasons[7] == "drone flight" and reasons[20] == "surfaced"
    assert reasons[35] == "tag floating" and reasons[16] == "underwater with no whale"
    assert set(wv.why) <= set(pl.REASONS) | {"whale", "padding"}
    assert "hq_score" in wv.columns and "tier" not in wv.columns
    # fish score is information only: same keep decision with a high fish score
    fs2 = fs.assign(fish=0.99)
    assert pl.apply_detections(w, fs2, frames, pl.ScanConfig(preset="sensitive"), 40.0)[1] == kv
    drops = pl.drop_stretches(wv, kv)
    assert drops[0] == {"start": 0.0, "end": 5.0, "reason": "tag in air"}
    assert {"start": 19.0, "end": 21.0, "reason": "surfaced"} in drops
    covered = sum(d["end"] - d["start"] for d in drops) + sum(b - a for a, b in kv)
    assert covered == pytest.approx(40.0)


def test_offset_check_finds_a_wrong_offset():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 5000, 30))
    depth = pd.DataFrame({"t": np.r_[t, t + 3.5], "kind": ["exit"] * 30 + ["entry"] * 30})
    pic = depth.assign(t=depth.t + rng.normal(0, 0.1, len(depth)))
    ok = pl.offset_check(pic, depth)
    assert ok["verdict"] == "ok" and abs(ok["median_picture_minus_depth_s"]) < 0.1
    wrong = pl.offset_check(pic.assign(t=pic.t + 2.0), depth)
    assert wrong["verdict"] == "OFFSET LOOKS WRONG"
    assert wrong["median_picture_minus_depth_s"] == pytest.approx(2.0, abs=0.1)
    # extra false picture events do not move it
    junk = pd.DataFrame({"t": rng.uniform(0, 5000, 40), "kind": "exit"})
    assert pl.offset_check(pd.concat([pic, junk]), depth)["median_picture_minus_depth_s"] == pytest.approx(0, abs=0.1)
    assert pl.offset_check(pic.iloc[:0], depth)["n_pairs"] == 0


def test_frame_scores_filters_and_options():
    fq = detect.patch_score_column
    rows = [  # frame, query, score, box, fish score at the same patch
        (0, "whale", 0.8, (0, 0, 1900, 1000), 0.1),       # covers the frame: dropped unless near_whale_score <= 0.8
        (1, "whale", 0.5, (1750, 1055, 1900, 1078), 0.0),  # in the clock: always dropped
        (2, "whale", 0.4, (100, 100, 400, 300), 0.6),      # fish scores higher at this box: vetoed with fish_veto
        (3, "fish", 0.7, (100, 100, 200, 200), 0.7),
    ]
    b = pd.DataFrame([dict(frame=f, query=q, score=s, x0=x[0], y0=x[1], x1=x[2], y1=x[3], **{fq("fish"): fs,
                                                                                            fq("whale"): s})
                      for f, q, s, x, fs in rows])
    b = pd.concat([b, pd.DataFrame([dict(frame=4, query="whale", score=0.9, x0=10, y0=10, x1=200, y1=150)])],
                  ignore_index=True)   # 190 px long: a small (fish-sized or far) whale
    frames = np.arange(5)
    off = pl.frame_scores(b, frames, pl.ScanConfig(near_whale_score=None), 1920, 1080).set_index("frame")
    assert list(off.whale) == [0.0, 0.0, 0.4, 0.0, 0.0]
    assert off.fish[3] == 0.7 and off.small[4] == 0.9
    base = pl.frame_scores(b, frames, pl.ScanConfig(), 1920, 1080).set_index("frame")
    assert base.whale[0] == 0.8                     # near whale filling the frame: kept by default (score >= 0.6)
    assert base.cover[0] == pytest.approx(1900 * 1000 / (1920 * 1080))
    assert base.whale_px[2] == pytest.approx(300)
    big = pl.frame_scores(b, frames, pl.ScanConfig(min_whale_px=450), 1920, 1080).set_index("frame")
    assert big.whale[2] == 0.0 and big.small[2] == 0.4
    veto = pl.frame_scores(b, frames, pl.ScanConfig(fish_veto=True), 1920, 1080).set_index("frame")
    assert veto.whale[2] == 0.0


def test_hq_parts_and_small_reason(data_dirs):
    S = pl.HQ_SCALES
    col = {"cover": "cover", "score": "top_score", "bright": "bright", "contrast": "contrast", "sharp": "sharp",
           "bubbles": "bubbles", "depth": "depth_m"}
    df = pd.DataFrame({col[k]: [S[k][1], S[k][0], 0.5 * (S[k][0] + S[k][1])] for k in col})
    df.loc[2, "cover"] = np.nan
    hq = pl.hq_parts(df)
    assert hq.hq_score[0] == pytest.approx(1.0) and hq.hq_score[1] == pytest.approx(0.0)
    assert np.isnan(hq.hq_score[2])
    src = make_video(data_dirs["footage"] / "v.mp4", n=40 * 10, fps=10, gop=10)
    sensors, vclock = _sensor_windows()
    w = pl.build_windows(pl.Recording(src, "mp4"), pl.ScanConfig(), sensors, vclock, None, None)
    frames = np.unique(w.loc[w.scanned, ["frame_0", "frame_1"]].to_numpy().ravel())
    fs = pd.DataFrame({"frame": frames, "whale": 0.0, "small": 0.0, "fish": 0.0})
    fs.loc[fs.frame == int(w.frame_0[14]), "small"] = 0.9
    ww, keep = pl.apply_detections(w, fs, frames, pl.ScanConfig(), 40.0)
    assert keep == [] and ww.loc[14, "why"] == "small (fish or far whale)"
    assert not ww.hq_whale.any()


class FakeDetector:
    """'Whale' box (score 0.5) on frames whose drawn index lies in 5-7 s at 10 frames/s."""
    calls = 0

    def __init__(self, queries):
        self.queries = queries
        self.name = "fake"

    def detect(self, images):
        FakeDetector.calls += len(images)
        out = []
        for im in images:
            i = decode_index(im)
            rows = []
            if 50 <= i < 70:
                rows.append(dict(query="whale", score=0.5, x0=50.0, y0=50.0, x1=150.0, y1=120.0,
                                 **{detect.patch_score_column(q): 0.5 if q == "whale" else 0.05 for q in self.queries}))
            out.append(pd.DataFrame(rows, columns=["query", "score", "x0", "y0", "x1", "y1"]
                                    + [detect.patch_score_column(q) for q in self.queries]))
        return out


def fake_water(path):
    """Picture water state: air for the first 2 s, under water after."""
    n = int(np.ceil(video.frame_table(path).duration))
    k = np.arange(n)
    win = pd.DataFrame({"k": k, "t0": k * 1.0, "t1": k + 1.0, "state": np.where(k < 2, "air", "underwater"),
                        "score": np.where(k < 2, 0.9, 0.1)})
    return win, pd.DataFrame({"t": [2.0], "kind": ["entry"]})


def test_scan_end_to_end_without_sensor_log(data_dirs):
    src = make_video(data_dirs["footage"] / "clip.mp4", n=150, fps=10, gop=10, audio=True)
    _age(src)
    before = (src.stat().st_size, src.stat().st_mtime_ns)
    FakeDetector.calls = 0
    cfg = pl.ScanConfig(preset="sensitive", gpu_lock=None)
    out_root, clip_root = data_dirs["cache"] / "scan", data_dirs["cache"] / "clips"
    sums = pl.scan(data_dirs["footage"], cfg, cut=True, mode="copy", log=lambda *a: None,
                   detector_factory=FakeDetector, water_fn=fake_water, out_root=out_root, clip_root=clip_root)
    assert len(sums) == 1
    s = sums[0]
    assert FakeDetector.calls == 26      # 13 underwater windows x 2 frames
    assert s["kept_s"] == pytest.approx(6.0)  # whale windows 5, 6 -> 3-9 s
    assert s["dropped_s"]["surfaced"] == pytest.approx(2.0)
    assert s["dropped_s"]["underwater with no whale"] == pytest.approx(7.0)
    w = pd.read_csv(out_root / "clip" / "sensitive" / "windows" / "clip.mp4.csv")
    assert list(w.columns[:4]) == ["recording", "k", "t0", "t1"]
    assert w.loc[w.k == 5, "why"].item() == "whale" and w.loc[w.k == 0, "water_source"].item() == "picture"
    vids = clips.load_intervals(out_root / "clip" / "sensitive" / "intervals.json")
    assert [(k.start, k.end) for k in vids[0]["keep"]] == [(3.0, 9.0)]
    assert {d.reason for d in vids[0]["drop"]} == {"surfaced", "underwater with no whale"}
    assert s["clips"]["n"] == 1 and s["clips"]["bytes"] > 0
    idx = pd.read_csv(clip_root / "clip" / "clips_index.csv")
    assert idx.tier.iloc[0] in ("whale", "hq_whale")
    clip = idx.clip_file.iloc[0]
    got = [decode_index(f.image) for f in video.iter_frames(clip)]
    assert got[0] <= 30 and got[-1] == 89           # copy mode starts on the keyframe at or before 3.0 s
    assert (src.stat().st_size, src.stat().st_mtime_ns) == before
    # a re-run reuses the detector cache and gives the same answer; the strict preset needs no new detections
    pl.scan(data_dirs["footage"], cfg, log=lambda *a: None, detector_factory=FakeDetector, water_fn=fake_water,
            out_root=out_root, clip_root=clip_root)
    s2 = pl.scan(data_dirs["footage"], pl.ScanConfig(preset="strict", gpu_lock=None), log=lambda *a: None,
                 detector_factory=FakeDetector, water_fn=fake_water, out_root=out_root, clip_root=clip_root)[0]
    assert FakeDetector.calls == 26
    assert s2["kept_s"] == pytest.approx(6.0)
    summary = json.loads((out_root / "clip" / "strict" / "summary.json").read_text())
    assert summary["label"] == "strict"
