import csv
import json

import pytest

from conftest import decode_index, make_video
from uwid import clips, video


def _src(data_dirs):
    # sparse keyframes: one every 2 s
    return make_video(data_dirs["footage"] / "src.mp4", n=240, fps=30, gop=60, audio=True)


def _intervals(tmp_path, keep, drop=()):
    f = tmp_path / "iv.json"
    f.write_text(json.dumps({"format": clips.INTERVALS_FORMAT, "videos": [
        {"source": "src.mp4", "keep": [dict(start=s, end=e, reason=r) for s, e, r in keep],
         "drop": [dict(start=s, end=e, reason=r) for s, e, r in drop]}]}))
    return f


def _read_csv(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def test_pad_and_merge():
    I = clips.Interval
    out = clips.pad_and_merge([I(5, 6, "b"), I(1, 2, "a"), I(2.3, 3, "a")], pad=0.1, merge_gap=0.2, t_min=0, t_max=5.5)
    assert [(round(i.start, 3), round(i.end, 3), i.reason) for i in out] == [(0.9, 3.1, "a"), (4.9, 5.5, "b")]


def test_reencode_is_frame_exact(data_dirs, tmp_path):
    src = _src(data_dirs)
    before = (src.stat().st_size, src.stat().st_mtime_ns)
    iv = _intervals(tmp_path, [(3.0, 4.5, "whale")], [(0.0, 3.0, "air")])
    res = clips.run(iv, mode="reencode", read_clock=False, log=lambda *a: None)
    row = res["clips"][0]
    assert row["status"] == "ok" and row["start_frame"] == 90 and row["end_frame"] == 135
    fr = list(video.iter_frames(row["clip_file"]))
    assert len(fr) == 45
    assert [decode_index(f.image) for f in fr] == list(range(90, 135))
    assert (src.stat().st_size, src.stat().st_mtime_ns) == before
    idx = _read_csv(data_dirs["cache"] / "clips" / "clips_index.csv")
    assert idx[0]["reason"] == "whale" and idx[0]["mode"] == "reencode"
    dropped = _read_csv(data_dirs["cache"] / "clips" / "dropped.csv")
    assert dropped[0]["reason"] == "air"
    assert video.probe(row["clip_file"])["has_audio"]


def test_copy_starts_on_keyframe_before(data_dirs, tmp_path):
    _src(data_dirs)
    iv = _intervals(tmp_path, [(3.0, 4.5, "whale")])
    res = clips.run(iv, mode="copy", read_clock=False, log=lambda *a: None)
    row = res["clips"][0]
    assert row["status"] == "ok", row["status"]
    assert row["start_frame"] == 60 and abs(row["start_s"] - 2.0) < 1e-3  # keyframe at 2 s
    fr = list(video.iter_frames(row["clip_file"]))
    codes = [decode_index(f.image) for f in fr]
    assert codes[0] == 60 and codes == list(range(60, 60 + len(codes)))
    assert codes[-1] >= 134  # nothing wanted is lost
    assert row["end_frame"] == 60 + len(codes) and row["verified_frames"] == len(codes)


def test_auto_mode_and_merge(data_dirs, tmp_path):
    _src(data_dirs)
    iv = _intervals(tmp_path, [(2.1, 2.5, "near keyframe"), (2.6, 3.0, "close"), (5.0, 6.0, "far from keyframe")])
    res = clips.run(iv, mode="auto", max_lead=0.5, merge_gap=0.2, read_clock=False, log=lambda *a: None)
    modes = [(r["mode"], r["start_frame"]) for r in res["clips"]]
    assert modes == [("copy", 60), ("reencode", 150)]
    assert res["clips"][0]["reason"] == "near keyframe; close"


def test_dry_run_writes_nothing(data_dirs, tmp_path):
    _src(data_dirs)
    iv = _intervals(tmp_path, [(3.0, 4.5, "whale")], [(0, 3, "air")])
    res = clips.run(iv, mode="auto", dry_run=True, read_clock=False, log=lambda *a: None)
    assert res["clips"][0]["status"] == "dry run"
    assert not (data_dirs["cache"] / "clips").exists()


def test_outside_interval_is_reported(data_dirs, tmp_path):
    _src(data_dirs)
    iv = _intervals(tmp_path, [(50.0, 60.0, "too late")])
    res = clips.run(iv, read_clock=False, log=lambda *a: None)
    assert not res["clips"] and "NOT CUT" in res["dropped"][0]["reason"]


def test_refuses_writing_into_footage(data_dirs, tmp_path):
    _src(data_dirs)
    iv = _intervals(tmp_path, [(3.0, 4.5, "whale")])
    with pytest.raises(PermissionError):
        clips.run(iv, out_dir=data_dirs["footage"] / "clips", read_clock=False, log=lambda *a: None)
