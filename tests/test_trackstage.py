"""uwid.trackstage without a GPU: scan reading on a made-up scan folder, sensor lens-out mapping, the blocked-view
rule, the start screen, and removing objects from a tracker session."""

from __future__ import annotations

import json
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd

from conftest import make_video
from uwid import trackstage as T


def _fake_scan(root, video, dep="DepX"):
    sd = root / "scan" / dep / "sensitive"
    (sd / "windows").mkdir(parents=True)
    iv = {"format": "uwid-intervals/1", "config": {"threshold": 0.25, "min_whale_px": 300.0},
          "videos": [{"source": str(video), "duration": 3.0, "hq": [],
                      "keep": [{"start": 0.0, "end": 3.0, "advanced_ok": True, "hq_spans": [[1.0, 2.0]]},
                               {"start": 0.0, "end": 3.0, "advanced_ok": False, "hq_spans": [[0.0, 1.0]]}]}]}
    (sd / "intervals.json").write_text(json.dumps(iv))
    w = pd.DataFrame(dict(t0=[0.0, 1.0, 2.0], t1=[1.0, 2.0, 3.0], sensor_t0=[100.0, 101.0, 102.0],
                          depth_m=[5.0, 6.0, 7.0], frame_0=[7, 37, 67], frame_1=[22, 52, 82]))
    w.to_csv(sd / "windows" / f"{video.name}.csv", index=False)
    td = root / "tagdata" / dep
    td.mkdir(parents=True)
    pd.DataFrame(dict(start_t=[101.0], end_t=[102.5])).to_csv(td / "surfacings.csv", index=False)
    det = root / "scan" / dep / "cache" / "det" / video.name
    det.mkdir(parents=True)
    (det / "meta.json").write_text(json.dumps({"decode_size": [320, 240]}))
    pd.DataFrame(dict(frame=[37, 37], t=[1.23, 1.23], query=["whale", "whale"], score=[0.8, 0.1],
                      x0=[10.0, 10.0], y0=[10.0, 10.0], x1=[200.0, 30.0], y1=[150.0, 30.0])
                 ).to_parquet(det / "part_0.parquet")
    np.save(det / "part_0.frames.npy", np.array([7, 22, 37, 52]))
    return dep


def test_scan_reading(tmp_path, data_dirs):
    video = make_video(data_dirs["footage"] / "rec.mov", n=90)
    dep = _fake_scan(data_dirs["cache"], video)
    h = T.hq_spans(dep)
    assert len(h) == 1 and h.iloc[0].start == 1.0 and h.iloc[0].depth_med == 6.0
    assert bool(h.iloc[0].lens_out)                    # surfacing 101-102.5 s sensor = 1-2.5 s video
    lo = T.lens_out_spans(dep, video.name)
    assert np.allclose(lo[0], (1.3, 2.1))              # + 0.3 s lag, - 0.4 s lead
    s = T.scan_starts(dep, video.name, 0, 90)
    # only the box that passes threshold and size: 190 px at 320 wide = 1140 px at 1920 >= 300
    assert len(s) == 1 and s.iloc[0].frame == 37 and abs(s.iloc[0].x1 - 1200.0) < 1e-6


def test_blocked_rule_and_start_screen():
    fps = 30.0
    n = 90
    frames = np.arange(0, n, 3)
    labels = np.zeros((len(frames), 45, 80), np.uint8)
    labels[(frames >= 30) & (frames < 45)] = T.BUBBLES          # 0.5 s of bubbles
    labels[(frames >= 60) & (frames < 63), :, :] = T.BUBBLES    # one sampled frame: too short to count
    times = np.arange(n) / fps
    m, info = T.blocked_frames(frames, labels, 0, n, fps, times, lens_out=[])
    blocked = np.nonzero(m)[0]
    assert blocked.min() >= 28 and blocked.max() <= 46 and not m[60:64].any()
    m2, _ = T.blocked_frames(frames, labels, 0, n, fps, times, lens_out=[(2.5, 2.6)])
    assert m2[int(2.55 * fps)]
    lab = np.zeros((45, 80), np.uint8)
    lab[:, :40] = T.SURFACE
    assert T.clutter_share(lab, [0, 0, 960, 1080]) == 1.0
    assert T.clutter_share(lab, [960, 0, 1920, 1080]) == 0.0


def test_remove_objects():
    sess = SimpleNamespace(_obj_id_to_idx=OrderedDict([(0, 0), (5, 1), (7, 2)]),
                           _obj_idx_to_id=OrderedDict([(0, 0), (1, 5), (2, 7)]), obj_ids=[0, 5, 7],
                           point_inputs_per_obj={0: "a", 1: "b", 2: "c"}, mask_inputs_per_obj={0: 1, 1: 2, 2: 3},
                           output_dict_per_obj={0: "o0", 1: "o5", 2: "o7"}, frames_tracked_per_obj={0: {}, 1: {}, 2: {}},
                           obj_with_new_inputs=[5])
    T.remove_objects(sess, [5])
    assert sess.obj_ids == [0, 7] and sess._obj_id_to_idx[7] == 1
    assert sess.output_dict_per_obj == {0: "o0", 1: "o7"} and sess.obj_with_new_inputs == []


def test_lens_out_needs_the_picture():
    # WP11a follow-up A: a depth lens-out blocks a frame only where the picture water model is not "underwater"
    fps, n = 30.0, 90
    frames = np.arange(0, n, 3)
    labels = np.zeros((len(frames), 45, 80), np.uint8)
    times = np.arange(n) / fps
    pic = times >= 2.0                                         # picture: under water until 2 s, then air
    m, info = T.blocked_frames(frames, labels, 0, n, fps, times, lens_out=[(1.0, 2.9)], picture_not_under=pic)
    assert not m[: int(1.9 * fps)].any() and m[int(2.5 * fps)]
    assert "picture agrees" in info["lens_source"]


def test_frame_store_gives_the_decoded_frames(tmp_path):
    # WP13 flag: frames read from a store equal the decoded ones (index, time, every pixel); without a store (or
    # with an incomplete one) the old decode path is used
    v = str(make_video(tmp_path / "s.mp4", n=60, fps=30, gop=15))
    f0, f1 = 7, 41
    direct = list(T.span_frames(v, f0, f1, None))[: f1 - f0]
    assert T.stored(tmp_path / "store", v, f0, f1) is None
    d = T.store_frames(v, f0, f1, tmp_path / "store")
    assert T.stored(tmp_path / "store", v, f0, f1) == d
    fromstore = list(T.span_frames(v, f0, f1, tmp_path / "store"))
    assert len(fromstore) == len(direct) == f1 - f0
    for a, b in zip(direct, fromstore):
        assert a.index == b.index and abs(a.t - b.t) < 1e-9 and np.array_equal(a.image, b.image)
    meta = json.loads((d / "index.json").read_text())
    meta["complete"] = False
    (d / "index.json").write_text(json.dumps(meta))
    assert T.stored(tmp_path / "store", v, f0, f1) is None          # incomplete store: decode as before


def test_scan_key_for_names_with_spaces():
    # WP19: the scan stores "2025-12-20 CG.m4v" as "2025-12-20_CG.m4v"; names without spaces are unchanged
    assert T.scan_key("2025-12-20 CG.m4v") == "2025-12-20_CG.m4v"
    assert T.scan_key("CC-11-96-20250203-104242-000-00005.mov") == "CC-11-96-20250203-104242-000-00005.mov"
