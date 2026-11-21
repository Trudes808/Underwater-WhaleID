"""uwid.tracks without a GPU: stage-1 run finding, box coverage, the stored format, cleaning and the renderer on a
synthetic video with made-up masks."""

from __future__ import annotations

import json

import av
import numpy as np
import pandas as pd

from conftest import make_video
from uwid import tracks


def test_runs_of_bridges_one_second_gaps():
    whale = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1]
    win = pd.DataFrame(dict(window=range(len(whale)), whale=[bool(x) for x in whale], n=1, size=0.3))
    runs = tracks.runs_of(win, min_s=3, gap_s=1)
    assert [(r["t0"], r["t1"]) for r in runs] == [(1, 10)]


def test_box_covered():
    m = np.zeros((72, 128), bool)
    m[10:30, 10:40] = True
    back = np.zeros((72, 128), bool)
    back[54:] = True
    assert tracks.box_covered([8, 8, 42, 32], {1: m}, back)          # track inside the box
    assert not tracks.box_covered([60, 10, 100, 40], {1: m}, back)   # open water
    assert tracks.box_covered([0, 56, 128, 72], {1: m}, back)        # on the back


def _fake_run(tmp, video, n=30):
    """Write a run folder by hand: back = bottom quarter, whale 1 moves right, whale 2 duplicates whale 1 on frames
    10-14, whale 3 sits on the back, whale 4 appears on 2 frames."""
    meta = dict(video=str(video), t0=0.0, t1=n / 30, method="owl_sam3", label="test", frames=[0, n], n_frames=n,
                fps=30.0, store_size=[128, 72], events=[], seconds_total=1.0, seconds_model=1.0, gpu_peak_gb=0.0,
                done=True)
    wr = tracks._Writer(n, size=(128, 72))
    for i in range(n):
        back = np.zeros((72, 128), bool)
        back[54:] = True
        wr.add(i, i, i / 30, tracks.BACK_ID, back, 5.0)
        w1 = np.zeros((72, 128), bool)
        if i < 20 or i >= 25:                           # lost on frames 20-24
            w1[20:40, 10 + i:40 + i] = True
        wr.add(i, i, i / 30, 1, w1, 3.0)
        w2 = np.zeros((72, 128), bool)
        if 10 <= i < 15:
            w2[22:38, 12 + i:38 + i] = True
        wr.add(i, i, i / 30, 2, w2, 2.0)
        w3 = np.zeros((72, 128), bool)
        w3[58:70, 50:80] = True
        wr.add(i, i, i / 30, 3, w3, 2.0)
        w4 = np.zeros((72, 128), bool)
        if i in (3, 4):
            w4[5:10, 100:110] = True
        wr.add(i, i, i / 30, 4, w4, 1.0)
    out = tmp / "run"
    wr.save(out, meta)
    return out


def test_clean_summary_and_render(tmp_path):
    video = make_video(tmp_path / "v.mp4", n=30)
    run = _fake_run(tmp_path, video)
    tr = tracks.load_tracks(run)
    assert tr.roles[3] == "on_back" and tr.roles[4] == "flicker" and tr.roles[tracks.BACK_ID] == "back"
    assert tr.number == {1: 1} and tr.roles[2] == "duplicate"
    assert all((i, 2) in tr.hidden for i in range(10, 15))
    s = tracks.track_summary(tr).set_index("number")
    assert s.loc[1, "frames_seen"] == 25 and s.loc[1, "gaps_s"] == "0.7-0.8"
    res = tracks.render_tracks(tr, tmp_path / "out.mp4", tmp_path / "out.jpg", n_sheet=4)
    with av.open(res["mp4"]) as c:
        assert len(c.streams.audio) == 0
        v = c.streams.video[0]
        assert v.codec_context.name == "h264" and v.format.name == "yuv420p"
        assert sum(1 for _ in c.decode(v)) == 30
    assert (tmp_path / "out.jpg").exists()
    assert json.loads((run / "meta.json").read_text())["done"]
