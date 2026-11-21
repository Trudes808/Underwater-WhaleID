import shutil
import subprocess

import numpy as np
import pytest

from conftest import decode_index, make_video
from uwid import audio, video


def test_frame_table_and_sequential(tmp_path):
    p = make_video(tmp_path / "a.mp4", n=90, fps=30, gop=30)
    tab = video.frame_table(p)
    assert tab.n_frames == 90 and tab.n_hidden == 0
    assert np.allclose(tab.keyframe_times, [0.0, 1.0, 2.0], atol=1e-3)
    rs = tab.rate_summary()
    assert rs["constant"] and abs(rs["mean_fps"] - 30) < 1e-3
    frames = list(video.iter_frames(p))
    assert [f.index for f in frames] == list(range(90))
    assert [decode_index(f.image) for f in frames] == list(range(90))
    assert np.allclose([f.t for f in frames], np.arange(90) / 30, atol=1e-4)


def test_sampling_by_time(tmp_path):
    p = make_video(tmp_path / "a.mp4", n=90, fps=30, gop=30)
    fr = list(video.iter_frames(p, fps=2.0))
    assert [f.index for f in fr] == [0, 15, 30, 45, 60, 75]
    assert all(decode_index(f.image) == f.index for f in fr)
    fr = list(video.iter_frames(p, fps=4.0, start=1.2, end=2.0))
    assert [f.index for f in fr] == [36, 44, 51, 59]  # nearest frames to 1.2, 1.45, 1.7, 1.95


def test_random_access(tmp_path):
    p = make_video(tmp_path / "a.mp4", n=150, fps=30, gop=60)
    with video.VideoReader(p) as r:
        for t in [3.5, 0.0, 4.9, 1.01, 2.0, 1.99, 10.0, -1.0]:
            f = r.frame_at(t)
            want = int(np.clip(round(t * 30), 0, 149))
            assert f.index == want
            assert decode_index(f.image) == want


def test_probe_and_decode_check(tmp_path):
    p = make_video(tmp_path / "a.mp4", n=60, fps=30, gop=30, audio=True)
    pr = video.probe(p)
    assert pr["has_video"] and pr["has_audio"]
    vs = [s for s in pr["streams"] if s["type"] == "video"][0]
    assert (vs["width"], vs["height"]) == (320, 240)
    dc = video.decode_check(p, grab_times=[0.0, 1.0, 1.95])
    assert dc["n_decoded"] == 60 and dc["n_errors"] == 0
    assert [f.index for f in dc["frames"]] == [0, 30, 58]
    assert all(decode_index(f.image) == f.index for f in dc["frames"])


def test_fragmented_file(tmp_path):
    p = make_video(tmp_path / "frag.mov", n=120, fps=30, gop=30, movflags="frag_keyframe+empty_moov",
                   audio=True, audio_channels=1)
    st = video.mp4_structure(p)
    assert st["fragmented"] and st["n_moof"] >= 3
    tab = video.frame_table(p)
    assert tab.n_frames == 120
    want = tab.index_at(3.0)  # the muxer may leave a doubled first step, so ask the table
    with video.VideoReader(p) as r:
        f = r.frame_at(3.0)
        assert f.index == want and decode_index(f.image) == want and abs(f.t - 3.0) < 0.02


def test_truncated_tail_does_not_raise(tmp_path):
    p = make_video(tmp_path / "a.mp4", n=120, fps=30, gop=30)
    q = tmp_path / "cut.mp4"
    data = p.read_bytes()
    q.write_bytes(data[: int(len(data) * 0.6)])
    tab = video.frame_table(q)  # header still lists all frames
    dc = video.decode_check(q)
    assert 0 < dc["n_decoded"] < 120
    assert list(video.iter_frames(q))  # no exception
    with video.VideoReader(q) as r:
        r.frame_at(3.9)  # may be None, must not raise


def test_edit_list_hidden_frames(tmp_path):
    """A stream copy that starts mid-GOP gets an edit list hiding the pre-roll: those frames are not counted."""
    from uwid.clips import ffmpeg_exe
    p = make_video(tmp_path / "a.mp4", n=120, fps=30, gop=60)
    q = tmp_path / "el.mp4"
    subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-ss", "1.5", "-i", str(p), "-c", "copy", str(q)],
                   check=True)
    tab = video.frame_table(q)
    frames = list(video.iter_frames(q))
    assert tab.n_frames == len(frames)
    if tab.n_hidden == 0:
        pytest.skip("this ffmpeg build wrote no edit list")
    assert tab.n_hidden + tab.n_frames == tab.n_packets
    assert decode_index(frames[0].image) == 45  # first shown frame is the one at 1.5 s
    assert video.probe(q)["edit_lists"]


def test_audio_range(tmp_path):
    p = make_video(tmp_path / "a.mp4", n=90, fps=30, gop=30, audio=True, audio_rate=32000)
    a = audio.read_audio(p, 1.0, 2.0)
    assert a.sample_rate == 32000 and a.channels == 2
    assert abs(len(a.samples) - 32000) <= 1100 and abs(a.t0 - 1.0) < 0.05
    spec = np.abs(np.fft.rfft(a.samples))
    freq = np.fft.rfftfreq(len(a.samples), 1 / 32000)
    assert abs(freq[spec.argmax()] - 440) < 5
    q = make_video(tmp_path / "b.mp4", n=10, fps=30, gop=30)
    assert audio.read_audio(q) is None
