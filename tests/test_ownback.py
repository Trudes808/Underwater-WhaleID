"""uwid.ownback (WP14 / WP17): the own-back template, its block frames and cache, and the tracking flag
``own_back_template`` in the clean-up (a run folder with ``own_back.npz``); flag off = unchanged output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uwid import ownback as O
from uwid import tracks


def maps_with_back(n=20, rows=(38, 45), extra=None):
    m = np.zeros((n, 45, 80), np.uint8)
    m[:, rows[0]:rows[1], :] = O.BACK
    if extra is not None:                       # something called back in a few frames only
        k, (y0, y1, x0, x1) = extra
        m[:k, y0:y1, x0:x1] = O.BACK
    return m


def test_template_keeps_steady_back_drops_passing_things():
    t = O.back_template(maps_with_back(extra=(3, (30, 38, 30, 40))), tau=0.5, dilate=0)
    assert t[40:].all() and not t[:38].any()          # steady back kept, 3-of-20 blob dropped
    t1 = O.back_template(maps_with_back(), tau=0.5, dilate=1)
    assert t1[37].all() and not t1[36].any()          # grown by one patch
    with pytest.raises(ValueError):
        O.back_template(np.zeros((45, 80)))


def test_upsample_own_back_and_flag_off_identity():
    t = np.zeros((45, 80), bool); t[40:] = True
    px = O.upsample(t, (720, 1280))
    assert px.shape == (720, 1280) and px[650:].all() and not px[:630].any()
    sam = np.zeros((720, 1280), bool); sam[560:] = True          # the back object swallowed a calf above the back
    ob = O.own_back(sam, px)
    calf = np.zeros_like(sam); calf[560:640, 400:700] = True
    assert O.share_inside(calf, sam) == 1.0 and O.share_inside(calf, ob) == 0.0
    assert O.own_back(sam, None) is sam and O.own_back(None, px) is None   # flag off: the very same mask
    assert O.box_on_back([0, 660, 400, 720], px) and not O.box_on_back([0, 400, 400, 720], px)


def test_template_frames_and_block_cache(tmp_path):
    w = pd.DataFrame(dict(t0=np.arange(0, 700.0), water=["underwater"] * 700, tag_state=["whale"] * 700,
                          scanned=[True] * 700, frame_0=np.arange(0, 700) * 30 + 7))
    w.loc[310:320, "water"] = "air"
    fr = O.template_frames(w, O.block_of(450.0))
    assert len(fr) == 29 and all(300 * 30 <= f < 600 * 30 + 30 for f in fr)
    calls = []

    def fake_maps(video, frames):
        calls.append(len(frames))
        return maps_with_back(len(frames)), frames
    tm, info = O.block_template("v.mov", w, 450.0, tmp_path, maps_fn=fake_maps)
    assert tm[40:].all() and info["frames"] == 29 and info["block"] == 1
    tm2, _ = O.block_template("v.mov", w, 451.0, tmp_path, maps_fn=fake_maps)
    assert calls == [29] and (tm2 == tm).all()          # second call read the block cache
    none, info = O.block_template("v.mov", w.iloc[:3], 1.0, tmp_path / "x", maps_fn=fake_maps)
    assert none is None and "fewer" in info["why_none"]


def _run(tmp, name, template=None):
    """A run where a calf (object 1) sits inside the back object above the usual back line."""
    n = 30
    meta = dict(video="x", t0=0.0, t1=1.0, method="online_sam3", label="t", frames=[0, n], n_frames=n, fps=30.0,
                store_size=[128, 72], done=True)
    wr = tracks._Writer(n, size=(128, 72))
    for i in range(n):
        back = np.zeros((72, 128), bool); back[50:] = True       # back object swallowed the calf (rows 50-60)
        wr.add(i, i, i / 30, tracks.BACK_ID, back, 5.0)
        calf = np.zeros((72, 128), bool); calf[50:58, 30:90] = True
        wr.add(i, i, i / 30, 1, calf, 3.0)
        own = np.zeros((72, 128), bool); own[64:70, 20:60] = True   # a patch of the mother's own back
        wr.add(i, i, i / 30, 2, own, 2.0)
    d = tmp / name
    wr.save(d, meta)
    if template is not None:
        O.save_run_template(d, template, {"test": True})
    return d


def test_clean_with_and_without_template(tmp_path):
    off = tracks.load_tracks(_run(tmp_path, "off"))
    assert off.own_back is None and off.roles[1] == "on_back" and off.roles[2] == "on_back"
    tm = np.zeros((45, 80), bool); tm[37:] = True                 # the usual back: rows >= 59 of 72
    on = tracks.load_tracks(_run(tmp_path, "on", tm))
    assert on.own_back is not None
    assert on.roles[1] in ("whale", "small") and on.roles[2] == "on_back"   # calf freed, own body stays own back
    full = tracks.load_tracks(_run(tmp_path, "full", np.ones((45, 80), bool)))
    assert full.roles == off.roles and full.back_share == off.back_share   # template covering all = flag off


# ------------------------------------------------------------------------------------------------ WP18

def test_default_on_and_opt_out(monkeypatch):
    import inspect
    from uwid import cli, trackstage
    assert inspect.signature(trackstage.run_span).parameters["own_back_template"].default is True
    assert inspect.signature(trackstage.run_deployment).parameters["own_back_template"].default is True
    seen = {}

    def fake_run_deployment(dep, preset, **kw):
        seen.update(kw)
        return []
    monkeypatch.setattr(trackstage, "run_deployment", fake_run_deployment)
    monkeypatch.setattr(trackstage, "write_index", lambda dep: "index.csv")
    cli.main(["tracks", "DepX"])
    assert seen["own_back_template"] is True
    cli.main(["tracks", "DepX", "--no-own-back-template"])
    assert seen["own_back_template"] is False


def test_despeckle_drops_far_small_keeps_large_or_close():
    m = np.zeros((720, 1280), bool)
    m[300:500, 300:700] = True                  # main piece, 80,000 px
    m[100:110, 1100:1110] = True                # far speck (100 px, ~400 px away): dropped
    m[300:320, 705:720] = True                  # small but close (gap 5 px): kept
    out, info = tracks.despeckle(m)
    assert info["pieces"] == 3 and info["dropped"] == 1 and info["dropped_px"] == 100
    assert not out[100:110, 1100:1110].any() and out[300:320, 705:720].all() and out[300:500, 300:700].all()
    cut = np.zeros((720, 1280), bool)
    cut[300:500, 100:500] = True                # a whale cut in two by the back / a flipper / the frame edge
    cut[300:500, 700:900] = True                # second half, far (200 px) but large (1/3): kept
    out2, info2 = tracks.despeckle(cut)
    assert info2["dropped"] == 0 and out2.sum() == cut.sum()
    assert tracks.despeckle(np.zeros((10, 10), bool))[1]["pieces"] == 0


def test_mask_applies_speck_rule_numbers_unchanged(tmp_path):
    n = 10
    meta = dict(video="x", t0=0.0, t1=1.0, method="online_sam3", label="t", frames=[0, n], n_frames=n, fps=30.0,
                store_size=[128, 72], done=True)
    wr = tracks._Writer(n, size=(128, 72))
    for i in range(n):
        back = np.zeros((72, 128), bool); back[64:] = True
        wr.add(i, i, i / 30, tracks.BACK_ID, back, 5.0)
        w = np.zeros((72, 128), bool); w[20:50, 20:80] = True; w[2:4, 120:122] = True   # whale + far speck
        wr.add(i, i, i / 30, 1, w, 3.0)
    wr.save(tmp_path / "r", meta)
    from uwid import trackstage
    raw = trackstage.finish(tracks.read_run(tmp_path / "r"))
    tr = trackstage.load_online(tmp_path / "r")
    assert tr.despeck and tr.number == raw.number and tr.roles == raw.roles
    assert tr.raw_mask(1, 0)[2:4, 120:122].all() and not tr.mask(1, 0)[2:4, 120:122].any()
    assert tr.mask(1, 0)[20:50, 20:80].all() and tr.mask(tracks.BACK_ID, 0).sum() == raw.mask(tracks.BACK_ID, 0).sum()
