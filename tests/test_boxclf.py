"""uwid.boxclf (whale / fish / other per detector box) and the window classes of uwid.patchseg (seafloor, sand,
dark), on synthetic boxes, frames and a small synthetic video; a tiny stand-in backbone replaces DINOv3."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from uwid import boxclf as B
from uwid import patchseg as ps


class TinyBackbone(nn.Module):
    """get_intermediate_layers like DINOv3: patch features = mean colour of each 16 px patch (3 channels), tiled to
    ``dim``; class token = mean colour."""

    def __init__(self, dim: int = 6):
        super().__init__()
        self.dim = dim
        self.embed_dim = dim
        self.w = nn.Parameter(torch.ones(1))

    def get_intermediate_layers(self, x, n=1, reshape=False, return_class_token=False, norm=True):
        p = nn.functional.avg_pool2d(x.float(), 16)                     # B, 3, h, w
        f = p.repeat(1, self.dim // 3, 1, 1)                             # B, dim, h, w
        if reshape:
            outs = [f]
        else:
            outs = [f.flatten(2).transpose(1, 2)]                        # B, N, dim
        if return_class_token:
            cls = f.mean((2, 3))
            return tuple((o, cls) for o in outs)
        return tuple(outs)


def scan_rows(frame=0):
    """Scan-cache rows: one box seen by two prompts (same coordinates) and a second box."""
    base = dict(frame=frame, t=0.0, score__whale=0.7, score__fish=0.3, score__school_of_fish=0.1,
                score__small_fish=0.2, score__air_bubbles=0.01)
    r = [dict(base, query="whale", score=0.7, x0=100, y0=100, x1=500, y1=300),
         dict(base, query="fish", score=0.3, x0=100, y0=100, x1=500, y1=300),
         dict(base, query="fish", score=0.6, x0=110, y0=105, x1=505, y1=310, score__whale=0.2, score__fish=0.6),
         dict(base, query="fish", score=0.55, x0=1500, y0=800, x1=1560, y1=830, score__whale=0.05, score__fish=0.55),
         dict(base, query="fish", score=0.1, x0=10, y0=10, x1=20, y1=20, score__whale=0.02, score__fish=0.1,
              score__small_fish=0.05)]
    return pd.DataFrame(r)


def test_unique_and_group():
    u = B.unique_boxes(scan_rows())
    assert len(u) == 3                       # the 0.1 box is dropped, the two-prompt box is one patch box
    two = u[(u.x0 == 100)].iloc[0]
    assert two.queries == ["fish", "whale"] and len(two.rows) == 2
    assert two.s_fish == pytest.approx(0.3) and two.s_whale == pytest.approx(0.7)
    g = B.group_objects(u)
    assert g.obj.nunique() == 2              # the two overlapping boxes are one object
    assert g.rep.sum() == 2
    top = g[(g.obj == g.loc[g.x0 == 100, "obj"].iloc[0]) & g.rep].iloc[0]
    assert top.x0 == 100                      # best score (whale 0.7) represents the object


def test_geometry_and_scores():
    xy = np.array([[0, 0, 960, 540], [100, 100, 200, 150]], float)
    g = B.geometry(xy, 1920, 1080)
    assert g.shape == (2, len(B.GEOM_NAMES))
    assert np.exp(g[0, 0]) == pytest.approx(960) and g[0, 7] == 1 and g[1, 7] == 0
    g4 = B.geometry(xy * 2, 3840, 2160)       # pixels are in 1920 px units whatever the frame size
    assert np.allclose(g4[:, :4], g[:, :4])
    s = B.score_features(B.unique_boxes(scan_rows()))
    assert s.shape[1] == len(B.SCORE_NAMES) and np.isfinite(s).all()


def test_crop_embeddings_shapes_and_padding():
    img = np.zeros((1080, 1920, 3), np.uint8); img[:, :960] = 200
    m = TinyBackbone(6)
    e = B.crop_embeddings(m, img, np.array([[900, 500, 1020, 580], [-50, -50, 50, 50]], float), (1.0, 2.0))
    assert e[1.0].shape == (2, 12) and e[2.0].shape == (2, 12)
    assert np.isfinite(e[1.0]).all()


def _toy_model(blocks, dim):
    rng = np.random.default_rng(0)
    return B.LinearBoxModel(np.zeros(dim), np.ones(dim), rng.normal(size=(3, dim)), np.zeros(3), B.FeatureSpec(blocks))


def test_model_save_load_and_classify(tmp_path):
    dim = len(B.GEOM_NAMES) + len(B.SCORE_NAMES)
    mdl = _toy_model(("geom", "scores"), dim)
    p = tmp_path / "m.npz"
    mdl.save(p)
    m2 = B.LinearBoxModel.load(p)
    X = np.random.default_rng(1).normal(size=(4, dim))
    assert np.allclose(mdl.proba(X), m2.proba(X)) and np.allclose(m2.proba(X).sum(1), 1)
    rows = scan_rows()
    out = B.classify(rows, np.zeros((1080, 1920, 3), np.uint8), clf=p)
    assert list(out.index) == list(rows.index)
    assert out.loc[4, "boxclf_class"] == ""                         # below the score floor
    labelled = out[out.boxclf_class != ""]
    assert set(labelled.boxclf_class) <= set(B.CLASSES)
    assert np.allclose(labelled[["p_whale", "p_fish", "p_other"]].sum(1), 1)
    # rows of one object share the answer
    obj = labelled.groupby("boxclf_obj").p_whale.nunique()
    assert (obj == 1).all() and len(obj) == 2


def test_classify_with_image_features(tmp_path):
    m = TinyBackbone(6)
    dim = 12 + 12 + len(B.GEOM_NAMES) + len(B.SCORE_NAMES)
    mdl = _toy_model(("tight", "ctx", "geom", "scores"), dim)
    out = B.classify(scan_rows(), np.full((1080, 1920, 3), 90, np.uint8), model=m, clf=mdl)
    assert (out.boxclf_class != "").sum() == 4


def test_from_sklearn_roundtrip():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4)); y = np.repeat([0, 1, 2], 30); X[y == 1] += 2; X[y == 2] -= 2
    s = StandardScaler().fit(X); lr = LogisticRegression().fit(s.transform(X), y)
    m = B.LinearBoxModel.from_sklearn(s, lr, B.FeatureSpec(("geom",)))
    assert np.allclose(m.proba(X), lr.predict_proba(s.transform(X)), atol=1e-6)


# ------------------------------------------------------------------------------------------------ patchseg windows

def _frame(maps_code: int, grey: float, emb=None):
    m = np.full((45, 80), ps.WATER, np.uint8)
    m[30:, :] = maps_code
    return {"maps": {"h": m}, "grey_mean": grey, "emb": emb if emb is not None else np.zeros(4)}


def test_window_decisions_rules():
    sea = [_frame(ps.SEAFLOOR, 80), _frame(ps.SEAFLOOR, 90)]
    r = ps.window_decisions(sea, "h", rules={"seafloor_share": 0.2})
    assert r["seafloor"] and not r["dark"] and r["share_seafloor"] == pytest.approx(15 / 45)
    back = [_frame(ps.BACK, 80), _frame(ps.BACK, 80)]
    assert not ps.window_decisions(back, "h")["seafloor"]
    dark = [_frame(ps.SEAFLOOR, 5), _frame(ps.SEAFLOOR, 12)]
    r = ps.window_decisions(dark, "h")
    assert r["dark"] and not r["seafloor"]                  # a black window is dark, not seafloor
    half = [_frame(ps.SEAFLOOR, 5), _frame(ps.SEAFLOOR, 60)]
    assert not ps.window_decisions(half, "h")["dark"]       # dark only when every frame is dark


def test_sand_model(tmp_path):
    m = ps.SandModel(np.zeros(4), np.ones(4), np.array([1.0, 0, 0, 0]), 0.0, {"x": 1})
    m.save(tmp_path / "s.npz")
    m2 = ps.SandModel.load(tmp_path / "s.npz")
    assert m2.meta == {"x": 1}
    fr = [_frame(ps.WATER, 80, np.array([3.0, 0, 0, 0])), _frame(ps.WATER, 80, np.array([3.0, 0, 0, 0]))]
    r = ps.window_decisions(fr, "h", sand=m2)
    assert r["sand"] and r["sand_p"] > 0.9


def test_brightness_and_embedding():
    img = np.zeros((720, 1280, 3), np.uint8); img[:360] = 255
    b = ps.brightness_stats(img)
    assert 120 < b["grey_mean"] < 135 and b["grey_p95"] == pytest.approx(255, abs=1)
    e = ps.frame_embedding(torch.ones(45, 80, 6))
    assert e.shape == (30,)


def test_window_classes_on_synthetic_video(tmp_path):
    from conftest import make_video
    v = make_video(tmp_path / "v.mp4", n=90, fps=30)
    clf = ps.PatchClassifier(6, len(ps.CLASSES))
    with torch.no_grad():
        clf.net[-1].weight.zero_(); clf.net[-1].bias.zero_(); clf.net[-1].bias[ps.WATER] = 5.0
    path = tmp_path / "head.pt"
    ps.save_classifier(clf, path, {})
    df = ps.window_classes(v, model=TinyBackbone(6), seafloor_clf=path, sand_clf=None)
    assert list(df.t0) == [0.0, 1.0, 2.0]
    assert (df.n_frames == 2).all() and not df.seafloor.any()
    assert df.share_water.min() == pytest.approx(1.0)
    assert df.attrs["video_s"] == pytest.approx(3.0, abs=0.05)


def test_decode_child_failure_raises_not_hangs(tmp_path):
    bad = tmp_path / "missing.mp4"
    with pytest.raises(RuntimeError):
        list(ps._iter_decoded(bad, [0, 1], (64, 36), stall_s=20, log=lambda *_: None))
