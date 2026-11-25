"""uwid.export on a small synthetic video and a hand-made tracker run (no GPU, no scan): the track document's
fields, polygons in 0..1 that rasterise back to the masks, and a clip whose decoded frames match the document."""

from __future__ import annotations

import json

import numpy as np

from conftest import make_video
from uwid import export as E
from uwid import tracks
from uwid import trackstage as T


def _run(tmp_path, video: str, f0: int, n: int):
    wr = tracks._Writer(n, size=(128, 72))
    for i in range(n):
        back = np.zeros((72, 128), bool)
        back[54:, :] = True
        wr.add(i, f0 + i, (f0 + i) / 30, tracks.BACK_ID, back, 1.0)
        m = np.zeros((72, 128), bool)
        m[10:40, 10 + i:50 + i] = True
        m[12:20, 80:90] = True                     # a second piece of the same outline
        wr.add(i, f0 + i, (f0 + i) / 30, 1, m, 3.0)
    d = tmp_path / "run"
    meta = dict(video=video, t0=f0 / 30, t1=(f0 + n - 1) / 30, method="online_sam3", label="t", frames=[f0, f0 + n],
                n_frames=n, fps=30.0, store_size=[128, 72], done=True, blocked=[[f0 + 5, f0 + 7]],
                endings=[], refused=[dict(i=3, frame=f0 + 3, t=(f0 + 3) / 30, reason="box classifier: fish",
                                          box=[100.0, 100.0, 300.0, 200.0])])
    wr.save(d, meta)
    return d


def test_track_json_and_clip(tmp_path):
    v = str(make_video(tmp_path / "s.mp4", n=60, fps=30, gop=15))
    f0, n = 6, 24
    d = _run(tmp_path, v, f0, n)
    tr = T.load_online(d)
    tracks.link_lone(tr)
    doc = E.track_json(tr, "hq", "DepX", "s", (0.2, 1.0), "c.mp4", "c.jpg")
    for k in ("format", "version", "group", "frames", "clip", "numbers", "back", "blocked", "endings", "links",
              "refused_starts", "n_frames", "fps"):
        assert k in doc
    assert doc["n_frames"] == n and len(doc["frames"]["index"]) == n and doc["frames"]["index"][0] == f0
    assert doc["blocked"][0]["j0"] == 5 and doc["blocked"][0]["j1"] == 7
    assert doc["refused_starts"][0]["box"] == [round(100 / 1920, 4), round(100 / 1080, 4), round(300 / 1920, 4),
                                               round(200 / 1080, 4)]
    nb = doc["numbers"][0]
    assert nb["number"] == 1 and len(nb["frames"]) == n
    fr = nb["frames"][4]
    assert len(fr["polygons"]) == 2                      # both pieces
    for p in fr["polygons"]:
        assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in p)
    got = E.rasterize(fr["polygons"], 128, 72)
    want = tr.mask(1, 4)
    iou = (got & want).sum() / (got | want).sum()
    assert iou > 0.9
    assert doc["back"] and doc["back"][0]["j"] == 0 and doc["back"][1]["j"] == E.BACK_EVERY
    json.dumps(E._js(doc))
    mp4, jpg = tmp_path / "c.mp4", tmp_path / "c.jpg"
    res = E.write_clip(tr, mp4, jpg, best_j=3)
    k, size = E.count_frames(mp4)
    assert res["padded"] == 0 and k == n and size == (E.CLIP_W, E.CLIP_H) and jpg.exists()
