import numpy as np
import pandas as pd

from uwid import eval_presence as ep


def _labels(tmp_path):
    rows = [  # fps 30: window k = frames 30k .. 30k+29
        dict(frame=0, state="air", other_whales=0, faint_whales=0, fish="none", seafloor="none", back="none", view="clear"),
        dict(frame=15, state="underwater", other_whales=0, faint_whales=0, fish="none", seafloor="none", back="back", view="clear"),
        dict(frame=30, state="underwater", other_whales=2, faint_whales=0, fish="few", seafloor="none", back="back", view="bubbles"),
        dict(frame=45, state="underwater", other_whales=0, faint_whales=0, fish="none", seafloor="visible", back="back", view="clear"),
        dict(frame=60, state="underwater", other_whales=0, faint_whales=1, fish="none", seafloor="unsure", back="back", view="clear"),
        dict(frame=75, state="underwater", other_whales=0, faint_whales=0, fish="none", seafloor="none", back="back", view="clear"),
    ]
    p = tmp_path / "frames.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return ep.load_frame_labels(p)


def test_window_truth(tmp_path):
    t = ep.window_truth(_labels(tmp_path), fps=30.0).set_index("window")
    assert list(t.index) == [0, 1, 2]
    assert t.loc[0, "state"] == "crossing" and t.loc[1, "state"] == "underwater"
    assert t.loc[1, "whale"] and t.loc[1, "fish"] and t.loc[1, "seafloor"] and t.loc[1, "bubbles"]
    assert t.loc[0, "whale_free"] and bool(t.loc[0, "back"])  # back on f15 only: any frame counts
    assert t.loc[2, "faint_only"] and not t.loc[2, "whale_free"] and t.loc[2, "seafloor_unsure"]


def test_window_scores_rules():
    fs = pd.DataFrame(dict(frame=[0, 10, 20, 30, 40, 50], score=[0.9, 0.1, 0.0, 0.5, 0.6, 0.7]))
    mx = ep.window_scores(fs, 30.0, rule="max").set_index("window").score
    k2 = ep.window_scores(fs, 30.0, rule="kth", k=2).set_index("window").score
    nb = ep.window_scores(fs, 30.0, rule="max", neighbours=1).set_index("window").score
    assert mx[0] == 0.9 and k2[0] == 0.1 and k2[1] == 0.6 and nb[1] == 0.9


def test_roc_and_operating_point():
    c = ep.roc([0.9, 0.8, 0.3, 0.2, np.nan], [True, True, False, True, False])
    assert c.iloc[-1].recall == 1.0 and c.iloc[0].recall == 0.0
    o = ep.operating_point(c, max_far=0.0)
    assert o["recall"] == 2 / 3 and o["fp"] == 0
    assert 0.5 < ep.auc(c) <= 1.0
    s = ep.summarize(pd.DataFrame(dict(score=[0.9, 0.1, 0.5], whale=[True, True, False],
                                       whale_free=[False, False, True])), 0.4)
    assert (s["hits"], s["false_alarms"]) == (1, 1)
