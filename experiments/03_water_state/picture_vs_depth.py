"""What the picture adds when a sensor log exists: picture events against the raw depth crossings.

For each method's events (window step 'chain', run.py predictions) matched within 1.5 s to the RAW depth crossings
(no lens lag): picture minus depth, per exit and entry. On 2025-02-03 the depth crossings are surfacings.csv
start_t / end_t (sensor time -> video time with the burned-clock fit). On Zenodo they are the port exposures of
zenodo_truth.py with no lag and the file-name time base, so the median offset is an automatic video-to-sensor sync
estimate from the picture.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402
from uwid import tagdata  # noqa: E402
from uwid import waterstate as ws  # noqa: E402

METHODS = ["color", "clip0", "clipLR", "dino"]


def raw_events(name):
    if name.startswith("CC-11-96"):
        vc = tagdata.video_clock(run.DEP, name + ".mov")
        s = pd.read_csv(tagdata.cache_dir(run.DEP) / "surfacings.csv")
        ex = pd.DataFrame({"t": vc.to_video(s.start_t.values), "kind": "exit"})
        en = pd.DataFrame({"t": vc.to_video(s.end_t.values), "kind": "entry"})
    else:
        e = pd.read_csv(run.ZTRUTH / f"{name}_port_exposures.csv")
        ex = pd.DataFrame({"t": e.port_out_t.values, "kind": "exit"})
        en = pd.DataFrame({"t": e.port_in_t.values, "kind": "entry"})
    return pd.concat([ex, en], ignore_index=True)


def offsets(name, method, kind_step="chain", tol=1.5):
    p = np.load(run.PRED / f"{method}__{name}.npz")
    ev = ws.events_from_binary(p["t"], p[f"air_{kind_step}"])
    raw = raw_events(name)
    rows = []
    for r in raw.itertuples():
        c = ev[(ev.kind == r.kind) & ((ev.t - r.t).abs() <= tol)]
        if len(c):
            j = (c.t - r.t).abs().idxmin()
            rows.append((name, method, r.kind, r.t, c.t[j] - r.t))
    return rows



def to_md(df, index=False):
    """Markdown table without the optional tabulate package."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in d.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)

def main():
    rows = []
    sets = {"check 2025-02-03": run.CHECK, "Zenodo": run.ZEN}
    for sname, names in sets.items():
        for n in names:
            if not (run.PRED / f"dino__{n}.npz").exists():
                continue
            for m in METHODS:
                rows += [(sname,) + r for r in offsets(n, m)]
    df = pd.DataFrame(rows, columns=["set", "video", "method", "kind", "depth_t", "pic_minus_depth"])
    df.to_csv(run.C / "picture_vs_depth.csv", index=False)
    g = df.groupby(["set", "method", "kind"]).pic_minus_depth
    out = pd.DataFrame({"n": g.size(), "median": g.median(), "q10": g.quantile(0.1), "q90": g.quantile(0.9),
                        "IQR": g.quantile(0.75) - g.quantile(0.25)}).round(2)
    print(to_md(out, index=True))


if __name__ == "__main__":
    main()
