"""Independent check against WP3's AI by-eye events on the 2025-02-03 recordings (labels/<stem>/events.json).

WP3 labelled two 180 s windows per piece on the .m4v conversions; its event frame is a decoded-frame index. The
.m4v and .mov hold the same frames in the same order (WP0), so the .mov presentation time of an event is
frame_table(mov).times[frame].

Reports, inside the labelled windows only:
1. sensor truth (depth + measured lens lags, uwid.eval_state) against the by-eye events: how exact is the truth;
2. the raw depth crossings (no lag) against the by-eye events;
3. each method's picture events (window step 'chain'; out-of-fold scores on the fit pieces) against the by-eye
   events: found within 0.5 s / 1.0 s, false events, timing error.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402
from uwid import eval_state as es  # noqa: E402
from uwid import tagdata  # noqa: E402
from uwid import video as uv  # noqa: E402
from uwid import waterstate as ws  # noqa: E402

LAB = run.REPO / "experiments/02_reference_labels/labels"
METHODS = ["sat", "color", "clip0", "clipLR", "dino"]


def byeye(name):
    f = LAB / name / "events.json"
    if not f.exists() or not (Path("/home/sat3737/NOAA_whales/underwater_cache/reference") / name / "DONE").exists():
        return None, None
    e = json.loads(f.read_text())
    times = np.asarray(uv.frame_table(run.VDIR / f"{name}.mov").times)
    m4v_times = np.asarray(uv.frame_table(run.VDIR / f"{name}.m4v").times)

    def mov_t(m4v_s):  # window bounds are m4v seconds: go through the frame index
        return float(times[min(int(np.argmin(np.abs(m4v_times - m4v_s))), len(times) - 1)])
    ev = pd.DataFrame({"t": [float(times[x["frame"]]) for x in e["events"]], "kind": [x["type"] for x in e["events"]]})
    wins = [(mov_t(w["start_s"]), mov_t(w["end_s"])) for w in e["windows"]]
    return ev, wins


def inside(df, wins, pad=0.0):
    if not len(df):
        return df
    m = np.zeros(len(df), bool)
    for a, b in wins:
        m |= (df.t.values >= a - pad) & (df.t.values <= b + pad)
    return df[m].reset_index(drop=True)



def to_md(df, index=False):
    """Markdown table without the optional tabulate package."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in d.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)

def main():
    rows, sets = [], {"fit (LOPO scores)": run.FIT, "check": run.CHECK}
    summary = {}
    for sname, names in sets.items():
        for n in names:
            ev, wins = byeye(n)
            if ev is None:
                continue
            ev = inside(ev, wins)
            dur = float(uv.frame_table(run.VDIR / f"{n}.mov").duration)
            secs = sum(b - a for a, b in wins)
            tl = es.sensor_timeline(run.DEP, f"{n}.mov", duration=dur)
            tl_raw = es.sensor_timeline(run.DEP, f"{n}.mov", duration=dur, out_lag=0.0, in_lead=0.0)
            cand = {"sensor truth (depth + lags)": inside(tl.events, wins),
                    "raw depth crossing": inside(tl_raw.events, wins)}
            for m in METHODS:
                f = run.PRED / f"{m}__{n}.npz"
                if f.exists():
                    p = np.load(f)
                    cand[f"picture: {m}"] = inside(ws.events_from_binary(p["t"], p["air_chain"]), wins)
            for cname, cev in cand.items():
                for tol in (0.5, 1.0):
                    r = es.match_events(ev, cev, tol=tol)
                    for kind, t, pt in r["pairs"]:
                        if tol == 1.0:
                            rows.append((sname, n[-5:], cname, kind, t, pt - t))
                    key = (sname, cname, tol)
                    s = summary.setdefault(key, {"n_truth": 0, "found": 0, "n_pred": 0, "seconds": 0.0})
                    s["n_truth"] += r["all"]["n_truth"]; s["found"] += r["all"]["found"]
                    s["n_pred"] += r["all"]["n_pred"]
                    if tol == 0.5:
                        s["seconds"] += secs
    df = pd.DataFrame(rows, columns=["set", "piece", "source", "kind", "byeye_t", "minus_byeye"])
    df.to_csv(run.C / "wp3_event_check.csv", index=False)
    out = []
    for (sname, cname, tol), s in summary.items():
        d = df[(df.set == sname) & (df.source == cname)]
        de, dn = d[d.kind == "exit"].minus_byeye, d[d.kind == "entry"].minus_byeye
        out.append((sname, cname, tol, f"{s['found']}/{s['n_truth']}", s["n_pred"] - s["found"],
                    f"{de.median():+.2f} [{de.min():+.2f}, {de.max():+.2f}]" if len(de) else "-",
                    f"{dn.median():+.2f} [{dn.min():+.2f}, {dn.max():+.2f}]" if len(dn) else "-",
                    f"{d.minus_byeye.abs().median():.2f}" if len(d) else "-", round(s["seconds"])))
    t = pd.DataFrame(out, columns=["set", "source", "tol_s", "found", "false", "exit: median [min, max] (s)",
                                   "entry: median [min, max] (s)", "median abs (s)", "labelled s (all pieces)"])
    print(to_md(t))


if __name__ == "__main__":
    main()
