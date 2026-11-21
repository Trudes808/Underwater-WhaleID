"""Render run.py's results.json as markdown tables (printed; pasted into tables.md)."""
import json
import sys
from pathlib import Path

import os
R = json.loads(Path(os.environ.get("WS_RESULTS", "/home/sat3737/NOAA_whales/underwater_cache/waterstate/results.json")).read_text())
METHODS = ["sat", "color", "clip0", "clipLR", "dino"]
NAMES = {"sat": "(a0) mean saturation, 1 number", "color": "(a) colour stats, 18 numbers", "clip0": "(b) CLIP zero-shot",
         "clipLR": "(b2) CLIP + logistic", "dino": "(c) DINOv3 + logistic"}
STEPNAME = {"raw": "none (frame threshold)", "pool": "pool 1 s", "hyst": "pool + hysteresis", "chain": "pool + 2-state chain"}


def pr(per, s):
    r = [x for x in per if x["state"] == s][0]
    if not r["n_truth"]:
        return "-"
    rec = f"{r['correct']}/{r['n_truth']}"
    prec = f"{r['correct']}/{r['n_pred']}" if r["n_pred"] else "0/0"
    return f"{rec} / {prec}"


def pct(per, s, key):
    r = [x for x in per if x["state"] == s][0]
    v = r[key]
    return "-" if v is None or v != v else f"{100 * v:.1f}"


def ev(e):
    a = e["all"]
    err = a.get("abs_err_median")
    err = "-" if err is None or err != err else f"{err:.2f}"
    return f"{a['found']}/{a['n_truth']}", f"{a['false']} ({a['false_per_h']:.1f}/h)", err


def main(sets=None):
    first = R["methods"][METHODS[0]]["sets"]
    sets = sets or list(first)
    out = []
    for sname in sets:
        h = first[sname]["raw"]["hours"]
        out.append(f"\n### {sname} ({h * 3600:.0f} s of video)\n")
        out.append("Window states: `correct/truth (recall) / correct/predicted (precision)`; events matched within 0.5 s "
                   "(and 1.0 s), false events per hour, median |timing error| of the 0.5 s matches.\n")
        out.append("| method | window step | underwater R / P | crossing R / P | air R / P | events found 0.5 s | 1.0 s "
                   "| false events | median abs err (s) |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for m in METHODS:
            if m not in R["methods"] or sname not in R["methods"][m]["sets"]:
                continue
            S = R["methods"][m]["sets"][sname]
            for k in ["raw", "pool", "hyst", "chain"]:
                x = S[k]
                f05, fp, err = ev(x["ev05"])
                f10, _, _ = ev(x["ev10"])
                out.append(f"| {NAMES[m] if k == 'raw' else ''} | {STEPNAME[k]} | {pr(x['per'], 'underwater')} | "
                           f"{pr(x['per'], 'crossing')} | {pr(x['per'], 'air')} | {f05} | {f10} | {fp} | {err} |")
        # frame level
        out.append("\nPer frame (5 frames/s, frame threshold, no window step; frames within 0.3 s of a true event "
                   "left out and counted apart):\n")
        out.append("| method | underwater R / P | air R / P | air recall % | air precision % | crossing frames called air |")
        out.append("|---|---|---|---|---|---|")
        for m in METHODS:
            if m not in R["methods"] or sname not in R["methods"][m]["sets"]:
                continue
            x = R["methods"][m]["sets"][sname]["raw"]
            out.append(f"| {NAMES[m]} | {pr(x['fper'], 'underwater')} | {pr(x['fper'], 'air')} | "
                       f"{pct(x['fper'], 'air', 'recall')} | {pct(x['fper'], 'air', 'precision')} | "
                       f"{x['cross_called_air']}/{x['cross_frames']} |")
            if m == "clip0" and "raw_bias0" in R["methods"][m]["sets"][sname]:
                y = R["methods"][m]["sets"][sname]["raw_bias0"]
                out.append(f"| (b) CLIP zero-shot, untuned threshold 0 | {pr(y['fper'], 'underwater')} | "
                           f"{pr(y['fper'], 'air')} | {pct(y['fper'], 'air', 'recall')} | "
                           f"{pct(y['fper'], 'air', 'precision')} | |")
        # ambiguous dropped, chain
        out.append("\nWindow step `pool + 2-state chain`, windows with a true event within 0.3 s of their edge "
                   "left out (truth timing +-0.3 s):\n")
        out.append("| method | underwater R / P | crossing R / P | air R / P |")
        out.append("|---|---|---|---|")
        for m in METHODS:
            if m not in R["methods"] or sname not in R["methods"][m]["sets"]:
                continue
            x = R["methods"][m]["sets"][sname]["chain"]
            out.append(f"| {NAMES[m]} | {pr(x['perA'], 'underwater')} | {pr(x['perA'], 'crossing')} | "
                       f"{pr(x['perA'], 'air')} |")
    out.append("\nTuned window steps (on the LOPO out-of-fold scores of 00004-00006):\n")
    out.append("| method | C | none | pool | hysteresis | chain |")
    out.append("|---|---|---|---|---|---|")
    for m in METHODS:
        st = R["methods"][m]["steps"]
        f = lambda s: ", ".join(f"{k}={v}" for k, v in s.items() if k != "kind")  # noqa: E731
        out.append(f"| {m} | {R['methods'][m]['C']} | {f(st['raw'])} | {f(st['pool'])} | {f(st['hyst'])} | "
                   f"{f(st['chain'])} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
