"""Turn my hand-written label lines into frames.csv (WP3 part 2).

    PY=.venv/bin/python
    $PY experiments/02_reference_labels/build_frames_csv.py LINES.txt <cache>/<stem>/index.json labels/<stem>/frames.csv

LINES.txt: one line per frame, `frame|state|view|other_whales|faint_whales|back|seafloor|fish|other|note`, optionally
followed by `|whale_size|whale_frame_share|whale_cut|quality|quality_note` (WP3b; required when other_whales >= 1 if
any line has them, empty otherwise; rule in README "Size and quality"); lines starting with # or without a leading
frame number are ignored. Every sampled frame in index.json must have exactly one line.
time_s comes from the decoder timestamp in index.json.
"""
import csv
import json
import sys
from collections import Counter

lines, index, out = sys.argv[1:4]
meta = json.load(open(index))
times = {int(k): v for k, v in meta["sample"].items()}
rows = {}
for ln in open(lines):
    p = ln.rstrip("\n").split("|")
    if not p[0].strip().isdigit():
        continue
    if len(p) not in (10, 15):
        sys.exit(f"bad line ({len(p)} fields): {ln}")
    p = [x.strip() for x in p] + [""] * (15 - len(p))
    f = int(p[0])
    if f in rows:
        sys.exit(f"duplicate frame {f}")
    rows[f] = dict(frame=f, time_s=f"{times[f]:.3f}", state=p[1], view=p[2], other_whales=p[3], faint_whales=p[4], back=p[5],
                   seafloor=p[6], fish=p[7], other=p[8], note=p[9], whale_size=p[10], whale_frame_share=p[11],
                   whale_cut=p[12], quality=p[13], quality_note=p[14])
missing = sorted(set(times) - set(rows)); extra = sorted(set(rows) - set(times))
if missing or extra:
    sys.exit(f"missing {missing} extra {extra}")
allowed = dict(state={"air", "crossing", "underwater", "unusable"}, view={"clear", "bubbles", "dark", "glare", "sand"},
               seafloor={"none", "visible", "unsure"}, fish={"none", "few", "school"})
for r in rows.values():
    for k, ok in allowed.items():
        if r[k] not in ok:
            sys.exit(f"frame {r['frame']}: bad {k} {r[k]!r}")
    int(r["other_whales"]); int(r["faint_whales"])
    has_q = any(x["quality"] for x in rows.values())
    if has_q and (int(r["other_whales"]) >= 1) != bool(r["quality"]):
        sys.exit(f"frame {r['frame']}: size / quality must be filled exactly when other_whales >= 1")
    if r["quality"]:
        if r["quality"] not in ("hq", "ok", "low") or r["whale_cut"] not in ("no", "yes"):
            sys.exit(f"frame {r['frame']}: bad quality / cut")
        r["whale_size"] = f"{float(r['whale_size']):.1f}"; r["whale_frame_share"] = f"{float(r['whale_frame_share']):.2f}"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(next(iter(rows.values())))); w.writeheader(); w.writerows(rows[f] for f in sorted(rows))
print(len(rows), "rows")
for k in ["state", "view", "other_whales", "faint_whales", "back", "seafloor", "fish", "quality", "whale_size"]:
    print(k, sorted(Counter(r[k] for r in rows.values()).items()))
