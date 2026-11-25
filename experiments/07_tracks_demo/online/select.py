"""WP11a held-out stretches. The RULE, written down 2026-11-23 before any tracker output on these stretches:

Eligible: hq spans of kept stretches with advanced_ok in the scan's `sensitive` intervals.json. A span is the hq span
itself, widened to its whole kept stretch when shorter than 8 s (2 s of tracking says nothing), then capped at 30 s
around the hq span's centre (staying inside the stretch).
- 2025-02-03 (4 stretches): recordings other than 00006 and 00007; spans not overlapping any stretch used in WP8 / WP8b
  (00004 650-668, 00005 365.5-381.5 and 736-754); no sensor lens-out inside; median window depth >= 1 m. Take the
  longest such hq span per recording, then the 4 longest of those (one per recording).
- CC-09-85 (4 stretches): no sensor lens-out inside the span; the 4 longest hq spans, at most 2 per recording.
- Breaths on purpose (2, outside the 8): CC-09-85 hq spans with a sensor lens-out inside, the 2 longest from
  different recordings.
- Old stretches run online for the comparison with the after-the-fact policy (2): demo clips 01 (dev 24-36 s) and
  05 (00006 456-476 s), WP5a start boxes as in WP8.

    .venv/bin/python experiments/07_tracks_demo/online/select.py     # -> results/selection.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from uwid import trackstage as T  # noqa: E402

D1, D2 = "2025-2-3 Mini-mother 11-96", "TestSubset 2025-1-2 CATS -85"
USED = {"00004": [(650, 668)], "00005": [(365.5, 381.5), (736, 754)]}


def widen(r, min_s=8.0, cap=30.0):
    a, b = r.start, r.end
    if b - a < min_s:
        a, b = r.stretch_start, r.stretch_end
    c = (r.start + r.end) / 2
    if b - a > cap:
        a, b = max(a, c - cap / 2), min(b, c + cap / 2)
        if b - a < cap:          # shifted against a stretch end
            a, b = (r.stretch_start, min(r.stretch_end, r.stretch_start + cap)) if a == r.stretch_start else \
                   (max(r.stretch_start, r.stretch_end - cap), r.stretch_end)
    return round(a, 2), round(b, 2)


def main():
    rows = []
    h = T.hq_spans(D1)
    h["rec"] = h.recording.str[-9:-4]
    h = h[~h.rec.isin(["00006", "00007"]) & ~h.lens_out & (h.depth_med >= 1.0)]
    h = h[[not any(r.start < b and r.end > a for a, b in USED.get(r.rec, [])) for r in h.itertuples()]]
    best = h.sort_values("dur", ascending=False).groupby("rec").head(1).sort_values("dur", ascending=False).head(4)
    for r in best.itertuples():
        a, b = widen(r)
        rows.append(dict(set="held-out", deployment=D1, recording=r.recording, source=r.source, t0=a, t1=b,
                         hq=f"{r.start:g}-{r.end:g}", depth_med=round(r.depth_med, 1)))
    h = T.hq_spans(D2)
    h["rec"] = h.recording.str[-9:-4]
    ok = h[~h.lens_out].sort_values("dur", ascending=False)
    picked, per = [], {}
    for r in ok.itertuples():
        if per.get(r.rec, 0) < 2 and len(picked) < 4:
            picked.append(r)
            per[r.rec] = per.get(r.rec, 0) + 1
    for r in picked:
        a, b = widen(r)
        rows.append(dict(set="held-out", deployment=D2, recording=r.recording, source=r.source, t0=a, t1=b,
                         hq=f"{r.start:g}-{r.end:g}", depth_med=round(r.depth_med, 1)))
    br = h[h.lens_out].sort_values("dur", ascending=False)
    seen = set()
    for r in br.itertuples():
        if r.rec in seen or len(seen) == 2:
            continue
        seen.add(r.rec)
        a, b = widen(r)
        rows.append(dict(set="breath", deployment=D2, recording=r.recording, source=r.source, t0=a, t1=b,
                         hq=f"{r.start:g}-{r.end:g}", depth_med=round(r.depth_med, 1)))
    d = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    d.to_csv(HERE / "results" / "selection.csv", index=False)
    print(d.drop(columns="source").to_string())


if __name__ == "__main__":
    main()
