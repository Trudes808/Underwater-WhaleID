"""The clock-offset check on real data with a deliberately wrong offset (does it notice?).

Re-uses the cached picture events and clock fits of `uwid scan`; maps the picture events with offsets 2.03 (the
measured one), 3.03, 1.53 and 12.03 s and runs `pipeline.offset_check`. Writes results/offset_demo.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from uwid import paths, pipeline as pl, video  # noqa: E402

DEP = "2025-2-3 Mini-mother 11-96"
cfg = pl.ScanConfig()
dep = [d for d in pl.discover(paths.footage_dir() / DEP)][0]
sensors = pl.SensorTimeline(dep.name, dep.sensor_csv, cfg, log=lambda *a: None)
cache = paths.cache_dir("scan") / DEP / "cache"
dev = sensors.depth_events(cfg)
out = {}
for off in (2.03, 3.03, 1.53, 12.03):
    pe, de = [], []
    for rec in dep.recordings:
        fit = json.loads((cache / f"{rec.key}.clock.json").read_text())
        vc = pl._video_clock(rec, sensors, fit, off)
        ev = pd.read_csv(cache / f"{rec.key}.water_events.csv")
        dur = video.frame_table(rec.path).duration
        ev = ev[(ev.t > 1) & (ev.t < dur - 1)]
        pe.append(pd.DataFrame({"t": vc.to_sensor(ev.t.to_numpy()), "kind": ev.kind.to_numpy()}))
        a, b = float(vc.to_sensor(0.0)), float(vc.to_sensor(dur))
        de.append(dev[(dev.t > a + 1) & (dev.t < b - 1)])
    r = pl.offset_check(pd.concat(pe), pd.concat(de))
    r.pop("pairs")
    out[str(off)] = {k: r.get(k) for k in ("verdict", "median_picture_minus_depth_s", "n_pairs", "n_depth_events",
                                           "best_shift_s")}
    out[str(off)]["implied_offset_s"] = off + r.get("median_picture_minus_depth_s", float("nan"))
print(json.dumps(out, indent=1))
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "offset_demo.json").write_text(json.dumps(out, indent=1))
