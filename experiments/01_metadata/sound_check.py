"""Does the sound track separate 'surfaced' from 'underwater'? Threshold test on band levels.

For every stable .mov: band levels (dBFS) in 0.25 s frames over the whole file -> audio_levels.parquet. Frames are
labelled from the sensor: 'air' inside a sensor surfacing, 'water' when the tag is on the whale and at least 3 s
from any surfacing. For each band the best single threshold is chosen on the first video and applied to the rest
(balanced accuracy reported for both).

    .venv/bin/python experiments/01_metadata/sound_check.py "2025-2-3 Mini-mother 11-96"
"""
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import tagdata as td  # noqa: E402

VIDEO_ROOT = Path("/home/sat3737/NOAA_whales/underwater_video")
BANDS = [(20, 200), (200, 1000), (1000, 4000), (4000, 8000), (8000, 15900)]
HOP = 0.25


def levels(video: Path) -> pd.DataFrame:
    c = av.open(str(video))
    a = c.streams.audio[0]
    sr = a.codec_context.sample_rate
    xs = [fr.to_ndarray().astype(np.float64).mean(0) for fr in c.decode(a)]
    c.close()
    x = np.concatenate(xs)
    n = int(HOP * sr)
    k = len(x) // n
    seg = x[:k * n].reshape(k, n) * np.hanning(n)
    S = np.abs(np.fft.rfft(seg, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    rows = {"pts": (np.arange(k) + 0.5) * HOP, "rms_db": 10 * np.log10((x[:k * n].reshape(k, n) ** 2).mean(1) + 1e-20),
            "peak": np.abs(x[:k * n].reshape(k, n)).max(1)}
    for lo, hi in BANDS:
        rows[f"b{lo}_{hi}"] = 10 * np.log10(S[:, (f >= lo) & (f < hi)].sum(1) + 1e-20)
    out = pd.DataFrame(rows)
    out["file"] = video.name
    return out


def main(dep):
    out = td.cache_dir(dep)
    surf = pd.read_csv(out / "surfacings.csv")
    st = pd.read_parquet(out / "states_50hz.parquet")
    fits = pd.read_csv(out / "clock_fits_mov.csv")
    sync = json.loads((out / "sync_summary.json").read_text())
    off = sync["clock_minus_sensor_from_drop_s"]["mean"]
    allv = []
    for r in fits.itertuples():
        p = VIDEO_ROOT / dep / "Video" / r.file
        t1 = time.time()
        lv = levels(p)
        ts = r.clock_at_pts0_t - off + r.clock_rate * lv.pts.to_numpy()
        lab = np.full(len(ts), "", dtype=object)
        whale = np.interp(ts, st.t, (st.state == "whale").to_numpy().astype(float)) > 0.5
        near = np.zeros(len(ts), bool)
        inside = np.zeros(len(ts), bool)
        for s in surf.itertuples():
            inside |= (ts >= s.start_t) & (ts <= s.end_t)
            near |= (ts >= s.start_t - 3) & (ts <= s.end_t + 3)
        lab[inside] = "air"
        lab[whale & ~near] = "water"
        lv["label"] = lab
        lv["sensor_t"] = ts
        allv.append(lv)
        print(r.file, f"{time.time() - t1:.0f}s", "rms dB median", round(lv.rms_db.median(), 1), "range q01-q99",
              lv.rms_db.quantile([.01, .99]).round(1).tolist(), "peak max", round(lv.peak.max(), 4))
    A = pd.concat(allv, ignore_index=True)
    A.to_parquet(out / "audio_levels.parquet", index=False)
    res = {"per_file": {}, "threshold_test": {}}
    for f, g in A.groupby("file"):
        res["per_file"][f] = {c: {"air_median": float(g[g.label == "air"][c].median()),
                                  "water_median": float(g[g.label == "water"][c].median()),
                                  "water_q05_q95": [float(x) for x in g[g.label == "water"][c].quantile([.05, .95])],
                                  "n_air": int((g.label == "air").sum()), "n_water": int((g.label == "water").sum())}
                              for c in ["rms_db"] + [f"b{lo}_{hi}" for lo, hi in BANDS]}
    files = sorted(A.file.unique())
    fit, rest = A[A.file == files[0]], A[A.file.isin(files[1:])]
    for c in ["rms_db"] + [f"b{lo}_{hi}" for lo, hi in BANDS]:
        best = None
        for thr in np.quantile(fit[c], np.linspace(0.01, 0.99, 99)):
            for sign in (1, -1):
                pa = sign * fit[c] > sign * thr
                ba = 0.5 * (pa[fit.label == "air"].mean() + (~pa[fit.label == "water"]).mean())
                if best is None or ba > best[0]:
                    best = (ba, thr, sign)
        ba_fit, thr, sign = best
        pr = sign * rest[c] > sign * thr
        ba_test = 0.5 * (pr[rest.label == "air"].mean() + (~pr[rest.label == "water"]).mean()) if len(rest) else np.nan
        res["threshold_test"][c] = {"fit_file": files[0], "threshold": float(thr), "air_if": ">" if sign > 0 else "<",
                                    "balanced_acc_fit": float(ba_fit), "balanced_acc_test": float(ba_test),
                                    "n_test_air": int((rest.label == "air").sum()), "n_test_water": int((rest.label == "water").sum())}
    (out / "sound_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["threshold_test"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-2-3 Mini-mother 11-96")
