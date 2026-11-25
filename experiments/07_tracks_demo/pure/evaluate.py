"""WP8c: old policy against the purity guards, on every reused run. CPU only.

    .venv/bin/python experiments/07_tracks_demo/pure/evaluate.py fires      # which guard fires where (old runs)
    .venv/bin/python experiments/07_tracks_demo/pure/evaluate.py trade      # policies x strengths -> results/trade.csv
    .venv/bin/python experiments/07_tracks_demo/pure/evaluate.py fishstats  # first-second stats of fish vs whale tracks

Truth:
- gap cut runs: the uncut run of the same stretch (old policy). A displayed track (numbered whale or `sN small`)
  MOVED when, over its frames, its best-overlapping uncut whale (IoU >= 0.3) is one whale for >= 5 frames and another
  whale for >= 5 frames. Fragments = displayed tracks per uncut whale (>= 5 frames on it). Coverage = uncut whale
  frames (shown to the tracker) that some displayed track overlaps with IoU >= 0.3.
- demo stretches and real cases: the moved numbers I saw by eye in WP8 / WP8b (EVENTS below, AI by-eye); a move is
  fixed when the policy ends the track inside the event's window, or the track is not drawn there.
- not whale-like displayed tracks: median `whale` patch share inside the outline < 0.25 (WP5b classifier, fitted on
  the dev clip), i.e. outlines on water / bubbles / surface / air.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as RU  # noqa: E402
from runs import tracks  # noqa: E402

import synthetic as SY  # noqa: E402  (gap/)
import cases as CA  # noqa: E402
import blocked as BL  # noqa: E402

from uwid import video as uv  # noqa: E402

LABELS = Path("/home/sat3737/NOAA_whales/underwater_cache/tracks/pure")

# by-eye moved numbers / joins from WP8 and WP8b: (run, old-policy display number, t_from, t_to, what)
EVENTS = [
    ("demo07", "s1", 822.3, 824.9, "fish -> calf (s1 small)"),
    ("demo01", 3, 25.5, 26.5, "joins two whales, then settles on one"),
    ("demo01", 2, 29.7, 30.5, "swallows the spotted whale"),
    ("demo02", 3, 72.4, 73.2, "joins the huge fin and the near whale (fin probably its own)"),
    ("R02a", 1, 11.8, 18.4, "exhalation bubble cloud -> whale after the breath"),
    ("R02a", 4, 11.8, 21.1, "stale outline on a wrong piece after the breath"),
    ("R02a", 6, 11.8, 21.1, "stale outline on a wrong piece after the breath"),
    ("R02b", 1, 11.8, 18.4, "exhalation bubble cloud -> whale after the breath"),
    ("R02c", 1, 11.8, 18.4, "exhalation bubble cloud -> whale after the breath"),
]


def run_key_dir(key: str) -> Path:
    if key.startswith("demo"):
        return RU.demo_dir(key[4:])
    return RU.real_dir(key[1:3], key[3])


# ------------------------------------------------------------------------------------------------ policies

OLD = tracks.PurityConfig(max_block_s=None, lone_block_s=None, max_absent_s=None, min_iou_prev=None,
                          max_area_ratio=None, swallow=None, max_drift=None, grow_from_small=None,
                          start_clutter=None, fish_speed=None, blocked_no_outline=False)


def policy(level: str) -> tracks.PurityConfig:
    """Named strengths of the full guard set."""
    base = dict(blocked_no_outline=True)
    table = {
        "loose": dict(max_block_s=4.0, lone_block_s=8.0, max_absent_s=2.0, min_iou_prev=0.15, max_area_ratio=3.0,
                      swallow=0.7, max_drift=0.3, grow_from_small=3.0, start_clutter=0.7, fish_speed=4.0),
        "medium": dict(max_block_s=2.0, lone_block_s=8.0, max_absent_s=1.0, min_iou_prev=0.3, max_area_ratio=2.5,
                       swallow=0.5, max_drift=0.15, grow_from_small=2.0, start_clutter=0.5, fish_speed=2.5),
        "strict": dict(max_block_s=1.0, lone_block_s=4.0, max_absent_s=0.5, min_iou_prev=0.45, max_area_ratio=2.0,
                       swallow=0.35, max_drift=0.10, grow_from_small=1.5, start_clutter=0.3, fish_speed=1.5),
    }
    return tracks.PurityConfig(**base, **table[level])


def single(guard: str, level: str = "medium") -> tracks.PurityConfig:
    """Only one guard of the medium set switched on (to see which guard does the work)."""
    full = policy(level).__dict__
    keep = {"block": ["max_block_s", "lone_block_s"], "absent": ["max_absent_s"],
            "jump": ["min_iou_prev", "max_area_ratio"], "swallow": ["swallow"], "drift": ["max_drift"],
            "grow": ["grow_from_small"], "start": ["start_clutter"], "fish": ["fish_speed"],
            "noblock": []}[guard]
    kw = {k: (full[k] if k in keep else None) for k in OLD.__dict__
          if k not in ("blocked_no_outline", "whale_size", "fish_size")}
    return tracks.PurityConfig(**kw, blocked_no_outline=(guard in ("block", "noblock")))


# ------------------------------------------------------------------------------------------------ inputs

_LAB: dict = {}


def labels_for(video: str):
    stem = tracks.stem_of(video)
    if stem not in _LAB:
        p = LABELS / f"labels_{stem}.npz"
        _LAB[stem] = np.load(p) if p.exists() else None
    return _LAB[stem]


def blocked_for(tr) -> list[tuple[int, int]]:
    """The WP8b rule on any run: bubbles from the patch labels (every 3rd frame), lens out of water from the depth
    sensor (2025-02-03) or from the `air` share (dev clip)."""
    if tr.meta.get("hide"):
        return []                   # the hidden ranges are the blocked views
    z = labels_for(tr.meta["video"])
    f0, n = tr.meta["frames"][0], tr.n
    frames = np.arange(f0, f0 + n)
    lens = np.zeros(n, bool)
    bub = np.zeros(n, bool)
    if z is not None:
        sel = (z["frames"] >= f0 - 6) & (z["frames"] < f0 + n + 6)
        sf, lab = z["frames"][sel], z["labels"][sel]
        if len(sf):
            nb = np.maximum((lab != BL.BACK).reshape(len(lab), -1).sum(1), 1)
            bs = (lab == BL.BUBBLES).reshape(len(lab), -1).sum(1) / nb
            air = (lab == BL.AIR).reshape(len(lab), -1).mean(1)
            near = np.abs(frames[:, None] - sf[None, :]).argmin(1)
            bub = bs[near] >= BL.BUBBLE_SHARE
            if tr.meta["video"] == tracks.DEV_VIDEO:
                lens = air[near] >= BL.AIR_SHARE
    if tr.meta["video"] != tracks.DEV_VIDEO:
        s = pd.read_csv(CA.SURFACINGS)
        tab = uv.frame_table(tr.meta["video"])
        times = tab.times[frames]
        for r in s[s.in_video.astype(bool)].itertuples():
            if f"-{int(r.piece):05d}" in Path(tr.meta["video"]).stem:
                lens |= (times >= r.pts_exit_pred - BL.LENS_PAD_S) & (times <= r.pts_entry_pred + BL.LENS_PAD_S)
    m = BL.clean(lens | bub, tr.meta["fps"])
    return [(int(frames[a]), int(frames[b - 1]) + 1) for a, b in BL.runs(m)]


def feats_for(d: Path):
    p = d / "pure_feats.npz"
    return dict(np.load(p)) if p.exists() else None


def apply(d: Path, cfg) -> tracks.Tracks:
    raw = tracks.read_run(d)
    return tracks.purify(raw, cfg, blocked_for(raw), feats_for(d))


# ------------------------------------------------------------------------------------------------ measures

def displayed(tr) -> list[int]:
    return [k for k, r in tr.roles.items() if r in ("whale", "small")]


def label_of(tr, k) -> str:
    return str(tr.number[k]) if k in tr.number else f"s{tr.small_number.get(k, '?')}"


def gap_measures(cfg, details: list | None = None) -> dict:
    moved = moved_whale = frag = segs = cov_n = cov_d = 0
    nbw = 0
    for p in SY.plan():
        d = tracks.tracks_dir(p["video"], 0, 0, "owl_sam3").with_name(p["name"])
        base = tracks.load_tracks(SY.base_dir(p["base"]))
        cut = apply(d, cfg)
        bf0, cf0 = base.meta["frames"][0], cut.meta["frames"][0]
        bw = displayed(base)
        cw = displayed(cut)
        frames = [f for f in range(cf0, cf0 + cut.n) if not (p["ga"] <= f < p["gb"]) and 0 <= f - bf0 < base.n]
        hits = {c: {} for c in cw}
        covered = {b: [0, 0] for b in bw}
        distinct = set()          # pairs of uncut tracks seen at the same time as separate outlines = two animals
        for f in frames[::2]:
            bm = {b: base.mask(b, f - bf0) for b in bw if base.packed[b][f - bf0].any()}
            for b1 in bm:
                for b2 in bm:
                    if b1 < b2 and tracks._mask_iou(bm[b1], bm[b2]) < 0.2:
                        distinct.add((b1, b2))
            cm = {c: cut.mask(c, f - cf0) for c in cw if cut.visible(c, f - cf0)}
            for b, m in bm.items():
                covered[b][1] += 1
                if any(tracks._mask_iou(m, x) >= 0.3 for x in cm.values()):
                    covered[b][0] += 1
            for c, x in cm.items():
                sc = {b: tracks._mask_iou(m, x) for b, m in bm.items()}
                if sc:
                    b = max(sc, key=sc.get)
                    if sc[b] >= 0.3:
                        hits[c][b] = hits[c].get(b, 0) + 1
        for c, h in hits.items():
            on = [b for b, n in h.items() if n >= 3]          # every 2nd frame: 3 samples ~ 5-6 frames
            two = any((min(x, y), max(x, y)) in distinct for x in on for y in on if x != y)
            if two:
                moved += 1
                if c in cut.number:
                    moved_whale += 1
                if details is not None:
                    details.append(dict(run=p["name"], gap_s=p["g"], track=label_of(cut, c),
                                        uncut_whales="/".join(label_of(base, b) for b in on),
                                        samples="/".join(str(h[b]) for b in on)))
            if on:
                segs += 1
        per_b = {}
        for c, h in hits.items():
            for b, n in h.items():
                if n >= 3:
                    per_b[b] = per_b.get(b, 0) + 1
        frag += sum(per_b.values())
        nbw += len(per_b)
        cov_n += sum(v[0] for v in covered.values())
        cov_d += sum(v[1] for v in covered.values())
    return dict(gap_moved=moved, gap_moved_whale_numbers=moved_whale, gap_tracks=segs, gap_tracks_per_whale=round(frag / max(nbw, 1), 2),
                gap_coverage=round(cov_n / max(cov_d, 1), 3))


def event_fixed(cfg) -> list[dict]:
    rows = []
    for key, num, ta, tb, what in EVENTS:
        d = run_key_dir(key)
        old = tracks.load_tracks(d)
        inv = {v: k for k, v in old.number.items()}
        sinv = {f"s{v}": k for k, v in old.small_number.items()}
        k = inv.get(num) if not isinstance(num, str) else sinv.get(num)
        new = apply(d, cfg)
        f0, fps = old.meta["frames"][0], old.meta["fps"]
        tab = uv.frame_table(old.meta["video"])
        ia, ib = tab.index_at(ta) - f0, tab.index_at(tb) - f0
        # which new displayed tracks carry raw track k inside the window: one number across the whole window = not fixed
        segs = [nk for nk in displayed(new) if nk // 1000 == k]
        vis_in = [nk for nk in segs if new.packed[nk][max(ia, 0):ib].any()]
        spans = [nk for nk in vis_in if new.packed[nk][max(ia, 0):max(ia, 0) + int(0.4 * fps)].any()
                 and new.packed[nk][max(ib - int(0.4 * fps), 0):ib].any()]
        rows.append(dict(run=key, number=num, window=f"{ta}-{tb}", what=what, raw=k,
                         fixed=bool(k is not None and not spans),
                         reasons="; ".join(sorted({r for i, n_, r, _ in new.endings if ia <= i < ib}))))
    return rows


def stretch_measures(cfg) -> dict:
    """Over the 7 demo stretches and 18 real-case runs: displayed whale numbers, displayed tracks that are not
    whale-like, whale-like outlined frames (sampled every 3rd frame)."""
    nums = fakes = wl = 0
    for d in [RU.demo_dir(c) for c in "01 02 03 04 05 06 07".split()] + \
             [RU.real_dir(c, v) for c in CA.REAL for v in "abc"]:
        if not (d / "meta.json").exists():
            continue
        tr = apply(d, cfg)
        f = feats_for(d)
        nums += len(tr.number)
        if f is None:
            continue
        for k in displayed(tr):
            raw = k // 1000 if k >= 1000 else k
            sel = (f["obj"] == raw)
            ii = f["i"][sel]
            ok = np.array([tr.packed[k][i].any() for i in ii]) if len(ii) else np.zeros(0, bool)
            if not ok.any():
                continue
            ws = f["cls"][sel][ok][:, 1].astype(float)
            if np.median(ws) < 0.25:
                fakes += 1
            wl += int((ws >= 0.25).sum())
    return dict(stretch_numbers=nums, stretch_not_whale_like=fakes, stretch_whale_like_frames=wl)


def all_configs() -> list:
    medium_settle = policy("medium")
    medium_settle.fish_settle = 1.0
    return [("old", OLD)] + [(f"full {lv}", policy(lv)) for lv in ("loose", "medium", "strict")] + \
           [("full medium + settle", medium_settle)] + \
           [(f"only {g}", single(g)) for g in ("noblock", "block", "absent", "jump", "swallow", "drift", "grow",
                                                "start", "fish")]


def cfg_file(name: str) -> Path:
    return HERE / "results" / f"trade_{name.replace(' ', '_').replace('+', 'plus')}.json"


def cmd_trade(only=None):
    rows = []
    for name, cfg in all_configs():
        if only and name not in only:
            continue
        out = cfg_file(name)
        if out.exists():
            continue
        r = dict(policy=name)
        r.update(gap_measures(cfg))
        ev = event_fixed(cfg)
        r["events_fixed"] = f"{sum(e['fixed'] for e in ev)}/{len(ev)}"
        r.update(stretch_measures(cfg))
        rows.append(r)
        print(r, flush=True)
        out.write_text(json.dumps(r))
        pd.DataFrame(ev).to_csv(HERE / "results" / f"events_{out.stem[6:]}.csv", index=False)


def cmd_moved(only=None):
    for name in (only or ("old", "full medium", "full strict")):
        cfg = dict(all_configs())[name]
        det = []
        gap_measures(cfg, det)
        d = pd.DataFrame(det)
        d.to_csv(HERE / "results" / f"moved_{name.replace(' ', '_')}.csv", index=False)
        print(name)
        print(d.to_string())


def cmd_merge():
    rows = [json.loads(cfg_file(n).read_text()) for n, _ in all_configs() if cfg_file(n).exists()]
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "results" / "trade.csv", index=False)
    print(d.to_string())


def cmd_fires():
    cfg = policy("medium")
    out = []
    for d in RU.all_dirs():
        if "_rev" in d.name or d.name.startswith("gap_"):
            continue
        raw = tracks.read_run(d)
        cuts, drops = tracks.purity_cuts(raw, cfg, blocked_for(raw), feats_for(d))
        for k, lst in cuts.items():
            for i, reason in lst:
                out.append(dict(run=d.name, obj=k, i=i, t=round(raw.meta["t0"] + i / raw.meta["fps"], 2), reason=reason))
    df = pd.DataFrame(out)
    df.to_csv(HERE / "results" / "fires_medium.csv", index=False)
    print(df.groupby(["run", "reason"]).size().to_string())


def cmd_fishstats():
    """First second of each displayed track: largest size and speed in body lengths per second."""
    rows = []
    for key in ("demo05", "demo07", "demo01", "demo03", "demo06", "R04a", "R06a"):
        d = run_key_dir(key)
        tr = tracks.load_tracks(d)
        t = tr.table[tr.table.area > 0].copy()
        t["size"] = np.maximum(t.x1 - t.x0, t.y1 - t.y0) / tracks.W
        t["cx"], t["cy"] = (t.x0 + t.x1) / 2, (t.y0 + t.y1) / 2
        for k in displayed(tr):
            g = t[t.obj == k].sort_values("i")
            g = g[g.i < g.i.iloc[0] + tr.meta["fps"]]
            if len(g) < 5:
                continue
            path = np.hypot(np.diff(g.cx), np.diff(g.cy)).sum()
            body = max(float(g["size"].median()) * tracks.W, 1)
            rows.append(dict(run=key, track=label_of(tr, k), first_s_max_size=round(float(g["size"].max()), 3),
                             body_lengths_per_s=round(path / body * tr.meta["fps"] / len(g), 2)))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "fishstats.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["trade", "fires", "fishstats", "merge", "moved"])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    (HERE / "results").mkdir(exist_ok=True)
    if a.cmd == "trade":
        cmd_trade(a.only)
    else:
        if a.cmd == "moved":
            cmd_moved(a.only)
        else:
            {"fires": cmd_fires, "fishstats": cmd_fishstats, "merge": cmd_merge}[a.cmd]()
