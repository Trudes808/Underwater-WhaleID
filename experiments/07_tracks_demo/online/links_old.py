"""WP11a link layer, set on the OLD stretches only (dev clip, 00006, 00007; the 2025-02-03 gap runs of 00006 / 00007
and the dev group), strict purity after the fact as in WP8c, then `tracks.link_lone` with the LinkConfig defaults.

Truth: gap runs against their uncut run (a link is right when both fragments sit on the same uncut track); elsewhere
my by-eye knowledge of these clips from WP8 / WP8b (AI by-eye).

    .venv/bin/python experiments/07_tracks_demo/online/links_old.py     # -> results/links_old.csv, links_old_runs.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pure"))
import evaluate as EV  # noqa: E402  (pure/, puts gap/ on the path too)
import synthetic as SY  # noqa: E402
from runs import tracks  # noqa: E402

RUNS = ["demo05", "demo07", "R04a", "R04b", "R05a", "R05b", "R06a", "R06b", "demo01", "demo02", "R01a", "R01b",
        "R02a", "R02b"]
MULTI = {"demo01", "demo02", "R01a", "R01b", "R02a", "R02b"}     # dev clip: several whales


def seg_emb(tr, d: Path):
    """pure_feats rows (raw objects) re-keyed to the purified segments (k * 1000 + s)."""
    f = EV.feats_for(d)
    if f is None:
        return None
    rows_i, rows_o, rows_e = [], [], []
    for nk, p in tr.packed.items():
        raw = nk // 1000 if nk >= 1000 else nk
        vis = p.reshape(tr.n, -1).any(1)
        sel = f["obj"] == raw
        for i, e in zip(f["i"][sel], f["emb"][sel]):
            if 0 <= i < tr.n and vis[i]:
                rows_i.append(i), rows_o.append(nk), rows_e.append(e)
    return dict(i=np.array(rows_i), obj=np.array(rows_o), emb=np.array(rows_e))


def breaths_for(tr):
    v = tr.meta["video"]
    if "CC-11-96" not in v:
        return []
    from uwid import video as uv
    s = pd.read_csv("/home/sat3737/NOAA_whales/underwater_cache/tagdata/2025-2-3 Mini-mother 11-96/surfacings.csv")
    tab = uv.frame_table(v)
    out = []
    for r in s[s.in_video.astype(bool)].itertuples():
        if f"-{int(r.piece):05d}" in Path(v).stem:
            out.append((tab.index_at(r.pts_exit_pred), tab.index_at(r.pts_entry_pred)))
    return out


def detector_for(tr):
    b, _, _ = tracks.stage1_boxes(tr.meta["video"])
    f0 = tr.meta["frames"][0]
    return b[(b.frame >= f0) & (b.frame < f0 + tr.n)][["frame", "x0", "y0", "x1", "y1"]]


def linked(tr_dir: Path, cfg=None):
    raw = tracks.read_run(tr_dir)
    blocked = EV.blocked_for(raw)
    tr = tracks.purify(raw, EV.policy("strict"), blocked, EV.feats_for(tr_dir))
    tr.meta["blocked"] = [list(x) for x in blocked]
    n_before = len(tr.number)
    tracks.link_lone(tr, detector_for(tr), seg_emb(tr, tr_dir), breaths_for(tr), cfg)
    return tr, n_before


def gap_truth(cfg=None):
    rows = []
    for p in SY.plan():
        if p["base"] == "B1":
            scene = "multi"
        else:
            scene = "lone" if p["base"] in ("B3", "B4", "B6") else "calf + fish"
        d = tracks.tracks_dir(p["video"], 0, 0, "owl_sam3").with_name(p["name"])
        tr, n_before = linked(d, cfg)
        base = tracks.load_tracks(SY.base_dir(p["base"]))
        bf0, cf0 = base.meta["frames"][0], tr.meta["frames"][0]
        bw = [k for k, r in base.roles.items() if r in ("whale", "small")]

        def owner(k):
            cnt = {}
            for i in range(0, tr.n, 3):
                if not tr.visible(k, i) or not (0 <= i + cf0 - bf0 < base.n):
                    continue
                m = tr.mask(k, i)
                sc = {b: tracks._mask_iou(m, base.mask(b, i + cf0 - bf0)) for b in bw if base.packed[b][i + cf0 - bf0].any()}
                if sc:
                    b = max(sc, key=sc.get)
                    if sc[b] >= 0.3:
                        cnt[b] = cnt.get(b, 0) + 1
            return max(cnt, key=cnt.get) if cnt else None
        for L in tr.links:
            oa, ob = owner(L["a"]), owner(L["b"])
            rows.append(dict(run=p["name"], scene=scene, g=p["g"], **{k: L.get(k) for k in ("gap_s", "lone", "made", "why",
                                                                                          "size_ratio", "app_dist")},
                             same_uncut=(oa is not None and oa == ob)))
        # fragments per uncut whale, with and without the layer
        own = {k: owner(k) for k in tr.number}
        per_b_chain, per_b_frag = {}, {}
        for k, b in own.items():
            if b is None:
                continue
            per_b_frag[b] = per_b_frag.get(b, 0) + 1
            per_b_chain.setdefault(b, set()).add(tr.number[k])
        rows.append(dict(run=p["name"], scene=scene, g=p["g"], why="_fragments",
                         frag=sum(per_b_frag.values()), chains=sum(len(v) for v in per_b_chain.values()),
                         whales=len(per_b_frag)))
    return pd.DataFrame(rows)


def main():
    rows, runs = [], []
    for key in RUNS:
        tr, n_before = linked(EV.run_key_dir(key))
        for L in tr.links:
            rows.append(dict(run=key, scene="multi" if key in MULTI else "lone", **{k: L.get(k) for k in (
                "t_end", "gap_s", "breath", "others", "detector", "lone", "size_ratio", "app_dist", "made", "why")}))
        runs.append(dict(run=key, scene="multi" if key in MULTI else "lone", whale_numbers_without=n_before,
                         whale_numbers_with=len(set(tr.number.values())), links=sum(L["made"] for L in tr.links)))
    pd.DataFrame(rows).to_csv(HERE / "results" / "links_old.csv", index=False)
    pd.DataFrame(runs).to_csv(HERE / "results" / "links_old_runs.csv", index=False)
    print(pd.DataFrame(runs).to_string())
    print(pd.DataFrame(rows).to_string())
    g = gap_truth()
    g.to_csv(HERE / "results" / "links_old_gap.csv", index=False)
    lk = g[g.why != "_fragments"]
    print(lk.groupby(["scene", "made", "same_uncut"]).size())
    fr = g[g.why == "_fragments"].groupby("scene")[["frag", "chains", "whales"]].sum()
    print(fr.assign(frag_per_whale=(fr.frag / fr.whales).round(2), chains_per_whale=(fr.chains / fr.whales).round(2)))


if __name__ == "__main__":
    main()
