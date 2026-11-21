"""SAM 3 on the 31 dev mask frames: whale instances found by grade (SAM 3's own masks against the reference), what
"fish", "seafloor", "whale back" and "air bubbles" cover, and how much of the back is called seafloor / whale.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04_whale_fish_detection/sam3_masks.py run
    .venv/bin/python experiments/04_whale_fish_detection/sam3_masks.py report --thr 0.5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (CLASS_IDS, DEV_VIDEO, PROMPTS, detect, instances, load_masks, mask_frames, to_md)

OUT = detect.detect_cache_dir(DEV_VIDEO, "sam3_maskframes")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    r = detect.Sam3Runner(PROMPTS["sam3_all"], min_score=0.1, keep_masks=True)
    mf = mask_frames()
    rows, store = [], {}
    for f, img in detect.iter_frames_cv(DEV_VIDEO, frames=mf):
        df = r.detect([img])[0]
        for i, (_, row) in enumerate(df.iterrows()):
            key = f"{f}_{i}"
            store[key] = np.packbits(r.last_masks[0][i])
            rows.append(dict(frame=f, key=key, **row.to_dict()))
    pd.DataFrame(rows).to_csv(OUT / "rows.csv", index=False)
    np.savez_compressed(OUT / "masks.npz", **store)
    print("sam3 mask-frame instances:", len(rows))


def unpack(store, key):
    return np.unpackbits(store[key])[: 1920 * 1080].reshape(1080, 1920).astype(bool)


def report(thr):
    rows = pd.read_csv(OUT / "rows.csv")
    store = np.load(OUT / "masks.npz")
    res = []
    cover = []
    for f in mask_frames():
        cls, _, _ = load_masks(f)
        ins = [i for i in instances(f) if i["cls"] == "whale"]
        w = rows[(rows.frame == f) & (rows["query"] == "whale") & (rows.score >= thr)].sort_values("score", ascending=False)
        pm = {k: unpack(store, k) for k in w.key}
        used = set()
        for i in sorted(ins, key=lambda x: -x["area"]):
            best, bk = 0.0, None
            for k, m in pm.items():
                if k in used:
                    continue
                inter = (m & i["mask"]).sum()
                iou = inter / max((m | i["mask"]).sum(), 1)
                if iou > best:
                    best, bk = iou, k
            found = bk is not None and best >= 0.3
            if found:
                used.add(bk)
            res.append(dict(frame=f, grade=i["grade"], cut=i["cut"], bubbles=i["behind_bubbles"], small=i["small"],
                            found=found, mask_iou=best if found else np.nan))
        # unmatched whale masks: what they sit on
        for k in set(pm) - used:
            m = pm[k]
            fr = {c: float((cls[m] == v).mean()) if m.any() else 0 for c, v in CLASS_IDS.items()}
            cause = "whale_part" if fr["whale"] >= 0.25 else max(("back", "bubbles", "surface", "air", "water"),
                                                                    key=lambda c: fr[c])
            cover.append(dict(frame=f, query="whale", cause=cause))
        # union of each prompt's masks vs the reference classes
        for q in ("whale", "fish", "seafloor", "sand", "whale back", "air bubbles"):
            ks = rows[(rows.frame == f) & (rows["query"] == q) & (rows.score >= thr)].key
            u = np.zeros((1080, 1920), bool)
            for k in ks:
                u |= unpack(store, k)
            for c, v in CLASS_IDS.items():
                ref = cls == v
                if ref.sum() >= 500:
                    cover.append(dict(frame=f, query=q, ref=c, ref_px=int(ref.sum()), covered=float((u & ref).sum() / ref.sum()),
                                      pred_px=int(u.sum())))
    d = pd.DataFrame(res)
    groups = [("all", d.index == d.index)] + [(g, d.grade == g) for g in ("clear", "hazy", "faint")] + \
             [("cut by frame edge", d.cut), ("behind bubbles", d.bubbles), ("small", d.small)]
    print(f"### SAM 3 'whale' at score >= {thr}: reference whale instances found (mask IoU >= 0.3) on 31 mask frames\n")
    print(to_md(pd.DataFrame([dict(group=n, n=int(m.sum()), found=int(d.found[m].sum()),
                                   median_mask_iou=round(d.mask_iou[m & d.found].median(), 3),
                                   iou_ge_05=int((d.mask_iou[m & d.found] >= 0.5).sum())) for n, m in groups])))
    c = pd.DataFrame(cover)
    un = c[c.get("cause").notna()] if "cause" in c else pd.DataFrame()
    if len(un):
        print("\nunmatched 'whale' masks by what they sit on:", un.groupby("cause").size().to_dict())
    cc = c[c.get("ref").notna()] if "ref" in c else c
    piv = cc.groupby(["query", "ref"]).apply(lambda g: (g.covered * g.ref_px).sum() / g.ref_px.sum()).unstack()
    print(f"\nshare of each reference class covered by the union of a prompt's masks (score >= {thr}):\n")
    print(to_md(piv.round(3).reset_index()))
    d.to_csv(OUT / f"instances_{thr}.csv", index=False)
    cc.to_csv(OUT / f"cover_{thr}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    run() if a.cmd == "run" else report(a.thr)


if __name__ == "__main__":
    main()
