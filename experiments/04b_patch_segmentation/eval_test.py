"""Across deployments: the classifier fitted on the 31 dev mask frames, scored on the 2025-02-03 recordings.

    .venv/bin/python experiments/04b_patch_segmentation/eval_test.py [--pred w1280_h0] [--tag ""]

For every recording with WP3 labels (experiments/02_reference_labels/labels/<stem>/frames.csv):
- whale presence per labelled frame and per 1 s window (same rules as eval_dev.py), AUC and the counts at the
  thresholds chosen on the dev clip (false alarms <= 10 % on the dev clip's held-out windows; nothing tuned here);
- back: share of the frame called back on frames with and without the back;
- fish: frames where any patch is called fish, against the fish label;
- seafloor: what the classifier calls the pixels Mask2Former calls sand / rock / earth / floor / mountain, on frames
  labelled seafloor visible / unsure / none;
- masks (when WP3's masks exist, underwater_cache/reference/<stem>/masks/): pixel scores as on the dev clip,
  including the seafloor row, single frame and window average, with and without the "unknown" cut.
Writes results/test_<pred><tag>.json.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import pandas as pd
import torch

from uwid import eval_presence as ep
from uwid import patchseg as ps
import common as C
from eval_dev import presence_tables

OUT = C.HERE / "results"
SEAFLOOR_IDS = [46, 34, 13, 3, 16]


def dev_thresholds(pred):
    d = json.loads((OUT / f"dev_windows_{pred}.json").read_text())["presence"]
    return {k: v["op_far10"]["threshold"] for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", default="w1280_h0"); ap.add_argument("--tag", default="")
    ap.add_argument("--unknown", type=float, default=None, help="distance cut for 'unknown' (from unknown_calib)")
    ap.add_argument("--stems", default="")
    a = ap.parse_args()
    fixed = dev_thresholds(a.pred)
    if a.unknown is None:
        uk = OUT / "unknown_calib.json"
        a.unknown = json.loads(uk.read_text())["thr_p99"] if uk.exists() else None
    res = {"dev_thresholds": fixed, "unknown_thr": a.unknown, "videos": {}}
    stems = a.stems.split(",") if a.stems else C.TEST_STEMS
    pooled = []
    kept = []
    pooled_masks = {k: C.Scorer() for k in ('single', 'window', 'single_unknown')}
    for stem in stems:
        lp = C.REPO / "experiments/02_reference_labels/labels" / stem / "frames.csv"
        pp = ps.CACHE / f"pred_{stem}_{a.pred}{a.tag}.npz"
        if not lp.exists() or not pp.exists():
            continue
        labels = ep.load_frame_labels(lp)
        idx = json.loads((C.REF / stem / "index.json").read_text())
        d = np.load(pp)
        frames, wins, probs, dist = list(d["frames"]), d["window"], d["probs"], d["dist"]
        r = {"n_labelled": len(labels)}
        r["presence"] = presence_tables(np.array(frames), wins, probs, labels, idx["fps"], fixed=fixed, keep=kept)
        # per labelled frame: class shares
        rows = []
        m2f = np.load(ps.CACHE / f"m2f_{stem}.npz") if (ps.CACHE / f"m2f_{stem}.npz").exists() else None
        for _, lr in labels.iterrows():
            f = int(lr.frame)
            if f not in frames:
                continue
            p = probs[frames.index(f)].astype(np.float32)
            lab = p.argmax(-1)
            share = np.bincount(lab.ravel(), minlength=len(ps.CLASSES)) / lab.size
            row = {"frame": f, "state": lr.state, "back_label": lr.back_present, "fish_label": lr.fish_present,
                   "seafloor": lr.seafloor, "whale": lr.whale, "faint": lr.faint,
                   **{f"share_{ps.CLASSES[c]}": share[c] for c in range(len(ps.CLASSES))},
                   "fish_patches": int((lab == ps.FISH).sum()),
                   "unknown_share": float((dist[frames.index(f)] > a.unknown).mean()) if a.unknown else None}
            if m2f is not None and f in list(m2f["frames"]):
                ml = m2f["labels"][list(m2f["frames"]).index(f)]
                sf = cv2.resize(np.isin(ml, SEAFLOOR_IDS).astype(np.uint8), (lab.shape[1], lab.shape[0]),
                                interpolation=cv2.INTER_AREA) > 0
                row["m2f_seafloor_share"] = float(sf.mean())
                for c in range(len(ps.CLASSES)):
                    row[f"m2fsf_as_{ps.CLASSES[c]}"] = int(((lab == c) & sf).sum())
                if a.unknown:
                    row["m2fsf_unknown"] = int(((dist[frames.index(f)] > a.unknown) & sf).sum())
                    row["nonsf_unknown"] = int(((dist[frames.index(f)] > a.unknown) & ~sf).sum())
                row["m2fsf_patches"] = int(sf.sum())
            rows.append(row)
        fr = pd.DataFrame(rows); fr["stem"] = stem; pooled.append(fr)
        uw = fr[fr.state == "underwater"]
        r["back"] = {"frames_with_back": int(uw.back_label.sum()),
                     "mean_back_share_with": float(uw[uw.back_label].share_back.mean()),
                     "mean_back_share_without": float(uw[~uw.back_label].share_back.mean()) if (~uw.back_label).any() else None,
                     "frames_with_back_share_gt5pct": int((uw[uw.back_label].share_back > 0.05).sum()),
                     "auc_back_share": None}
        if uw.back_label.nunique() == 2:
            r["back"]["auc_back_share"] = ep.auc(ep.roc(uw.share_back, uw.back_label))
        r["fish"] = {"fish_frames": int(fr.fish_label.sum()),
                     "fish_frames_with_fish_patch": int((fr.fish_label & (fr.fish_patches > 0)).sum()),
                     "nofish_frames_with_fish_patch": int((~fr.fish_label & (fr.fish_patches > 0)).sum()),
                     "nofish_frames": int((~fr.fish_label).sum())}
        r["shares_underwater_mean"] = {c: float(uw[f"share_{c}"].mean()) for c in ps.CLASSES}
        r["seafloor_labels"] = fr.seafloor.value_counts().to_dict()
        if "m2fsf_patches" in fr:
            for sfl in ["visible", "unsure", "none"]:
                g = fr[fr.seafloor == sfl]
                if len(g) and g.m2fsf_patches.sum() > 0:
                    tot = g.m2fsf_patches.sum()
                    r.setdefault("m2f_seafloor_called", {})[sfl] = {
                        "frames": len(g), "frames_with_m2f_seafloor": int((g.m2fsf_patches > 0).sum()),
                        **{c: float(g[f"m2fsf_as_{c}"].sum() / tot) for c in ps.CLASSES},
                        "unknown": float(g.m2fsf_unknown.sum() / tot) if a.unknown else None}
        # masks
        mdir = C.REF / stem / "masks"
        if mdir.exists() and (C.REF / stem / "DONE").exists():
            single, win, single_u = C.Scorer(), C.Scorer(), C.Scorer()
            mframes = sorted(int(p.name.split("_")[0]) for p in mdir.glob("*_class.png"))
            used = []
            for f in mframes:
                if f not in frames:
                    continue
                used.append(f)
                cm, inst, meta = C.read_mask(C.test_folder(stem), f)
                i = frames.index(f)
                p0 = torch.from_numpy(probs[i].astype(np.float32))
                pred = C.pred_codes(p0, cm.shape)
                single.add(pred, cm, inst, meta); pooled_masks['single'].add(pred, cm, inst, meta)
                if a.unknown:
                    du = cv2.resize(dist[i].astype(np.float32), (cm.shape[1], cm.shape[0])) > a.unknown
                    pu = pred.copy(); pu[du] = ps.UNKNOWN
                    single_u.add(pu, cm, inst, meta); pooled_masks['single_unknown'].add(pu, cm, inst, meta)
                k = wins[i]
                sel = [j for j in range(len(frames)) if wins[j] == k]
                pw = C.pred_codes(torch.from_numpy(probs[sel].astype(np.float32).mean(0)), cm.shape)
                win.add(pw, cm, inst, meta); pooled_masks['window'].add(pw, cm, inst, meta)
            r["masks"] = {"frames": used, "single": single.summary(), "window": win.summary(),
                          "single_unknown": single_u.summary() if a.unknown else None}
        res["videos"][stem] = r
        print(stem, json.dumps({k: v for k, v in r.items() if k not in ("presence", "masks")}, default=float)[:1500])
        for k, v in r["presence"].items():
            print("  ", k, "auc", round(v["auc"], 3), "whale/free/faint", v["n_whale"], v["n_free"], v["n_faint_only"],
                  "at dev thr:", {x: v.get("at_fixed", {}).get(x) for x in ["threshold", "hits", "n_pos", "false_alarms", "n_neg"]})
        if "masks" in r:
            for k in ["single", "window"]:
                s = r["masks"][k]
                print("   masks", k, len(r["masks"]["frames"]), {c: C.fmt(v, 2) for c, v in s["iou_band"].items()},
                      {g: round(v["pixel_recall"], 2) for g, v in s["whale_recall_by_grade"].items()}, s["confusions"])
                print("   seafloor row (truth seafloor -> predicted share):",
                      np.round(np.array(s["conf"][7]) / max(sum(s["conf"][7]), 1), 3).tolist())
    # pooled over recordings
    pres = {}
    for name in (kept[0].keys() if kept else []):
        tab = pd.concat([u[name] for u in kept], ignore_index=True)
        t = tab[tab.whale | tab.whale_free]
        curve = ep.roc(t.score, t.whale)
        pres[name] = {"auc": ep.auc(curve), "n_whale": int(t.whale.sum()), "n_free": int(t.whale_free.sum()),
                      "n_faint_only": int(tab.faint_only.sum()),
                      "at_dev_thr": ep.summarize(tab, fixed[name]) if name in fixed else None,
                      "op_far10": {k: float(v) for k, v in ep.operating_point(curve, max_far=0.10).items()},
                      "op_far2": {k: float(v) for k, v in ep.operating_point(curve, max_far=0.02).items()},
                      "sweep": {int(th): {k: ep.summarize(tab, th)[k] for k in ("hits", "n_pos", "false_alarms", "n_neg")}
                                for th in (1, 4, 8, 16, 32, 64)}}
        print("POOLED", name, "auc", round(pres[name]["auc"], 3), pres[name]["n_whale"], pres[name]["n_free"],
              {k: pres[name]["at_dev_thr"][k] for k in ("threshold", "hits", "n_pos", "false_alarms", "n_neg")},
              "sweep", {k: (v["hits"], v["false_alarms"]) for k, v in pres[name]["sweep"].items()})
    res["pooled_presence"] = pres
    res["pooled_masks"] = {k: v.summary() for k, v in pooled_masks.items() if v.frames}
    for k, s_ in res["pooled_masks"].items():
        print("POOLED masks", k, s_["frames"], {c: C.fmt(v, 2) for c, v in s_["iou_band"].items()},
              {g: (round(v["pixel_recall"], 2), v["instances"]) for g, v in s_["whale_recall_by_grade"].items()})
        cf = np.array(s_["conf"])
        for c in range(len(ps.CLASSES)):
            if cf[c].sum():
                print("   truth", ps.CLASSES[c], int(cf[c].sum()), "->", {ps.CLASSES[j] if j < len(ps.CLASSES) else "unknown": round(cf[c, j] / cf[c].sum(), 3) for j in range(cf.shape[1]) if cf[c, j] / cf[c].sum() > 0.01})
    if pooled:
        allf = pd.concat(pooled)
        allf.to_csv(ps.CACHE / f"test_frames_{a.pred}{a.tag}.csv", index=False)  # per-frame table: cache, not repo
    OUT.mkdir(exist_ok=True)
    (OUT / f"test_{a.pred}{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
