"""The tagged whale's own blow at a breath (orchestrator request 2026-11-22, after WP9): shallow boxes the scan
tiers hq / ok.

Source: the scan's window tables (`underwater_cache/scan/<deployment>/sensitive/windows/*.csv`), rows with tier hq or
ok and depth_m < 1; the box = the window's best box (best_frame, bx0..by1, detector pixels = 1920 px wide). Windows
of one recording closer than 2 s to an already chosen one are skipped (near copies); at most 150 boxes.
Labels: AI by-eye labels by the WP10 agent in breath_labels.json: whale (another whale really in the box), other
(blow / bubble burst / surface / mirrored back / the tagged whale's own body), unsure.

    pick     choose, decode the frames (1920 px), labelling sheets (box in white, a number, no scores)
    feat     (GPU) crop embeddings -> CACHE/breath/feat.npz
    score    before / after re-fit, leave one source out -> results/breath.json, CACHE/predictions_breath.parquet
    sheet    review_sheets/underwater/boxclf_breath_01.jpg (_02): half blow boxes, half real shallow whales
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402
from uwid import boxclf as B  # noqa: E402

warnings.filterwarnings("ignore")
SCAN = Path("/home/sat3737/NOAA_whales/underwater_cache/scan")
DEPL = {"d1": ("2025-2-3 Mini-mother 11-96", D.VIDEO / "2025-2-3 Mini-mother 11-96" / "Video"),
        "d2": ("TestSubset 2025-1-2 CATS -85", D.VIDEO / "TestSubset 2025-1-2 CATS -85" / "Video")}
X = D.CACHE / "breath"
CAP = 150


def pick():
    X.mkdir(parents=True, exist_ok=True)
    rows = []
    for src, (name, vdir) in DEPL.items():
        w = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(SCAN / name / "sensitive" / "windows" / "*.csv")))])
        s = w[w.tier.isin(["hq", "ok"]) & (w.depth_m < 1)].sort_values(["recording", "t0"])
        for rec, g in s.groupby("recording"):
            last = -1e9
            for r in g.itertuples():
                if r.t0 - last < 2:
                    continue
                last = r.t0
                rows.append(dict(source=src, stem=rec, k=int(r.k), t0=float(r.t0), frame=int(r.best_frame),
                                 bx0=r.bx0, by0=r.by0, bx1=r.bx1, by1=r.by1, tier=r.tier, depth_m=r.depth_m,
                                 whale_px=r.whale_px, cover=r.cover))
    s = pd.DataFrame(rows)
    if len(s) > CAP:
        s = s.sample(CAP, random_state=20261122)
    s = s.sort_values(["source", "stem", "t0"]).reset_index(drop=True)
    s["id"] = np.arange(1, len(s) + 1)
    # the object of the scan's detector rows that the box belongs to
    objs = []
    for r in s.itertuples():
        g = load_rows(r.source, r.stem, r.frame)
        u = B.group_objects(B.unique_boxes(g))
        iou = D.iou_xyxy(np.array([[r.bx0, r.by0, r.bx1, r.by1]]), u[["x0", "y0", "x1", "y1"]].to_numpy(float))[0]
        o = int(u.obj.iloc[int(np.argmax(iou))])
        rep = u[(u.obj == o) & u.rep].iloc[0]
        objs.append(dict(obj=o, x0=rep.x0, y0=rep.y0, x1=rep.x1, y1=rep.y1, match_iou=float(iou.max())))
    s = pd.concat([s, pd.DataFrame(objs)], axis=1)
    from uwid import video
    for (src, stem), g in s.groupby(["source", "stem"]):
        for fr in video.iter_frames_at(DEPL[src][1] / f"{stem}.mov", sorted(set(g.frame)), size=(1920, 1080)):
            cv2.imwrite(str(X / f"{stem}_{fr.index:06d}.jpg"), cv2.cvtColor(fr.image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    s.to_csv(X / "chosen.csv", index=False)
    print(len(s), "boxes", s.groupby(["source", "tier"]).size().to_dict())
    sd = X / "sheets"; sd.mkdir(exist_ok=True)
    tiles = []
    for r in s.itertuples():
        img = cv2.imread(str(X / f"{r.stem}_{r.frame:06d}.jpg"))
        sm = cv2.resize(img, (640, 360))
        sc = 640 / 1920
        cv2.rectangle(sm, (int(r.x0 * sc), int(r.y0 * sc)), (int(r.x1 * sc), int(r.y1 * sc)), (255, 255, 255), 2)
        t = np.zeros((385, 640, 3), np.uint8); t[:360] = sm
        cv2.putText(t, f"#{r.id}", (8, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(t)
    for k in range(0, len(tiles), 8):
        tt = tiles[k:k + 8]
        while len(tt) < 8:
            tt.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(sd / f"label_{k // 8 + 1:02d}.jpg"), np.vstack([np.hstack(tt[i:i + 2]) for i in range(0, 8, 2)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


_ROWS: dict = {}


def load_rows(src, stem, frame):
    key = (src, stem)
    if key not in _ROWS:
        name = DEPL[src][0]
        _ROWS[key] = pd.concat([pd.read_parquet(p) for p in
                                sorted(glob.glob(str(SCAN / name / "cache" / "det" / f"{stem}.mov" / "part_*.parquet")))])
    df = _ROWS[key]
    return df[df.frame == frame]


def feat():
    import torch
    from uwid import patchseg
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8.5e9 / torch.cuda.get_device_properties(0).total_memory))
    s = pd.read_csv(X / "chosen.csv")
    model = patchseg.load_backbone()
    T, C = [], []
    for r in s.itertuples():
        img = cv2.cvtColor(cv2.imread(str(X / f"{r.stem}_{r.frame:06d}.jpg")), cv2.COLOR_BGR2RGB)
        e = B.crop_embeddings(model, img, np.array([[r.x0, r.y0, r.x1, r.y1]]), (1.0, 2.0))
        T.append(e[1.0][0]); C.append(e[2.0][0])
    np.savez(X / "feat.npz", id=s.id.to_numpy(), tight=np.stack(T), ctx=np.stack(C))
    (X / "feat.DONE").write_text("ok")


def table():
    from table import baseline
    s = pd.read_csv(X / "chosen.csv")
    lab = json.loads((HERE / "breath_labels.json").read_text())["labels"]
    rows = []
    for r in s.itertuples():
        g = load_rows(r.source, r.stem, r.frame)
        u = B.group_objects(B.unique_boxes(g))
        grp = u[u.obj == r.obj]; rep = grp[grp.rep].iloc[0]
        ba, bb = baseline(grp, 1920, 1080)
        px = max(r.x1 - r.x0, r.y1 - r.y0)
        tr = lab.get(str(r.id), "unsure")
        rows.append(dict(id=r.id, source=r.source, stem=r.stem, frame=r.frame, x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1,
                         px=px, band=D.size_band(px), depth_m=r.depth_m, tier=r.tier, truth=tr,
                         truth3=tr if tr in ("whale", "other", "fish") else "ambiguous", base_a=ba, base_b=bb,
                         **{c: rep[c] for c in ("s_whale", "s_fish", "s_fish_prompt", "s_school_of_fish",
                                                "s_small_fish", "s_air_bubbles")}))
    t = pd.DataFrame(rows)
    z = np.load(X / "feat.npz")
    pos = {int(i): k for k, i in enumerate(z["id"])}
    ii = [pos[int(i)] for i in t.id]
    xy = t[["x0", "y0", "x1", "y1"]].to_numpy(float)
    parts = {"tight": z["tight"][ii].astype(np.float32), "ctx": z["ctx"][ii].astype(np.float32),
             "geom": B.geometry(xy, 1920, 1080), "scores": B.score_features(t)}
    return t, parts


def score():
    import train as T
    obj, parts = T.load_table()
    ext, ext_parts = T.load_extra()
    br, br_parts = table()
    blocks = T.FEATURE_SETS["tight+ctx"]
    CL = {c: i for i, c in enumerate(T.CLS3)}
    lab = obj.truth3.isin(T.CLS3).to_numpy()
    summ = json.loads((HERE / "results" / "train_summary.json").read_text())
    out, preds = {}, []
    for fold, (train_src, test_src) in T.FOLDS.items():
        if test_src == "dev":
            continue
        C = summ[f"tight+ctx|{fold}"]["C"]
        tr = lab & obj.source.isin(train_src).to_numpy()
        Xr, yr = T.X_of(parts, blocks, tr), obj.truth3[tr].map(CL).to_numpy()
        btr = (br.source.isin(train_src) & br.truth3.isin(T.CLS3)).to_numpy()
        bte = (br.source == test_src).to_numpy() & br.truth3.isin(T.CLS3).to_numpy()
        ete = ((ext.source == test_src) & ext.truth3.isin(T.CLS3)).to_numpy()
        rte = lab & (obj.source == test_src).to_numpy()
        for when, (Xt, yt) in {"before": (Xr, yr),
                               "after": (np.concatenate([Xr, T.X_of(br_parts, blocks, btr)]),
                                         np.concatenate([yr, br.truth3[btr].map(CL).to_numpy()]))}.items():
            s, m, _ = T.fit(Xt, yt, None, C=C)

            def pr(Xq):
                P = m.predict_proba(s.transform(Xq)); f = np.zeros((len(Xq), 3)); f[:, m.classes_] = P
                return np.array(T.CLS3)[f.argmax(1)], f
            pb, fb = pr(T.X_of(br_parts, blocks, bte))
            d = br[bte].copy(); d["pred"], d["p_whale"], d["fold"], d["when"] = pb, fb[:, 0], fold, when
            preds.append(d)
            pe, _ = pr(T.X_of(ext_parts, blocks, ete))
            e = ext[ete]
            prf, _ = pr(T.X_of(parts, blocks, rte))
            r = obj[rte]
            big = r.band.isin(["300-500", "500-800"]).to_numpy()
            out[f"{fold}|{when}"] = {
                "blow_boxes": int((d.truth3 == "other").sum()),
                "blow_called_whale_rule_A": int(((d.truth3 == "other") & (d.base_a == "whale")).sum()),
                "blow_called_whale_clf": int(((d.truth3 == "other") & (d.pred == "whale")).sum()),
                "blow_kept_A_plus_veto": int(((d.truth3 == "other") & (d.base_a == "whale") & (d.pred == "whale")).sum()),
                "shallow_whales": int((d.truth3 == "whale").sum()),
                "shallow_whales_lost_clf": int(((d.truth3 == "whale") & (d.pred != "whale")).sum()),
                "shallow_whales_lost_A_plus_veto": int(((d.truth3 == "whale") & ~((d.base_a == "whale") & (d.pred == "whale"))).sum()),
                "extra_fish": int((e.truth3 == "fish").sum()),
                "extra_fish_called_whale": int(((e.truth3 == "fish") & (pe == "whale")).sum()),
                "extra_whales_lost": int(((e.truth3 == "whale") & (pe != "whale")).sum()),
                "ref_fish_300_800": int(((r.truth3 == "fish").to_numpy() & big).sum()),
                "ref_fish_300_800_called_whale": int(((r.truth3 == "fish").to_numpy() & big & (prf == "whale")).sum()),
                "ref_whale_ge300_lost": int(((r.truth3 == "whale").to_numpy() & (r.px >= 300).to_numpy() & (prf != "whale")).sum()),
                "ref_whale_ge300": int(((r.truth3 == "whale").to_numpy() & (r.px >= 300).to_numpy()).sum()),
            }
            print(fold, when, out[f"{fold}|{when}"], flush=True)
    (HERE / "results" / "breath.json").write_text(json.dumps(out, indent=1))
    pd.concat(preds, ignore_index=True).to_parquet(D.CACHE / "predictions_breath.parquet", index=False)
    # delivered models. The leave-one-source-out runs above show the re-fit with breath boxes is not better (2025-01-02:
    # blow boxes kept 0 -> 4 of 43), so the DEFAULT stays the model without them (reference + extra boxes, as train.py
    # saves it); the one with them is saved beside it for comparison.
    ref_X = [T.X_of(parts, blocks, lab), T.X_of(ext_parts, blocks, ext.truth3.isin(T.CLS3).to_numpy())]
    ref_y = [obj.truth3[lab].map(CL).to_numpy(), ext.truth3[ext.truth3.isin(T.CLS3)].map(CL).to_numpy()]
    brX, bry = T.X_of(br_parts, blocks, br.truth3.isin(T.CLS3).to_numpy()), br.truth3[br.truth3.isin(T.CLS3)].map(CL).to_numpy()
    for path, Xs, ys, what in ((B.DEFAULT_WEIGHTS, ref_X, ref_y, "dev + d1 + d2 reference boxes + 81 extra 300-800 px boxes"),
                               (B.CACHE / "boxclf_v1_with_breath.npz", ref_X + [brX], ref_y + [bry],
                                "dev + d1 + d2 reference boxes + 81 extra boxes + 133 shallow breath boxes")):
        Xa, ya = np.concatenate(Xs), np.concatenate(ys).astype(int)
        s, m, C = T.fit(Xa, ya, None, C=0.0003)
        meta = {"trained_on": what + " (AI by-eye labels; ambiguous / unsure left out)", "n": int(len(ya)),
                "counts": {c: int((ya == i).sum()) for c, i in CL.items()}, "C": C, "features": list(blocks)}
        B.LinearBoxModel.from_sklearn(s, m, B.FeatureSpec(tuple(blocks)), meta).save(path)
        print("saved", path, meta)


def sheet():
    P = pd.read_parquet(D.CACHE / "predictions_breath.parquet")
    LOSO = {"d1": "test d1", "d2": "test d2"}
    P = P[(P.when == "before") & (P.fold == P.source.map(LOSO))]   # the delivered model has no breath boxes
    rng = np.random.default_rng(3)
    blow = P[P.truth3 == "other"]; wh = P[P.truth3 == "whale"]
    n = min(12, len(blow), len(wh))
    sel = pd.concat([blow.iloc[rng.choice(len(blow), n, replace=False)], wh.iloc[rng.choice(len(wh), min(len(wh), 12), replace=False)]])
    col = {"whale": (40, 60, 255), "fish": (0, 220, 255), "other": (170, 170, 170)}
    tiles = []
    for r in sel.itertuples():
        img = cv2.imread(str(X / f"{r.stem}_{r.frame:06d}.jpg"))
        sm = cv2.resize(img, (480, 270)); sc = 480 / 1920
        keep = (r.base_a == "whale") and (r.pred == "whale")
        dec = "whale" if keep else ("other" if r.pred != "fish" else "fish")
        cv2.rectangle(sm, (int(r.x0 * sc), int(r.y0 * sc)), (int(r.x1 * sc), int(r.y1 * sc)), col[dec], 2)
        t = np.zeros((320, 480, 3), np.uint8); t[:270] = sm
        cv2.putText(t, f"truth {'blow/surface/back' if r.truth3 == 'other' else 'whale'} | decision {dec} (p whale {r.p_whale:.2f})",
                    (5, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(t, f"{r.source} {D.piece(r.stem)} f{r.frame} depth {r.depth_m:.1f} m  {r.px:.0f} px  was {r.tier}",
                    (5, 311), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(t)
    for old in D.SHEETS.glob("boxclf_breath_*.jpg"):
        old.unlink()
    for k in range(0, len(tiles), 12):
        tt = tiles[k:k + 12]
        while len(tt) % 3:
            tt.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(D.SHEETS / f"boxclf_breath_{k // 12 + 1:02d}.jpg"),
                    np.vstack([np.hstack(tt[i:i + 3]) for i in range(0, len(tt), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    {"pick": pick, "feat": feat, "score": score, "sheet": sheet}[sys.argv[1]]()
