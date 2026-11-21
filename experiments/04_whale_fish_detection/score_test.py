"""Score the 2025-02-03 test recordings with the operating point fixed on the development video (nothing is tuned
here). Needs the OWLv2 run ``owl_owl_all_s15`` of each recording (``logs/owl_test.sh``) and WP3's labels
``experiments/02_reference_labels/labels/<stem>/frames.csv``.

    .venv/bin/python experiments/04_whale_fish_detection/score_test.py            # every recording with labels
    .venv/bin/python experiments/04_whale_fish_detection/score_test.py --sheets   # + review sheets

Operating point (from operating_point.py / report_dev.py on the dev video): OWLv2-base prompt "whale", boxes in the
burned-in clock or covering > 60 % of the frame dropped, 2 frames per 1 s window (every 15th frame), a window has a
whale when both frames have a box >= 0.25. Single-frame numbers use the labelled frames at the same threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import FISH_Q, REPO, SHEETS, detect, ep, fps_of, frame_scores, tag_boxes, to_md

VD = Path("/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96/Video")
LABELS = REPO / "experiments/02_reference_labels/labels"
RUN, QUERY, THR, STRIDE = "owl_owl_all_s15", "whale", 0.25, 15
PIECES = ["00004", "00005", "00006", "00007", "00008", "00009", "00010"]   # the pieces WP3 labelled


def test_videos():
    return [v for p in PIECES for v in sorted(VD.glob(f"CC-11-96-20250203-*-{p}.m4v"))]
DROP = ("clock", "whole")


def score_one(video: Path, sheets: bool):
    stem = video.stem
    lab_csv = LABELS / stem / "frames.csv"
    if not lab_csv.exists():
        return None, f"{stem}: no labels yet"
    b, frames = detect.load_cached(video, RUN)
    if len(frames) == 0:
        return None, f"{stem}: no detections cached"
    labels = ep.load_frame_labels(lab_csv)
    fps = fps_of(str(video))
    b = tag_boxes(b)
    grid = frames[frames % STRIDE == 0]
    fs = frame_scores(b, grid, [QUERY], DROP)
    truth = ep.window_truth(labels, fps)
    win = truth.merge(ep.window_scores(fs, fps, rule="kth", k=2), on="window", how="left")
    win_any = truth.merge(ep.window_scores(fs, fps, rule="max"), on="window", how="left")
    fr = labels.merge(frame_scores(b, frames, [QUERY], DROP), on="frame", how="left")
    fr["faint_only"] = ~fr.whale & fr.faint
    groups = lambda t, n: {"bubbles": t.bubbles & t.whale, "clear view": ~t.bubbles & t.whale,
                           "1 whale": (t[n] == 1) & t.whale, ">=2 whales": (t[n] >= 2) & t.whale,
                           "underwater negatives": t.whale_free & (t.state == "underwater"),
                           "air/crossing negatives": t.whale_free & (t.state != "underwater")}
    res = dict(video=stem, n_labelled=len(labels),
               window_both=ep.summarize(win, THR, groups=groups(win, "max_whales")),
               window_any=ep.summarize(win_any, THR, groups=groups(win_any, "max_whales")),
               single_frame=ep.summarize(fr, THR, groups=groups(fr, "other_whales")))
    for k, t, s in (("window_both", win, "score"), ("single_frame", fr, "score")):
        m = t.whale | t.whale_free
        c = ep.roc(t.loc[m, s], t.loc[m, "whale"])
        res[k]["auc"] = ep.auc(c) if t.loc[m, "whale"].any() and (~t.loc[m, "whale"]).any() else None
    win.assign(video=stem).to_csv(detect.detect_cache_dir(video, RUN) / "test_windows.csv", index=False)
    fr.assign(video=stem).to_csv(detect.detect_cache_dir(video, RUN) / "test_frames.csv", index=False)
    if sheets:
        test_sheets(video, b, win, fps)
    return (win.assign(video=stem), fr.assign(video=stem), res), None


def test_sheets(video, b, win, fps):
    from make_sheets import draw, grid
    stem = video.stem
    wb = b[(b["query"] == QUERY) & ~b.clock & ~b.whole]
    fa = win[win.whale_free & (win.score >= THR)]
    miss = win[win.whale & (win.score.fillna(0) < THR)]
    pick = lambda ws: [int(f) for w in ws for f in range(int(np.ceil(w * fps)), int(np.ceil((w + 1) * fps)))
                       if f % STRIDE == 0][:48]
    fa_f, miss_f = pick(fa.window)[:32], pick(miss.window)[:32]
    imgs = dict(detect.iter_frames_cv(video, frames=sorted(set(fa_f + miss_f))))
    t = lambda f, thr: f"{stem[-5:]} f{f} t={f / fps:.1f}s win {ep.window_of(f, fps)} | OWLv2 'whale' >= {thr}"
    grid([draw(imgs[f], wb[(wb.frame == f) & (wb.score >= THR)], t(f, THR)) for f in fa_f if f in imgs], 4,
         SHEETS / f"detect_test_{stem[-5:]}_false_alarms.jpg",
         f"{stem}: windows labelled whale-free that fired (both frames >= {THR})")
    grid([draw(imgs[f], wb[(wb.frame == f) & (wb.score >= 0.1)], t(f, 0.1)) for f in miss_f if f in imgs], 4,
         SHEETS / f"detect_test_{stem[-5:]}_misses.jpg",
         f"{stem}: whale windows missed (boxes >= 0.1 drawn)")


FISH_THR = 0.5   # from the dev clip's false-alarm side only (no fish there): ~1.9 both-frame windows per minute


def fish_test(videos):
    """Fish on the test recordings: per labelled frame and per window (any fish prompt, max score), the curve, the
    fixed dev threshold, instance recall against WP3's fish boxes (fish_calls.json), and fish boxes lying on a
    whale box of the same frame."""
    import json as _json
    from common import iou_one
    fr_all, win_all, inst = [], [], []
    on_whale = dict(fish_boxes=0, on_whale_box=0)
    for v in videos:
        stem = v.stem
        lab_csv = LABELS / stem / "frames.csv"
        b, frames = detect.load_cached(v, RUN)
        if not lab_csv.exists() or len(frames) == 0:
            continue
        fps = fps_of(str(v))
        b = tag_boxes(b)
        labels = ep.load_frame_labels(lab_csv)
        fb = b[b["query"].isin(FISH_Q) & ~b.clock & ~b.whole]
        fr = labels.merge(frame_scores(b, frames, FISH_Q, DROP), on="frame", how="left").assign(video=stem)
        fr_all.append(fr)
        grid = frames[frames % STRIDE == 0]
        w = ep.window_truth(labels, fps).merge(
            ep.window_scores(frame_scores(b, grid, FISH_Q, DROP), fps, rule="kth", k=2), on="window", how="left")
        win_all.append(w.assign(video=stem))
        calls = LABELS / stem / "fish_calls.json"
        if calls.exists():
            for c in _json.loads(calls.read_text()):
                sel = fb[fb.frame == c["frame"]]
                for bx in c["boxes"]:
                    a = np.array(bx, float)
                    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
                    best = 0.0
                    for _, r in sel.iterrows():
                        iou = iou_one(a, np.array([[r.x0, r.y0, r.x1, r.y1]]))[0]
                        inside = r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 and (r.x1 - r.x0) <= 8 * max(a[2] - a[0], 20)
                        if iou >= 0.1 or inside:
                            best = max(best, r.score)
                    inst.append(dict(video=stem, frame=c["frame"], size=float(max(a[2] - a[0], a[3] - a[1])), best=best))
        wb = b[(b["query"] == QUERY) & (b.score >= THR) & ~b.clock & ~b.whole]
        for f, g in fb[fb.score >= FISH_THR].groupby("frame"):
            wf = wb[wb.frame == f][["x0", "y0", "x1", "y1"]].to_numpy(float)
            for _, r in g.iterrows():
                on_whale["fish_boxes"] += 1
                if len(wf) and iou_one(np.array([r.x0, r.y0, r.x1, r.y1]), wf).max() >= 0.5:
                    on_whale["on_whale_box"] += 1
    if not fr_all:
        print("fish: nothing to score")
        return None
    F, Wn, I = pd.concat(fr_all), pd.concat(win_all), pd.DataFrame(inst)
    F = F.assign(fish=F.fish_present.astype(bool))
    Wn = Wn.assign(fish=Wn.fish.astype(bool))
    out = {}
    for name, t in (("labelled frames", F), ("1 s windows (both frames)", Wn)):
        c = ep.roc(t.score, t.fish)
        s = ep.summarize(t.assign(neg=~t.fish), FISH_THR, positive="fish", negative="neg")
        out[name] = dict(n_fish=int(t.fish.sum()), n_none=int((~t.fish).sum()), auc=ep.auc(c),
                         hits=s["hits"], false_alarms=s["false_alarms"],
                         recall_at_FA05=ep.operating_point(c, max_far=0.05)["recall"])
        print(f"fish, {name}: fish {out[name]['n_fish']} / none {out[name]['n_none']}; AUC {out[name]['auc']:.3f}; "
              f"at dev thr {FISH_THR}: hits {s['hits']}/{s['n_pos']}, false {s['false_alarms']}/{s['n_neg']}; "
              f"recall at FA<=5 %: {out[name]['recall_at_FA05']:.3f}")
    if len(I):
        for t in (0.1, 0.2, 0.3, FISH_THR):
            print(f"  fish instances (WP3 boxes, {len(I)}): found at >= {t}: {int((I.best >= t).sum())}")
        out["instances"] = dict(n=len(I), **{f"found_{t}": int((I.best >= t).sum()) for t in (0.1, 0.2, 0.3, FISH_THR)},
                                median_size_px=float(I["size"].median()))
    out["fish_boxes_on_whale_boxes"] = on_whale
    print("  fish boxes >= %.1f lying on a 'whale' box >= %.2f (IoU >= 0.5):" % (FISH_THR, THR), on_whale)
    return out


SAM3_RUN = "sam3_sam3_test_s99999999_lab"


def sam3_test(videos, owl_frames=None):
    """SAM 3 on the labelled test frames (single frames): whale, fish, and seafloor (none labelled: all false)."""
    rows = []
    for v in videos:
        lab_csv = LABELS / v.stem / "frames.csv"
        b, frames = detect.load_cached(v, SAM3_RUN)
        if not lab_csv.exists() or len(frames) == 0:
            continue
        b = tag_boxes(b)
        lab = ep.load_frame_labels(lab_csv)
        lab = lab[lab.frame.isin(frames)]
        for q in ("whale", "fish", "seafloor"):
            fs = frame_scores(b, frames, [q], ("clock",))
            area = b[(b["query"] == q)].groupby("frame").area_frac.sum() if "area_frac" in b else None
            t = lab.merge(fs, on="frame", how="left")
            t["area"] = t.frame.map(area).fillna(0) if area is not None else 0
            rows.append(t.assign(video=v.stem[-5:], query=q))
    if not rows:
        print("SAM 3 test: nothing cached")
        return None
    d = pd.concat(rows)
    out = {}
    for q, pos in (("whale", "whale"), ("fish", "fish_present")):
        t = d[d["query"] == q]
        m = (t.whale | t.whale_free) if q == "whale" else t.index == t.index
        c = ep.roc(t.score[m], t[pos][m])
        neg = ~t[pos] & m
        out[q] = dict(n_pos=int(t[pos][m].sum()), n_neg=int(neg.sum()), auc=ep.auc(c),
                      **{f"at_{th}": f"{int(((t.score >= th) & t[pos] & m).sum())} hits, {int(((t.score >= th) & neg).sum())} false"
                         for th in (0.3, 0.5, 0.7)},
                      recall_FA0=ep.operating_point(c, max_far=0.0)["recall"])
    t = d[(d["query"] == "seafloor") & (d.state == "underwater")]
    out["seafloor_false"] = {f"score>={th}": int((t.score >= th).sum()) for th in (0.3, 0.5, 0.7)}
    out["seafloor_false"]["n_underwater_frames"] = int(len(t))
    out["seafloor_false"]["area>5%"] = int((t.area > 0.05).sum())
    print("SAM 3 on labelled test frames:", json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--fish-sheets", action="store_true",
                    help="sheet of the confident fish boxes (>= FISH_THR, best per window) of each whole recording")
    a = ap.parse_args()
    if a.fish_sheets:
        from make_sheets import fish_sheet
        for v in test_videos():
            if len(detect.load_cached(v, RUN)[1]):
                fish_sheet(v, RUN, FISH_THR, f"detect_fish_test_{v.stem[-5:]}.jpg", stride=STRIDE)
    fish_res = fish_test(test_videos())
    sam3_res = sam3_test(test_videos())
    wins, frs, rows = [], [], []
    for v in test_videos():
        out, msg = score_one(v, a.sheets)
        if out is None:
            print(msg)
            continue
        w, f, r = out
        wins.append(w)
        frs.append(f)
        rows.append(r)
    if not rows:
        print("nothing scored")
        return
    W_, F_ = pd.concat(wins), pd.concat(frs)
    pooled = dict(video="pooled",
                  window_both=ep.summarize(W_, THR), window_any=ep.summarize(W_.assign(score=np.nan), THR),
                  single_frame=ep.summarize(F_, THR))
    tab = []
    for r in rows + [pooled]:
        for k in ("single_frame", "window_any", "window_both"):
            s = r.get(k)
            if not s or (r["video"] == "pooled" and k == "window_any"):
                continue
            tab.append(dict(video=r["video"] if r["video"] == "pooled" else r["video"][-5:], scoring=k, hits=f"{s['hits']}/{s['n_pos']}",
                            false_alarms=f"{s['false_alarms']}/{s['n_neg']}",
                            faint_only_fired=f"{s['fire_other']}/{s['n_other']}",
                            bubbles=f"{s.get('hits_bubbles', '')}/{s.get('n_bubbles', '')}",
                            one_whale=f"{s.get('hits_1 whale', '')}/{s.get('n_1 whale', '')}",
                            uw_neg_FA=f"{s.get('hits_underwater negatives', '')}/{s.get('n_underwater negatives', '')}",
                            auc=s.get("auc")))
    print(to_md(pd.DataFrame(tab)))
    m = W_.whale | W_.whale_free
    c = ep.roc(W_.score[m], W_.whale[m])
    print("\npooled window curve (both-frames score):")
    for far in (0.0, 0.01, 0.02, 0.05, 0.1):
        o = ep.operating_point(c, max_far=far)
        print(f"  FA <= {far:.2f}: recall {o['recall']:.3f} ({o['tp']}/{o['n_pos']}) at thr {o['threshold']:.3f}, FA {o['fp']}/{o['n_neg']}")
    out = Path(__file__).resolve().parent / "test_results.json"
    out.write_text(json.dumps(dict(rows=rows, pooled=pooled, auc_pooled_window=ep.auc(c), fish=fish_res, sam3=sam3_res), indent=1,
                              default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
