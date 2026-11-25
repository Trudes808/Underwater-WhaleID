"""Review sheets of WP7 (to `review_sheets/underwater/scan_*.jpg`) and the dropped-window check.

    PY=.venv/bin/python; S=experiments/06_clips_e2e/sheets.py
    $PY $S timelines                 # scan_timeline_<piece>.jpg for 00004, 00012 and the dev clip
    $PY $S dropcheck                 # 150 random 'underwater with no whale' windows + 30 'small' ones, seed below
    $PY $S dropcheck --tally         # after the calls are written to results/dropcheck_calls.json
    $PY $S hq [--dep NAME]           # scan_hq_top / border / rejected (scan_d2_* for CATS -85), decisions drawn
    $PY $S bands [--dep NAME]        # scan_hq_band_050 / 040 / 030: 24 random windows per hq band below 0.6
    $PY $S midsize                   # scan_midsize: kept windows resting on a 300-500 px box, both deployments
    $PY $S d2_timelines | d2_clips | d2_dropcheck [--tally] | d2_spans     # second deployment (scan_d2_*)
    $PY $S lost                      # labelled clear-whale frames that fall outside the kept stretches
    $PY $S clips                     # first / middle / last frame of 12 kept clips + 3 clips verified by decoding
    $PY $S kept --dep NAME           # a sample of kept windows of any scanned deployment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from uwid import paths, pipeline as pl, sheets, video  # noqa: E402

DEP = "2025-2-3 Mini-mother 11-96"
DEV = "2025-12-20_CG"
SCAN = paths.cache_dir("scan")
OUTD = paths.sheets_dir(create=True)
RES = HERE / "results"
SEED = 20261120
LABEL = "sensitive"


def windows(dep=DEP, label=LABEL) -> pd.DataFrame:
    d = SCAN / dep / label / "windows"
    return pd.concat([pd.read_csv(f).assign(key=f.name[:-4]) for f in sorted(d.glob("*.csv"))], ignore_index=True)


def source_of(dep, key, label=None) -> Path:
    iv = json.loads((SCAN / dep / (label or LABEL) / "intervals.json").read_text())
    for v in iv["videos"]:
        p = Path(v["source"])
        if f"{p.stem.replace(' ', '_')}.{p.suffix[1:]}" == key:
            return p
    raise KeyError(key)


DECISION_COL = {"whale": (255, 50, 50), "small": (255, 165, 0), "fish": (40, 210, 255), "other": (170, 170, 170)}


def draw_decisions(img: np.ndarray, objs: pd.DataFrame, det_w: int) -> np.ndarray:
    """Draw the pipeline's decision per object (one box each): red whale, orange small, cyan fish (information)."""
    import cv2
    sc = img.shape[1] / float(det_w)
    th = max(2, int(round(img.shape[1] / 640)))
    fs = img.shape[1] / 2000.0
    for _, o in objs.iloc[::-1].iterrows():          # whales drawn last, on top
        col = DECISION_COL[o.label]
        x0, y0, x1, y1 = (int(round(v * sc)) for v in (o.x0, o.y0, o.x1, o.y1))
        cv2.rectangle(img, (x0, y0), (x1, y1), col, th * (2 if o.label == "whale" else 1))
        txt = f"{o.label} {o.score:.2f}" + (f" {o.px:.0f}px" if o.label != "fish" else "")
        if bool(o.get("clf", False)):
            txt += " (clf)"
        big = 2.0 if o.label == "whale" else 1.4
        # near the top edge (under the tile's time label) the text goes inside the box at its bottom
        ty = y1 - 10 if y0 < img.shape[0] * 0.12 else y0 - 6
        tx = min(max(x0 + 4, 4), img.shape[1] - int(fs * big * 20 * len(txt)))
        cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs * big, (0, 0, 0), th * 4)
        cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs * big, col, th + (1 if o.label == "whale" else 0))
    return img


def grab(dep, rows: pd.DataFrame, which: str = "best", draw_boxes: bool = True, cfg=None, label=None):
    """One frame per window row: the frame with the largest whale box (``best``) or the window centre, with the
    pipeline's decision drawn (one box per object, :func:`uwid.pipeline.decision_objects`)."""
    cfg = cfg or pl.ScanConfig()
    out = []
    for key, g in rows.groupby("key", sort=False):
        src = source_of(dep, key, label)
        rec = pl.Recording(src, src.suffix[1:])
        boxes, _ = pl.load_boxes(SCAN / dep / "cache", rec)
        boxes = pl.attach_box_classes(boxes, pl.load_picture_classes(SCAN / dep / "cache", rec)[0])
        det_w, det_h = pl.detector_frame_size(SCAN / dep / "cache", rec)
        with video.VideoReader(src) as r:
            for i, row in g.iterrows():
                if "show_frame" in row and np.isfinite(row.get("show_frame", np.nan)):
                    f_idx = int(row.show_frame)
                elif which == "best" and "best_frame" in row and np.isfinite(row.get("cover", np.nan)):
                    f_idx = int(row.best_frame)
                elif which == "best" and np.isfinite(row.get("whale_1", np.nan)) and row.get("whale_1", 0) > row.get(
                        "whale_0", 0):
                    f_idx = int(row.frame_1)
                elif which == "best":
                    f_idx = int(row.frame_0)
                elif isinstance(which, (int, np.integer)):
                    f_idx = int(which)
                else:
                    f_idx = r.table.index_at(row.t0 + 0.5)
                fr = r.frame_by_index(f_idx)
                if fr is None:
                    continue
                img = fr.image.copy()
                if draw_boxes and len(boxes):
                    objs = pl.decision_objects(boxes[boxes.frame == f_idx], cfg, det_w, det_h,
                                               shallow=bool(row.get("shallow_rule", False)))
                    img = draw_decisions(img, objs, det_w)
                out.append((i, fr, img))
    order = {i: k for k, i in enumerate(rows.index)}
    return sorted(out, key=lambda x: order[x[0]])


def write_sheets(items, name: str, cols: int, tile: int, per: int, title: str, caption):
    paths_out = []
    for s in range(0, len(items), per):
        chunk = items[s:s + per]
        imgs = [im for _, _, im in chunk]
        labels = [caption(i, fr)[0] for i, fr, _ in chunk]
        caps = [caption(i, fr)[1] for i, fr, _ in chunk]
        out = OUTD / f"{name}_{s // per + 1:02d}.jpg"
        sheets.make_sheet(imgs, labels, caps, cols=cols, tile_width=tile,
                          title=f"{title} ({s + 1}-{s + len(chunk)} of {len(items)})", out=out)
        paths_out.append(str(out))
    return paths_out


# ------------------------------------------------------------------------------------------- timelines

STATE_COL = {"air": "#9e9e9e", "flight": "#ff9800", "floating": "#9c27b0", "whale": "#2e7d32", "unknown": "#e0e0e0"}
WATER_COL = {"underwater": "#1565c0", "crossing": "#64b5f6", "surfaced": "#ffeb3b", "air": "#bdbdbd",
             "waterline": "#ce93d8", "unknown": "#eeeeee"}


def timeline(dep, key, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = {lb: pd.read_csv(SCAN / dep / lb / "windows" / f"{key}.csv") for lb in ("sensitive", "strict")
          if (SCAN / dep / lb / "windows" / f"{key}.csv").exists()}
    w = ws["sensitive"]
    fig, ax = plt.subplots(5, 1, figsize=(22, 8.5), sharex=True,
                           gridspec_kw={"height_ratios": [0.6, 0.6, 2.2, 1.0, 0.6]})
    x = w.t0.to_numpy()
    for a, col, cmap in ((ax[0], "tag_state", STATE_COL), (ax[1], "water", WATER_COL)):
        for st, c in cmap.items():
            m = w[col] == st
            if m.any():
                a.bar(x[m], 1, width=1.0, align="edge", color=c, label=st)
        a.set_yticks([])
        a.legend(loc="upper left", bbox_to_anchor=(1.0, 1.15), fontsize=8, frameon=False)
    src = "/".join(sorted(set(w.tag_source))) + " | " + "/".join(sorted(set(w.water_source)))
    ax[0].set_ylabel("tag", rotation=0, ha="right")
    ax[1].set_ylabel("water", rotation=0, ha="right")
    ax[2].plot(x, w.whale_0, ".", ms=2, color="#c62828", label="whale score, frame +0.25 s")
    ax[2].plot(x, w.whale_1, ".", ms=2, color="#6a1b9a", label="frame +0.75 s")
    ax[2].plot(x, w.small_score, "x", ms=2, color="#ff9800", label="small box score (not kept)")
    ax[2].axhline(0.25, color="k", lw=0.8)
    ax[2].set_ylim(0, 1)
    ax[2].set_ylabel("whale\nscore", rotation=0, ha="right")
    ax[2].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax[3].plot(x, w.fish_score, ".", ms=2, color="#00838f", label="fish score (info)")
    ax[3].plot(x, w.hq_score, ".", ms=3, color="#2e7d32", label="hq score")
    ax[3].set_ylim(0, 1)
    ax[3].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    for k, (lb, ww) in enumerate(ws.items()):
        for col, y0, c in (("kept", 1 - k * 0.5, "#43a047"),):
            m = ww[col].astype(bool)
            ax[4].bar(ww.t0[m], 0.45, bottom=y0 - 0.5, width=1.0, align="edge", color=c)
        m = ww.hq_whale.astype(bool)
        ax[4].bar(ww.t0[m], 0.2, bottom=1 - k * 0.5 - 0.5, width=1.0, align="edge", color="#ffd600")
    ax[4].set_yticks([0.25, 0.75])
    ax[4].set_yticklabels(list(ws)[::-1] if len(ws) == 2 else list(ws))
    ax[4].set_xlabel("video time (s)")
    ax[4].set_ylabel("kept\n(yellow = hq)", rotation=0, ha="right")
    kept = {lb: ww.kept.sum() / 60 for lb, ww in ws.items()}
    fig.suptitle(f"{title}: kept " + ", ".join(f"{lb} {v:.1f} min" for lb, v in kept.items())
                 + f" of {len(w) / 60:.1f} min   (sources: {src})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return str(out)


def cmd_timelines(a):
    w = windows()
    keys = sorted(w.key.unique())
    outs = []
    for piece in ("00004", "00012"):
        key = next(k for k in keys if f"-{piece}." in k)
        outs.append(timeline(DEP, key, OUTD / f"scan_timeline_{piece}.jpg", f"{DEP} {piece}"))
    dk = next((SCAN / DEV / "sensitive" / "windows").glob("*.csv")).name[:-4]
    outs.append(timeline(DEV, dk, OUTD / "scan_timeline_dev.jpg", "2025-12-20 CG (no sensor log)"))
    print("\n".join(outs))


# ------------------------------------------------------------------------------------------- dropped check

def cmd_dropcheck(a):
    RES.mkdir(exist_ok=True)
    w = windows()
    pool = w[w.why.isin(["underwater with no whale", "nothing detected"])]
    small = w[(w.why == "small (fish or far whale)")]
    rng = np.random.default_rng(SEED)
    pick = pool.loc[np.sort(rng.choice(pool.index, size=min(150, len(pool)), replace=False))]
    pick_s = small.loc[np.sort(rng.choice(small.index, size=min(30, len(small)), replace=False))]
    sample = pd.concat([pick.assign(set="no whale"), pick_s.assign(set="small")])
    sample["id"] = np.arange(len(sample))
    sample[["id", "set", "key", "k", "t0", "clock", "whale_score", "small_score", "fish_score", "depth_m"]].to_csv(
        RES / "dropcheck_sample.csv", index=False)
    calls_f = RES / "dropcheck_calls.json"
    if a.tally:
        calls = json.loads(calls_f.read_text())
        sample["call"] = sample.id.map(lambda i: calls.get(str(i), "?"))
        tab = sample.groupby(["set", "call"]).size().unstack(fill_value=0)
        print(tab)
        found = sample[sample.call.isin(["large whale", "small whale", "faint whale"])]
        res = {"seed": SEED, "pool_no_whale": int(len(pool)), "pool_small": int(len(small)),
               "counts": {s: g.call.value_counts().to_dict() for s, g in sample.groupby("set")},
               "found": found[["id", "set", "key", "k", "clock", "call", "whale_score", "small_score"]].to_dict("records")}
        (RES / "dropcheck_result.json").write_text(json.dumps(res, indent=1, default=str))
        if len(found):
            items = grab(DEP, found, which="centre")
            cap = lambda i, fr: (f"#{found.loc[i, 'id']} {found.loc[i, 'key'][-9:-4]} {found.loc[i, 'clock']}",
                                 f"{found.loc[i, 'call']} | dropped as {found.loc[i, 'set']}, whale "
                                 f"{found.loc[i, 'whale_score']:.2f} small {found.loc[i, 'small_score']:.2f}")
            print(write_sheets(items, "scan_dropcheck_found", 2, 960, 6,
                               "whales found in dropped windows (AI by-eye)", cap))
        return
    items = grab(DEP, sample, which="centre")
    cap = lambda i, fr: (f"#{sample.loc[i, 'id']}  {sample.loc[i, 'key'][-9:-4]}  {sample.loc[i, 'clock']}",
                         f"{sample.loc[i, 'set']}: whale {sample.loc[i, 'whale_score']:.2f} small "
                         f"{sample.loc[i, 'small_score']:.2f} fish {sample.loc[i, 'fish_score']:.2f}")
    print(write_sheets(items, "scan_dropcheck", 2, 960, 6,
                       f"dropped windows, seed {SEED}: centre frame (boxes >= 0.2 drawn)", cap))


# ------------------------------------------------------------------------------------------- hq sheets

def hq_caption(w):
    def cap(i, fr):
        r = w.loc[i]
        parts = " ".join(f"{k[3:]}{r[k]:.2f}" for k in ("hq_cover", "hq_score_", "hq_bright", "hq_contrast",
                                                        "hq_sharp", "hq_bubbles", "hq_depth") if k in r and
                         np.isfinite(r[k]))
        return (f"{r.key[-9:-4]} {r.clock} hq {r.hq_score:.2f}",
                f"cov {r.cover:.2f} sc {r.top_score:.2f} br {r.bright:.2f} ct {r.contrast:.3f} sh {r.sharp:.1f} "
                f"bub {r.bubbles:.2f} d {r.depth_m:.0f}m")
    return cap


def prefix(dep) -> str:
    """Sheet name prefix: ``scan_`` for the first deployment, ``scan_d2_`` for CATS -85."""
    return "scan_d2_" if "CATS -85" in dep else "scan_"


def hq_reason(cfg):
    def reason(r):
        if r.cover < cfg.hq_min_cover:
            return f"whale box {r.cover:.2f} of frame < {cfg.hq_min_cover}"
        parts = {k[3:]: r[k] for k in ("hq_cover", "hq_score", "hq_bright", "hq_contrast", "hq_sharp", "hq_bubbles",
                                         "hq_depth") if k in r and k != "hq_score" and np.isfinite(r[k])}
        low = min(parts, key=parts.get)
        return f"hq {r.hq_score:.2f} < {cfg.hq_threshold}; lowest part {low} {parts[low]:.2f}"
    return reason


def cmd_hq(a):
    """Best 40, 40 nearest the threshold, 40 random rejects; same windows as before for the first deployment
    (results/hq_sheet_windows.json is reused when it exists)."""
    dep = a.dep
    w = windows(dep)
    cfg = pl.ScanConfig()
    pre = prefix(dep)
    large = w[w.whale & w.cover.notna()].copy()
    saved = RES / ("hq_sheet_windows.json" if pre == "scan_" else "hq_sheet_windows_d2.json")
    if saved.exists() and not a.redraw_new:
        sv = json.loads(saved.read_text())
        pick = lambda lst: w.set_index(["key", "k"]).loc[[(d["key"], d["k"]) for d in lst]].reset_index().set_index(
            pd.Index([w.index[(w.key == d["key"]) & (w.k == d["k"])][0] for d in lst]))
        top, border, rej = pick(sv["top"]), pick(sv["border"]), pick(sv["rejected"])
    else:
        top = large.sort_values("hq_score", ascending=False).head(40)
        cov = large[(large.cover >= cfg.hq_min_cover)]
        border = cov.iloc[(cov.hq_score - cfg.hq_threshold).abs().argsort()[:40]].sort_values("hq_score",
                                                                                               ascending=False)
        rej = large[~large.hq_whale]
        rng = np.random.default_rng(SEED)
        rej = rej.loc[np.sort(rng.choice(rej.index, size=min(40, len(rej)), replace=False))]
        RES.mkdir(exist_ok=True)
        saved.write_text(json.dumps({
            "top": top[["key", "k", "clock", "hq_score", "cover", "depth_m"]].to_dict("records"),
            "border": border[["key", "k", "clock", "hq_score", "cover", "depth_m"]].to_dict("records"),
            "rejected": rej[["key", "k", "clock", "hq_score", "cover"]].to_dict("records")}, indent=1, default=str))
    reason = hq_reason(cfg)
    outs = []
    cap = hq_caption(w)
    outs += write_sheets(grab(dep, top), pre + "hq_top", 3, 640, 12, "best 40 windows by hq score (sensitive); "
                         "boxes = decision: red whale, orange small, cyan fish (info)", cap)
    outs += write_sheets(grab(dep, border), pre + "hq_border", 3, 640, 12,
                         f"40 windows nearest the hq threshold {cfg.hq_threshold} (cover >= {cfg.hq_min_cover})", cap)
    cap_r = lambda i, fr: (cap(i, fr)[0], reason(rej.loc[i]))
    outs += write_sheets(grab(dep, rej), pre + "hq_rejected", 3, 640, 12,
                         f"40 random large-whale windows that are not hq_whale (seed {SEED})", cap_r)
    print("\n".join(outs))


def cmd_bands(a):
    """About 24 random windows per hq band below the threshold (cover >= hq_min_cover), seed SEED."""
    dep = a.dep
    w = windows(dep)
    cfg = pl.ScanConfig()
    cov = w[w.whale & w.cover.notna() & (w.cover >= cfg.hq_min_cover)]
    rng = np.random.default_rng(SEED)
    cap = hq_caption(w)
    res = {}
    outs = []
    for lo in (0.5, 0.4, 0.3):
        band = cov[(cov.hq_score >= lo) & (cov.hq_score < lo + 0.1)]
        pk = band.loc[np.sort(rng.choice(band.index, size=min(a.n, len(band)), replace=False))] if len(band) else band
        res[f"{lo:.1f}-{lo + 0.1:.1f}"] = {"pool": int(len(band)), "drawn": int(len(pk)),
                                           "windows": pk[["key", "k", "clock", "hq_score"]].to_dict("records")}
        if len(pk):
            outs += write_sheets(grab(dep, pk.sort_values("hq_score", ascending=False)),
                                 f"{prefix(dep)}hq_band_{int(lo * 100):03d}", 3, 640, 12,
                                 f"hq {lo:.1f}-{lo + 0.1:.1f}: {len(pk)} of {len(band)} windows at random (seed {SEED})",
                                 cap)
    RES.mkdir(exist_ok=True)
    (RES / f"hq_bands_{'d2' if 'CATS' in dep else 'd1'}.json").write_text(json.dumps(
        {"seed": SEED, "bands": res}, indent=1, default=str))
    print("\n".join(outs))


def cmd_midsize(a):
    """Kept whale windows that rest on a box of 300-500 px (the band where a near fish can pass as a whale)."""
    deps = [d.strip() for d in a.deps.split(",")]
    allw, counts = [], {}
    for dep in deps:
        w = windows(dep)
        kept_whale = w[w.whale]
        # the window's largest whale box (in its best frame) is the one the keep decision rests on
        m = kept_whale[(kept_whale.whale_px >= 300) & (kept_whale.whale_px < 500)]
        counts[dep] = {"whale_windows": int(len(kept_whale)), "rest_on_300_500px": int(len(m)),
                       "hq_whale_among_them": int(m.hq_whale.sum())}
        allw.append(m.assign(dep=dep))
    pool = pd.concat(allw)
    pool["uid"] = np.arange(len(pool))
    pool = pool.set_index("uid", drop=False)
    rng = np.random.default_rng(SEED)
    pk = pool.loc[np.sort(rng.choice(pool.index, size=min(40, len(pool)), replace=False))]
    items = []
    for dep, g in pk.groupby("dep", sort=False):
        items += grab(dep, g)
    order = {i: k for k, i in enumerate(pk.index)}
    items.sort(key=lambda x: order[x[0]])
    cap = lambda i, fr: (f"{'d2 ' if 'CATS' in pk.loc[i, 'dep'] else ''}{pk.loc[i, 'key'][-9:-4]} {pk.loc[i, 'clock']}",
                         f"box {pk.loc[i, 'whale_px']:.0f}px whale {pk.loc[i, 'top_score']:.2f} fish "
                         f"{pk.loc[i, 'fish_score']:.2f} hq {pk.loc[i, 'hq_score']:.2f}")
    outs = write_sheets(items, "scan_midsize", 3, 640, 12,
                        f"kept windows resting on a 300-500 px whale box, 40 at random of {len(pool)} (seed {SEED})", cap)
    RES.mkdir(exist_ok=True)
    (RES / "midsize.json").write_text(json.dumps({"seed": SEED, "counts": counts, "drawn": pk[[
        "dep", "key", "k", "clock", "whale_px", "top_score", "fish_score"]].to_dict("records")}, indent=1, default=str))
    print(json.dumps(counts, indent=1))
    print("\n".join(outs))


# ------------------------------------------------------------------------------------------- lost labelled whales

def cmd_lost(a):
    ev = json.loads((RES / "eval.json").read_text())
    lost = pd.DataFrame(ev["runs"][a.label]["lost_whale_frames"])
    if lost.empty:
        print("nothing lost")
        return
    items = []
    for p, g in lost.groupby("piece"):
        src = next((paths.footage_dir() / DEP / "Video").glob(f"CC-*-{p}.mov"))
        with video.VideoReader(src) as r:
            for i, row in g.iterrows():
                fr = r.frame_by_index(int(row.frame))
                items.append((i, fr, fr.image))
    cap = lambda i, fr: (f"{lost.loc[i, 'piece']} f{lost.loc[i, 'frame']} t={lost.loc[i, 't_mov']:.1f}",
                         f"{lost.loc[i, 'why']}; whale {lost.loc[i, 'whale_score']}")
    print(write_sheets(items, f"scan_lost_{a.label.replace('+', '_')}", 2, 960, 6,
                       f"labelled clear-whale frames outside the kept stretches [{a.label}]", cap))


# ------------------------------------------------------------------------------------------- clips

def cmd_clips(a):
    import cv2
    from uwid import timecode
    cdir = paths.cache_dir("clips") / DEP
    idx = pd.read_csv(cdir / "clips_index.csv")
    idx = idx[idx.status.str.startswith(("ok", "exists"))].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pick = idx.loc[np.sort(rng.choice(idx.index, size=min(12, len(idx)), replace=False))]
    items = []
    for i, r in pick.iterrows():
        tab = video.frame_table(r.clip_file)
        with video.VideoReader(r.clip_file) as rd:
            for pos, j in (("first", 0), ("middle", tab.n_frames // 2), ("last", tab.n_frames - 1)):
                fr = rd.frame_by_index(j)
                items.append(((i, pos), fr, fr.image))
    cap = lambda i, fr: (f"{Path(pick.loc[i[0], 'clip_file']).name[-30:]} {i[1]}",
                         f"{pick.loc[i[0], 'tier']} {pick.loc[i[0], 'clock_start']} - {pick.loc[i[0], 'clock_end']}")
    outs = write_sheets(items, "scan_clips", 3, 640, 18, "first / middle / last frame of 12 kept clips (sensitive, copy)",
                        cap)
    # verify 3 clips by decoding: first and last frame against the source frames the index names
    ver = []
    for _, r in pick.head(3).iterrows():
        ct = video.frame_table(r.clip_file)
        st = video.frame_table(r.source)
        with video.VideoReader(r.clip_file) as rc, video.VideoReader(r.source) as rs:
            c0, c1 = rc.frame_by_index(0), rc.frame_by_index(ct.n_frames - 1)
            s0, s1 = rs.frame_by_index(int(r.start_frame)), rs.frame_by_index(int(r.end_frame) - 1)

        def ncc(x, y):
            x = cv2.cvtColor(x.image, cv2.COLOR_RGB2GRAY).astype(float).ravel()
            y = cv2.cvtColor(y.image, cv2.COLOR_RGB2GRAY).astype(float).ravel()
            x, y = x - x.mean(), y - y.mean()
            return float((x @ y) / np.sqrt((x @ x) * (y @ y)))
        gs = timecode.load_glyphs()
        clk = lambda f: (timecode.read_clock(f.image, gs) or type("x", (), {"text": None})).text
        ver.append({"clip": Path(r.clip_file).name, "index_start_s": r.start_s, "index_end_s": r.end_s,
                    "wanted": [r.wanted_start_s, r.wanted_end_s], "source_start_t": float(st.times[int(r.start_frame)]),
                    "clip_frames": ct.n_frames, "index_frames": int(r.n_frames), "clip_duration_s": ct.duration,
                    "index_duration_s": float(r.end_s - r.start_s), "ncc_first": ncc(c0, s0), "ncc_last": ncc(c1, s1),
                    "clock_first": [clk(c0), clk(s0)], "clock_last": [clk(c1), clk(s1)]})
    src_bytes = sum(Path(s).stat().st_size for s in idx.source.unique())
    clip_bytes = int(sum(Path(c).stat().st_size for c in idx.clip_file))
    res = {"verified": ver, "n_clips": int(len(idx)), "clip_bytes": clip_bytes, "source_bytes": int(src_bytes),
           "clip_seconds": float((idx.end_s - idx.start_s).sum()),
           "tiers": idx.tier.value_counts().to_dict(), "sheets": outs}
    RES.mkdir(exist_ok=True)
    (RES / "clips_check.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def cmd_whole(a):
    """Every whale window whose whale box covers more than 60 % of the frame (the near-whale rule), by depth."""
    w = windows(a.dep, a.label)
    big = w[w.whale & (w.cover > 0.6)].sort_values("depth_m")
    cap = lambda i, fr: (f"{big.loc[i, 'key'][-9:-4]} {big.loc[i, 'clock']} d {big.loc[i, 'depth_m']:.1f} m",
                         f"cover {big.loc[i, 'cover']:.2f} score {big.loc[i, 'top_score']:.2f} water {big.loc[i, 'water']}")
    print(write_sheets(grab(a.dep, big), f"scan_wholeframe_{a.label.replace('+', '_')}", 4, 480, 24,
                       f"whale windows with a box over 60 % of the frame [{a.label}], shallowest first", cap))


def cmd_kept(a):
    w = windows(a.dep, LABEL)
    k = w[w.whale]
    rng = np.random.default_rng(SEED)
    pick = k.loc[np.sort(rng.choice(k.index, size=min(a.n, len(k)), replace=False))] if len(k) else k
    if pick.empty:
        print("no kept windows")
        return
    cap = lambda i, fr: (f"{pick.loc[i, 'key'][-12:-4]} {pick.loc[i, 'clock']}",
                         f"whale {pick.loc[i, 'whale_score']:.2f} cover {pick.loc[i, 'cover']:.2f} "
                         f"hq {pick.loc[i, 'hq_score']:.2f} {pick.loc[i, 'tag_state']}/{pick.loc[i, 'water']}")
    tag = a.dep.split()[1] if " " in a.dep else a.dep
    print(write_sheets(grab(a.dep, pick), f"scan_kept_{tag.replace('-', '')}", 3, 640, 12,
                       f"{a.dep}: random kept whale windows (seed {SEED})", cap))


# ------------------------------------------------------------------------------------------- second deployment

D2 = "TestSubset 2025-1-2 CATS -85"
SEED_D2 = 20261122


def _key(dep, piece):
    return next(k for k in windows(dep).key.unique() if k.endswith(f"-{piece}.mov"))


def cmd_d2_timelines(a):
    outs = []
    for piece, what in (("00005", "boat"), ("00017", "middle"), ("00023", "late")):
        outs.append(timeline(D2, _key(D2, piece), OUTD / f"scan_d2_timeline_{piece}.jpg", f"CATS -85 {piece} ({what})"))
    print("\n".join(outs))


def cmd_d2_clips(a):
    """First / middle / last frame of every kept stretch of the middle recordings and of 20 (seeded) of the late."""
    iv = json.loads((SCAN / D2 / LABEL / "intervals.json").read_text())
    rows = []
    for v in iv["videos"]:
        piece = Path(v["source"]).stem[-5:]
        for j, k in enumerate(v["keep"]):
            rows.append({"piece": piece, "src": v["source"], "j": j, **{x: k[x] for x in ("start", "end", "tier",
                                                                                         "description")}})
    df = pd.DataFrame(rows)
    mid = df[df.piece.isin(["00010", "00011", "00012", "00017"])]
    late = df[df.piece.isin(["00023", "00024"])]
    rng = np.random.default_rng(SEED_D2)
    late = late.loc[np.sort(rng.choice(late.index, size=min(20, len(late)), replace=False))] if len(late) else late
    outs = []
    for name, sel in (("scan_d2_clips_middle", mid), ("scan_d2_clips_late", late)):
        items = []
        for i, r in sel.iterrows():
            with video.VideoReader(r.src) as rd:
                for pos, t in (("first", r.start + 0.1), ("middle", (r.start + r.end) / 2), ("last", r.end - 0.2)):
                    fr = rd.frame_at(t)
                    if fr is not None:
                        items.append(((i, pos), fr, fr.image))
        cap = lambda i, fr: (f"{sel.loc[i[0], 'piece']} #{sel.loc[i[0], 'j']} {i[1]} t={fr.t:.0f}s",
                             f"{sel.loc[i[0], 'tier']}: {sel.loc[i[0], 'description'][:95]}")
        if items:
            outs += write_sheets(items, name, 3, 640, 18, f"{name[8:]}: first / middle / last frame of kept stretches"
                                 + (f" (20 at random, seed {SEED_D2})" if "late" in name else " (all)"), cap)
    (RES / "d2_clips_sheet.json").write_text(json.dumps({"middle": mid.to_dict("records"),
                                                         "late": late.to_dict("records")}, indent=1, default=str))
    print("\n".join(outs))


def cmd_d2_dropcheck(a):
    """100 windows drawn at random (seed SEED_D2) from 'nothing detected', centre frame, 2 per row at 960 px."""
    w = windows(D2)
    pool = w[w.why == "nothing detected"]
    rng = np.random.default_rng(SEED_D2)
    pick = pool.loc[np.sort(rng.choice(pool.index, size=min(100, len(pool)), replace=False))].copy()
    pick["id"] = np.arange(len(pick))
    RES.mkdir(exist_ok=True)
    cols = ["id", "key", "k", "t0", "clock", "whale_score", "small_score", "fish_score", "depth_m", "water_picture",
            "bubbles"]
    pick[cols].to_csv(RES / "dropcheck_d2_sample.csv", index=False)
    calls_f = RES / "dropcheck_d2_calls.json"
    if a.tally and calls_f.exists():
        calls = json.loads(calls_f.read_text())
        pick["call"] = pick.id.map(lambda i: calls.get(str(i), "?"))
        print(pick.call.value_counts())
        (RES / "dropcheck_d2_result.json").write_text(json.dumps(
            {"seed": SEED_D2, "pool": int(len(pool)), "counts": pick.call.value_counts().to_dict(),
             "calls": pick[["id", "key", "k", "clock", "call", "depth_m"]].to_dict("records")}, indent=1, default=str))
        return
    items = grab(D2, pick, which="centre")
    cap = lambda i, fr: (f"#{pick.loc[i, 'id']} {pick.loc[i, 'key'][-9:-4]} {pick.loc[i, 'clock']}",
                         f"d {pick.loc[i, 'depth_m']:.0f} m, whale {pick.loc[i, 'whale_score']:.2f} small "
                         f"{pick.loc[i, 'small_score']:.2f} fish {pick.loc[i, 'fish_score']:.2f}")
    print(write_sheets(items, "scan_d2_dropcheck", 2, 960, 6,
                       f"'nothing detected' windows of CATS -85, 100 at random (seed {SEED_D2}), centre frame", cap))


def cmd_d2_spans(a):
    """A dozen windows inside the labeller's sand / dark spans and on the seafloor, with the decision drawn."""
    lab = HERE.parents[1] / "experiments/02_reference_labels/labels"
    w = windows(D2)
    rng = np.random.default_rng(SEED_D2)
    outs = []
    for name, fname, col, kinds in (("scan_d2_sand", "blocked_spans.csv", "kind", ("sand",)),
                                    ("scan_d2_seafloor", "seafloor_spans.csv", "seafloor", ("visible",))):
        sp = pd.concat([pd.read_csv(f) for f in sorted(lab.glob(f"CC-09-85-*/{fname}"))])
        sp = sp[sp[col].isin(kinds)]
        sel = []
        for r in sp.itertuples():
            m = (w.key == f"{r.recording}.mov") & (w.t0 >= r.start_s) & (w.t1 <= r.end_s)
            sel.append(w[m])
        sel = pd.concat(sel)
        # first the windows where something fired (to see what), then random others; 12 in all
        fired = sel[sel.active.astype(bool)] if "active" in sel else sel.iloc[:0]
        n_f = min(6, len(fired))
        pick = pd.concat([fired.loc[np.sort(rng.choice(fired.index, n_f, replace=False))] if n_f else fired.iloc[:0],
                          sel.drop(fired.index).pipe(lambda x: x.loc[np.sort(rng.choice(x.index, min(12 - n_f, len(x)),
                                                                                       replace=False))])])
        cap = lambda i, fr, pk=pick: (f"{pk.loc[i, 'key'][-9:-4]} {pk.loc[i, 'clock']} {pk.loc[i, 'why']}",
                                      f"tier {pk.loc[i, 'tier']} water(pic) {pk.loc[i, 'water_picture']} whale "
                                      f"{pk.loc[i, 'whale_score']:.2f} fish {pk.loc[i, 'fish_score']:.2f} bub "
                                      f"{pk.loc[i, 'bubbles']:.2f}")
        outs += write_sheets(grab(D2, pick, which="best"), name, 3, 640, 12,
                             f"{name[8:]} spans (labeller): {len(sel)} windows, {int(sel.active.sum())} with something "
                             f"detected; {n_f} of those + {len(pick) - n_f} others at random (seed {SEED_D2})", cap)
    print("\n".join(outs))


def cmd_veto_dropped(a):
    """Windows that were active only through a large "whale" box the classifier calls other (no longer active since
    WP11b): 60 at random over both deployments (seed SEED_D2), best frame of the window with the decision drawn."""
    rows = []
    for dep in (DEP, D2):
        w = windows(dep)
        if "vetoed_other_only" in w:
            rows.append(w[w.vetoed_other_only.astype(bool)].assign(dep=dep))
    pool = pd.concat(rows)
    pool["uid"] = np.arange(len(pool))
    pool = pool.set_index("uid", drop=False)
    rng = np.random.default_rng(SEED_D2)
    pk = pool.loc[np.sort(rng.choice(pool.index, size=min(60, len(pool)), replace=False))]
    items = []
    for dep, g in pk.groupby("dep", sort=False):
        # the frame holding the vetoed box: the frame with the higher veto_other score in the frame_0 / frame_1 pair
        g = g.copy()
        show = []
        for key, gg in g.groupby("key"):
            src = source_of(dep, key)
            cls = pl.load_picture_classes(SCAN / dep / "cache", pl.Recording(src, src.suffix[1:]))[0]
            other = set(cls[cls.boxclf_class == "other"].frame.astype(int))
            show += [(i, r.frame_1 if r.frame_1 in other and r.frame_0 not in other else r.frame_0)
                     for i, r in gg.iterrows()]
        g["show_frame"] = pd.Series(dict(show)).astype(float)
        items += grab(dep, g, which="centre")
    order = {i: k for k, i in enumerate(pk.index)}
    items.sort(key=lambda x: order[x[0]])
    cap = lambda i, fr: (f"#{i} {'d2 ' if 'CATS' in pk.loc[i, 'dep'] else ''}{pk.loc[i, 'key'][-9:-4]} {pk.loc[i, 'clock']}",
                         f"vetoed 'other' {pk.loc[i, 'veto_other_score']:.2f}, depth {pk.loc[i, 'depth_m']:.1f} m")
    counts = {d: int((pool.dep == d).sum()) for d in pool.dep.unique()}
    RES.mkdir(exist_ok=True)
    (RES / "veto_dropped_sample.json").write_text(json.dumps({"seed": SEED_D2, "pool": counts, "drawn": pk[[
        "dep", "key", "k", "clock", "veto_other_score", "depth_m"]].to_dict("records")}, indent=1, default=str))
    print(counts)
    print(write_sheets(items, "scan_veto_dropped", 3, 640, 12,
                       f"windows no longer active: only box vetoed to 'other' (60 of {len(pool)}, seed {SEED_D2}); "
                       "centre frame", cap))


def cmd_float(a):
    """Windows of 'tag floating' time kept since WP9 (detector on floating time where the picture sees water):
    2025-02-03 00012 (run under _float_check/) and CATS -85, best frame with the decision drawn."""
    items, rows = [], []
    for dep in ("_float_check/" + DEP, D2):
        w = windows(dep)
        k = w[(w.tag_state == "floating") & w.active.astype(bool)]
        rows.append(k.assign(dep=dep))
        if len(k):
            items += [(f"{dep}|{i}", fr, im) for i, fr, im in grab(dep, k)]
    allk = pd.concat(rows)
    cap = lambda i, fr: (f"{'d2 ' if 'CATS' in i else ''}{allk[allk.dep == i.split('|')[0]].loc[int(i.split('|')[1]), 'key'][-9:-4]}"
                         f" {allk[allk.dep == i.split('|')[0]].loc[int(i.split('|')[1]), 'clock']}",
                         f"floating, water(pic) {allk[allk.dep == i.split('|')[0]].loc[int(i.split('|')[1]), 'water']}, "
                         f"tier {allk[allk.dep == i.split('|')[0]].loc[int(i.split('|')[1]), 'tier']}")
    print({d: int((allk.dep == d).sum()) for d in allk.dep.unique()})
    if items:
        print(write_sheets(items, "scan_d2_float", 3, 640, 12,
                           "kept windows of 'tag floating' time (2025-02-03 00012 and CATS -85 00024), decision drawn", cap))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["timelines", "dropcheck", "hq", "lost", "clips", "kept", "whole", "bands",
                                    "midsize", "d2_timelines", "d2_clips", "d2_dropcheck", "d2_spans", "float", "veto_dropped"])
    ap.add_argument("--redraw-new", action="store_true", help="hq: draw new windows instead of the saved ones")
    ap.add_argument("--deps", default=DEP + ",TestSubset 2025-1-2 CATS -85")
    ap.add_argument("--tally", action="store_true")
    ap.add_argument("--label", default="sensitive")
    ap.add_argument("--dep", default=DEP)
    ap.add_argument("-n", type=int, default=24)
    a = ap.parse_args()
    {"whole": cmd_whole, "timelines": cmd_timelines, "dropcheck": cmd_dropcheck, "hq": cmd_hq, "lost": cmd_lost,
     "clips": cmd_clips, "kept": cmd_kept, "bands": cmd_bands, "midsize": cmd_midsize,
     "d2_timelines": cmd_d2_timelines, "d2_clips": cmd_d2_clips, "d2_dropcheck": cmd_d2_dropcheck,
     "d2_spans": cmd_d2_spans, "float": cmd_float,
     "veto_dropped": cmd_veto_dropped}[a.cmd](a)
