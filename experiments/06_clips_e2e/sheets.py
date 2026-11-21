"""Review sheets of WP7 (to `review_sheets/underwater/scan_*.jpg`) and the dropped-window check.

    PY=.venv/bin/python; S=experiments/06_clips_e2e/sheets.py
    $PY $S timelines                 # scan_timeline_<piece>.jpg for 00004, 00012 and the dev clip
    $PY $S dropcheck                 # 150 random 'underwater with no whale' windows + 30 'small' ones, seed below
    $PY $S dropcheck --tally         # after the calls are written to results/dropcheck_calls.json
    $PY $S hq                        # scan_hq_top / border / rejected
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


def source_of(dep, key) -> Path:
    iv = json.loads((SCAN / dep / LABEL / "intervals.json").read_text())
    for v in iv["videos"]:
        p = Path(v["source"])
        if f"{p.stem.replace(' ', '_')}.{p.suffix[1:]}" == key:
            return p
    raise KeyError(key)


def grab(dep, rows: pd.DataFrame, which: str = "best", draw_boxes: bool = True, cfg=None):
    """One frame per window row: the frame with the largest whale box (``best``) or the window centre."""
    import cv2
    cfg = cfg or pl.ScanConfig()
    out = []
    for key, g in rows.groupby("key", sort=False):
        src = source_of(dep, key)
        boxes, _ = pl.load_boxes(SCAN / dep / "cache", pl.Recording(src, src.suffix[1:]))
        with video.VideoReader(src) as r:
            for i, row in g.iterrows():
                if which == "best" and np.isfinite(row.get("whale_1", np.nan)) and row.get("whale_1", 0) > row.get(
                        "whale_0", 0):
                    f_idx = int(row.frame_1)
                elif which == "best":
                    f_idx = int(row.frame_0)
                else:
                    f_idx = r.table.index_at(row.t0 + 0.5)
                fr = r.frame_by_index(f_idx)
                if fr is None:
                    continue
                img = fr.image.copy()
                if draw_boxes and len(boxes):
                    b = boxes[(boxes.frame == f_idx) & (boxes["query"].isin([cfg.whale_query] + list(cfg.fish_queries)))
                              & (boxes.score >= 0.2)]
                    for _, bb in b.iterrows():
                        col = (255, 60, 60) if bb["query"] == "whale" else (60, 200, 255)
                        cv2.rectangle(img, (int(bb.x0), int(bb.y0)), (int(bb.x1), int(bb.y1)), col, 3)
                        cv2.putText(img, f"{bb['query'][:5]} {bb.score:.2f}", (int(bb.x0) + 4, int(bb.y0) + 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
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
    pool = w[(w.why == "underwater with no whale")]
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
        found = sample[sample.call.isin(["large whale", "small whale"])]
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


def cmd_hq(a):
    w = windows()
    w = w.rename(columns={"hq_score": "hq_score"})
    large = w[w.whale & w.cover.notna()].copy()
    top = large.sort_values("hq_score", ascending=False).head(40)
    cfg = pl.ScanConfig()
    border = large[(large.cover >= cfg.hq_min_cover)].iloc[
        (large[(large.cover >= cfg.hq_min_cover)].hq_score - cfg.hq_threshold).abs().argsort()[:40]]
    border = border.sort_values("hq_score", ascending=False)
    rej = large[~large.hq_whale]
    rng = np.random.default_rng(SEED)
    rej = rej.loc[np.sort(rng.choice(rej.index, size=min(40, len(rej)), replace=False))]

    def reason(r):
        if r.cover < cfg.hq_min_cover:
            return f"whale box {r.cover:.2f} of frame < {cfg.hq_min_cover}"
        parts = {k[3:]: r[k] for k in ("hq_cover", "hq_score", "hq_bright", "hq_contrast", "hq_sharp", "hq_bubbles",
                                         "hq_depth") if k in r and k != "hq_score" and np.isfinite(r[k])}
        low = min(parts, key=parts.get)
        return f"hq {r.hq_score:.2f} < {cfg.hq_threshold}; lowest part {low} {parts[low]:.2f}"

    outs = []
    cap = hq_caption(w)
    outs += write_sheets(grab(DEP, top), "scan_hq_top", 3, 640, 12, "best 40 windows by hq score (sensitive)", cap)
    outs += write_sheets(grab(DEP, border), "scan_hq_border", 3, 640, 12,
                         f"40 windows nearest the hq threshold {cfg.hq_threshold} (cover >= {cfg.hq_min_cover})", cap)
    cap_r = lambda i, fr: (cap(i, fr)[0], reason(rej.loc[i]))
    outs += write_sheets(grab(DEP, rej), "scan_hq_rejected", 3, 640, 12,
                         f"40 random large-whale windows that are not hq_whale (seed {SEED})", cap_r)
    RES.mkdir(exist_ok=True)
    (RES / "hq_sheet_windows.json").write_text(json.dumps({
        "top": top[["key", "k", "clock", "hq_score", "cover", "depth_m"]].to_dict("records"),
        "border": border[["key", "k", "clock", "hq_score", "cover", "depth_m"]].to_dict("records"),
        "rejected": rej[["key", "k", "clock", "hq_score", "cover"]].to_dict("records")}, indent=1, default=str))
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
        src = next((paths.footage_dir() / DEP / "Video").glob(f"*-{p}.mov"))
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["timelines", "dropcheck", "hq", "lost", "clips", "kept"])
    ap.add_argument("--tally", action="store_true")
    ap.add_argument("--label", default="sensitive")
    ap.add_argument("--dep", default=DEP)
    ap.add_argument("-n", type=int, default=36)
    a = ap.parse_args()
    {"timelines": cmd_timelines, "dropcheck": cmd_dropcheck, "hq": cmd_hq, "lost": cmd_lost, "clips": cmd_clips,
     "kept": cmd_kept}[a.cmd](a)
