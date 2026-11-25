"""WP18: own-back template as the default; far specks dropped from whale outlines (CPU only).

    dev       the speck rule on the dev clip's stored runs (2025-12-20 CG) only: counts, to keep or change the values
              before the test deployments are looked at -> results/wp18_dev.json
    retro     the 135 WP13 runs get the template a default run would store (own_back.npz + meta key `own_back`);
              meta.json backed up to ownback/wp18/meta_backup/, list in tracks/wp18_ownback_added.csv (undo:
              delete each listed own_back.npz and restore the backed-up meta.json)
    check     numbers per span before links / after links on the 135 WP13 runs (against wp13_spans.csv), the 17
              stretches (against WP17) and the dev clip's online run
    stats     speck counts per deployment and dev; sheets speck_fix_changed_01 / speck_fix_kept_01
    extra     tracks/extra_stretches.csv (the 17 replay stretches and the folder that holds the default result)
    calf      ownback_replay_calf_v2.mp4 and ownback_replay_calf_v2_01.jpg (last 4 s, v1 above v2)

    .venv/bin/python experiments/10_own_back/wp18.py dev|retro|check|stats|extra|calf
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
import wp17 as P  # noqa: E402
from uwid import ownback as O  # noqa: E402
from uwid import paths, tracks, video  # noqa: E402
from uwid import trackstage as T  # noqa: E402

OUT = C.OUT / "wp18"
TR = C.CACHE / "tracks"
SEED = 20261124
DEV_RUNS = ["2025-12-20_CG/2025-12-20 CG/0024.00-0036.00_online", "2025-12-20_CG/0005.00-0022.00_owl_sam3",
            "2025-12-20_CG/0024.00-0048.00_owl_sam3", "2025-12-20_CG/0061.00-0076.00_owl_sam3"]


def load_any(d: Path) -> tracks.Tracks:
    m = json.loads((d / "meta.json").read_text())
    if m.get("method") == "online_sam3":
        return T.load_online(d)
    tr = tracks.load_tracks(d)
    tr.despeck = True
    return tr


def speck_rows(tr: tracks.Tracks, label: str) -> list[dict]:
    rows = []
    for k, n in tr.number.items():
        for i in range(tr.n):
            if not tr.packed[k][i].any():
                continue
            m = tr.raw_mask(k, i)
            _, info = tracks.despeckle(m)
            rows.append(dict(run=label, obj=k, number=n, i=i, **info))
    return rows


def summarize(df: pd.DataFrame) -> dict:
    if not len(df):
        return {}
    multi = df[df.pieces > 1]; ch = df[df.dropped > 0]
    sh = ch.dropped_px / ch.area_px
    return dict(object_frames=len(df), with_2plus_pieces=len(multi), changed=len(ch),
                multi_all_kept=int((multi.dropped == 0).sum()),
                dropped_share_median=round(float(sh.median()), 4) if len(ch) else 0.0,
                dropped_share_max=round(float(sh.max()), 4) if len(ch) else 0.0,
                numbers=int(df.groupby(["run", "number"]).ngroups),
                numbers_touched=int(ch.groupby(["run", "number"]).ngroups) if len(ch) else 0)


def step_dev():
    rows = []
    for r in DEV_RUNS:
        rows += speck_rows(load_any(TR / r), r)
    df = pd.DataFrame(rows)
    res = summarize(df)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "wp18_dev.json").write_text(json.dumps(res, indent=1))
    df.to_csv(OUT / "speck_dev.csv", index=False)
    print(res)


def step_retro():
    s = pd.read_csv(TR / "wp13_spans.csv")
    bk = OUT / "meta_backup"; bk.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in s.itertuples():
        d = Path(r.folder)
        meta = json.loads((d / "meta.json").read_text())
        if "own_back" in meta:
            rows.append(dict(folder=str(d), added_npz=(d / "own_back.npz").exists(), meta_key="own_back", note="already")); continue
        stem = Path(r.recording).stem
        t = float(meta["t0"])
        w = T.windows_table(r.deployment, r.recording)

        def no_maps(v, f):
            raise RuntimeError(f"block {O.block_of(t)} of {stem} not cached")
        tm, info = O.block_template(None, w, t, paths.cache_dir("tracks") / r.deployment / stem / "ownback", maps_fn=no_maps)
        key = d.relative_to(TR).as_posix().replace("/", "__")
        shutil.copy2(d / "meta.json", bk / f"{key}.meta.json")
        meta["own_back"] = info
        if tm is not None:
            O.save_run_template(d, tm, info)
        (d / "meta.json").write_text(json.dumps(meta))
        rows.append(dict(folder=str(d), added_npz=tm is not None, meta_key="own_back", meta_backup=str(bk / f"{key}.meta.json"),
                         note="" if tm is not None else info.get("why_none", "")))
    pd.DataFrame(rows).to_csv(TR / "wp18_ownback_added.csv", index=False)
    print(len(rows), "runs;", sum(x["added_npz"] for x in rows), "own_back.npz added")


def step_check():
    s = pd.read_csv(TR / "wp13_spans.csv")
    res = {}
    b1 = b2 = a1 = a2 = 0
    diff = []
    for r in s.itertuples():
        tr = T.load_online(r.folder)
        n1 = len(set(tr.number.values()))
        tr = T.link(tr, r.deployment)
        n2 = len(set(tr.number.values()))
        a1 += n1; a2 += n2; b1 += int(r.numbers_purity); b2 += int(r.numbers)
        if n1 != int(r.numbers_purity) or n2 != int(r.numbers):
            diff.append(dict(folder=r.folder, before=[int(r.numbers_purity), int(r.numbers)], after=[n1, n2]))
    res["wp13"] = dict(spans=len(s), before_links=[b1, a1], after_links=[b2, a2], spans_differing=diff)
    st = pd.read_csv(P.OUT / "stretches.csv")
    w17 = []
    for r in st.itertuples():
        d = T.span_dir(P.DEP[r.src], Path(P.stem_of(r.src, f"{int(r.piece):05d}")).name + ".mov", float(r.t0), float(r.t1))
        d = d.with_name(d.name + "_ownback")
        tr = T.load_online(d)
        w17.append(dict(stretch=f"{int(r.piece):05d} {r.t0}-{r.t1}", wp17=int(r.numbers_on), now=len(set(tr.number.values()))))
    res["stretches"] = dict(total_wp17=sum(x["wp17"] for x in w17), total_now=sum(x["now"] for x in w17),
                            differing=[x for x in w17 if x["wp17"] != x["now"]])
    dv = T.load_online(TR / DEV_RUNS[0])
    res["dev_online"] = dict(numbers=len(set(dv.number.values())))
    (HERE / "results" / "wp18_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def collect_test():
    rows = []
    s = pd.read_csv(TR / "wp13_spans.csv")
    for r in s.itertuples():
        rows += [dict(x, deployment=r.deployment, kind="hq span") for x in speck_rows(T.load_online(r.folder), r.folder)]
    st = pd.read_csv(P.OUT / "stretches.csv")
    for r in st.itertuples():
        d = T.span_dir(P.DEP[r.src], P.stem_of(r.src, f"{int(r.piece):05d}") + ".mov", float(r.t0), float(r.t1))
        d = d.with_name(d.name + "_ownback")
        rows += [dict(x, deployment=P.DEP[r.src], kind="replay stretch") for x in speck_rows(T.load_online(d), str(d))]
    return pd.DataFrame(rows)


def tile_pair(d: str, k: int, i: int, tr_cache: dict) -> np.ndarray:
    tr = tr_cache.setdefault(d, load_any(Path(d)))
    f = int(tr.meta["frames"][0]) + i
    img = next(iter(video.iter_frames_at(tr.meta["video"], [f], size=(1280, 720)))).image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    tiles = []
    for m, lab in ((tr.raw_mask(k, i), "before (stored outline)"), (tr.mask(k, i), "after (speck rule)")):
        a = img.copy()
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(a, cs, -1, (40, 60, 255), 3)
        a = cv2.resize(a, (480, 270), interpolation=cv2.INTER_AREA)
        bar = np.zeros((22, 480, 3), np.uint8)
        cv2.putText(bar, f"{lab}  no. {tr.number[k]}  {Path(d).parent.name[-5:]} t {tr.meta['t0'] + i / tr.meta['fps']:.1f} s",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([a, bar]))
    return np.hstack(tiles)


def safe_write(name, img):
    p = C.SHEETS / name
    mine_f = C.OUT / "sheets_written.txt"
    mine = set(mine_f.read_text().split()) if mine_f.exists() else set()
    if p.exists() and name not in mine:
        raise SystemExit(f"{p} exists and was not written by WP14-WP18")
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    mine.add(name); mine_f.write_text("\n".join(sorted(mine)) + "\n")
    print("wrote", p)


def step_stats():
    df = collect_test()
    df.to_csv(OUT / "speck_test.csv", index=False)
    dev = pd.read_csv(OUT / "speck_dev.csv").assign(deployment="2025-12-20 CG (dev)", kind="dev runs")
    res = {}
    for dep, g in pd.concat([df, dev]).groupby("deployment"):
        res[dep] = summarize(g)
    res["all test (hq spans + stretches)"] = summarize(df)
    (HERE / "results" / "wp18_speck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    rng = np.random.default_rng(SEED)
    cache = {}
    ch = df[df.dropped > 0]
    pick = ch.iloc[rng.choice(len(ch), min(12, len(ch)), replace=False)] if len(ch) else ch
    if len(pick):
        rows = [tile_pair(r.run, int(r.obj), int(r.i), cache) for r in pick.itertuples()]
        safe_write("speck_fix_changed_01.jpg", np.vstack([np.hstack(rows[j:j + 2]) if j + 1 < len(rows) else
                                                           np.hstack([rows[j], np.zeros_like(rows[j])]) for j in range(0, len(rows), 2)]))
    kept = df[(df.pieces > 1) & (df.dropped == 0)]
    pick = kept.iloc[rng.choice(len(kept), min(12, len(kept)), replace=False)] if len(kept) else kept
    if len(pick):
        rows = [tile_pair(r.run, int(r.obj), int(r.i), cache) for r in pick.itertuples()]
        safe_write("speck_fix_kept_01.jpg", np.vstack([np.hstack(rows[j:j + 2]) if j + 1 < len(rows) else
                                                        np.hstack([rows[j], np.zeros_like(rows[j])]) for j in range(0, len(rows), 2)]))


def step_extra():
    st = P.stretches()
    rows = []
    for r in st.itertuples():
        d = T.span_dir(r.deployment, Path(r.video).name, float(r.t0), float(r.t1))
        d = d.with_name(d.name + "_ownback")
        note = "calf stretch (user-confirmed calf)" if r.kind == "calf" else "WP14 own-back candidate"
        if json.loads((d / "meta.json").read_text()).get("own_back", {}).get("why_none"):
            note += "; no template in its block (tag floating): as before"
        rows.append(dict(deployment=r.deployment, recording=Path(r.video).name, t0=r.t0, t1=r.t1, folder=str(d), note=note))
    pd.DataFrame(rows).to_csv(TR / "extra_stretches.csv", index=False)
    print(len(rows), "rows ->", TR / "extra_stretches.csv")


def step_calf():
    import wp17_sheets as S
    st = P.stretches()
    r = st[st.kind == "calf"].iloc[0]
    base = T.span_dir(r.deployment, Path(r.video).name, float(r.t0), float(r.t1))
    d = base.with_name(base.name + "_ownback")
    v1 = T.load_online(d); v1.despeck = False
    v2 = T.load_online(d)
    fps = float(v2.meta["fps"])
    idx = np.linspace(v2.n - 1 - int(4 * fps), v2.n - 1, 8).round().astype(int).tolist()
    fr = S.decode(v2, idx)
    rows = []
    for tr, lab in ((v1, "v1 (WP17)"), (v2, "v2 (speck rule)")):
        tiles = [S.draw(tr, fr, i, f"{lab}  t {tr.meta['t0'] + i / fps:.1f} s") for i in idx]
        rows += [np.hstack(tiles[:4]), np.hstack(tiles[4:])]
    head = np.zeros((30, rows[0].shape[1], 3), np.uint8)
    cv2.putText(head, "2025-02-03 00005 calf stretch, last 4 s, flag on: rows 1-2 v1, rows 3-4 v2 (far specks dropped "
                "from whale outlines); purple = own back", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    safe_write("ownback_replay_calf_v2_01.jpg", np.vstack([head] + rows))
    v2.back = np.stack([np.packbits(S.own_mask(v2, i), axis=1) for i in range(v2.n)])
    tmp = OUT / "ownback_replay_calf_v2.mp4"
    tracks.render_tracks(v2, tmp, None)
    p = C.SHEETS / "ownback_replay_calf_v2.mp4"
    if p.exists():
        raise SystemExit(f"{p} exists")
    tmp.replace(p)
    print("wrote", p, round(p.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"dev": step_dev, "retro": step_retro, "check": step_check, "stats": step_stats, "extra": step_extra,
     "calf": step_calf}[sys.argv[1]]()
