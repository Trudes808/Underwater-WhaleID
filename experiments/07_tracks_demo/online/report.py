"""WP11a numbers: per span summary (index), the online vs after-the-fact comparison on the two old stretches, and the
videos for the user.

    .venv/bin/python experiments/07_tracks_demo/online/report.py summary
    .venv/bin/python experiments/07_tracks_demo/online/report.py compare
    .venv/bin/python experiments/07_tracks_demo/online/report.py render
    .venv/bin/python experiments/07_tracks_demo/online/report.py judge <span-index> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pure"))
sys.path.insert(0, str(HERE))
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("online_run_gpu", HERE / "run_gpu.py")
RG = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(RG)
from uwid import trackstage as T, tracks  # noqa: E402

SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
VIDEOS = [  # (index in run_gpu.items() or ("enc", n) for a whole encounter, file name)
    (("enc", 0), "demo_online_01_cats85_00024_whole_encounter_near_whale_breath"),
    (("enc", 1), "demo_online_02_0203_00008_whole_encounter"),
    (5, "demo_online_03_cats85_00023_near_whale_42m"),
    (9, "demo_online_04_cats85_00017_breath_blow_no_number"),
    (0, "demo_online_05_0203_00004_very_near_whale"),
]
LINK_VIDEOS = [(11, "demo_link_01_calf_with_fish")]     # online run + lone links (the old stretch, WP5a starts)


def encounter(n: int):
    dep, src, a, b = RG.ENCOUNTERS[n]
    ds = [T.span_dir(dep, Path(src).name, x, y) for x, y in T.pieces(a, b)]
    tr, emb = T.stitch(ds)
    T.finish(tr)
    n0 = len(tr.number)
    T.link(tr, dep, "scan", emb=emb)
    T.write_links(tr, T.encounter_dir(dep, Path(src).name, a, b) / "links.csv")
    tr.meta["label"] = "scan boxes -> SAM 3 tracker, purity online + lone links (whole encounter, stitched)"
    return tr, n0


def dirs():
    return [T.span_dir(dep, Path(src).name, a, b) for dep, src, a, b, _ in RG.items()]


def loaded(idx: int, link: bool = True):
    dep, src, a, b, start = RG.items()[idx]
    tr = T.load_online(dirs()[idx])
    n0 = len(tr.number)
    if link:
        T.link(tr, dep, start)
    return tr, n0


def cmd_summary():
    rows, links = [], []
    for idx, ((dep, src, a, b, start), d) in enumerate(zip(RG.items(), dirs())):
        if not (d / "summary.json").exists():
            continue
        s = T.summarize(d)
        tr, n0 = loaded(idx)
        m = tr.meta
        ref = pd.Series([r["reason"].split(":")[0].split(" on ")[0] for r in m.get("refused", [])]).value_counts().to_dict()
        made = [L for L in tr.links if L.get("made") and L["why"] == "linked"]
        rows.append(dict(idx=idx, rec=Path(src).stem[-5:], dep=dep[:10], t0=a, t1=b, start=start,
                         numbers_purity=n0, numbers_linked=len(set(tr.number.values())), links=len(made),
                         lone_unlinked=sum(1 for L in tr.links if L["why"] in ("checks contradict", "gap too long")),
                         not_lone=sum(1 for L in tr.links if L["why"] == "not lone"),
                         endings=json.dumps(s["endings_by_reason"]), refused=json.dumps(ref),
                         blocked=s["blocked_frames"], outlined=s["frames_with_whale_outline"], frames=s["frames"],
                         gpu_s_per_s=s["gpu_s_per_video_s"], peak_gb=s["gpu_peak_gb"]))
        for L in tr.links:
            links.append(dict(idx=idx, **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in L.items()}))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "spans.csv", index=False)
    pd.DataFrame(links).to_csv(HERE / "results" / "links_heldout.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())


def cmd_compare():
    import evaluate as EV                                   # pure/
    out = []
    for key, idx in (("demo01", 10), ("demo05", 11)):
        on, _ = loaded(idx, link=False)
        aft = EV.apply(EV.run_key_dir(key), EV.policy("strict"))
        f_on, f_af = on.meta["frames"][0], aft.meta["frames"][0]
        frames = [f for f in range(max(f_on, f_af), min(f_on + on.n, f_af + aft.n))]
        maps = {}
        for f in frames[::2]:
            for k in on.number:
                if not on.visible(k, f - f_on):
                    continue
                m = on.mask(k, f - f_on)
                sc = {j: tracks._mask_iou(m, aft.mask(j, f - f_af)) for j in aft.number if aft.visible(j, f - f_af)}
                if sc:
                    j = max(sc, key=sc.get)
                    if sc[j] >= 0.3:
                        maps.setdefault(k, {}).setdefault(j, 0)
                        maps[k][j] += 1
        one_to_one = sum(1 for k, v in maps.items() if len([j for j, c in v.items() if c >= 3]) == 1)
        out.append(dict(stretch=key, online_numbers=len(on.number), after_numbers=len(aft.number),
                        online_endings=len(on.meta["endings"]), after_endings=len(aft.endings),
                        online_tracks_matching_one_after_track=f"{one_to_one}/{len(maps)}",
                        online_outlined=sum(1 for i in range(on.n) if any(on.visible(k, i) for k in on.number)),
                        after_outlined=sum(1 for i in range(aft.n) if any(aft.visible(k, i) for k in aft.number)),
                        online_reasons=json.dumps(pd.Series([e["reason"].split(" ")[0] for e in on.meta["endings"]]).value_counts().to_dict()),
                        after_reasons=json.dumps(pd.Series([e[2].split(" ")[0] for e in aft.endings]).value_counts().to_dict())))
    d = pd.DataFrame(out)
    d.to_csv(HERE / "results" / "online_vs_after.csv", index=False)
    print(d.T.to_string())


def cmd_render():
    for key, name in VIDEOS + LINK_VIDEOS:
        if isinstance(key, tuple):
            tr, n0 = encounter(key[1])
        else:
            if not (dirs()[key] / "meta.json").exists():
                continue
            tr, n0 = loaded(key)
            tr.meta["label"] = "SAM 3 tracker, purity online + lone links"
        r = tracks.render_tracks(tr, SHEETS / f"{name}.mp4", SHEETS / f"{name}.jpg", crf=20)
        print(name, n0, "->", len(set(tr.number.values())), "numbers;", sum(1 for L in tr.links if L["why"] == "linked"),
              "links;", len(tr.endings), "endings;", r["mb"], "MB", flush=True)


def cmd_judge(idx: int, out: str, step: float = 0.5):
    import cv2
    from uwid import video as uv
    tr, _ = loaded(idx)
    f0 = tr.meta["frames"][0]
    sw, sh = tr.meta["store_size"]
    every = max(1, int(round(step * tr.meta["fps"])))
    tiles = []
    blk = set(f for a, b in tr.meta["blocked"] for f in range(a, b))
    for fr in uv.iter_frames_at(tr.meta["video"], list(range(f0, f0 + tr.n, every))):
        img = cv2.resize(fr.image, (sw, sh), interpolation=cv2.INTER_AREA)
        o = tracks.draw_frame(img, tr, fr.index - f0, f"t {fr.t:.2f}" + ("  BLOCKED" if fr.index in blk else ""), {})
        tiles.append(cv2.resize(o, (480, 270), interpolation=cv2.INTER_AREA))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[j:j + 6]) for j in range(0, len(tiles), 6)]
    Path(out).mkdir(parents=True, exist_ok=True)
    for p in range(0, len(rows), 6):
        fn = Path(out) / f"online_{idx:02d}_{p // 6}.jpg"
        cv2.imwrite(str(fn), cv2.cvtColor(np.vstack(rows[p:p + 6]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "compare", "render", "judge"])
    ap.add_argument("idx", nargs="*", type=int)
    ap.add_argument("--out")
    ap.add_argument("--step", type=float, default=0.5)
    a = ap.parse_args()
    if a.cmd == "judge":
        for i in a.idx:
            cmd_judge(i, a.out, a.step)
    else:
        {"summary": cmd_summary, "compare": cmd_compare, "render": cmd_render}[a.cmd]()
