"""WP11c: per selected stretch (results/selection_breath.csv) the tool's own result (numbers, endings, links with
evidence) and 1 s sheets for the AI by-eye check (written outside the repo).

    .venv/bin/python experiments/07_tracks_demo/online/report_breath.py summary
    .venv/bin/python experiments/07_tracks_demo/online/report_breath.py sheets <row> --out <dir>
    .venv/bin/python experiments/07_tracks_demo/online/report_breath.py render <row> <name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from uwid import paths, trackstage as T, tracks  # noqa: E402

SEL = HERE / "results" / "selection_breath.csv"


def loaded(i: int):
    r = pd.read_csv(SEL).iloc[i]
    name = Path(r.source).name
    if r.t1 - r.t0 > 26:
        ds = [T.span_dir(r.deployment, name, a, b) for a, b in T.pieces(r.t0, r.t1)]
        tr, emb = T.stitch(ds)
        T.finish(tr)
        n0 = len(tr.number)
        T.link(tr, r.deployment, "scan", emb=emb)
        T.write_links(tr, T.encounter_dir(r.deployment, name, r.t0, r.t1) / "links.csv")
    else:
        tr = T.load_online(T.span_dir(r.deployment, name, r.t0, r.t1))
        n0 = len(tr.number)
        T.link(tr, r.deployment, "scan")
    return r, tr, n0


def cmd_summary():
    out = []
    for i in range(len(pd.read_csv(SEL))):
        r, tr, n0 = loaded(i)
        fps, t0 = tr.meta["fps"], tr.meta["t0"]
        L = T.links_table(tr)
        out.append(dict(row=i, set=r.set, recording=r.recording[-9:-4], t0=r.t0, t1=r.t1,
                        blocked=[(round(t0 + (a - tr.meta["frames"][0]) / fps, 2),
                                  round(t0 + (b - tr.meta["frames"][0]) / fps, 2)) for a, b in tr.meta["blocked"]],
                        numbers_purity=n0, numbers_links=len(set(tr.number.values())),
                        links_made=int(L.made.sum()) if len(L) else 0, links_refused=int((~L.made).sum()) if len(L) else 0,
                        endings=[(round(t0 + e[0] / fps, 2), e[1], e[2]) for e in tr.endings],
                        refused_starts=len(tr.meta.get("refused", []))))
        print(out[-1])
        if len(L):
            print(L.to_string())
    pd.DataFrame(out).to_csv(HERE / "results" / "breath_summary.csv", index=False)


def cmd_sheets(i: int, out: str, step: float = 1.0):
    import cv2
    from uwid import video as uv
    r, tr, _ = loaded(i)
    f0 = tr.meta["frames"][0]
    sw, sh = tr.meta["store_size"]
    every = max(1, int(round(step * tr.meta["fps"])))
    blk = set(f for a, b in tr.meta["blocked"] for f in range(a, b))
    tiles = []
    for fr in uv.iter_frames_at(tr.meta["video"], list(range(f0, f0 + tr.n, every))):
        img = cv2.resize(fr.image, (sw, sh), interpolation=cv2.INTER_AREA)
        o = tracks.draw_frame(img, tr, fr.index - f0, f"t {fr.t:.2f}" + ("  BLOCKED" if fr.index in blk else ""), {})
        tiles.append(cv2.resize(o, (480, 270), interpolation=cv2.INTER_AREA))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[j:j + 6]) for j in range(0, len(tiles), 6)]
    Path(out).mkdir(parents=True, exist_ok=True)
    for p in range(0, len(rows), 6):
        fn = Path(out) / f"wp11c_{i:02d}_{p // 6}.jpg"
        cv2.imwrite(str(fn), cv2.cvtColor(np.vstack(rows[p:p + 6]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(fn)


def cmd_render(i: int, name: str):
    sheets = paths.sheets_dir(create=True)
    if (sheets / f"{name}.mp4").exists():
        raise SystemExit(f"{name}.mp4 exists already")
    r, tr, _ = loaded(i)
    tr.meta["label"] = "scan boxes -> SAM 3 tracker, purity online + lone links (WP11c, rules frozen)"
    print(tracks.render_tracks(tr, sheets / f"{name}.mp4", sheets / f"{name}.jpg", crf=28))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "sheets", "render"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "summary":
        cmd_summary()
    elif a.cmd == "sheets":
        cmd_sheets(int(a.args[0]), a.out)
    else:
        cmd_render(int(a.args[0]), a.args[1])
