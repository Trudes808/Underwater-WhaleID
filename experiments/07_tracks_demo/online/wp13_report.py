"""WP13: summary and review pack of the tracker run on all hq spans (rules frozen as of WP11c). CPU only.

Spans = what `uwid tracks <deployment>` runs: every hq span of a kept stretch with advanced_ok in the scan's
`sensitive` intervals.json, capped at 30 s around its centre (none is longer today); the run starts at the span's
first scan start-box frame.

REVIEW-PACK RULES, written 2026-11-23 before the draw (tool's decision only on every sheet):
a. Sample: displayed numbers (after the link layer) whose life (first to last outlined frame) is >= 3 s; numpy
   default_rng(20261123); 12 per deployment drawn without replacement from that deployment's numbers sorted by
   (recording, span, number); a deployment with fewer than 12 gives all of them and the other fills up to 24.
   Each as 6 frames evenly spaced over its life.
b. Every link made: A's frame 1 s before its last outlined frame, A's last, B's first, B's frame 1 s after its first.
c. Bubble suspects: every number whose first outlined frame lies within 10 s after the end of a lens-out run (the
   frozen blocked rule's lens part: depth lens-out AND picture air / crossing, runs of the whole recording); its first
   frame and its largest frame.
d. Calf suspects: every tracked whale-sized object the tool refused as the tagged whale's own back (role `on_back`:
   >= 60 % of its outline inside the back region) whose life is >= 3 s; the plain frame beside the outlined frame at
   its largest outline. (There is no separate "own back" start refusal in the tool: this role is the refusal.)
Lists c and d go to the cache as CSV.

    .venv/bin/python experiments/07_tracks_demo/online/wp13_report.py summary
    .venv/bin/python experiments/07_tracks_demo/online/wp13_report.py sheets
    .venv/bin/python experiments/07_tracks_demo/online/wp13_report.py videos
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from uwid import paths, trackstage as T, tracks  # noqa: E402

DEPS = ["TestSubset 2025-1-2 CATS -85", "2025-2-3 Mini-mother 11-96"]
SHORT = {DEPS[0]: "CC-09-85", DEPS[1]: "2025-02-03"}
CACHE = paths.cache_dir("tracks")
SEED = 20261123


def span_items(dep: str) -> list[tuple[str, float, float]]:
    s = T.hq_spans(dep)
    out = []
    for r in s.itertuples():
        a, b = r.start, r.end
        if b - a > 30:
            c = (a + b) / 2
            a, b = c - 15, c + 15
        out.append((r.source, a, b))
    return out


def _sb():
    spec = importlib.util.spec_from_file_location("select_breath", HERE / "select_breath.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def loaded(dep: str, d: Path):
    tr = T.load_online(d)
    n0 = len(tr.number)
    roles = dict(tr.roles)
    T.link(tr, dep, "scan")
    return tr, n0, roles


def numbers_of(tr) -> list[dict]:
    """One row per displayed number: its objects, first / last outlined frame, outlined frames, largest frame."""
    fps, t0 = tr.meta["fps"], tr.meta["t0"]
    t = tr.table[(tr.table.area > 0)]
    out = []
    for num in sorted(set(tr.number.values())):
        objs = [k for k, v in tr.number.items() if v == num]
        g = t[t.obj.isin(objs)]
        g = g[[(int(i), int(o)) not in tr.hidden for i, o in zip(g.i, g.obj)]]
        if not len(g):
            continue
        big = g.loc[g.area.idxmax()]
        out.append(dict(number=num, objs=";".join(map(str, objs)), i_first=int(g.i.min()), i_last=int(g.i.max()),
                        t_first=round(t0 + g.i.min() / fps, 2), t_last=round(t0 + g.i.max() / fps, 2),
                        life_s=round((g.i.max() - g.i.min()) / fps, 2), outlined_s=round(g.i.nunique() / fps, 2),
                        i_big=int(big.i), obj_big=int(big.obj), area_big=int(big.area),
                        box_big=f"{big.x0:.0f},{big.y0:.0f},{big.x1:.0f},{big.y1:.0f}"))
    return out


def on_back_of(tr, roles) -> list[dict]:
    fps, t0 = tr.meta["fps"], tr.meta["t0"]
    t = tr.table[tr.table.area > 0]
    out = []
    for k, r in roles.items():
        if r != "on_back" or k == tracks.BACK_ID:
            continue
        g = t[t.obj == k]
        if not len(g):
            continue
        big = g.loc[g.area.idxmax()]
        out.append(dict(obj=int(k), t_first=round(t0 + g.i.min() / fps, 2), t_last=round(t0 + g.i.max() / fps, 2),
                        life_s=round((g.i.max() - g.i.min()) / fps, 2), back_share=round(tr.back_share.get(k, np.nan), 3),
                        i_big=int(big.i), area_big=int(big.area),
                        box_big=f"{big.x0:.0f},{big.y0:.0f},{big.x1:.0f},{big.y1:.0f}"))
    return out


def cmd_summary():
    sb = _sb()
    spans, nums, links, backs, bubbles = [], [], [], [], []
    for dep in DEPS:
        lens_cache = {}
        for src, a, b in span_items(dep):
            name = Path(src).name
            d = T.span_dir(dep, name, a, b)
            if not (d / "meta.json").exists() or not json.loads((d / "meta.json").read_text()).get("done"):
                spans.append(dict(deployment=dep, recording=name, t0=a, t1=b, done=False))
                continue
            tr, n0, roles = loaded(dep, d)
            m = tr.meta
            L = T.links_table(tr)
            T.write_links(tr, d / "links.csv")
            ns = numbers_of(tr)
            for x in ns:
                x.update(deployment=dep, recording=name, span=d.name, folder=str(d))
            nums += ns
            for x in on_back_of(tr, roles):
                x.update(deployment=dep, recording=name, span=d.name, folder=str(d))
                backs.append(x)
            if name not in lens_cache:
                w = T.windows_table(dep, name)
                lens_cache[name] = sb.breaths(dep, name, w) if w is not None else []
            for x in ns:
                prev = [e for s_, e in lens_cache[name] if 0 <= x["t_first"] - e <= 10]
                if prev:
                    bubbles.append(dict(x, lens_out_end=max(prev), after_s=round(x["t_first"] - max(prev), 2)))
            if len(L):
                L.insert(0, "span", d.name)
                L.insert(0, "recording", name)
                L.insert(0, "deployment", dep)
                links.append(L)
            spans.append(dict(deployment=dep, recording=name, t0=a, t1=b, done=True, folder=str(d),
                              video_s=round(m["n_frames"] / m["fps"], 2), gpu_s=m["seconds_total"],
                              numbers_purity=n0, numbers=len(set(tr.number.values())), small=len(tr.small_number),
                              track_s=round(sum(x["outlined_s"] for x in ns), 2), blocked_frames=m.get("n_blocked", 0),
                              links_made=int(L.made.sum()) if len(L) else 0,
                              refused=json.dumps(Counter(r["reason"] for r in m.get("refused", []))),
                              endings=len(tr.endings), on_back_ge3s=sum(x["life_s"] >= 3 for x in on_back_of(tr, roles))))
            print(dep[:10], name[-9:-4], f"{a:.1f}-{b:.1f}", spans[-1]["numbers"], flush=True)
    S, N = pd.DataFrame(spans), pd.DataFrame(nums)
    Lk = pd.concat(links, ignore_index=True) if links else pd.DataFrame()
    S.to_csv(CACHE / "wp13_spans.csv", index=False)
    N.to_csv(CACHE / "wp13_numbers.csv", index=False)
    Lk.to_csv(CACHE / "wp13_links.csv", index=False)
    B = pd.DataFrame(backs)
    (B[B.life_s >= 3] if len(B) else B).to_csv(CACHE / "wp13_calf_suspects.csv", index=False)
    pd.DataFrame(bubbles).to_csv(CACHE / "wp13_bubble_suspects.csv", index=False)
    for dep in DEPS:
        T.write_index(dep)
    # per deployment
    rows = []
    for dep in DEPS:
        s = S[(S.deployment == dep) & S.done]
        n = N[N.deployment == dep] if len(N) else N
        lk = Lk[Lk.deployment == dep] if len(Lk) else Lk
        ref = Counter()
        for r in s.refused:
            ref.update(json.loads(r))
        rows.append(dict(deployment=SHORT[dep], spans=len(S[S.deployment == dep]), spans_done=len(s),
                         video_s=round(s.video_s.sum(), 1), numbers=int(s.numbers.sum()),
                         numbers_before_links=int(s.numbers_purity.sum()), numbers_ge3s=int((n.life_s >= 3).sum()) if len(n) else 0,
                         track_s=round(s.track_s.sum(), 1), blocked_frames=int(s.blocked_frames.sum()),
                         links_made=int(lk.made.sum()) if len(lk) else 0,
                         links_not_made=json.dumps(lk[~lk.made].why.value_counts().to_dict()) if len(lk) else "{}",
                         refused_starts=json.dumps(dict(ref)), small=int(s.small.sum()),
                         gpu_min=round(s.gpu_s.sum() / 60, 1),
                         gpu_min_per_span_min=round(s.gpu_s.sum() / max(s.video_s.sum(), 1e-6), 2),
                         bubble_suspects=int(sum(1 for x in bubbles if x["deployment"] == dep)),
                         calf_suspects=int(sum(1 for x in backs if x["deployment"] == dep and x["life_s"] >= 3))))
    R = pd.DataFrame(rows)
    R.to_csv(HERE / "results" / "wp13_summary.csv", index=False)
    print(R.T.to_string())


# ------------------------------------------------------------------------------------------------ sheets

def _tile(img, text, w=480, h=270):
    import cv2
    t = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    for c, th in (((0, 0, 0), 4), ((255, 255, 0), 1)):
        cv2.putText(t, text, (6, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, th + 1)
    return t


def _frames(tr, idx: list[int], outline=True) -> dict[int, np.ndarray]:
    import cv2
    from uwid import video as uv
    f0 = tr.meta["frames"][0]
    sw, sh = tr.meta["store_size"]
    out = {}
    for fr in uv.iter_frames_at(tr.meta["video"], [f0 + i for i in idx]):
        img = cv2.resize(fr.image, (sw, sh), interpolation=cv2.INTER_AREA)
        i = fr.index - f0
        out[i] = (tracks.draw_frame(img, tr, i, f"t {fr.t:.2f}", {}) if outline else img)
    return out


def _write(rows: list[list[np.ndarray]], stem: str, per_sheet: int = 6) -> list[Path]:
    import cv2
    sheets = paths.sheets_dir(create=True)
    if not rows:
        return []
    width = max(len(r) for r in rows)
    blank = np.zeros_like(rows[0][0])
    rows = [np.hstack(r + [blank] * (width - len(r))) for r in rows]
    out = []
    for p in range(0, len(rows), per_sheet):
        fn = sheets / f"{stem}_{p // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(fn), cv2.cvtColor(np.vstack(rows[p:p + per_sheet]), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append(fn)
    return out


_TR: dict = {}


def tr_of(dep, folder):
    if folder not in _TR:
        _TR.clear()
        _TR[folder] = loaded(dep, Path(folder))
    return _TR[folder]


def cmd_sheets():
    N = pd.read_csv(CACHE / "wp13_numbers.csv")
    # a. random sample
    rng = np.random.default_rng(SEED)
    elig = N[N.life_s >= 3].sort_values(["deployment", "recording", "span", "number"]).reset_index(drop=True)
    take = {dep: min(12, int((elig.deployment == dep).sum())) for dep in DEPS}
    short = 24 - sum(take.values())
    for dep in DEPS:
        extra = min(short, int((elig.deployment == dep).sum()) - take[dep])
        take[dep] += extra
        short -= extra
    pick = []
    for dep in DEPS:
        idx = elig.index[elig.deployment == dep].to_numpy()
        pick += list(rng.choice(idx, size=take[dep], replace=False)) if take[dep] else []
    sample = elig.loc[sorted(pick)]
    sample.to_csv(CACHE / "wp13_sample_a.csv", index=False)
    rows = []
    for r in sample.itertuples():
        tr, _, _ = tr_of(r.deployment, r.folder)
        idx = [int(round(x)) for x in np.linspace(r.i_first, r.i_last, 6)]
        fr = _frames(tr, idx)
        rows.append([_tile(fr[i], f"a {SHORT[r.deployment]} {r.recording[-9:-4]} n{r.number} {tr.meta['t0'] + i / tr.meta['fps']:.1f}s")
                     for i in idx if i in fr])
    print(_write(rows, "wp13_a_sample"))
    # b. links made
    Lk = pd.read_csv(CACHE / "wp13_links.csv") if (CACHE / "wp13_links.csv").stat().st_size > 2 else pd.DataFrame()
    rows = []
    for r in (Lk[Lk.made].itertuples() if len(Lk) else []):
        folder = str(T.span_dir(r.deployment, r.recording, *[float(x) for x in r.span.split("_")[0].split("-")]))
        tr, _, _ = tr_of(r.deployment, folder)
        fps = tr.meta["fps"]
        t = tr.table[tr.table.area > 0]
        la, fb = int(t[t.obj == r.a].i.max()), int(t[t.obj == r.b].i.min())
        idx = [max(0, la - int(fps)), la, fb, min(tr.n - 1, fb + int(fps))]
        fr = _frames(tr, idx)
        rows.append([_tile(fr[i], f"b {r.recording[-9:-4]} {r.a}->{r.b} gap {r.gap_s}s {tr.meta['t0'] + i / fps:.1f}s")
                     for i in idx if i in fr])
    print(_write(rows, "wp13_b_links"))
    # c. bubble suspects
    Bs = pd.read_csv(CACHE / "wp13_bubble_suspects.csv") if (CACHE / "wp13_bubble_suspects.csv").stat().st_size > 2 \
        else pd.DataFrame()
    rows, row = [], []
    for r in (Bs.itertuples() if len(Bs) else []):
        tr, _, _ = tr_of(r.deployment, r.folder)
        fr = _frames(tr, [r.i_first, r.i_big])
        row += [_tile(fr[i], f"c {r.recording[-9:-4]} n{r.number} +{r.after_s}s {tr.meta['t0'] + i / tr.meta['fps']:.1f}s")
                for i in (r.i_first, r.i_big) if i in fr]
        if len(row) >= 6:
            rows.append(row[:6])
            row = row[6:]
    if row:
        rows.append(row)
    print(_write(rows, "wp13_c_bubble_suspects"))
    # d. calf suspects
    Cs = pd.read_csv(CACHE / "wp13_calf_suspects.csv") if (CACHE / "wp13_calf_suspects.csv").stat().st_size > 2 \
        else pd.DataFrame()
    rows, row = [], []
    for r in (Cs.itertuples() if len(Cs) else []):
        tr, _, _ = tr_of(r.deployment, r.folder)
        plain = _frames(tr, [r.i_big], outline=False)
        img = tr.mask(r.obj, r.i_big)
        if r.i_big not in plain:
            continue
        import cv2
        o = plain[r.i_big].copy()
        cs, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cs, -1, (255, 80, 255), 3)
        lab = f"d {r.recording[-9:-4]} obj{r.obj} {r.t_first:.1f}-{r.t_last:.1f}s back {r.back_share}"
        row += [_tile(plain[r.i_big], lab + " plain"), _tile(o, lab)]
        if len(row) >= 6:
            rows.append(row[:6])
            row = row[6:]
    if row:
        rows.append(row)
    print(_write(rows, "wp13_d_calf_suspects"))


def cmd_videos(n: int = 6, max_mb: float = 300.0):
    N = pd.read_csv(CACHE / "wp13_numbers.csv").sort_values("life_s", ascending=False)
    sheets = paths.sheets_dir(create=True)
    done, total = set(), 0.0
    for r in N.itertuples():
        if len(done) == n:
            break
        if r.folder in done:
            continue
        name = f"demo_wp13_{len(done) + 1:02d}_{SHORT[r.deployment]}_{r.recording[-9:-4]}_{r.t_first:.0f}s_n{r.number}"
        if (sheets / f"{name}.mp4").exists():
            raise SystemExit(f"{name}.mp4 exists already")
        tr, _, _ = loaded(r.deployment, Path(r.folder))
        tr.meta["label"] = "scan boxes -> SAM 3 tracker, purity online + lone links (WP13, all hq spans, rules frozen)"
        res = tracks.render_tracks(tr, sheets / f"{name}.mp4", sheets / f"{name}.jpg", crf=28)
        total += res["mb"]
        done.add(r.folder)
        print(name, r.life_s, res["mb"], "MB; total", round(total, 1), flush=True)
        if total > max_mb:
            print("over the size cap: stop")
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "sheets", "videos"])
    a = ap.parse_args()
    {"summary": cmd_summary, "sheets": cmd_sheets, "videos": cmd_videos}[a.cmd]()
