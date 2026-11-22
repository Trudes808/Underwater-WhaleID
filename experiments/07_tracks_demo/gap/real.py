"""WP8b step 3-5: real blocked views, three ways, and the side-by-side videos.

Variants (same stretch, same chain, OWLv2 boxes + SAM 3 tracker):
  a  straight through (as in the WP8 demos)
  b  blocked frames (blocked.py rule) not shown to the tracker: it meets a jump
  c  blocked frames shown, but nothing from them enters the tracker's memory
No new track starts on a blocked frame in b and c.

    .venv/bin/python experiments/07_tracks_demo/gap/real.py eval            # -> results/real_<case>.json, table
    .venv/bin/python experiments/07_tracks_demo/gap/real.py render [ids]    # side-by-side videos a | b
    .venv/bin/python experiments/07_tracks_demo/gap/real.py judge 01 --out <dir>   # dense a|b|c sheets for the eye
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import REAL, tracks  # noqa: E402

from uwid import timecode, video as uv  # noqa: E402

SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
VARIANTS = {"a": "straight through", "b": "blocked frames cut out of the tracker's view",
            "c": "blocked frames shown, memory not updated"}


def blocked(cid: str) -> dict:
    return json.loads((HERE / "results" / f"blocked_{cid}.json").read_text())


def run_dir(cid: str, var: str) -> Path:
    v, t0, t1 = REAL[cid][:3]
    if var == "a":
        return tracks.tracks_dir(v, t0, t1, "owl_sam3")
    return tracks.tracks_dir(v, 0, 0, "owl_sam3").with_name(f"real_{cid}_{'cut' if var == 'b' else 'gate'}")


def run(cid: str, var: str, log=print) -> Path:
    v, t0, t1 = REAL[cid][:3]
    b = blocked(cid)
    rng = [tuple(r) for r in b["ranges"]]
    if var == "a":
        return tracks.run_tracks(v, t0, t1, log=log)
    kw = dict(hide=rng) if var == "b" else dict(gate=rng)
    return tracks.run_tracks(v, t0, t1, name=run_dir(cid, var).name, log=log, **kw)


def episodes(ranges, fps: float, join_s: float = 2.5):
    """Blocked ranges closer than ``join_s`` form one episode for scoring (the tracker still sees the open frames
    between them in variants b and c)."""
    out = []
    for a, e in ranges:
        if out and a - out[-1][1] < join_s * fps:
            out[-1][1] = e
        else:
            out.append([a, e])
    return [tuple(x) for x in out]


def _visible_ids(tr, frames, f0, min_frames=3):
    ids = {}
    for k, num in tr.number.items():
        c = sum(1 for f in frames if 0 <= f - f0 < tr.n and tr.visible(k, f - f0))
        if c >= min_frames:
            ids[num] = c
    return set(ids)


def evaluate(cid: str) -> dict:
    b = blocked(cid)
    out = dict(case=cid, what=REAL[cid][4], blocked_s=b["ranges_s"], variants={})
    blocked_frames = [f for a, e in b["ranges"] for f in range(a, e)]
    for var in VARIANTS:
        d = run_dir(cid, var)
        if not (d / "meta.json").exists():
            continue
        tr = tracks.load_tracks(d)
        f0 = tr.meta["frames"][0]
        fps = tr.meta["fps"]
        gaps = []
        for a, e in episodes(b["ranges"], fps):
            before = _visible_ids(tr, range(a - int(fps), a), f0)
            after = _visible_ids(tr, range(e + int(0.5 * fps), e + int(2.5 * fps)), f0)
            nb = sum(1 for f in blocked_frames if a <= f < e)
            gaps.append(dict(gap_s=round((e - a) / fps, 2), blocked_s=round(nb / fps, 2), ids_before=sorted(before), ids_after=sorted(after),
                             same_number=sorted(before & after), new_after=sorted(after - before),
                             gone=sorted(before - after)))
        on_blocked = sum(1 for f in blocked_frames if 0 <= f - f0 < tr.n and any(tr.visible(k, f - f0) for k in tr.number))
        out["variants"][var] = dict(whale_ids=len(tr.number), gaps=gaps,
                                    blocked_frames_with_outline=f"{on_blocked}/{len(blocked_frames)}",
                                    gpu_s_per_video_s=round(tr.meta["seconds_total"] / (tr.n / fps), 2))
    (HERE / "results" / f"real_{cid}.json").write_text(json.dumps(out, indent=1))
    return out


def cmd_eval():
    for cid in REAL:
        try:
            r = evaluate(cid)
        except FileNotFoundError as e:
            print(cid, "missing", e)
            continue
        print(f"{cid} {r['what']}  blocked {r['blocked_s']}")
        for var, x in r["variants"].items():
            g = "; ".join(f"gap {q['gap_s']} s: before {q['ids_before']} after {q['ids_after']} same {q['same_number']} "
                          f"new {q['new_after']}" for q in x["gaps"])
            print(f"   {var}: ids {x['whale_ids']}, outlines on blocked frames {x['blocked_frames_with_outline']} | {g}")


def _half(img, tr, i, header, lost):
    return tracks.draw_frame(img, tr, i, header, lost)


def render(cid: str, crf: int = 20):
    v, t0, t1, short = REAL[cid][:4]
    ta, tb = tracks.load_tracks(run_dir(cid, "a")), tracks.load_tracks(run_dir(cid, "b"))
    b = blocked(cid)
    blk = set(f for a, e in b["ranges"] for f in range(a, e))
    f0, f1 = ta.meta["frames"]
    assert tb.meta["frames"][0] == f0
    sw, sh = ta.meta["store_size"]
    hw, hh = 960, 540
    tab = uv.frame_table(v)
    glyphs = timecode.load_glyphs()
    import datetime as _dt
    import re
    m = re.search(r"-(20\d{2})(\d{2})(\d{2})-", Path(v).name)
    hint = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    out = SHEETS / f"demo_gap_{cid}_{short}.mp4"
    cmd = [tracks._ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * hw}x{hh}",
           "-r", f"{ta.meta['fps']:.5f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = ta.n
    sheet_at = set(np.linspace(0, n - 1, 8).round().astype(int).tolist())
    # put two sheet frames inside the first blocked range and one right after it, so the sheet shows the gap
    if b["ranges"]:
        a0, e0 = b["ranges"][0]
        extra = [a0 + (e0 - a0) // 2 - f0, min(n - 1, e0 + 20 - f0)]
        sheet_at = set(sorted(sheet_at)[:6] + extra)
    sheet = []
    last = {"a": {}, "b": {}}
    for i, fr in enumerate(uv.iter_frames(v, start=float(tab.times[f0]) - 1e-3, end=float(tab.times[f1 - 1]) + 1e-3)):
        if i >= n:
            break
        rd = timecode.read_clock(fr.image, glyphs, date_hint=hint)
        clock = rd.text if rd is not None else "--"
        img = cv2.resize(fr.image, (sw, sh), interpolation=cv2.INTER_AREA)
        halves = []
        for var, tr in (("a", ta), ("b", tb)):
            lost = {}
            for k, num in tr.number.items():
                if tr.visible(k, i):
                    last[var][num] = fr.t
                elif num in last[var]:
                    lost[num] = last[var][num]
            hdr = f"{var.upper()}: {VARIANTS[var]}   t = {fr.t:7.2f} s   {clock}"
            if var == "b" and fr.index in blk:
                im = (img.astype(np.float32) * 0.55).astype(np.uint8)
                band = im[:44].astype(np.float32) * 0.35
                im[:44] = band.astype(np.uint8)
                cv2.putText(im, hdr, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                for j, txt in enumerate(["view blocked:", "not shown to the tracker"]):
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
                    x, y = (sw - tw) // 2, sh // 2 - 10 + j * 50
                    cv2.putText(im, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 6, cv2.LINE_AA)
                    cv2.putText(im, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 210, 80), 3, cv2.LINE_AA)
            else:
                im = _half(img, tr, i, hdr, lost)
            halves.append(cv2.resize(im, (hw, hh), interpolation=cv2.INTER_AREA))
        frame = np.hstack(halves)
        frame[:, hw - 1:hw + 1] = 255
        pr.stdin.write(frame.tobytes())
        if i in sheet_at:
            sheet.append(frame)
    pr.stdin.close()
    pr.wait()
    tiles = [cv2.resize(s, (2 * hw // 2, hh // 2), interpolation=cv2.INTER_AREA) for s in sheet]
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[j:j + 2]) for j in range(0, len(tiles), 2)]
    cv2.imwrite(str(out.with_suffix(".jpg")), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"rendered {out.name}: {out.stat().st_size / 1e6:.1f} MB")


def judge(cid: str, out_dir: str, step: float = 0.33, t_from=None, t_to=None):
    """Rows of a | b | c at every ``step`` s (blocked frames marked)."""
    v = REAL[cid][0]
    trs = {var: tracks.load_tracks(run_dir(cid, var)) for var in VARIANTS if (run_dir(cid, var) / "meta.json").exists()}
    b = blocked(cid)
    blk = set(f for a, e in b["ranges"] for f in range(a, e))
    ta = trs["a"]
    f0 = ta.meta["frames"][0]
    sw, sh = ta.meta["store_size"]
    every = max(1, int(round(step * ta.meta["fps"])))
    rows = []
    for fr in uv.iter_frames_at(v, list(range(f0, f0 + ta.n, every))):
        if (t_from is not None and fr.t < t_from) or (t_to is not None and fr.t > t_to):
            continue
        i = fr.index - f0
        img = cv2.resize(fr.image, (sw, sh), interpolation=cv2.INTER_AREA)
        cells = []
        for var, tr in trs.items():
            hdr = f"{var} t={fr.t:.2f}" + ("  BLOCKED" if fr.index in blk else "")
            im = tracks.draw_frame(img, tr, i, hdr, {})
            cells.append(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack(cells))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in range(0, len(rows), 8):
        fn = out / f"gapjudge_{cid}_{p // 8}.jpg"
        cv2.imwrite(str(fn), cv2.cvtColor(np.vstack(rows[p:p + 8]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(fn)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "render", "judge"])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--step", type=float, default=0.33)
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    a = ap.parse_args()
    if a.cmd == "eval":
        cmd_eval()
    elif a.cmd == "render":
        for cid in a.ids or list(REAL):
            render(cid)
    else:
        for cid in a.ids:
            judge(cid, a.out, a.step, a.t0, a.t1)
