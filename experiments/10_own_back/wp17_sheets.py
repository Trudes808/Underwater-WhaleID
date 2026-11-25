"""WP17 sheets for the user (tool's decision only): whale numbers as outlines with their number, and the own-back line
the tool uses (purple: with the flag off the back object, with the flag on the back object inside the template).

    ownback_replay_calf_01.jpg      the calf stretch, flag off (stored run) above flag on, the same 8 frames
    ownback_replay_calf.mp4         the calf stretch with the flag on
    ownback_replay_candidates_NN.jpg  one row of 6 frames per candidate stretch with the flag on (unclear cases first)

    .venv/bin/python experiments/10_own_back/wp17_sheets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402
import wp17 as P  # noqa: E402
from uwid import tracks, video  # noqa: E402
from uwid import trackstage as T  # noqa: E402

TW, TH = 480, 270
COLS = [(40, 60, 255), (0, 200, 255), (60, 220, 60), (255, 120, 40), (255, 0, 200), (0, 255, 255)]
MINE = C.OUT / "sheets_written.txt"
DEPN = {"d1": "2025-02-03", "d2": "2025-01-02"}


def safe_write(name, img=None, src=None):
    p = C.SHEETS / name
    mine = set(MINE.read_text().split()) if MINE.exists() else set()
    if p.exists() and name not in mine:
        raise SystemExit(f"{p} exists and was not written by WP14 / WP17")
    if img is not None:
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    elif src is not None:
        Path(src).replace(p)
    mine.add(name); MINE.write_text("\n".join(sorted(mine)) + "\n")
    print("wrote", p)


def own_mask(tr, i):
    b = tr.mask(tracks.BACK_ID, i)
    return b & tr.own_back if tr.own_back is not None else b


def draw(tr, frames_img: dict, i: int, head: str) -> np.ndarray:
    f = int(tr.meta["frames"][0]) + i
    img = cv2.cvtColor(frames_img[f], cv2.COLOR_RGB2BGR) if f in frames_img else np.zeros((720, 1280, 3), np.uint8)
    if tr.back is not None:
        cs, _ = cv2.findContours(own_mask(tr, i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (200, 80, 160), 4)
    for o, n in tr.number.items():
        if (i, o) in tr.hidden or not tr.packed[o][i].any():
            continue
        m = tr.mask(o, i).astype(np.uint8)
        c = COLS[(n - 1) % len(COLS)]
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, c, 4)
        ys, xs = np.nonzero(m)
        cv2.putText(img, str(n), (int(xs.mean()), max(30, int(ys.min()) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 4)
    t = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
    bar = np.zeros((24, TW, 3), np.uint8)
    cv2.putText(bar, head, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([t, bar])


def decode(tr, idx):
    f0 = int(tr.meta["frames"][0])
    return {fr.index: fr.image for fr in video.iter_frames_at(tr.meta["video"], [f0 + i for i in idx], size=(1280, 720))}


def calf():
    st = P.stretches()
    r = st[st.kind == "calf"].iloc[0]
    base = T.span_dir(r.deployment, Path(r.video).name, float(r.t0), float(r.t1))
    off = T.load_online(base)
    on = T.load_online(base.with_name(base.name + "_ownback"))
    fo, fn = int(off.meta["frames"][0]), int(on.meta["frames"][0])
    n = min(off.n - (fn - fo if fn > fo else 0), on.n)
    abs_frames = np.linspace(max(fo, fn), max(fo, fn) + n - 1, 8).round().astype(int)
    fr = {fr.index: fr.image for fr in video.iter_frames_at(on.meta["video"], abs_frames.tolist(), size=(1280, 720))}
    fps = float(on.meta["fps"])
    rows = []
    for tr, lab in ((off, "flag OFF (stored run)"), (on, "flag ON (own back inside the template)")):
        f0 = int(tr.meta["frames"][0])
        tiles = [draw(tr, fr, int(a - f0), f"{lab}  t {tr.meta['t0'] + (a - f0) / fps:.1f} s") for a in abs_frames]
        rows.append(np.hstack(tiles[:4])); rows.append(np.hstack(tiles[4:]))
    head = np.zeros((30, rows[0].shape[1], 3), np.uint8)
    cv2.putText(head, "2025-02-03 00005 607-627 s (calf in front of the tagged mother): top two rows flag off, bottom two flag on; "
                "purple = own back as the tool uses it; coloured = whale numbers", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    safe_write("ownback_replay_calf_01.jpg", np.vstack([head] + rows))
    # video with the flag on: the renderer draws the back object, so give it the own back in its place
    vt = T.load_online(base.with_name(base.name + "_ownback"))
    w = int(vt.meta["store_size"][0])
    vt.back = np.stack([np.packbits(own_mask(vt, i), axis=1) for i in range(vt.n)])
    tmp = P.OUT / "ownback_replay_calf.mp4"
    tracks.render_tracks(vt, tmp, None)
    safe_write("ownback_replay_calf.mp4", src=tmp)
    print("mp4 MB", round((C.SHEETS / "ownback_replay_calf.mp4").stat().st_size / 1e6, 1))


def candidates():
    st = P.stretches()
    st = st[st.kind == "candidate"]
    rows = []
    for r in st.itertuples():
        base = T.span_dir(r.deployment, Path(r.video).name, float(r.t0), float(r.t1))
        d = base.with_name(base.name + "_ownback")
        if not (d / "meta.json").exists():
            continue
        tr = T.load_online(d)
        idx = np.linspace(0, tr.n - 1, 6).round().astype(int).tolist()
        fr = decode(tr, idx)
        fps = float(tr.meta["fps"])
        tmpl = "template" if tr.own_back is not None else "NO template (too few frames in the block): as flag off"
        tiles = [draw(tr, fr, i, f"{DEPN[r.src]} {r.piece}  t {tr.meta['t0'] + i / fps:.1f} s") for i in idx]
        bar = np.zeros((26, TW * 6, 3), np.uint8)
        txt = f"{DEPN[r.src]} {r.piece} {r.t0:.0f}-{r.t1:.0f} s, flag ON ({tmpl}); numbers: {len(set(tr.number.values()))}"
        if r.unclear:
            txt += "   [unclear case]"
        cv2.putText(bar, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255) if r.unclear else (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, np.hstack(tiles)]))
    for k in range(0, len(rows), 4):
        safe_write(f"ownback_replay_candidates_{k // 4 + 1:02d}.jpg", np.vstack(rows[k:k + 4]))


if __name__ == "__main__":
    calf(); candidates()
