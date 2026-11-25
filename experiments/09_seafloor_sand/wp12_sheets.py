"""WP12 sheets for the user (decision only: one outline per region).

    masks     seafloor_masks_NN.jpg: the painted masks on the user-confirmed frames (seafloor brown, back purple,
              ignored grey)
    heldout   seafloor_heldout_found_NN.jpg / seafloor_heldout_missed_01.jpg: the held-out fold head's seafloor region
              on window frames of the episode it did not see
    false     seafloor_false_NN.jpg: the worst false seafloor window per recording / source (its head did not see it;
              2025-01-02 shown with fold 2, the fold with the most false windows)
    sand      sand_candidates_best_01.jpg: sediment frames from candidate_calls.csv for the user to confirm, with the
              places where the old sand label is wrong marked

Existing files are never replaced unless this script wrote them (names listed in wp12/sheets_written.txt).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "08_whale_fish_boxes")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import candidates as CA  # noqa: E402
import data as D  # noqa: E402
import wp12 as W  # noqa: E402
from uwid import patchseg as ps  # noqa: E402

TW, TH = 640, 360
MINE = W.W12 / "sheets_written.txt"
COL = {"seafloor": (60, 120, 200), "back": (200, 80, 160), "unsure": (170, 170, 170)}


def safe_write(name: str, img: np.ndarray):
    p = D.SHEETS / name
    mine = set(MINE.read_text().split()) if MINE.exists() else set()
    if p.exists() and name not in mine:
        raise SystemExit(f"{p} exists and was not written by WP12: not replaced")
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    mine.add(name)
    MINE.write_text("\n".join(sorted(mine)) + "\n")
    print("wrote", p)


def outline(img, mask, color, thick=2):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, [c for c in cs if cv2.contourArea(c) > 30], -1, color, thick)


def caption(img, lines, h=46):
    bar = np.zeros((h, img.shape[1], 3), np.uint8)
    for i, (t, c) in enumerate(lines):
        cv2.putText(bar, t, (6, 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def grid(tiles, cols=3):
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def masks():
    fs = W.folders()
    f = fs["wp12"][1]
    tiles = []
    for fr in f.frames():
        im = cv2.resize(cv2.imread(str(W.W12 / "frames" / f"{fr:06d}.jpg")), (TW, TH), interpolation=cv2.INTER_AREA)
        cm = cv2.resize(f.class_mask(fr), (TW, TH), interpolation=cv2.INTER_NEAREST)
        outline(im, cm == ps.SEAFLOOR, COL["seafloor"], 3)
        outline(im, cm == ps.BACK, COL["back"], 2)
        outline(im, cm == 255, COL["unsure"], 1)
        tiles.append(caption(im, [(f"2025-01-02 00017  t {fr / W.FPS:.0f} s  (episode E{W.episode_of(fr / W.FPS)})", (255, 255, 255)),
                                  ("brown = seafloor   purple = back   grey = not used (edge / sediment)", (200, 200, 200))]))
    for k in range(0, len(tiles), 12):
        safe_write(f"seafloor_masks_{k // 12 + 1:02d}.jpg", grid(tiles[k:k + 12]))


def _map_tile(kind, stem, z, i, head, text_lines):
    p = (D.SF_CACHE / "frames_tmp" / stem / f"{int(z['frame'][i]):06d}.jpg") if kind == "dense" else \
        [r for r in D.recordings() if r.stem == stem][0].image_path(int(z["frame"][i]))
    im = cv2.resize(cv2.imread(str(p)), (TW, TH), interpolation=cv2.INTER_AREA)
    m = z["maps"][i, list(z["heads"]).index(head)]
    sea = cv2.resize((m == ps.SEAFLOOR).astype(np.uint8), (TW, TH), interpolation=cv2.INTER_NEAREST)
    outline(im, sea, COL["seafloor"], 3)
    return caption(im, text_lines)


def heldout():
    import json
    thr = json.loads((HERE / "results" / "wp12_windows.json").read_text())["thresholds"]
    stem = W.S17
    z = dict(np.load(D.SF_CACHE / "frames_v2" / f"dense_{stem}.npz"))
    t = z["t"]
    found, missed = [], []
    for k in (1, 2, 3, 4):
        h = f"v2_fold{k}"
        hi = list(z["heads"]).index(h)
        ii = [i for i in range(len(t)) if W.episode_of(t[i]) == k]
        # window decision = mean share of the window's frames
        win = {}
        for i in ii:
            win.setdefault(int(np.floor(t[i])), []).append(i)
        for w, idx in sorted(win.items()):
            share = float(np.mean([(z["maps"][j, hi] == ps.SEAFLOOR).mean() for j in idx]))
            call = share >= thr[h]
            i = idx[0]
            tile = _map_tile("dense", stem, z, i, h, [
                (f"00017 t {t[i]:.1f} s  episode E{k} (held out: its head never saw it)", (255, 255, 255)),
                (f"window called seafloor: {'yes' if call else 'NO'}  (share {share:.2f}, threshold {thr[h]})",
                 (120, 220, 120) if call else (80, 80, 255))])
            (found if call else missed).append((k, w, tile))
    rng = np.random.default_rng(0)
    pick = []
    for k in (1, 2, 3, 4):
        fk = [x for x in found if x[0] == k]
        pick += [fk[j] for j in sorted(rng.choice(len(fk), min(6, len(fk)), replace=False))] if fk else []
    tiles = [x[2] for x in pick]
    for j in range(0, len(tiles), 12):
        safe_write(f"seafloor_heldout_found_{j // 12 + 1:02d}.jpg", grid(tiles[j:j + 12]))
    if missed:
        safe_write("seafloor_heldout_missed_01.jpg", grid([x[2] for x in missed[:12]]))
    print("found", len(found), "missed", len(missed))


def false():
    import json
    thr = json.loads((HERE / "results" / "wp12_windows.json").read_text())["thresholds"]
    tiles = []

    def worst(kind, stem, z, head, label, mask_fn=None, unit=None):
        t = z["t"] if unit is None else unit
        hi = list(z["heads"]).index(head)
        k = np.floor(t).astype(int)
        best = None
        for w in np.unique(k):
            if mask_fn is not None and not mask_fn(w):
                continue
            idx = np.nonzero(k == w)[0]
            share = float(np.mean([(z["maps"][j, hi] == ps.SEAFLOOR).mean() for j in idx]))
            if best is None or share > best[0]:
                best = (share, idx[0], w)
        if best is None:
            return
        share, i, w = best
        call = share >= thr[head]
        tiles.append(_map_tile(kind, stem, z, i, head, [
            (f"{label}  window {w} s", (255, 255, 255)),
            (f"worst window: called seafloor {'YES (false)' if call else 'no'}  share {share:.2f} (thr {thr[head]}, {head})",
             (80, 80, 255) if call else (120, 220, 120))]))

    D2 = {D.piece(r.stem): r.stem for r in D.recordings(("d2",))}
    z17 = dict(np.load(D.SF_CACHE / "frames_v2" / f"dense_{W.S17}.npz"))
    worst("dense", W.S17, z17, "v2_fold2", "00017 withdrawn 'visible' 842-1390 (hard negative)", lambda w: w >= 842)
    worst("dense", W.S17, z17, "v2_fold2", "00017 sediment / black / unsure 536-842", lambda w: 536 <= w < 842)
    for piece in ("00010", "00011", "00012", "00023", "00024"):
        p = D.SF_CACHE / "frames_v2" / f"dense_{D2[piece]}.npz"
        if p.exists():
            z = dict(np.load(p))
            if piece == "00023":
                worst("dense", D2[piece], z, "v2_fold2", "00023 withdrawn 'visible' 0-306 (hard negative)",
                      lambda w: w < 306 and not 286 <= w < 301)
                worst("dense", D2[piece], z, "v2_fold2", "00023 weak positive 286-301 (not fitted on)", lambda w: 286 <= w < 301)
            else:
                worst("dense", D2[piece], z, "v2_fold2", f"2025-01-02 {piece}")
    p = D.SF_CACHE / "frames_v2" / f"dense_{CA.D.recordings(('d1',))[2].stem}.npz"
    if p.exists():
        worst("dense", CA.D.recordings(("d1",))[2].stem, dict(np.load(p)), "v2_test_d1", "2025-02-03 00006 (head not fitted on 2025-02-03)")
    for rec in D.recordings(("dev",)):
        z = dict(np.load(D.SF_CACHE / "frames_v2" / f"label_{rec.stem}.npz"))
        worst("label", rec.stem, z, "v2_test_dev", "dev clip (head not fitted on it)")
    for j in range(0, len(tiles), 12):
        safe_write(f"seafloor_false_{j // 12 + 1:02d}.jpg", grid(tiles[j:j + 12]))


def sand():
    calls = pd.read_csv(HERE / "candidate_calls.csv")
    sed = calls[(calls.call == "sediment") & (calls.piece == 17)].t.tolist() + [330, 426, 434, 548, 744]
    sed = sorted(set(sed))[:22]
    wrong = {}
    for t in sed:
        if 327 <= t < 332 or 422 <= t < 440 or 449 <= t < 459 or 510 <= t < 524 or 610 <= t < 615:
            wrong[t] = "old sand label: none here (cloud not labelled)"
    rows = [(t, "sediment", wrong.get(t, "")) for t in sed]
    rows += [(411, "bottom, not sand", "old sand label: SAND here, but it is the bottom (seafloor sheet)"),
             (660, "back in haze, no cloud", "old sand label: SAND here, but no cloud is seen")]
    tiles = []
    stem = W.S17
    for t, call, note in rows:
        f = CA.OUT / stem / f"{int(round(t * 10)):06d}.jpg"
        im = cv2.imread(str(f))
        a = cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA)
        b = cv2.resize(CA.stretch(im), (320, 180), interpolation=cv2.INTER_AREA)
        tile = np.zeros((TH + 46, TW + 320, 3), np.uint8)
        tile[:TH, :TW] = a; tile[:180, TW:] = b
        cv2.putText(tile, "contrast-stretched copy", (TW + 10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if note:
            cv2.putText(tile, note[:46], (TW + 10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 160, 255), 1, cv2.LINE_AA)
            cv2.putText(tile, note[46:92], (TW + 10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 160, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, f"2025-01-02 00017  t {t:.0f} s  call: {call}", (6, TH + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (40, 200, 200) if call == "sediment" else (80, 160, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    safe_write("sand_candidates_best_01.jpg", grid(tiles))


if __name__ == "__main__":
    {"masks": masks, "heldout": heldout, "false": false, "sand": sand}[sys.argv[1]]()
