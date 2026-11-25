"""Step 1 and 2 of 08_whale_fish_boxes: the table of labelled detector boxes and what the scan's rules call them.

Unit = one OBJECT: OWLv2 patch boxes of one frame (all prompts) whose best whale / fish prompt score is >= 0.2, grouped
at IoU >= 0.5 (uwid.boxclf.unique_boxes + group_objects); the object's best box is its geometry. Only frames
labelled `underwater` (mask frames outside frames.csv: underwater unless most pixels are `air`). Boxes in the
burned-in clock are dropped.

Truth (AI by-eye labels):
- mask frames: share of the box covered by whale / fish / back / unsure pixels (fw, ff, fb, fu):
  fish if ff >= 0.10 and fw < 0.10 (fish and whale both >= 0.10: ambiguous); whale if fw >= 0.15 and ff < fw / 2;
  ambiguous if fu >= 0.5; back if fb >= 0.3; nothing otherwise.
- frames without masks: fish if the box matches a box of fish_calls.json (IoU >= 0.3, or >= 70 % of the box inside
  the call, or the call >= 70 % inside the box and the box at most 4x the call's area); a box holding two or more
  calls is fish when no whale is in the frame, else ambiguous; on a fish frame an unmatched box under 150 px is
  ambiguous (uncounted spindles). Whale: only on frames with exactly one clear other whale: the anchor is the best
  "whale" box (score >= 0.1, not a fish) whose longest side is 0.55-1.6x the labelled whale size; a box with IoU >= 0.5
  to the anchor or >= 60 % inside it is whale, a box less than 10 % inside it is `other` (back or nothing) when no
  faint whale is counted, anything else is ambiguous. No anchor, two or more whales, or faint whales only: the
  non-fish boxes are ambiguous. Frames with no whale, no faint whale and no fish: every box is `other`.
- `other` = back or nothing; back and nothing are told apart on mask frames only.

Baseline (scan rules, uwid.pipeline defaults): an object is `whale` when one of its boxes has a "whale" row with
score >= 0.25 that passes the whole-frame rule (area <= 0.6 of the frame, or score >= 0.6) and is >= 300 px long,
`small` when such a box is shorter, `fish` when a fish-prompt row scores >= 0.5, else `none`. Rule B adds the fish
veto: the whale box counts only when its whale score beats its best fish score (a vetoed box -> `fish`).

    .venv/bin/python experiments/08_whale_fish_boxes/table.py   -> CACHE/objects.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402
from uwid import boxclf as B  # noqa: E402
from uwid import detect  # noqa: E402

THR, FISH_THR, MIN_PX, WHOLE, NEAR = 0.25, 0.5, 300.0, 0.6, 0.6


def baseline(members: pd.DataFrame, W: int, H: int) -> tuple[str, str]:
    """(rule A, rule B) labels of one object from its member patch boxes."""
    area = (members.x1 - members.x0) * (members.y1 - members.y0) / (W * H)
    px = np.maximum(members.x1 - members.x0, members.y1 - members.y0) * 1920.0 / W
    has_w = members.queries.apply(lambda q: B.WHALE_Q in q)
    ok = has_w & (members.s_whale >= THR) & ((area <= WHOLE) | (members.s_whale >= NEAR))
    fish = members.queries.apply(lambda q: any(x in q for x in B.FISH_QS)) & (members.s_fish >= FISH_THR)

    def lab(okw):
        if (okw & (px >= MIN_PX)).any():
            return "whale"
        if okw.any():
            return "small"
        return None
    a = lab(ok) or ("fish" if fish.any() else "none")
    okb = ok & (members.s_whale > members.s_fish)
    b = lab(okb) or ("fish" if (ok & ~okb).any() or fish.any() else "none")
    return a, b


def mask_truth(cm: np.ndarray, box) -> tuple[str, dict]:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(cm.shape[1], x1), min(cm.shape[0], y1)
    reg = cm[y0:y1, x0:x1]
    n = max(1, reg.size)
    fw, ff, fb, fu = [(reg == c).sum() / n for c in (1, 3, 2, 255)]
    info = dict(fw=fw, ff=ff, fb=fb, fu=fu)
    if ff >= 0.10 and fw >= 0.10:
        return "ambiguous", info
    if ff >= 0.10:
        return "fish", info
    if fw >= 0.15 and ff < fw / 2:
        return "whale", info
    if fu >= 0.5:
        return "ambiguous", info
    if fb >= 0.3:
        return "back", info
    return "nothing", info


def fish_match(box, calls) -> tuple[bool, int]:
    """(matches one fish call, number of calls >= 70 % inside the box)."""
    if not calls:
        return False, 0
    c = np.asarray(calls, float)
    b = np.asarray(box, float)[None]
    iou = D.iou_xyxy(b, c)[0]
    b_in_c = D.inside_frac(b, c)[0]
    c_in_b = D.inside_frac(c, b)[:, 0]
    a_b = (b[0, 2] - b[0, 0]) * (b[0, 3] - b[0, 1]); a_c = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    single = (iou >= 0.3) | (b_in_c >= 0.7) | ((c_in_b >= 0.7) & (a_b <= 4 * a_c))
    return bool(single.any()), int((c_in_b >= 0.7).sum())


def build() -> pd.DataFrame:
    rows = []
    for rec in D.recordings():
        owl = pd.read_parquet(D.CACHE / "owl" / f"{rec.stem}.parquet")
        lab = owl[owl.kind == "label"].drop(columns=["kind"])
        fc = rec.frames_csv().set_index("frame")
        masks = set(rec.mask_frames())
        calls = rec.fish_calls()
        for f in rec.labelled_frames():
            img_p = rec.image_path(f)
            if img_p is None:
                continue
            cm = None
            if f in masks:
                cm = cv2.imread(str(rec.ref / "masks" / f"{f:06d}_class.png"), cv2.IMREAD_UNCHANGED)
                if rec.source == "dev":
                    cm = cm.copy(); cm[cm == 3] = 255     # the dev clip's "fish" are not fish (user)
            if f in fc.index:
                L = fc.loc[f]
                if L.state != "underwater":
                    continue
            else:
                L = None
                if cm is not None and (cm == 6).mean() > 0.5:
                    continue
            H, W = cm.shape if cm is not None else (1080, 1920)
            g = lab[lab.frame == f]
            u_all = B.unique_boxes(g, min_score=0.0) if len(g) else pd.DataFrame()
            if not len(u_all):
                continue
            u_all = u_all[~detect.in_clock(u_all, W, H).to_numpy()].reset_index(drop=True)
            u = u_all[np.fmax(u_all.s_whale, u_all.s_fish) >= B.MIN_SCORE].reset_index(drop=True)
            if not len(u):
                continue
            u = B.group_objects(u)
            fcalls = calls.get(f, [])
            # anchor of the one clear whale on frames without masks
            anchor = None
            n_w = int(L.other_whales) if L is not None and str(L.other_whales) != "" else 0
            n_faint = int(L.faint_whales) if L is not None and str(L.faint_whales) != "" else 0
            fish_lab = (L.fish if L is not None else "none") or "none"
            other_txt = str(L.other) if L is not None else ""
            if cm is None and n_w == 1 and L is not None and str(L.whale_size) != "":
                E = float(L.whale_size) * 1920
                cand = u_all[(u_all.s_whale >= 0.1) & u_all.queries.apply(lambda q: B.WHALE_Q in q)].copy()
                cand["px"] = np.maximum(cand.x1 - cand.x0, cand.y1 - cand.y0) * 1920.0 / W
                cand = cand[(cand.px >= 0.55 * E) & (cand.px <= 1.6 * E)]
                cand = cand[[not fish_match(r[["x0", "y0", "x1", "y1"]].to_numpy(float), fcalls)[0]
                             for _, r in cand.iterrows()]] if len(cand) else cand
                if len(cand):
                    anchor = cand.sort_values("s_whale").iloc[-1][["x0", "y0", "x1", "y1"]].to_numpy(float)
            for o, grp in u.groupby("obj"):
                rep = grp[grp.rep].iloc[0]
                box = rep[["x0", "y0", "x1", "y1"]].to_numpy(float)
                px = max(box[2] - box[0], box[3] - box[1]) * 1920.0 / W
                info = {}
                why = ""
                if cm is not None:
                    truth, info = mask_truth(cm, box)
                    why = "mask"
                else:
                    single, n_in = fish_match(box, fcalls)
                    if single:
                        truth, why = "fish", "fish call"
                    elif n_in >= 2:
                        truth, why = ("fish", "school box") if n_w == 0 else ("ambiguous", "school box with whale")
                    elif fish_lab != "none" and not fcalls:
                        truth, why = "ambiguous", "fish labelled, no boxes"
                    elif fish_lab != "none" and px < 150:
                        truth, why = "ambiguous", "small box on a fish frame"
                    elif n_w == 0:
                        if n_faint > 0:
                            truth, why = "ambiguous", "faint whale only"
                        elif "unsure" in other_txt and px < 150:
                            truth, why = "ambiguous", "unsure specks"
                        else:
                            truth, why = "other", "no whale, no fish"
                    elif n_w >= 2:
                        truth, why = "ambiguous", "2+ whales, no mask"
                    elif anchor is None:
                        truth, why = "ambiguous", "no anchor box"
                    else:
                        iou = D.iou_xyxy(box[None], anchor[None])[0, 0]
                        ins = D.inside_frac(box[None], anchor[None])[0, 0]
                        if iou >= 0.5 or ins >= 0.6:
                            truth, why = "whale", "anchor"
                        elif ins < 0.1 and n_faint == 0:
                            truth, why = "other", "off the anchor"
                        else:
                            truth, why = "ambiguous", "near the anchor" if ins >= 0.1 else "faint whale counted"
                ba, bb = baseline(grp, W, H)
                rows.append(dict(source=rec.source, stem=rec.stem, piece=D.piece(rec.stem), frame=int(f),
                                 obj=int(o), x0=box[0], y0=box[1], x1=box[2], y1=box[3], W=W, H=H, px=px,
                                 band=D.size_band(px), n_boxes=len(grp), truth=truth, why=why, mask=cm is not None,
                                 s_whale=rep.s_whale, s_fish=rep.s_fish, s_fish_prompt=rep.s_fish_prompt,
                                 s_school_of_fish=rep.s_school_of_fish, s_small_fish=rep.s_small_fish,
                                 s_air_bubbles=rep.s_air_bubbles, obj_whale=grp.s_whale.max(),
                                 obj_fish=grp.s_fish.max(), base_a=ba, base_b=bb, **info))
    df = pd.DataFrame(rows)
    df["truth3"] = df.truth.replace({"back": "other", "nothing": "other"})
    return df


if __name__ == "__main__":
    df = build()
    df.to_parquet(D.CACHE / "objects.parquet", index=False)
    print(len(df), "objects")
    print(pd.crosstab([df.source, df.truth], df.band))
    print(df.groupby(["source", "why"]).size())
