"""WP19 bundle checker (docs/EXPORT_FORMAT.md).

Part (i), standard library only (what a web backend without this repo's environment can rely on):
    /usr/bin/python3 -I experiments/12_export/check_bundle.py structure [bundle]
  walks index.json, opens every referenced file, checks every JSON against the documented fields, every polygon /
  box coordinate in 0..1, frame lists against n_frames, CSV headers.

Part (ii), this repo's .venv:
    .venv/bin/python experiments/12_export/check_bundle.py media [bundle]
  every clip: decoded frame count and size against its track JSON; 20 random (span, number, frame) picks (seed
  20261124) of the exported polygon rasterised at 1280 x 720 against the cleaned track mask (tr.mask); and the
  review picture review_sheets/underwater/export_check_02.jpg (first run: _01): 12 random frames drawn ONLY from the
  bundle (clip frame + JSON polygons), seed 20261125, the first 3 from the dev clip.
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import sys

DEFAULT = "/home/sat3737/NOAA_whales/underwater_cache/export/web_v1"
SEED_IOU, SEED_PICTURE = 20261124, 20261125
PICTURE = "export_check_02.jpg"   # first run: export_check_01.jpg (seed 20261125, no dev-clip rule)
MIN_DEV_TILES = 3                 # WP19 final run: at least 3 of the 12 frames from the dev clip

INDEX_KEYS = {"format", "version", "kind", "created", "command", "readme", "coordinates", "deployments"}
DEP_KEYS = {"id", "name", "tag", "date", "preset", "sensor_log", "recordings", "counts", "stretches", "spans"}
REC_KEYS = {"file", "source", "duration_s", "n_frames", "fps", "width", "height", "codec", "codec_tag", "container",
            "recording", "first_frame_clock", "clock_fit", "other_forms", "windows"}
SPAN_KEYS = {"group", "note", "recording", "t0", "t1", "track", "clip", "poster", "n_frames", "numbers", "links_made",
             "first_t", "last_t"}
TRACK_KEYS = {"format", "version", "kind", "group", "note", "deployment", "recording", "span", "tracked", "fps", "n_frames",
              "frames", "clip", "coordinates", "simplify_px", "mask_size", "back_every", "numbers", "small", "back",
              "blocked", "endings", "links", "refused_starts", "purity", "method", "start_source", "blocked_rule",
              "own_back_template"}
NUMBER_KEYS = {"number", "kind", "objects", "first_j", "last_j", "first_t", "last_t", "n_frames", "frames"}
LINK_KEYS = {"made", "why", "number_a", "number_b", "object_a", "object_b", "t_end", "gap_s", "breath", "limit_s",
             "others", "others_numbers", "lone", "detector_before", "detector_during", "detector_after", "size_ratio",
             "appearance_distance", "checks_ok"}
STRETCH_KEYS = {"format", "version", "kind", "deployment", "preset", "scan_created", "recordings"}
WINDOW_COLS = ["time_s", "end_s", "clock", "depth_m", "tag_state", "water_state", "water_source", "water_picture",
               "p_air_picture", "scanned", "whale", "whale_score", "n_whale", "whale_px", "small", "small_score",
               "fish", "fish_score", "n_fish", "active", "tier", "kept", "why", "seafloor_share", "sand", "dark",
               "silhouette", "bubbles", "hq_score", "whale_box_x0", "whale_box_y0", "whale_box_x1", "whale_box_y1"]


class Problems(list):
    def need(self, cond, msg):
        if not cond:
            self.append(msg)


def _keys(p: Problems, obj: dict, want: set, where: str):
    miss = want - set(obj)
    p.need(not miss, f"{where}: missing {sorted(miss)}")


def _coords_ok(poly) -> bool:
    return len(poly) >= 3 and all(len(q) == 2 and 0 <= q[0] <= 1 and 0 <= q[1] <= 1 for q in poly)


def check_track(p: Problems, root: str, rel: str, span: dict) -> dict:
    doc = json.load(open(os.path.join(root, rel)))
    _keys(p, doc, TRACK_KEYS, rel)
    n = doc["n_frames"]
    p.need(len(doc["frames"]["index"]) == n and len(doc["frames"]["t"]) == n, f"{rel}: frame lists != n_frames")
    p.need(all(a < b for a, b in zip(doc["frames"]["t"], doc["frames"]["t"][1:])), f"{rel}: times not increasing")
    p.need(span["n_frames"] == n, f"{rel}: index n_frames differs")
    if doc["clip"]:
        p.need(doc["clip"]["n_frames"] == n, f"{rel}: clip n_frames differs")
        for k in ("path", "poster"):
            p.need(os.path.isfile(os.path.join(root, doc["clip"][k])), f"{rel}: missing {doc['clip'][k]}")
    stats = dict(polys=0, points=0, frames=0)
    for nb in doc["numbers"] + doc["small"]:
        _keys(p, nb, NUMBER_KEYS, f"{rel} number {nb.get('number')}")
        js = [f["j"] for f in nb["frames"]]
        p.need(js == sorted(set(js)) and all(0 <= j < n for j in js), f"{rel} number {nb['number']}: frame j")
        p.need(nb["n_frames"] == len(js) and nb["first_j"] == js[0] and nb["last_j"] == js[-1],
               f"{rel} number {nb['number']}: first / last / n_frames")
        for f in nb["frames"]:
            stats["frames"] += 1
            b = f["box"]
            p.need(len(b) == 4 and all(0 <= x <= 1 for x in b) and b[0] < b[2] and b[1] < b[3],
                   f"{rel} number {nb['number']} j {f['j']}: box")
            for poly in f["polygons"]:
                stats["polys"] += 1
                stats["points"] += len(poly)
                p.need(_coords_ok(poly), f"{rel} number {nb['number']} j {f['j']}: polygon out of 0..1")
    for f in doc["back"]:
        p.need(0 <= f["j"] < n and f["j"] % doc["back_every"] == 0, f"{rel}: back j")
        for poly in f["polygons"]:
            p.need(_coords_ok(poly), f"{rel}: back polygon out of 0..1")
    for b in doc["blocked"]:
        p.need(0 <= b["j0"] < b["j1"] <= n, f"{rel}: blocked range")
    for e in doc["endings"]:
        p.need(0 <= e["j"] < n and e.get("reason"), f"{rel}: ending")
    for L in doc["links"]:
        _keys(p, L, LINK_KEYS, f"{rel} link")
    for r in doc["refused_starts"]:
        p.need(0 <= r["j"] < n and (r["box"] is None or all(0 <= x <= 1.0001 for x in r["box"])), f"{rel}: refused")
    return stats


def structure(root: str) -> int:
    p = Problems()
    idx = json.load(open(os.path.join(root, "index.json")))
    _keys(p, idx, INDEX_KEYS, "index.json")
    p.need(os.path.isfile(os.path.join(root, idx["readme"])), "README.md missing")
    tot = dict(deployments=0, recordings=0, windows=0, spans=0, tracks=0, clips=0, polys=0, points=0, frames=0)
    for d in idx["deployments"]:
        tot["deployments"] += 1
        _keys(p, d, DEP_KEYS, f"deployment {d.get('name')}")
        st = json.load(open(os.path.join(root, d["stretches"])))
        _keys(p, st, STRETCH_KEYS, d["stretches"])
        for r in d["recordings"]:
            tot["recordings"] += 1
            _keys(p, r, REC_KEYS, f"{d['id']} recording {r.get('file')}")
            p.need(r["recording"] in st["recordings"], f"{d['id']}: {r['recording']} not in stretches.json")
            if r["windows"]:
                with open(os.path.join(root, r["windows"]), newline="") as fh:
                    rd = csv.reader(fh)
                    head = next(rd)
                    rows = sum(1 for _ in rd)
                p.need(head == WINDOW_COLS, f"{r['windows']}: columns differ")
                p.need(rows > 0, f"{r['windows']}: no rows")
                tot["windows"] += 1
        p.need(d["counts"]["tracked_spans"]["hq"] + d["counts"]["tracked_spans"]["extra"] == len(d["spans"]),
               f"{d['id']}: span count")
        for s in d["spans"]:
            tot["spans"] += 1
            _keys(p, s, SPAN_KEYS, f"{d['id']} span")
            st_ = check_track(p, root, s["track"], s)
            tot["tracks"] += 1
            for k in ("polys", "points", "frames"):
                tot[k] += st_[k]
            if s["clip"]:
                tot["clips"] += os.path.isfile(os.path.join(root, s["clip"]))
    print(json.dumps(tot))
    print(f"{len(p)} problems")
    for m in p[:30]:
        print(" ", m)
    return 1 if p else 0


# ------------------------------------------------------------------------------------------------ part (ii)

def _spans(root):
    idx = json.load(open(os.path.join(root, "index.json")))
    return [(d, s) for d in idx["deployments"] for s in d["spans"]]


def media(root: str) -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    import numpy as np
    from uwid import export as E, trackstage as T
    bad, n_clips = [], 0
    for d, s in _spans(root):
        doc = json.load(open(os.path.join(root, s["track"])))
        if not s["clip"]:
            continue
        n_clips += 1
        k, size = E.count_frames(os.path.join(root, s["clip"]))
        if k != doc["n_frames"] or tuple(size) != (doc["clip"]["width"], doc["clip"]["height"]):
            bad.append((s["clip"], k, doc["n_frames"], size))
    print(f"clips: {n_clips}, frame count or size differing: {len(bad)}", bad[:5])
    # IoU of exported polygons against the cleaned masks
    rng = random.Random(SEED_IOU)
    picks, ious = [], []
    spans = _spans(root)
    while len(picks) < 20:
        d, s = rng.choice(spans)
        doc = json.load(open(os.path.join(root, s["track"])))
        if not doc["numbers"]:
            continue
        nb = rng.choice(doc["numbers"])
        fr = rng.choice(nb["frames"])
        picks.append((d, s, doc, nb, fr))
    for d, s, doc, nb, fr in picks:
        folder = T.span_dir(doc["deployment"], doc["recording"] + ".mov", doc["span"]["t0"], doc["span"]["t1"])
        if doc["group"] == "extra":
            folder = folder.with_name(folder.name + "_ownback")
        if not (folder / "meta.json").exists():
            folder = T.span_dir(doc["deployment"], doc["recording"], doc["span"]["t0"], doc["span"]["t1"])
        tr = T.load_online(folder)
        T.link(tr, doc["deployment"], tr.meta.get("start_source", "scan"))
        w, h = tr.meta["store_size"]
        m = np.zeros((h, w), bool)
        for k in nb["objects"]:
            if tr.visible(k, fr["j"]):
                m |= tr.mask(k, fr["j"])
        got = E.rasterize(fr["polygons"], w, h)
        iou = float((got & m).sum() / max((got | m).sum(), 1))
        ious.append(iou)
        print(f"  {d['id']} {doc['recording'][-5:]} {doc['span']['t0']:.0f}-{doc['span']['t1']:.0f} "
              f"number {nb['number']} j {fr['j']}: IoU {iou:.4f} ({int(m.sum())} px)")
    print(f"IoU polygons vs masks over 20 picks: min {min(ious):.4f}, median {statistics.median(ious):.4f}")
    picture(root)
    return 1 if bad else 0


def picture(root: str, n: int = 12) -> str:
    import av
    import cv2
    import numpy as np
    rng = random.Random(SEED_PICTURE)
    spans = [(d, s) for d, s in _spans(root) if s["clip"] and s["numbers"]]
    tiles = []
    colours = [(255, 80, 80), (80, 220, 255), (255, 220, 60), (120, 255, 120), (255, 120, 255), (255, 160, 60)]
    dev = [(d, s) for d, s in spans if d["id"].startswith("2025-12-20")]
    while len(tiles) < n:
        # the first MIN_DEV_TILES tiles come from the dev clip (when it has tracked spans), the rest from all spans
        d, s = rng.choice(dev if dev and len(tiles) < MIN_DEV_TILES else spans)
        doc = json.load(open(os.path.join(root, s["track"])))
        js = sorted({f["j"] for nb in doc["numbers"] for f in nb["frames"]})
        j = rng.choice(js)
        with av.open(os.path.join(root, s["clip"])) as c:
            img = None
            for k, f in enumerate(c.decode(c.streams.video[0])):
                if k == j:
                    img = f.to_ndarray(format="rgb24")
                    break
        if img is None:
            continue
        H, W = img.shape[:2]
        img = img.copy()
        for f in [f for f in doc["back"] if f["j"] == j - j % doc["back_every"]]:
            for poly in f["polygons"]:
                cv2.polylines(img, [np.round(np.array(poly) * [W, H]).astype(np.int32)], True, (200, 200, 200), 1)
        nums = []
        for nb in doc["numbers"]:
            for f in nb["frames"]:
                if f["j"] == j:
                    col = colours[(nb["number"] - 1) % len(colours)]
                    for poly in f["polygons"]:
                        cv2.polylines(img, [np.round(np.array(poly) * [W, H]).astype(np.int32)], True, col, 3)
                    x0, y0 = int(f["box"][0] * W), int(f["box"][1] * H)
                    cv2.putText(img, str(nb["number"]), (x0 + 4, max(24, y0 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                col, 2)
                    nums.append(str(nb["number"]))
        lab = (f"{d['id']} {doc['recording'][-5:]} t {doc['frames']['t'][j]:.2f}s (clip frame {j}) "
               f"number {','.join(nums)}")
        t = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 336), (640, 360), (0, 0, 0), -1)
        cv2.putText(t, lab, (6, 353), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        tiles.append(t)
    sheet = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, n, 4)])
    out = "/home/sat3737/NOAA_whales/review_sheets/underwater/" + PICTURE
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("picture", out)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "structure"
    root = sys.argv[2] if len(sys.argv) > 2 else DEFAULT
    sys.exit(structure(root) if cmd == "structure" else media(root))
