"""WP15 phase 2 report (triposg env, CPU). Stages, run in this order:

    envs/triposg/bin/python experiments/10_mesh3d/trial_report.py render    # per-mesh renders, IoU, eye sheets
    envs/triposg/bin/python experiments/10_mesh3d/trial_report.py sheets    # user sheets mesh3d_trial_*.jpg
    (AI by-eye calls written by hand to underwater_cache/mesh3d/trial/eye_calls.csv BEFORE the next stages)
    envs/triposg/bin/python experiments/10_mesh3d/trial_report.py pairs     # chamfer within / across numbers
    envs/triposg/bin/python experiments/10_mesh3d/trial_report.py summary   # tables (needs eye_calls.csv)

All under underwater_cache/mesh3d/trial/; user sheets in review_sheets/underwater/; small numbers in
experiments/10_mesh3d/results/trial_*.json.
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import mesh_tools as mt  # noqa: E402

ARM = os.environ.get("ARM", "A")   # A = whole-outline picks (trial/), B = whales cut by the frame (trial_partial/)
T = Path("/home/sat3737/NOAA_whales/underwater_cache/mesh3d/" + ("trial" if ARM == "A" else "trial_partial"))
PICKS = "picks.csv" if ARM == "A" else "picks_partial.csv"
SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
RESULTS = Path(__file__).parent / "results"
RANDOM_SEED = 20261124          # sheet (ii): random 24
PAIR_SEED = 7                    # sample of different-number pairs
DEP = {"2025-2-3 Mini-mother 11-96": "2025-02-03", "TestSubset 2025-1-2 CATS -85": "CC-09-85 (2025-01-02)"}
TILE = 270


def picks() -> pd.DataFrame:
    p = pd.read_csv(T / PICKS)
    p["has_mesh"] = [(T / "meshes" / f"{i}.ply").exists() for i in p.id]
    return p


def tile_label(r) -> str:
    rec = r.video.split("-")[-1].split(".")[0]
    return f"{DEP[r.deployment]}  {rec}  t={r.t:.1f}s  number {r.number}"


def text(img, s, org, sc=0.5):
    (w, h), b = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
    cv2.rectangle(img, (org[0] - 3, org[1] - h - 4), (org[0] + w + 3, org[1] + b + 2), (0, 0, 0), -1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 1, cv2.LINE_AA)


def render_one(args):
    r = args
    pts, nrm, share, _ = mt.load_points(T / "meshes" / f"{r['id']}.ply", n=300_000)
    alpha = cv2.imread(str(T / "inputs" / f"{r['id']}.png"), cv2.IMREAD_UNCHANGED)[..., 3] > 0
    R_cam, iou = mt.camera_rotation(pts, alpha)
    V = mt.pca_frame(pts)
    a1, a2, a3 = V
    views = [R_cam, np.stack([a1, a3, np.cross(a1, a3)]), np.stack([a1, a2, np.cross(a1, a2)])]
    tiles = [mt.render(pts, nrm, R, TILE) for R in views]
    ov = cv2.cvtColor(cv2.imread(str(T / "inputs" / f"{r['id']}_overlay.jpg")), cv2.COLOR_BGR2RGB)
    ov = cv2.resize(ov, (480, TILE), interpolation=cv2.INTER_AREA)
    strip = np.hstack([ov] + tiles)
    cv2.imwrite(str(T / "renders" / f"{r['id']}.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    np.save(T / "renders" / f"{r['id']}_pts.npy", pts[np.random.default_rng(0).choice(len(pts), 6000, False)]
            .astype(np.float32))
    info = json.loads((T / "meshes" / f"{r['id']}.json").read_text())
    return dict(id=r["id"], silhouette_iou=round(iou, 3), cam_is_plus_z=bool(np.allclose(R_cam, np.eye(3))),
                largest_piece_share=round(share, 3), seconds=info["seconds"],
                torch_peak_gb=info["torch_max_allocated_gb"], smi_peak_mib=info["smi_peak_mib"],
                faces=info.get("faces"), saved_faces=info.get("saved_faces"))


def stage_render():
    (T / "renders").mkdir(exist_ok=True)
    P = picks()
    P = P[P.has_mesh]
    with get_context("spawn").Pool(8) as pool:
        rows = pool.map(render_one, [r._asdict() for r in P.itertuples(index=False)], chunksize=2)
    M = pd.DataFrame(rows)
    M.to_csv(T / "mesh_metrics.csv", index=False)
    # eye sheets: 16 strips per sheet, labelled with the pick's index only (for the AI by-eye calls)
    P = P.reset_index(drop=True)
    (T / "eye").mkdir(exist_ok=True)
    for s in range(0, len(P), 16):
        tiles = []
        for k in range(s, min(s + 16, len(P))):
            im = cv2.imread(str(T / "renders" / f"{P.id[k]}.jpg"))
            text(im, f"{k:03d}", (6, 24), 0.8)
            tiles.append(im)
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
        cv2.imwrite(str(T / "eye" / f"eye_{s // 16 + 1:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    P[["id"]].to_csv(T / "eye" / "index.csv")
    print("render: done", len(M), "meshes; IoU median", M.silhouette_iou.median())


def labelled(r):
    im = cv2.imread(str(T / "renders" / f"{r.id}.jpg"))
    text(im, tile_label(r), (6, 20), 0.5)
    return im


def write_sheet(rows, path, cols=2, title=""):
    tiles = [labelled(r) for r in rows]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
    head = np.full((30, grid.shape[1], 3), 30, np.uint8)
    text(head, title, (8, 21), 0.55)
    cv2.imwrite(str(path), np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("sheet", path)


def stage_sheets():
    P = picks()
    M = pd.read_csv(T / "mesh_metrics.csv")
    P = P.merge(M, on="id", how="inner")      # keeps the pick list's order
    head = "each tile: frame + tracker outline | mesh (TripoSG, one frame) seen from the input view | side | top"
    if ARM == "B":
        for s in range(0, len(P), 12):
            write_sheet(list(P.iloc[s:s + 12].itertuples()), SHEETS / f"mesh3d_trial_partial_{s // 12 + 1:02d}.jpg",
                        title="whales cut by the frame, pick-list order. " + head)
        return
    best = P.sort_values("silhouette_iou", ascending=False).head(12)
    write_sheet(list(best.itertuples()), SHEETS / "mesh3d_trial_a_best_iou_01.jpg",
                title="12 best by outline agreement (IoU). " + head)
    rnd = P.sample(n=min(24, len(P)), random_state=RANDOM_SEED)
    for s in range(0, len(rnd), 12):
        write_sheet(list(rnd.iloc[s:s + 12].itertuples()), SHEETS / f"mesh3d_trial_b_random_{s // 12 + 1:02d}.jpg",
                    title=f"random {len(rnd)} (seed {RANDOM_SEED}). " + head)
    top = P.groupby("animal").size().sort_values(ascending=False, kind="stable").head(6).index
    for k, a in enumerate(top, 1):
        g = P[P.animal == a].sort_values("t")
        write_sheet(list(g.itertuples()), SHEETS / f"mesh3d_trial_c_number_{k:02d}.jpg",
                    title="all meshes of one number. " + head)


def pair_dist(args):
    a, b = args
    pa = np.load(T / "renders" / f"{a}_pts.npy").astype(np.float64)
    pb = np.load(T / "renders" / f"{b}_pts.npy").astype(np.float64)
    return mt.aligned_distance(pa, pb, n=3000)


def stage_pairs():
    P = picks()
    P = P[P.has_mesh]
    same, other = [], []
    for dep, g in P.groupby("deployment"):
        ids = list(g.id)
        an = dict(zip(g.id, g.animal))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                (same if an[ids[i]] == an[ids[j]] else other).append((dep, ids[i], ids[j]))
    rng = np.random.default_rng(PAIR_SEED)
    other_s = []
    for dep in P.deployment.unique():
        o = [x for x in other if x[0] == dep]
        n_same = sum(x[0] == dep for x in same)
        k = min(len(o), max(n_same, 50))
        other_s += [o[i] for i in rng.choice(len(o), k, replace=False)]
    jobs = same + other_s
    with get_context("spawn").Pool(8) as pool:
        d = pool.map(pair_dist, [(a, b) for _, a, b in jobs], chunksize=4)
    rows = [dict(deployment=dep, a=a, b=b, kind="same number" if k < len(same) else "different number",
                 chamfer_rms=round(x, 4)) for k, ((dep, a, b), x) in enumerate(zip(jobs, d))]
    pd.DataFrame(rows).to_csv(T / "pairs.csv", index=False)
    print("pairs: done", len(same), "same,", len(other_s), "different (sampled from", len(other), ")")


def stage_summary():
    P = picks()
    M = pd.read_csv(T / "mesh_metrics.csv")
    E = pd.read_csv(T / "eye_calls.csv")
    P = P.merge(M, on="id").merge(E[["id", "call"]], on="id", how="left")
    P["dep"] = P.deployment.map(DEP)
    P["elong_class"] = np.where(P.elong >= 2.5, "elongated (>= 2.5)", "compact (< 2.5)")
    P["size_class"] = np.where(P.area_share >= 0.06, "large (>= 6 %)", "medium (3-6 %)")
    ang = P.orient.to_numpy()
    P["angle_class"] = np.select([(ang < 30) | (ang >= 150), (ang >= 60) & (ang < 120)],
                                 ["near horizontal", "near vertical"], "diagonal")
    out = {}
    for col in ["dep", "elong_class", "size_class", "angle_class"]:
        out[col] = pd.crosstab(P[col], P.call, margins=True).to_dict()
    out["calls"] = P.call.value_counts().to_dict()
    out["iou"] = P.silhouette_iou.describe().round(3).to_dict()
    out["iou_by_call"] = P.groupby("call").silhouette_iou.describe().round(3).to_dict()
    out["iou_by_dep"] = P.groupby("dep").silhouette_iou.describe().round(3).to_dict()
    out["iou_below_0.7"] = int((P.silhouette_iou < 0.7).sum())
    out["seconds"] = P.seconds.describe().round(2).to_dict()
    out["torch_peak_gb"] = P.torch_peak_gb.describe().round(3).to_dict()
    out["smi_peak_mib"] = P.smi_peak_mib.describe().round(0).to_dict()
    if (T / "pairs.csv").exists():
        D = pd.read_csv(T / "pairs.csv")
        D["dep"] = D.deployment.map(DEP)
        out["pairs"] = D.groupby(["dep", "kind"]).chamfer_rms.describe().round(4).reset_index().to_dict("records")
        for dep, g in D.groupby("dep"):
            s, o = g[g.kind == "same number"].chamfer_rms.to_numpy(), g[g.kind != "same number"].chamfer_rms.to_numpy()
            if len(s) and len(o):
                auc = float(((o[None, :] > s[:, None]).sum() + 0.5 * (o[None, :] == s[:, None]).sum()) / (len(s) * len(o)))
                out.setdefault("pairs_auc", {})[dep] = dict(auc=round(auc, 3), same=len(s), different=len(o),
                                                           different_below_same_median=int((o < np.median(s)).sum()))
        # whale-like only
        wl = set(P[P.call == "whale"].id)
        g = D[D.a.isin(wl) & D.b.isin(wl)]
        out["pairs_whale_only"] = g.groupby(["dep", "kind"]).chamfer_rms.describe().round(4).reset_index().to_dict("records")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"trial_summary_{ARM}.json").write_text(json.dumps(out, indent=1, default=str))
    P.to_csv(T / "trial_table.csv", index=False)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    {"render": stage_render, "sheets": stage_sheets, "pairs": stage_pairs, "summary": stage_summary}[sys.argv[1]]()
