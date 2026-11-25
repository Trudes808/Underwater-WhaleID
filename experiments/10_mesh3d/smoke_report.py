"""WP15 smoke test, last step (triposg env, CPU): review sheets, silhouette match, consistency and proportions.

    envs/triposg/bin/python experiments/10_mesh3d/smoke_report.py PICKS.csv INPUTS MESHES

PICKS.csv: id, animal (key: same key = same tracked number), deployment, kind (AI by-eye viewpoint word), ...
Sheets -> review_sheets/underwater/mesh3d_smoke_<id>.jpg; numbers -> experiments/10_mesh3d/results/smoke_metrics.json
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import mesh_tools as mt  # noqa: E402

SHEETS = Path("/home/sat3737/NOAA_whales/review_sheets/underwater")
RESULTS = Path(__file__).parent / "results"
MODEL = "TripoSG (VAST-AI, MIT licence), fp16"


def views(pts, R_cam):
    V = mt.pca_frame(pts)
    a1, a2, a3 = V
    R_side = np.stack([a1, a3, np.cross(a1, a3)])
    R_top = np.stack([a1, a2, np.cross(a1, a2)])
    R_tq = mt.rot("x", 25) @ mt.rot("y", -40) @ R_side
    return [("as the camera saw it", R_cam), ("side (mesh long axis x 3rd axis)", R_side),
            ("top (long axis x 2nd axis)", R_top), ("three-quarter", R_tq)]


def label(img, text, org=(8, 22), s=0.55, c=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)


def main(picks, inputs, meshes):
    inputs, meshes = Path(inputs), Path(meshes)
    df = pd.read_csv(picks)
    RESULTS.mkdir(exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)
    rows, clouds = [], {}
    for r in df.itertuples():
        mp = meshes / f"{r.id}.ply"
        if not mp.exists():
            print("no mesh", r.id)
            continue
        info = json.loads((meshes / f"{r.id}.json").read_text())
        pts, nrm, share, _ = mt.load_points(mp)
        alpha = cv2.imread(str(inputs / f"{r.id}.png"), cv2.IMREAD_UNCHANGED)[..., 3] > 0
        R_cam, iou_cam = mt.camera_rotation(pts, alpha)
        prop = mt.proportions(pts)
        clouds[r.id] = pts
        rows.append(dict(id=r.id, animal=r.animal, deployment=r.deployment, kind=r.kind,
                         seconds=info["seconds"], torch_peak_gb=info["torch_max_allocated_gb"],
                         torch_reserved_gb=info["torch_max_reserved_gb"], smi_peak_mib=info["smi_peak_mib"],
                         faces=info.get("faces"), largest_piece_share=round(share, 3),
                         silhouette_iou=round(iou_cam, 3),
                         **{k: round(v, 3) for k, v in prop.items()}))
        # sheet: overlay left (height 600), 2 x 2 views right (300 px each)
        ov = cv2.imread(str(inputs / f"{r.id}_overlay.jpg"))
        ov = cv2.resize(ov, (int(ov.shape[1] * 600 / ov.shape[0]), 600), interpolation=cv2.INTER_AREA)
        label(ov, f"{r.id}: {r.video} t={r.t:.2f}s number {r.number} ({r.kind}, AI by-eye)")
        label(ov, "input = this frame + the tracker's outline (red)", (8, 46))
        tiles = []
        for name, R in views(pts, R_cam):
            t = mt.render(pts, nrm, R, 300)
            label(t, name, (6, 18), 0.45)
            tiles.append(t)
        label(tiles[0], f"silhouette IoU {iou_cam:.2f}", (6, 290), 0.45, (180, 255, 180))
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        sheet = np.hstack([ov, grid])
        foot = np.zeros((34, sheet.shape[1], 3), np.uint8)
        label(foot, f"mesh by {MODEL} from ONE frame; no ground truth; back side invented by the model. "
                    f"{info['seconds']:.0f} s, torch peak {info['torch_max_allocated_gb']:.1f} GB", (8, 22), 0.5)
        sheet = np.vstack([sheet, foot])
        out = SHEETS / f"mesh3d_smoke_{r.id}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("sheet", out, flush=True)
    # consistency: all pairs
    pairs = []
    ids = [x["id"] for x in rows]
    meta = {x["id"]: x for x in rows}
    for a, b in itertools.combinations(ids, 2):
        ka, kb = meta[a], meta[b]
        if ka["animal"] == kb["animal"]:
            kind = "same number"
        elif ka["deployment"] != kb["deployment"]:
            kind = "different animal (other deployment)"
        else:
            kind = "other number, same deployment"
        d = mt.aligned_distance(clouds[a], clouds[b])
        pairs.append(dict(a=a, b=b, kind=kind, chamfer_rms=round(d, 4)))
        print(a, b, kind, round(d, 4), flush=True)
    res = dict(model=MODEL, per_mesh=rows, pairs=pairs)
    (RESULTS / "smoke_metrics.json").write_text(json.dumps(res, indent=1))
    P = pd.DataFrame(pairs)
    if len(P):
        print(P.groupby("kind").chamfer_rms.describe().to_string())
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
