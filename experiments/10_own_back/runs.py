"""CPU: every stored online tracker run (tracks/<deployment>/<stem>/*_online, *_encounter) against the own-back
template. Per tracked object: frames seen, duration, median share of its outline inside the SAM back object (what
the tracker's `on_back` role uses today, >= 0.6 -> refused) and inside the proposed own back (SAM back object inside
the template). Per run: the SAM back object's area and the part of it outside the template (risk). Also the
template's drift: templates from each 300 s block of a recording against the whole-recording template.

    .venv/bin/python experiments/10_own_back/runs.py      -> ownback/runs_tracks.parquet, runs_back.parquet, drift.csv
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
import ownback as O  # noqa: E402

TAU, DIL = 0.5, 1


def templates(tau=TAU, dil=DIL):
    out = {}
    for f in sorted((C.OUT / "maps").glob("*.npz")):
        z = np.load(f)
        idx = pd.read_csv(C.OUT / "frames" / "index.csv")
        tf = set(idx[(idx.stem == f.stem) & (idx.kind == "template")].frame)
        sel = np.array([int(x) in tf for x in z["frames"]])
        if sel.sum() >= 10:
            out[f.stem] = (O.back_template(z["labels"][sel], tau, dil), z["frames"][sel], z["labels"][sel])
    return out


def drift(T):
    rows = []
    for stem, (tmpl, frames, labels) in T.items():
        whole = O.back_template(labels, TAU, 0)
        blocks = frames // (300 * 30)
        for b in np.unique(blocks):
            s = blocks == b
            if s.sum() < 5:
                continue
            tb = O.back_template(labels[s], TAU, 0)
            inter, uni = (tb & whole).sum(), (tb | whole).sum()
            rows.append(dict(stem=stem, block_start_s=int(b) * 300, frames=int(s.sum()), iou_with_whole=float(inter / max(uni, 1)),
                             block_area=float(tb.mean()), whole_area=float(whole.mean()),
                             block_outside_whole_dil=float((tb & ~O.back_template(labels, TAU, 1)).sum() / max(tb.sum(), 1))))
    return pd.DataFrame(rows)


def main():
    T = templates()
    d = drift(T)
    d.to_csv(C.OUT / "drift.csv", index=False)
    trows, brows = [], []
    for run in sorted(glob.glob(str(C.TRACKS / "*" / "*" / "*_online")) + glob.glob(str(C.TRACKS / "*" / "*" / "*_encounter"))):
        run = Path(run)
        if not (run / "masks.npz").exists() or not (run / "meta.json").exists():
            continue
        stem = run.parent.name
        if stem not in T:
            continue
        meta = json.loads((run / "meta.json").read_text())
        z = np.load(run / "masks.npz")
        if "obj_0" not in z.files:
            continue
        b0 = z["obj_0"]
        h = b0.shape[1]; w = int(meta.get("store_size", [C.W])[0]) if isinstance(meta.get("store_size"), list) else C.W
        tpx = O.upsample(T[stem][0], (h, w))
        n = len(b0)
        step = max(1, n // 60)
        for i in range(0, n, step):
            m = C.unpack(b0[i], w)
            if m.sum():
                brows.append(dict(run=run.name, stem=stem, i=i, back_area=float(m.mean()),
                                  back_outside_template=float((m & ~tpx).sum() / m.sum())))
        for k in z.files:
            if k == "obj_0" or not k.startswith("obj_"):
                continue
            p = z[k]
            seen = np.nonzero(p.reshape(len(p), -1).any(1))[0]
            if len(seen) < 3:
                continue
            sh_sam, sh_own, sh_t = [], [], []
            for i in seen[:: max(1, len(seen) // 40)]:
                m = C.unpack(p[i], w); b = C.unpack(b0[i], w)
                sh_sam.append(O.share_inside(m, b)); sh_own.append(O.share_inside(m, O.own_back(b, tpx)))
                sh_t.append(O.share_inside(m, tpx))
            trows.append(dict(run=run.name, stem=stem, obj=k, deployment=run.parent.parent.name,
                              t0=float(meta["t0"]), fps=float(meta.get("fps", 30)), first=int(seen[0]), last=int(seen[-1]),
                              n_seen=len(seen), dur_s=len(seen) / float(meta.get("fps", 30)),
                              in_sam_back=float(np.median(sh_sam)), in_own_back=float(np.median(sh_own)),
                              in_template=float(np.median(sh_t))))
    tr = pd.DataFrame(trows); br = pd.DataFrame(brows)
    tr.to_parquet(C.OUT / "runs_tracks.parquet", index=False); br.to_parquet(C.OUT / "runs_back.parquet", index=False)
    print(len(tr), "tracked objects in", tr.run.nunique() if len(tr) else 0, "runs")
    flip = tr[(tr.in_sam_back >= 0.6) & (tr.in_own_back < 0.6)]
    print("today on_back:", int((tr.in_sam_back >= 0.6).sum()), " would flip to free:", len(flip), " of them >= 3 s:", int((flip.dur_s >= 3).sum()))
    print(d.groupby("stem").iou_with_whole.min().round(2).to_dict())


if __name__ == "__main__":
    main()
