"""WP17: the own-back template behind the tracking flag `own_back_template` (off by default), replayed.

Steps (GPU steps under the lock with `timeout 900`, each stops itself after ~12 min and is resumable):
    blocks     (GPU) the tool's block templates (uwid.ownback.block_template, cached by the tool under
               tracks/<deployment>/<recording>/ownback/) for every block of WP13's 135 spans and of the stretches
    offline    (CPU) (a): for every start box of WP13's spans, the back test with the stored back object (today's rule)
               and with the back object inside the template; the clean-up with and without the template
    track      (GPU) (b): run_span with the flag on, suffix `_ownback`, the stretches shortest first, until the lock
               budget of the step is used; then (e): one 1080p stretch from a frame store, suffix `_ownback_store`
    track_off  (GPU) (c): flag off (suffix `_ownback_off`) only for stretches where a start box passes with the flag on
               but not with it off and no stored run of the same span exists
    report     (CPU) (c)-(e) tables -> results/wp17.json
    sheets     the user's sheets

Stretches: the calf stretch (00005 607-627 s, the span of the stored run) and the 17 WP14 candidates capped at 30 s
around their centre; the candidate 00005 596-667 s holds the calf stretch and is replaced by it.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 nice -n 10 .venv/bin/python experiments/10_own_back/wp17.py blocks
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402
from uwid import ownback as O  # noqa: E402
from uwid import paths, tracks  # noqa: E402
from uwid import trackstage as T  # noqa: E402

OUT = C.OUT / "wp17"
BUDGET_S = 12 * 60
TRACK_BUDGET_S = 40 * 60      # lock-held seconds for all tracking of (b), (c), (e) together
SPANS = C.CACHE / "tracks" / "wp13_spans.csv"
DEP = {"d1": "2025-2-3 Mini-mother 11-96", "d2": "TestSubset 2025-1-2 CATS -85"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def stretches() -> pd.DataFrame:
    c = pd.read_csv(C.OUT / "ownback_candidates.csv")
    rows = [dict(kind="calf", src="d1", piece="00005", t0=607.0, t1=627.0, note="calf stretch (user-confirmed calf)")]
    for r in c.itertuples():
        if r.src == "d1" and int(r.piece) == 5 and r.t0 <= 604 and r.t1 >= 629:
            continue                                            # holds the calf stretch: replaced by it
        mid, d = (r.t0 + r.t1) / 2, min(r.t1 - r.t0, 30.0)
        rows.append(dict(kind="candidate", src=r.src, piece=f"{int(r.piece):05d}", t0=round(mid - d / 2, 2),
                         t1=round(mid + d / 2, 2), note=""))
    s = pd.DataFrame(rows)
    s["unclear"] = [(p == "00011" and abs(t - 1053) < 2) or (p == "00012" and abs(t - 1553) < 2) for p, t in zip(s.piece, s.t0)]
    s["dur"] = s.t1 - s.t0
    s["deployment"] = s.src.map(DEP)
    s["video"] = [str(C.VID / DEP[a] / "Video" / f"{stem}.mov") for a, stem in zip(s.src, [stem_of(a, p) for a, p in zip(s.src, s.piece)])]
    order = s.kind.map({"calf": 0, "candidate": 1}) * 10 - s.unclear.astype(int) * 5
    return s.assign(order=order).sort_values(["order", "dur"]).reset_index(drop=True)


def stem_of(src, piece):
    import glob
    return Path(glob.glob(str(C.VID / DEP[src] / "Video" / f"*-{piece}.mov"))[0]).stem


def ob_cache(dep, stem):
    return paths.cache_dir("tracks") / dep / stem / "ownback"


# ------------------------------------------------------------------------------------------------ blocks (GPU)

def step_blocks():
    t0 = time.time()
    s = pd.read_csv(SPANS)
    need = {(r.deployment, Path(r.recording).stem, float(r.t0)) for r in s.itertuples()}
    st = stretches()
    need |= {(r.deployment, Path(r.video).stem, float(r.t0)) for r in st.itertuples()}
    todo = {}
    for dep, stem, t in need:
        todo.setdefault((dep, stem, O.block_of(t)), t)
    log(f"blocks: {len(todo)} blocks")
    model = clf = None
    done = 0
    for (dep, stem, b), t in sorted(todo.items()):
        cache = ob_cache(dep, stem) / f"block_{b:04d}.npz"
        if cache.exists():
            done += 1
            continue
        if time.time() - t0 > BUDGET_S:
            log("blocks: out of time, run again"); return
        if model is None:
            from uwid import patchseg as ps
            model = ps.load_backbone(); clf = ps.load_classifier(ps.CACHE / "clf_w1280_h0_all.pt")[0]
        w = T.windows_table(dep, f"{stem}.mov")
        tm, info = O.block_template(str(C.VID / dep / "Video" / f"{stem}.mov"), w, t, ob_cache(dep, stem), model, clf)
        log(f"block {stem[-5:]} {b}: {info}")
        done += 1
    (OUT / "blocks.DONE").write_text(time.strftime("%H:%M:%S"))
    log(f"blocks done ({done})")


def template_px(dep, stem, t):
    w = T.windows_table(dep, f"{stem}.mov")
    tm, info = O.block_template(None, w, t, ob_cache(dep, stem), maps_fn=lambda v, f: (_ for _ in ()).throw(RuntimeError("no cached block")))
    return (None, info) if tm is None else (O.upsample(tm, (tracks.STORE_SIZE[1], tracks.STORE_SIZE[0])), info)


# ------------------------------------------------------------------------------------------------ (a) offline

def back_test_replay(folder: Path, dep: str, name: str, tpx) -> dict:
    """Every scan start box of a stored run: dropped by the back test today? with the back object inside the
    template? Also covered by another stored outline at the previous frame (the other half of box_covered)."""
    meta = json.loads((folder / "meta.json").read_text())
    f0, f1 = meta["frames"]
    z = np.load(folder / "masks.npz")
    back = z["obj_0"] if "obj_0" in z.files else None
    others = [k for k in z.files if k != "obj_0"]
    w = int(meta["store_size"][0])
    sw, sh = tracks.STORE_SIZE
    sx, sy = sw / T.TRACK_W, sh / T.TRACK_H
    starts = T.scan_starts(dep, name, f0, f1)
    out = dict(boxes=0, back_now=0, back_flag=0, newly_pass=0, newly_pass_not_covered=0, frames_new=[])
    if back is None or not len(starts):
        return out
    nonempty = np.nonzero(back.reshape(len(back), -1).any(1))[0]
    for r in starts.itertuples():
        i = int(r.frame) - f0
        prev = nonempty[nonempty < i]
        if not len(prev):
            continue                          # first frame: no back object yet (as online)
        j = int(prev[-1])
        bm = np.unpackbits(back[j], axis=1)[:, :w].astype(bool)
        if max(r.x1 - r.x0, r.y1 - r.y0) < tracks.START_MIN_SIZE * tracks.W:
            continue
        out["boxes"] += 1
        box = tracks._scale_box([r.x0, r.y0, r.x1, r.y1], sx, sy)
        now = O.box_on_back(box, bm)
        flag = O.box_on_back(box, O.own_back(bm, tpx)) if tpx is not None else now
        out["back_now"] += int(now); out["back_flag"] += int(flag)
        if now and not flag:
            out["newly_pass"] += 1
            cover = {k: np.unpackbits(z[k][j], axis=1)[:, :w].astype(bool) for k in others}
            if not tracks.box_covered(box, cover, None):
                out["newly_pass_not_covered"] += 1
                out["frames_new"].append(int(r.frame))
    return out


def step_offline():
    s = pd.read_csv(SPANS)
    rows = []
    for r in s.itertuples():
        d = Path(r.folder)
        stem = Path(r.recording).stem
        tpx, info = template_px(r.deployment, stem, float(r.t0))
        bt = back_test_replay(d, r.deployment, r.recording, tpx)
        off = T.load_online(d)
        on = tracks.read_run(d)
        if tpx is not None:
            on.own_back = tpx
        on = T.finish(on)
        rows.append(dict(deployment=r.deployment, recording=stem[-5:], t0=r.t0, t1=r.t1, template=tpx is not None,
                         template_frames=info.get("frames"), **{k: v for k, v in bt.items() if k != "frames_new"},
                         new_frames=bt["frames_new"][:5],
                         roles_changed=sum(off.roles.get(k) != on.roles.get(k) for k in set(off.roles) | set(on.roles)),
                         numbers_off=len(set(off.number.values())), numbers_on=len(set(on.number.values())),
                         numbers_same_as_wp13=len(set(off.number.values())) == int(r.numbers_purity)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "offline_wp13.csv", index=False)
    log(df[["boxes", "back_now", "back_flag", "newly_pass", "newly_pass_not_covered", "roles_changed"]].sum().to_dict())
    log("spans with a newly passing box:", df[df.newly_pass_not_covered > 0][["recording", "t0", "t1", "newly_pass_not_covered"]].to_dict("records"))
    log("flag-off clean-up reproduces WP13 numbers:", int(df.numbers_same_as_wp13.sum()), "of", len(df))


# ------------------------------------------------------------------------------------------------ (b), (c), (e) tracking

def lock_used():
    p = OUT / "lock_seconds.json"
    return json.loads(p.read_text()) if p.exists() else {"tracking_s": 0.0, "runs": []}


def add_used(sec, what):
    u = lock_used(); u["tracking_s"] += sec; u["runs"].append([what, round(sec, 1)])
    (OUT / "lock_seconds.json").write_text(json.dumps(u, indent=1))


def run_one(r, flag: bool, suffix: str, frame_store=None, force=False):
    t = time.time()
    out = T.run_span(r.deployment, r.video, float(r.t0), float(r.t1), log=log, own_back_template=flag, suffix=suffix,
                     frame_store=frame_store, force=force)
    add_used(time.time() - t, f"{Path(r.video).stem[-5:]} {r.t0}-{r.t1}{suffix}")
    return out


def step_track():
    t0 = time.time()
    st = stretches()
    for r in st.itertuples():
        out = T.span_dir(r.deployment, Path(r.video).name, float(r.t0), float(r.t1))
        out = out.with_name(out.name + "_ownback")
        if (out / "meta.json").exists() and json.loads((out / "meta.json").read_text()).get("done"):
            continue
        est = r.dur * 12 + 60
        if lock_used()["tracking_s"] + est > TRACK_BUDGET_S:
            log(f"budget: {r.piece} {r.t0}-{r.t1} left out"); continue
        if time.time() - t0 + est > BUDGET_S:
            log("track: out of time, run again"); return
        log(f"track on: {r.piece} {r.t0}-{r.t1} ({r.kind})")
        run_one(r, True, "_ownback")
    # (e) one 1080p stretch both ways: the shortest 2025-02-03 candidate from a frame store as well
    e = st[(st.src == "d1") & (st.kind == "candidate")].sort_values("dur").iloc[0]
    out = T.span_dir(e.deployment, Path(e.video).name, float(e.t0), float(e.t1))
    out = out.with_name(out.name + "_ownback_store")
    if not (out / "meta.json").exists():
        root = OUT / "store"
        f0, f1, _ = T.span_start(e.deployment, e.video, float(e.t0), float(e.t1))
        T.store_frames(e.video, f0, f1, root, log=log)
        log(f"track on from the frame store: {e.piece} {e.t0}-{e.t1}")
        run_one(e, True, "_ownback_store", frame_store=root)
    (OUT / "track.DONE").write_text(time.strftime("%H:%M:%S"))
    log("track done")


def needs_off(r) -> bool:
    out = T.span_dir(r.deployment, Path(r.video).name, float(r.t0), float(r.t1))
    on = out.with_name(out.name + "_ownback")
    if not (on / "meta.json").exists():
        return False
    stored = out if (out / "meta.json").exists() else None
    if stored is not None:
        return False
    tpx = tracks.read_run(on).own_back
    bt = back_test_replay(on, r.deployment, Path(r.video).name, tpx)
    return bt["newly_pass_not_covered"] > 0


def step_track_off():
    t0 = time.time()
    for r in stretches().itertuples():
        if not needs_off(r):
            continue
        out = T.span_dir(r.deployment, Path(r.video).name, float(r.t0), float(r.t1))
        out = out.with_name(out.name + "_ownback_off")
        if (out / "meta.json").exists():
            continue
        est = r.dur * 12 + 60
        if lock_used()["tracking_s"] + est > TRACK_BUDGET_S or time.time() - t0 + est > BUDGET_S:
            log(f"off: {r.piece} {r.t0}-{r.t1} left out (budget)"); continue
        log(f"track off: {r.piece} {r.t0}-{r.t1}")
        run_one(r, False, "_ownback_off")
    (OUT / "track_off.DONE").write_text(time.strftime("%H:%M:%S"))
    log("track_off done")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    step = sys.argv[1]
    if step in ("blocks", "track", "track_off"):
        import torch
        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(min(1.0, 8.5e9 / torch.cuda.get_device_properties(0).total_memory))
    log(f"start {step}")
    {"blocks": step_blocks, "offline": step_offline, "track": step_track, "track_off": step_track_off}[step]()
