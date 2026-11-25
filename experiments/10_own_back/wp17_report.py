"""WP17 report (CPU): per stretch, numbers with the flag off and on, each number's share inside the back object (obj 0)
and inside the own back (obj 0 inside the template), how long it lasts; the frame-store check (e).

Flag-off result of a stretch: the stored run of the same span when one exists (the calf), the `_ownback_off` run
when one was made (a start box passes with the flag on but not off), else the `_ownback` run cleaned without the
template (no start box differs, so the tracker saw the same starts and only the clean-up can differ).

    .venv/bin/python experiments/10_own_back/wp17_report.py      -> results/wp17.json, ownback/wp17/stretches.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import wp17 as P  # noqa: E402
from uwid import tracks  # noqa: E402
from uwid import trackstage as T  # noqa: E402


def numbers_of(tr: tracks.Tracks, own: np.ndarray | None) -> list[dict]:
    w = int(tr.meta["store_size"][0]); fps = float(tr.meta["fps"])
    out = []
    by_num = {}
    for o, n in tr.number.items():
        by_num.setdefault(n, []).append(o)
    for n, objs in sorted(by_num.items()):
        seen = sorted({int(i) for o in objs for i in np.nonzero(tr.packed[o].reshape(tr.n, -1).any(1))[0]})
        s_b, s_o = [], []
        for i in seen[:: max(1, len(seen) // 40)]:
            m = np.zeros((int(tr.meta["store_size"][1]), w), bool)
            for o in objs:
                m |= tr.mask(o, i)
            b = tr.mask(tracks.BACK_ID, i) if tr.back is not None else np.zeros_like(m)
            s_b.append((m & b).sum() / max(m.sum(), 1))
            s_o.append((m & (b & own if own is not None else b)).sum() / max(m.sum(), 1))
        out.append(dict(number=int(n), objs=objs, first_s=round(float(tr.meta["t0"]) + seen[0] / fps, 2),
                        last_s=round(float(tr.meta["t0"]) + seen[-1] / fps, 2), lasts_s=round(len(seen) / fps, 2),
                        in_back_obj=round(float(np.median(s_b)), 3), in_own_back=round(float(np.median(s_o)), 3)))
    return out


def main():
    st = P.stretches()
    rows, det = [], {}
    for r in st.itertuples():
        base = T.span_dir(r.deployment, Path(r.video).name, float(r.t0), float(r.t1))
        on_d = base.with_name(base.name + "_ownback")
        if not (on_d / "meta.json").exists():
            rows.append(dict(piece=r.piece, t0=r.t0, t1=r.t1, kind=r.kind, done=False)); continue
        on = T.load_online(on_d)
        own = on.own_back
        tm_info = json.loads((on_d / "meta.json").read_text()).get("own_back", {})
        if (base / "meta.json").exists():
            off_d, how = base, "stored run"
            off = T.load_online(off_d)
        elif (base.with_name(base.name + "_ownback_off") / "meta.json").exists():
            off_d, how = base.with_name(base.name + "_ownback_off"), "run with the flag off"
            off = T.load_online(off_d)
        else:
            off_d, how = on_d, "same starts: the flag-on run cleaned without the template"
            t = tracks.read_run(on_d); t.own_back = None
            off = T.finish(t)
        n_on, n_off = numbers_of(on, own), numbers_of(off, own)
        roles_on = pd.Series(on.roles).value_counts().to_dict(); roles_off = pd.Series(off.roles).value_counts().to_dict()
        rows.append(dict(piece=r.piece, src=r.src, t0=r.t0, t1=r.t1, kind=r.kind, unclear=bool(r.unclear), done=True,
                         template=own is not None, template_frames=tm_info.get("frames"),
                         template_area=tm_info.get("area_share"), off_from=how,
                         numbers_off=len(n_off), numbers_on=len(n_on),
                         on_back_off=int(roles_off.get("on_back", 0)), on_back_on=int(roles_on.get("on_back", 0)),
                         new_numbers_ge3s=sum(1 for x in n_on if x["lasts_s"] >= 3) - sum(1 for x in n_off if x["lasts_s"] >= 3)))
        det[f"{r.piece} {r.t0}-{r.t1}"] = {"off": n_off, "on": n_on, "off_dir": str(off_d), "on_dir": str(on_d),
                                           "refused_on": json.loads((on_d / "meta.json").read_text()).get("refused", []),
                                           "own_back": tm_info}
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT / "stretches.csv", index=False)
    # (e) frame store vs decoded, flag on
    e = st[(st.src == "d1") & (st.kind == "candidate")].sort_values("dur").iloc[0]
    base = T.span_dir(e.deployment, Path(e.video).name, float(e.t0), float(e.t1))
    a, b = base.with_name(base.name + "_ownback"), base.with_name(base.name + "_ownback_store")
    za, zb = np.load(a / "masks.npz"), np.load(b / "masks.npz")
    diff = sum(int((za[k] != zb[k]).reshape(len(za[k]), -1).any(1).sum()) for k in za.files) if za.files == zb.files else None
    ta, tb = pd.read_parquet(a / "tracks.parquet"), pd.read_parquet(b / "tracks.parquet")
    ma, mb = json.loads((a / "meta.json").read_text()), json.loads((b / "meta.json").read_text())
    store = dict(stretch=f"{e.piece} {e.t0}-{e.t1}", objects_same=za.files == zb.files, mask_frames_differ=diff,
                 frames=int(ma["n_frames"]), table_equal=bool(ta.equals(tb)),
                 numbers=[len(set(T.load_online(a).number.values())), len(set(T.load_online(b).number.values()))],
                 endings_equal=ma["endings"] == mb["endings"], refused_equal=ma["refused"] == mb["refused"],
                 frames_from=[ma.get("frames_from"), mb.get("frames_from")],
                 seconds=[ma["seconds_total"], mb["seconds_total"]])
    off = pd.read_csv(P.OUT / "offline_wp13.csv")
    res = dict(offline=dict(spans=len(off), start_boxes=int(off.boxes.sum()), dropped_by_back_today=int(off.back_now.sum()),
                            dropped_with_flag=int(off.back_flag.sum()), newly_pass=int(off.newly_pass.sum()),
                            roles_changed=int(off.roles_changed.sum()), spans_without_template=int((~off.template).sum()),
                            flag_off_cleanup_equals_wp13=int(off.numbers_same_as_wp13.sum())),
               stretches=df.to_dict("records"), numbers=det, store_check=store,
               lock=json.loads((P.OUT / "lock_seconds.json").read_text()))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "wp17.json").write_text(json.dumps(res, indent=1, default=str))
    pd.set_option("display.width", 220)
    print(df.drop(columns=["src"]).to_string())
    for k, v in det.items():
        print(k, "OFF", [(x["number"], x["lasts_s"], x["in_back_obj"], x["in_own_back"]) for x in v["off"]],
              "ON", [(x["number"], x["first_s"], x["lasts_s"], x["in_back_obj"], x["in_own_back"]) for x in v["on"]])
    print(store); print(res["offline"])


if __name__ == "__main__":
    main()
