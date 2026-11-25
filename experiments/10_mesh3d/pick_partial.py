"""WP15 phase 2, arm B "whales cut by the frame" (repo .venv, CPU).

    .venv/bin/python experiments/10_mesh3d/pick_partial.py

Numbers: every number lasting >= 3 s with 0 passing frames in arm A (`trial/judged.csv`). Their frames are judged
again by the S0 rule WITHOUT the clause "outline >= 6 px from every frame edge" (every other clause, including
sharpness >= the number's median over the frames that pass the others, still applies). One frame per number: the
largest outlined area among the passing frames, ties by sharpness. Numbers with no such frame are dropped and
counted. Order of the list: life_s, longest first (the budget cut takes from the end).
Writes underwater_cache/mesh3d/trial_partial/{judged.csv, picks_partial.csv, pick_counts.json}; refuses to
overwrite picks_partial.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import framerule as fr  # noqa: E402
import pick_frames as pf  # noqa: E402
from uwid import trackstage  # noqa: E402

A = fr.MESH / "trial"
OUT = fr.MESH / "trial_partial"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "picks_partial.csv").exists():
        sys.exit("picks_partial.csv exists: the pick list is fixed, not overwritten")
    print("pick_partial: start", flush=True)
    JA = pd.read_csv(A / "judged.csv", keep_default_na=False)
    passA = JA[JA.why == ""].groupby(["folder", "number"]).size()
    N = pd.read_csv(fr.TRACKS / "wp13_numbers.csv")
    S = pd.read_csv(fr.TRACKS / "wp13_spans.csv")
    N = N[(N.life_s >= 3.0) & N.folder.isin(set(S.folder))]
    N = N[[passA.get((f, n), 0) == 0 for f, n in zip(N.folder, N.number)]]
    print("numbers >= 3 s with 0 arm-A frames:", len(N), dict(N.deployment.value_counts()), flush=True)
    hq = {dep: trackstage.hq_spans(dep) for dep in N.deployment.unique()}
    judged = []
    for folder, g in N.groupby("folder"):
        dep = g.deployment.iloc[0]
        df = pf.judge_span(dep, Path(folder), set(g.number.astype(int)), hq[dep], fr.Rule(), skip_edge=True)
        if len(df):
            judged.append(df)
        print(Path(folder).parent.name[-5:], Path(folder).name, "judged", len(df),
              "pass", int((df.why == "").sum()) if len(df) else 0, flush=True)
    J = pd.concat(judged, ignore_index=True)
    J.to_csv(OUT / "judged.csv", index=False)
    life = {(r.folder, r.number): r.life_s for r in N.itertuples()}
    picks, dropped = [], []
    for (folder, num), g in J.groupby(["folder", "number"]):
        ok = g[g.why == ""]
        if not len(ok):
            dropped.append(dict(folder=folder, number=int(num), reasons=g.why.value_counts().to_dict()))
            continue
        r = ok.sort_values(["area_share", "sharp"], ascending=False).iloc[0]
        rec = r.video.split("-")[-1].split(".")[0]
        picks.append(dict(id=f"{pf.DEP_SHORT[r.deployment]}_{rec}_{r.span[:7]}_n{num}_cut", run=folder, obj=r.obj,
                          i=r.i, frame=r.frame, t=r.t, video=r.video, number=int(num),
                          animal=f"{pf.DEP_SHORT[r.deployment]}_{rec}_{r.span}_n{num}", deployment=r.deployment,
                          kind="", sharp=r.sharp, area_share=r.area_share, orient=r.orient, elong=r.elong,
                          fill=r.fill, life_s=life[(folder, num)]))
    # numbers that never appear in J (no outlined frame at step 5) are dropped too
    seen = set(zip(J.folder, J.number))
    for r in N.itertuples():
        if (r.folder, r.number) not in seen:
            dropped.append(dict(folder=r.folder, number=int(r.number), reasons={"no judged frame": 1}))
    P = pd.DataFrame(picks).sort_values("life_s", ascending=False, kind="stable").reset_index(drop=True)
    P.to_csv(OUT / "picks_partial.csv", index=False)
    counts = dict(numbers=len(N), by_deployment=dict(N.deployment.value_counts()), picks=len(P),
                  picks_by_deployment=dict(P.deployment.value_counts()), dropped=len(dropped),
                  dropped_list=dropped,
                  frames_by_first_failing_clause={dep: J[J.deployment == dep].why.replace("", "passed")
                                                  .value_counts().to_dict() for dep in J.deployment.unique()})
    (OUT / "pick_counts.json").write_text(json.dumps(counts, indent=1, default=int))
    print(json.dumps({k: v for k, v in counts.items() if k != "dropped_list"}, indent=1, default=int))
    print("pick_partial: done", len(P), "picks", flush=True)


if __name__ == "__main__":
    main()
