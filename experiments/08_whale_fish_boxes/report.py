"""Tables of 08_whale_fish_boxes -> results/report.md (copied into tables.md) and results/report.json.

    .venv/bin/python experiments/08_whale_fish_boxes/report.py [feature set]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data as D  # noqa: E402

RES = HERE / "results"
BASE_MAP = {"whale": "whale", "small": "fish", "fish": "fish", "none": "other"}
KEY_BANDS = ("300-500", "500-800")


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [str(v) for v in r.values]) + " |")
    return "\n".join(out)


def counts(obj):
    t = pd.crosstab([obj.source, obj.truth], obj.band).reindex(columns=list(D.BANDS), fill_value=0)
    t["all"] = t.sum(1)
    t.index = [f"{a} {b}" for a, b in t.index]; t.index.name = "source truth"
    return t


def confusion(d, pred_col, labels):
    """rows truth3 x band, columns predicted label counts, as 'a/b/c' strings."""
    rows = {}
    for tr in ("whale", "fish", "other"):
        for band in D.BANDS:
            s = d[(d.truth3 == tr) & (d.band == band)]
            if not len(s):
                continue
            vc = s[pred_col].value_counts()
            rows[f"{tr} {band}"] = {"n": len(s), **{l: int(vc.get(l, 0)) for l in labels}}
    t = pd.DataFrame(rows).T
    t.index.name = "truth band"
    return t


def key_errors(d, pred_col, whale_label="whale"):
    """In the 300-800 px band: fish called whale; whale called not whale. Also >= 800 whales called not whale."""
    band = d.band.isin(KEY_BANDS)
    fish = d[band & (d.truth3 == "fish")]; whale = d[band & (d.truth3 == "whale")]
    big = d[(d.band == ">=800") & (d.truth3 == "whale")]
    oth = d[band & (d.truth3 == "other")]
    return {"fish_300_800": len(fish), "fish_called_whale": int((fish[pred_col] == whale_label).sum()),
            "whale_300_800": len(whale), "whale_not_whale": int((whale[pred_col] != whale_label).sum()),
            "whale_ge800": len(big), "whale_ge800_not_whale": int((big[pred_col] != whale_label).sum()),
            "other_300_800": len(oth), "other_called_whale": int((oth[pred_col] == whale_label).sum())}


def main(fs="tight+ctx"):
    obj = pd.read_parquet(D.CACHE / "objects.parquet")
    P = pd.read_parquet(D.CACHE / "predictions.parquet")
    lines = ["## Labelled objects (detector boxes >= 0.2, grouped), by source, truth and size band (px, 1920 px frame)", "",
             md_table(counts(obj)), ""]
    amb = obj[obj.truth == "ambiguous"].groupby(["source", "why"]).size().rename("n").reset_index()
    lines += ["Ambiguous objects (left out of training and scoring), by reason:", "",
              md_table(amb.set_index("source")), ""]
    why = obj[obj.truth != "ambiguous"].groupby(["source", "truth", "why"]).size().rename("n").reset_index()
    lines += ["How the truth was decided (non-ambiguous objects):", "", md_table(why.set_index("source")), ""]
    lab = obj[obj.truth3.isin(["whale", "fish", "other"])].copy()
    lab["base_a3"] = lab.base_a.map(BASE_MAP); lab["base_b3"] = lab.base_b.map(BASE_MAP)
    out = {"baseline": {}, "classifier": {}}
    lines += ["## Baseline (no training): the scan's rules", "",
              "Rule A = scan (whale score >= 0.25 and >= 300 px = whale; shorter = small; fish prompt >= 0.5 = fish; "
              "else none). Rule B = A + whale score must beat fish score. Columns: what the rule calls the object.", ""]
    for src in D.SOURCES:
        d = lab[lab.source == src]
        lines += [f"### Rule A on {D.SOURCE_NAME[src]}", "", md_table(confusion(d, "base_a", ["whale", "small", "fish", "none"])), ""]
        out["baseline"][f"A|{src}"] = key_errors(d, "base_a"); out["baseline"][f"B|{src}"] = key_errors(d, "base_b")
    lines += ["### Key errors of the baseline (300-800 px band; whales >= 800 px)", ""]
    kt = pd.DataFrame(out["baseline"]).T; kt.index.name = "rule|test source"
    lines += [md_table(kt), ""]
    # classifier
    lines += [f"## Classifier, leave one source out (feature set `{fs}` unless named)", ""]
    for fsn in sorted(P.features.unique()):
        rows = {}
        for fold in P.fold.unique():
            d = P[(P.features == fsn) & (P.fold == fold)]
            src = d.source.iloc[0]
            ke = key_errors(d, "pred")
            ba = key_errors(lab[lab.source == src], "base_a")
            rows[fold] = {"n": len(d), "fish 300-800: base A -> clf": f"{ba['fish_called_whale']} -> {ke['fish_called_whale']} of {ke['fish_300_800']}",
                          "whale 300-800 lost: base A -> clf": f"{ba['whale_not_whale']} -> {ke['whale_not_whale']} of {ke['whale_300_800']}",
                          "whale >=800 lost: base A -> clf": f"{ba['whale_ge800_not_whale']} -> {ke['whale_ge800_not_whale']} of {ke['whale_ge800']}",
                          "other 300-800 called whale: base A -> clf": f"{ba['other_called_whale']} -> {ke['other_called_whale']} of {ke['other_300_800']}",
                          "C": d.C.iloc[0]}
            out["classifier"][f"{fsn}|{fold}"] = ke
        t = pd.DataFrame(rows).T; t.index.name = "fold"
        lines += [f"### `{fsn}`", "", md_table(t), ""]
    for fold in P.fold.unique():
        d = P[(P.features == fs) & (P.fold == fold)]
        lines += [f"### Confusion by band, `{fs}`, fold {fold} (columns: classifier's call)", "",
                  md_table(confusion(d, "pred", ["whale", "fish", "other"])), ""]
    pe = D.CACHE / "predictions_extra.parquet"
    if pe.exists():
        E = pd.read_parquet(pe)
        lines += ["## Extra test boxes, 300-800 px (scan detector cache, AI by-eye labels by the WP10 agent; test only)", "",
                  "Every box here is one the scan keeps as a whale (whale score >= 0.25, 300-800 px). Half were drawn among "
                  "boxes whose best fish prompt scores >= 0.3. Ambiguous boxes (3) left out.", ""]
        rows = {}
        for fsn in sorted(E.features.unique()):
            for fold in E.fold.unique():
                d = E[(E.features == fsn) & (E.fold == fold)]
                f, w, o = d[d.truth3 == "fish"], d[d.truth3 == "whale"], d[d.truth3 == "other"]
                rows[f"{fsn} | {fold}"] = {
                    "fish: rule A / B / clf called whale": f"{(f.base_a == 'whale').sum()} / {(f.base_b == 'whale').sum()} / {(f.pred == 'whale').sum()} of {len(f)}",
                    "whale: rule A / B / clf lost": f"{(w.base_a != 'whale').sum()} / {(w.base_b != 'whale').sum()} / {(w.pred != 'whale').sum()} of {len(w)}",
                    "other: rule A / B / clf called whale": f"{(o.base_a == 'whale').sum()} / {(o.base_b == 'whale').sum()} / {(o.pred == 'whale').sum()} of {len(o)}"}
                out.setdefault("extra", {})[f"{fsn}|{fold}"] = {
                    "fish": len(f), "fish_A": int((f.base_a == 'whale').sum()), "fish_B": int((f.base_b == 'whale').sum()),
                    "fish_clf": int((f.pred == 'whale').sum()), "whale": len(w), "whale_lost_A": int((w.base_a != 'whale').sum()),
                    "whale_lost_B": int((w.base_b != 'whale').sum()), "whale_lost_clf": int((w.pred != 'whale').sum()),
                    "other": len(o), "other_clf_whale": int((o.pred == 'whale').sum())}
        t = pd.DataFrame(rows).T; t.index.name = "features | fold"
        lines += [md_table(t), ""]
    # the recommended use: the scan's rule A, then the classifier as a veto (whale only if both say whale)
    rows = {}
    for name, f in (("reference", D.CACHE / "predictions.parquet"), ("extra", D.CACHE / "predictions_extra.parquet")):
        if not f.exists():
            continue
        Q = pd.read_parquet(f)
        Q = Q[Q.features == fs]
        for fold in Q.fold.unique():
            d = Q[Q.fold == fold]
            A = d.base_a == "whale"; Cc = A & (d.pred == "whale")
            big = d.band.isin(["300-500", "500-800", ">=800"])
            r = {}
            for t in ("whale", "fish", "other"):
                x = big & (d.truth3 == t)
                r[f"{t} >= 300 px kept as whale: A -> A+veto"] = f"{int((A & x).sum())} -> {int((Cc & x).sum())} of {int(x.sum())}"
                out.setdefault("veto", {})[f"{name}|{fold}|{t}"] = [int((A & x).sum()), int((Cc & x).sum()), int(x.sum())]
            rows[f"{name} | {fold}"] = r
    if rows:
        t = pd.DataFrame(rows).T; t.index.name = "boxes | fold"
        lines += ["## Recommended use: scan rule A, then the classifier as a veto (`" + fs + "`)", "",
                  "A box stays a whale only when rule A calls it whale AND the classifier's best class is whale. "
                  "Below 300 px nothing changes (the size rule stays: small = fish or far whale).", "", md_table(t), ""]
    (RES / "report.md").write_text("\n".join(lines) + "\n")
    (RES / "report.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
