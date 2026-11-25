"""WP13 queue runner: ONE piece of the all-hq-spans tracker run, to be started under the GPU lock with timeout 900.
Same spans and same per-span code as `uwid tracks <deployment>` (trackstage.run_span on every hq span, capped at
30 s around its centre); only the order and the piece cutting differ (coordinator, 2026-11-23 22:10, CPU overload):

- Queue: 2025-02-03 (1080p) spans shortest first, then CC-09-85 (4K) spans shortest first.
- Before each span its wall time is estimated from the rate measured on the last span of the same class (4K / 1080p)
  and path (decoded / store): est = rate * span seconds + 30 s. A span that does not fit in what is left of this
  piece's budget ends the piece (it is first in the next one); a span whose estimate exceeds a whole piece is skipped
  ("too long" at the current rate; it stays at the end of the queue and is re-estimated in every piece).
- A span that was running when a piece was cut off (marker left in the state file) is "cut": it is never run again by
  the same path; with --use-store it may run once more from a complete frame store, else it is left over.
- --use-store: read a span's frames from `tracks/tmp_frames/` when the pre-decoder (`wp13_predecode.py`, no lock)
  has completed its store (trackstage frame_store flag); the store is deleted when the span is done.

    flock <lock> timeout 900 nice -n 10 .venv/bin/python -u wp13_queue.py [--use-store]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from uwid import paths, trackstage as T  # noqa: E402

CACHE = paths.cache_dir("tracks")
STATE = CACHE / "wp13_state.json"
STORE = CACHE / "tmp_frames"
DEP_HD, DEP_4K = "2025-2-3 Mini-mother 11-96", "TestSubset 2025-1-2 CATS -85"
OVERHEAD_S = 30.0
DEFAULT_RATE = {"hd/decoded": 15.0, "4k/decoded": 35.0, "hd/store": 10.0, "4k/store": 15.0}   # wall s per video s


def queue() -> list[dict]:
    out = []
    for dep, cls in ((DEP_HD, "hd"), (DEP_4K, "4k")):
        items = []
        for r in T.hq_spans(dep).itertuples():
            a, b = r.start, r.end
            if b - a > 30:
                c = (a + b) / 2
                a, b = c - 15, c + 15
            items.append(dict(key=f"{dep}|{Path(r.source).name}|{a:.2f}|{b:.2f}", dep=dep, src=r.source,
                              a=float(a), b=float(b), dur=float(b - a), cls=cls))
        out += sorted(items, key=lambda x: (x["dur"], x["key"]))
    return out


STORE_MAX = 3 * 1024 ** 3


def store_bytes(x: dict, f0: int, f1: int) -> int:
    return (f1 - f0) * (3840 * 2160 * 3 if x["cls"] == "4k" else 1920 * 1080 * 3)


def done(x: dict) -> bool:
    m = T.span_dir(x["dep"], Path(x["src"]).name, x["a"], x["b"]) / "meta.json"
    return m.exists() and json.loads(m.read_text()).get("done", False)


def load_state() -> dict:
    s = json.loads(STATE.read_text()) if STATE.exists() else {}
    s.setdefault("started", None)
    s.setdefault("cut", {})
    s.setdefault("rate", dict(DEFAULT_RATE))
    s.setdefault("runs", [])
    return s


def save_state(s: dict):
    STATE.write_text(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-s", type=float, default=840.0)
    ap.add_argument("--use-store", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    print(time.strftime("%H:%M:%S"), "WP13 piece start", "(store on)" if a.use_store else "(store off)", flush=True)
    st = load_state()
    if st["started"]:
        k, path = st["started"]["key"], st["started"]["path"]
        x = next((q for q in queue() if q["key"] == k), None)
        if x is not None and not done(x):
            st["cut"].setdefault(k, []).append(path)
            print("CUT", k, path, flush=True)
        st["started"] = None
        save_state(st)
    q = queue()
    n_run, skipped_long, left = 0, [], 0
    for x in q:
        if done(x):
            continue
        name = Path(x["src"]).name
        f0, f1 = None, None
        path = "decoded"
        if a.use_store:
            f0, f1, _ = T.span_start(x["dep"], x["src"], x["a"], x["b"])
            if T.stored(STORE, x["src"], f0, f1):
                path = "store"
        if path in st["cut"].get(x["key"], []):
            left += 1
            continue                                      # never the same path again after a cut-off
        rate = st["rate"].get(f"{x['cls']}/{path}", DEFAULT_RATE[f"{x['cls']}/{path}"])
        est = rate * x["dur"] + OVERHEAD_S
        remaining = a.budget_s - (time.time() - t_start)
        if est > a.budget_s:
            skipped_long.append(x["key"])
            left += 1
            continue
        if est > remaining:
            left += 1
            print(f"next span needs ~{est:.0f} s, {remaining:.0f} s left: new piece", flush=True)
            break
        st["started"] = dict(key=x["key"], path=path)
        save_state(st)
        t0 = time.time()
        print(time.strftime("%H:%M:%S"), "span start", x["key"], f"{x['dur']:.1f} s", path, f"est {est:.0f} s", flush=True)
        T.run_span(x["dep"], x["src"], x["a"], x["b"], frame_store=STORE if path == "store" else None)
        wall = time.time() - t0
        st["rate"][f"{x['cls']}/{path}"] = round(max(1.0, (wall - OVERHEAD_S) / max(x["dur"], 0.5)), 2)
        st["runs"].append(dict(key=x["key"], path=path, wall_s=round(wall, 1), dur=x["dur"],
                               end=time.strftime("%H:%M:%S")))
        st["started"] = None
        save_state(st)
        n_run += 1
        if path == "store" and f0 is not None:
            shutil.rmtree(T.store_dir(STORE, x["src"], f0, f1), ignore_errors=True)
        print(time.strftime("%H:%M:%S"), "SPAN_DONE", x["key"], f"{wall:.0f} s wall", path, flush=True)
    else:
        left = sum(1 for x in q if not done(x))
    for dep in (DEP_HD, DEP_4K):
        T.write_index(dep)
    n_done = sum(1 for x in q if done(x))
    print(f"PIECE_SUMMARY finished_this_piece={n_run} done={n_done}/{len(q)} left={len(q) - n_done} "
          f"too_long_now={len(skipped_long)} cut={len(st['cut'])} wall={time.time() - t_start:.0f}s", flush=True)
    def can_run(x):
        cut = set(st["cut"].get(x["key"], []))
        if "decoded" not in cut:
            return True
        if not a.use_store or "store" in cut:
            return False
        f0, f1, _ = T.span_start(x["dep"], x["src"], x["a"], x["b"])
        return store_bytes(x, f0, f1) <= STORE_MAX          # it can still come back from a store
    runnable = [x for x in q if not done(x) and x["key"] not in skipped_long and can_run(x)]
    if not runnable:
        print("NOTHING_RUNNABLE", flush=True)


if __name__ == "__main__":
    main()
