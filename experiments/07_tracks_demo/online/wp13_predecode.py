"""WP13 pre-decoder (CPU, nice 10, NO GPU lock): decodes the queue's next spans into `tracks/tmp_frames/` (raw
frames, trackstage.store_frames) so the locked tracker reads them instead of decoding. At most 3 GB in the store at a
time; spans bigger than that alone are not stored; stores of finished spans are deleted. Runs until every span is
done or `tracks/run_wp13.DONE` / `run_wp13.STOPPED` exists.

    nice -n 10 .venv/bin/python -u wp13_predecode.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
import wp13_queue as Q  # noqa: E402
from uwid import trackstage as T  # noqa: E402


def used() -> int:
    return sum(f.stat().st_size for f in Q.STORE.rglob("*") if f.is_file()) if Q.STORE.exists() else 0


def main():
    Q.STORE.mkdir(parents=True, exist_ok=True)
    for d in Q.STORE.iterdir():                      # incomplete stores from an earlier stop
        if d.is_dir() and not (d / "index.json").exists():
            shutil.rmtree(d, ignore_errors=True)
    print(time.strftime("%H:%M:%S"), "predecode start", flush=True)
    span_of = {}
    while not ((Q.CACHE / "run_wp13.DONE").exists() or (Q.CACHE / "run_wp13.STOPPED").exists()):
        q = Q.queue()
        todo = [x for x in q if not Q.done(x)]
        if not todo:
            break
        keep = set()
        made = False
        for x in todo:
            if x["key"] not in span_of:
                span_of[x["key"]] = T.span_start(x["dep"], x["src"], x["a"], x["b"])[:2]
            f0, f1 = span_of[x["key"]]
            d = T.store_dir(Q.STORE, x["src"], f0, f1)
            keep.add(d.name)
            if T.stored(Q.STORE, x["src"], f0, f1):
                continue
            size = Q.store_bytes(x, f0, f1)
            if size > Q.STORE_MAX:
                continue
            if used() + size > Q.STORE_MAX:
                break                                  # wait for the tracker to use (and delete) a store
            t0 = time.time()
            T.store_frames(x["src"], f0, f1, Q.STORE)
            print(time.strftime("%H:%M:%S"), "stored", x["key"], f"{size / 1e9:.2f} GB in {time.time() - t0:.0f} s",
                  flush=True)
            made = True
            break
        for d in Q.STORE.iterdir():                   # stores of spans that are done
            if d.is_dir() and d.name not in keep and (d / "index.json").exists():
                shutil.rmtree(d, ignore_errors=True)
        if not made:
            time.sleep(20)
    print(time.strftime("%H:%M:%S"), "predecode end", flush=True)


if __name__ == "__main__":
    main()
