"""WP11a follow-up A/B/C: re-run only the spans named by the review (picture-agrees lens-out rule, all link
candidates written to links.csv). The old run folders are kept beside as <name>_pre_fixA (patch pass reused).
Run under the GPU lock:  flock <lock> timeout 900 nice -n 10 .venv/bin/python -u run_followup.py"""
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from uwid import trackstage as T, tracks  # noqa: E402

V24 = ("TestSubset 2025-1-2 CATS -85", "/home/sat3737/NOAA_whales/underwater_video/TestSubset 2025-1-2 CATS -85/"
       "Video/CC-09-85-HRA-20250102-160604-000-00024.mov")
SPANS = [(*V24, 97.0, 120.0), (*V24, 108.0, 118.0), (*V24, 119.0, 142.0),
         ("2025-2-3 Mini-mother 11-96", tracks.TEST_DIR + "/CC-11-96-20250203-114256-000-00007.mov", 1459.0, 1475.0)]


def main():
    print(time.strftime("%H:%M:%S"), "WP11a follow-up start", flush=True)
    for dep, src, a, b in SPANS:
        d = T.span_dir(dep, Path(src).name, a, b)
        m = d / "meta.json"
        if m.exists():
            meta = json.loads(m.read_text())
            if meta.get("done") and meta.get("blocked_info", {}).get("lens_source", "").startswith("depth sensor where") \
                    or meta.get("done") and (d / "links.csv").exists():
                print(d.name, "done with the new rule", flush=True)
                continue
            old = d.with_name(d.name + "_pre_fixA")
            if not old.exists():
                shutil.move(str(d), str(old))
                d.mkdir(parents=True)
                if (old / "patch.npz").exists():
                    shutil.copy2(old / "patch.npz", d / "patch.npz")
        print(time.strftime("%H:%M:%S"), dep, Path(src).name, a, b, flush=True)
        T.run_span(dep, src, a, b, start_source="scan", force=True)
    for dep in {s[0] for s in SPANS}:
        T.write_index(dep)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
