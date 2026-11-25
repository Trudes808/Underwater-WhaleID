"""WP9 step 4: what scanning 'tag floating' time adds on 2025-02-03 piece 00012 (25 min of float).

Runs `uwid scan` on that one recording with the deployment's cache (clock fits, picture water states, detector boxes)
but writes its tables under <cache>/scan/_float_check/, so the deployment's own results are not replaced. Two runs:
floating scanned (the new default) and not (+nofloat, as before)."""
import json
import os
from pathlib import Path

from uwid import paths, pipeline as pl



def main():
    DEP = "2025-2-3 Mini-mother 11-96"
    SRC = paths.footage_dir() / DEP / "Video" / "CC-11-96-20250203-141328-000-00012.mov"
    root = paths.cache_dir("scan") / "_float_check"
    (root / DEP).mkdir(parents=True, exist_ok=True)
    link = root / DEP / "cache"
    if not link.exists():
        os.symlink(paths.cache_dir("scan") / DEP / "cache", link)
    out = {}
    for flo in (False, True):
        cfg = pl.ScanConfig(scan_floating=flo)
        s = pl.scan(SRC, cfg, out_root=root, log=lambda *m: print(*m, flush=True))[0]
        out[cfg.label()] = {k: s[k] for k in ("kept_s", "hq_windows", "hq_stretch_s", "dropped_s")}
    res = Path(__file__).parent / "results"
    (res / "float_check_00012.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":      # the detector's decoder is a spawned child that re-imports this file
    main()
