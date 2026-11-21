"""Build the compact sensor cache for one deployment (one pass over the tag CSV, about 10-20 min).

    nice -n 10 python experiments/01_metadata/build_cache.py "2025-2-3 Mini-mother 11-96"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from uwid import tagdata  # noqa: E402

VIDEO_ROOT = Path("/home/sat3737/NOAA_whales/underwater_video")

if __name__ == "__main__":
    dep = sys.argv[1] if len(sys.argv) > 1 else "2025-2-3 Mini-mother 11-96"
    csvs = sorted((VIDEO_ROOT / dep / "Data").glob("[!._]*.csv"))
    assert len(csvs) == 1, csvs
    out = tagdata.build_cache(csvs[0], dep, force="--force" in sys.argv, log=lambda s: print(s, flush=True))
    print("done", out)
