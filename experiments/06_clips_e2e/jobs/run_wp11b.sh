#!/bin/bash
# WP11b: box classifier veto + seafloor / sand / dark labels + 30 s activity guard, both deployments.
# GPU part: `uwid scan` under the shared lock with `timeout 900` and a 13-min GPU budget, repeated until every
# recording's picture classes are cached (summary.json "complete"); then CPU-only re-scores of the variants.
set -u
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
LOGD=/home/sat3737/NOAA_whales/underwater_cache/scan/logs
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
D1="/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96"
D2="/home/sat3737/NOAA_whales/underwater_video/TestSubset 2025-1-2 CATS -85"
SC=/home/sat3737/NOAA_whales/underwater_cache/scan
complete() { .venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('complete') else 1)" "$1"; }
echo "$(date) start"
for pair in "$D2|TestSubset 2025-1-2 CATS -85" "$D1|2025-2-3 Mini-mother 11-96"; do
  D="${pair%%|*}"; N="${pair##*|}"
  for i in 1 2 3 4 5 6 7 8; do
    echo "$(date) $N GPU pass $i"
    flock $LOCK timeout 900 nice -n 10 .venv/bin/uwid scan "$D" --no-gpu-lock --gpu-budget-min 13
    complete "$SC/$N/sensitive/summary.json" && break
    sleep 30
  done
  echo "$(date) $N variants (CPU, cached)"
  nice -n 10 .venv/bin/uwid scan "$D" --preset strict
  nice -n 10 .venv/bin/uwid scan "$D" --no-boxclf
  nice -n 10 .venv/bin/uwid scan "$D" --guard 9
done
echo "$(date) done"; touch $LOGD/run_wp11b.DONE
