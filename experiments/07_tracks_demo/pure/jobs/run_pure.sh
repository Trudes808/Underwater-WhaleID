#!/bin/bash
# WP8c GPU work: backward runs, then features. Detached:
#   nohup experiments/07_tracks_demo/pure/jobs/run_pure.sh > /home/sat3737/NOAA_whales/underwater_cache/tracks/run_pure.log 2>&1 &
cd /home/sat3737/NOAA_whales/Underwater-WhaleID || exit 1
export OMP_NUM_THREADS=8
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
for step in backward features; do
  echo "$(date +%T) $step"
  flock $LOCK timeout 1500 nice -n 10 .venv/bin/python -u experiments/07_tracks_demo/pure/run_gpu.py $step 2>&1 \
     | grep --line-buffered -v -i "warn\|Loading weights\|deprecated\| frame "
done
touch /home/sat3737/NOAA_whales/underwater_cache/tracks/run_pure.DONE
echo "$(date +%T) done"
