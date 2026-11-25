#!/bin/bash
# WP11c: the spans of results/selection_breath.csv through `uwid tracks --only`, under the GPU lock. Detached:
#   nohup experiments/07_tracks_demo/online/jobs/run_wp11c.sh "<deployment>" <rec:t0-t1> ... > <cache>/tracks/run_wp11c.log 2>&1 &
cd /home/sat3737/NOAA_whales/Underwater-WhaleID || exit 1
export OMP_NUM_THREADS=8
DEP="$1"; shift
L=/home/sat3737/NOAA_whales/underwater_cache/tracks/run_wp11c_last.txt
for attempt in $(seq 1 6); do
  echo "$(date +%T) lock wait (attempt $attempt)"
  flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 nice -n 10 .venv/bin/uwid tracks "$DEP" --only "$@" \
      --lock-minutes 10 2>&1 | grep --line-buffered -v -i "warn\|Loading weights\|deprecated\|online frame" | tee $L
  grep -q "spans; index" $L && ! grep -q "budget reached" $L && break
  sleep 20
done
touch /home/sat3737/NOAA_whales/underwater_cache/tracks/run_wp11c.DONE
echo "$(date +%T) done"
