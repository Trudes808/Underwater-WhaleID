#!/bin/bash
# WP11a follow-up: four spans under the GPU lock. Detached:
#   nohup experiments/07_tracks_demo/online/jobs/run_followup.sh > /home/sat3737/NOAA_whales/underwater_cache/tracks/run_followup.log 2>&1 &
cd /home/sat3737/NOAA_whales/Underwater-WhaleID || exit 1
export OMP_NUM_THREADS=8
L=/home/sat3737/NOAA_whales/underwater_cache/tracks/run_followup_last.txt
for attempt in $(seq 1 6); do
  echo "$(date +%T) lock wait (attempt $attempt)"
  flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 nice -n 10 .venv/bin/python -u \
      experiments/07_tracks_demo/online/run_followup.py 2>&1 | grep --line-buffered -v -i "warn\|Loading weights\|deprecated" | tee $L
  grep -q "all done" $L && break
  sleep 20
done
touch /home/sat3737/NOAA_whales/underwater_cache/tracks/run_followup.DONE
echo "$(date +%T) done"
