#!/bin/bash
# WP8b: repeat the GPU queue under the lock until everything is done (each locked command starts new items only in
# its first 12 minutes). Start detached:
#   nohup experiments/07_tracks_demo/gap/jobs/run_gap.sh > /home/sat3737/NOAA_whales/underwater_cache/tracks/run_gap.log 2>&1 &
cd /home/sat3737/NOAA_whales/Underwater-WhaleID || exit 1
export OMP_NUM_THREADS=8
for attempt in $(seq 1 20); do
  echo "$(date +%T) lock wait (attempt $attempt)"
  flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 1500 nice -n 10 .venv/bin/python -u \
      experiments/07_tracks_demo/gap/run_gpu.py --budget-s 720 2>&1 | grep -v -i "warn\|Loading weights\|deprecated" | tee /home/sat3737/NOAA_whales/underwater_cache/tracks/run_gap_last.txt
  echo "$(date +%T) released"
  grep -q "all done" /home/sat3737/NOAA_whales/underwater_cache/tracks/run_gap_last.txt && break
  sleep 20
done
touch /home/sat3737/NOAA_whales/underwater_cache/tracks/run_gap.DONE
