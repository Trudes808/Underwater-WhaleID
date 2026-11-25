#!/bin/bash
# WP11a: the online tracking queue under the GPU lock until done. Detached:
#   nohup experiments/07_tracks_demo/online/jobs/run_online.sh > /home/sat3737/NOAA_whales/underwater_cache/tracks/run_online.log 2>&1 &
cd /home/sat3737/NOAA_whales/Underwater-WhaleID || exit 1
export OMP_NUM_THREADS=8
L=/home/sat3737/NOAA_whales/underwater_cache/tracks/run_online_last.txt
for attempt in $(seq 1 20); do
  echo "$(date +%T) lock wait (attempt $attempt)"
  flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 nice -n 10 .venv/bin/python -u \
      experiments/07_tracks_demo/online/run_gpu.py --budget-s 360 2>&1 | grep --line-buffered -v -i "warn\|Loading weights\|deprecated" | tee $L
  grep -q "all done" $L && break
  sleep 20
done
touch /home/sat3737/NOAA_whales/underwater_cache/tracks/run_online.DONE
echo "$(date +%T) done"
