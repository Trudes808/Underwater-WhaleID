#!/bin/bash
# WP19: `uwid tracks 2025-12-20_CG` (the dev clip's hq spans, default settings) in pieces under the GPU lock.
#   nohup experiments/12_export/run_dev_tracks.sh > <cache>/tracks/run_dev_wp19.log 2>&1 &
cd /home/sat3737/NOAA_whales/Underwater-WhaleID || exit 1
export OMP_NUM_THREADS=8
C=/home/sat3737/NOAA_whales/underwater_cache/tracks
L=$C/run_dev_wp19_last.txt
for piece in $(seq 1 6); do
  echo "$(date +%T) lock wait: piece $piece"
  flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 nice -n 10 bash -c \
    'echo "$(date +%T) piece start"; exec .venv/bin/python -u -m uwid.cli tracks 2025-12-20_CG --lock-minutes 10' 2>&1 \
    | grep --line-buffered -v -i "warn\|Loading weights\|deprecated\|online frame" | tee $L
  if grep -q "spans; index" $L && ! grep -q "budget reached" $L; then echo "ALL_DONE $(date +%T)"; break; fi
  grep -q Traceback $L && { echo "STOP: error"; break; }
  sleep 10
done
touch $C/run_dev_wp19.DONE
