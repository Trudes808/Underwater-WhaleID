#!/bin/bash
# All stage-2 runs of WP8. Each GPU-locked command runs stretches in order and starts a new one only within its first
# 8 minutes (one stretch takes 3-7 min), so a locked command stays under ~15 min; finished stretches are skipped.
# Start detached:  nohup experiments/07_tracks_demo/jobs/run_all.sh > /home/sat3737/NOAA_whales/underwater_cache/tracks/run_all.log 2>&1 &
cd /home/sat3737/NOAA_whales/Underwater-WhaleID || exit 1
export OMP_NUM_THREADS=8
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
PY=.venv/bin/python
D=experiments/07_tracks_demo
ITEMS="01:owl_sam3 05:owl_sam3 01:text_sam3 05:text_sam3 01:owl_sam2 05:owl_sam2 02:owl_sam3 03:owl_sam3 04:owl_sam3 06:owl_sam3 07:owl_sam3"
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  echo "$(date +%T) lock wait (attempt $attempt)"
  flock $LOCK timeout 1500 nice -n 10 $PY -u $D/clips.py run $ITEMS --budget-s 480 2>&1 | grep -v -i warn
  echo "$(date +%T) released"
  left=$($PY -c "
import sys; sys.path.insert(0, '$D'); sys.path.insert(0, 'src')
from clips import CLIPS; from uwid import tracks
print(sum(not tracks.is_done(*CLIPS[i.split(':')[0]][:3], i.split(':')[1]) for i in '$ITEMS'.split()))")
  echo "left: $left"
  [ "$left" = "0" ] && break
  sleep 20
done
touch /home/sat3737/NOAA_whales/underwater_cache/tracks/run_all.DONE
