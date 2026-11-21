#!/bin/bash
# All stage-2 runs of WP8, one GPU-locked command per clip and method (each < ~12 min); finished runs are skipped.
# Start detached:  nohup experiments/07_tracks_demo/jobs/run_all.sh > /home/sat3737/NOAA_whales/underwater_cache/tracks/run_all.log 2>&1 &
cd /home/sat3737/NOAA_whales/Underwater-WhaleID || exit 1
export OMP_NUM_THREADS=8
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
PY=.venv/bin/python
D=experiments/07_tracks_demo
run() {  # clip method
  echo "$(date +%T) start $1 $2"
  flock $LOCK timeout 1500 nice -n 10 $PY -u $D/clips.py run "$1" --method "$2" 2>&1 | grep -v -i warn
  echo "$(date +%T) end $1 $2"
}
for c in 01 05; do run $c owl_sam3; done
for c in 01 05; do run $c text_sam3; done
for c in 01 05; do run $c owl_sam2; done
for c in 02 03 04 06 07; do run $c owl_sam3; done
touch /home/sat3737/NOAA_whales/underwater_cache/tracks/run_all.DONE
