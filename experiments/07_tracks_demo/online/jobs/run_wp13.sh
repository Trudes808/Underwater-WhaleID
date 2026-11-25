#!/bin/bash
# WP13: the tracker on all hq spans of both test deployments, rules frozen as of WP11c, as pieces of `wp13_queue.py`
# under the GPU lock (timeout 900 each; queue order, per-span estimates and cut-off handling are in wp13_queue.py).
# Version 2 (2026-11-23 22:30, after the CPU overload): v1 ran `uwid tracks` per deployment; its first piece
# (21:50-22:05) finished 3 short CC-09-85 spans and was cut off in 00023 120-138 (Python output was buffered, so the
# piece printed nothing).
# Stops by itself when the lock-held time passes 3.5 h, the tracks cache passes 6 GB or /home falls below 150 GB free.
#   nohup experiments/07_tracks_demo/online/jobs/run_wp13.sh [--use-store] >> <cache>/tracks/run_wp13.log 2>&1 &
cd /home/sat3737/NOAA_whales/Underwater-WhaleID || exit 1
export OMP_NUM_THREADS=8
C=/home/sat3737/NOAA_whales/underwater_cache/tracks
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
HELD=$C/run_wp13_held_s.txt            # lock-held seconds so far (survives a restart of this script)
[ -f $HELD ] || echo 0 > $HELD
OPT="$1"
for piece in $(seq 1 80); do
  held=$(cat $HELD)
  cache_kb=$(du -sk --exclude=tmp_frames $C | cut -f1)
  free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc 0-9)
  if [ "$held" -gt 12600 ]; then echo "STOP: lock-held time ${held} s > 3.5 h"; touch $C/run_wp13.STOPPED; exit 0; fi
  if [ "$cache_kb" -gt 6291456 ]; then echo "STOP: tracks cache ${cache_kb} kB > 6 GB"; touch $C/run_wp13.STOPPED; exit 0; fi
  if [ "$free_gb" -lt 150 ]; then echo "STOP: /home ${free_gb} GB free < 150"; touch $C/run_wp13.STOPPED; exit 0; fi
  [ "$free_gb" -lt 165 ] && echo "WARN_HOME /home ${free_gb} GB free"
  echo "$(date +%T) lock wait: piece $piece (held so far ${held} s, cache ${cache_kb} kB, /home ${free_gb} GB free)"
  L=$C/run_wp13_last.txt
  rm -f $C/run_wp13_piece_s0.txt
  flock $LOCK timeout 900 nice -n 10 bash -c "date +%s > $C/run_wp13_piece_s0.txt; \
      .venv/bin/python -u experiments/07_tracks_demo/online/wp13_queue.py $OPT" 2>&1 \
      | tee -a $C/run_wp13_raw.log | grep --line-buffered -v -i "warn\|Loading weights\|deprecated\|online frame" | tee $L
  s1=$(date +%s)
  if [ -f $C/run_wp13_piece_s0.txt ]; then
    s0=$(cat $C/run_wp13_piece_s0.txt); echo $(( held + s1 - s0 )) > $HELD
    echo "PIECE_END $(date +%T) piece $piece: $(grep -c SPAN_DONE $L) spans finished, lock held $(( s1 - s0 )) s, total $(cat $HELD) s"
  else
    echo "PIECE_END $(date +%T) piece $piece: lock not taken"
  fi
  grep -q "NOTHING_RUNNABLE" $L && break
  grep -q "Traceback" $L && { echo "STOP: error in piece"; touch $C/run_wp13.STOPPED; exit 1; }
  sleep 15
done
touch $C/run_wp13.DONE
echo "ALL_DONE $(date +%T)"
