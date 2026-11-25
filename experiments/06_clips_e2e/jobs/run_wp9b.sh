#!/bin/bash
# WP9 part 2: after run_wp9.sh. The first deployment re-scored from its cache with the new rules (activity kept,
# tiers, silhouette, floating time scanned), the old rule for comparison (+whaleonly+nofloat), and the floating check
# of 00012. Only 00012's floating frames are new detector work.
set -u
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
LOGD=/home/sat3737/NOAA_whales/underwater_cache/scan/logs
D1="/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96"
while [ ! -f $LOGD/run_wp9.DONE ]; do sleep 30; done
date
nice -n 10 .venv/bin/python experiments/06_clips_e2e/float_check.py
date
nice -n 10 .venv/bin/uwid scan "$D1"
nice -n 10 .venv/bin/uwid scan "$D1" --preset strict
nice -n 10 .venv/bin/uwid scan "$D1" --whale-only --no-floating
date; touch $LOGD/run_wp9b.DONE
