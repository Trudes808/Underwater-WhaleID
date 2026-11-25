#!/bin/bash
# WP9: floating time on 2025-02-03 00012, then the second deployment (CC-09-85) again with the clock read,
# the estimated clock offset ("tag" default -> auto for an unknown tag) and floating time scanned.
set -u
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
LOGD=/home/sat3737/NOAA_whales/underwater_cache/scan/logs
D2="/home/sat3737/NOAA_whales/underwater_video/TestSubset 2025-1-2 CATS -85"
date
nice -n 10 .venv/bin/python experiments/06_clips_e2e/float_check.py
date
nice -n 10 .venv/bin/uwid scan "$D2"
date
nice -n 10 .venv/bin/uwid scan "$D2" --preset strict
date; touch $LOGD/run_wp9.DONE
