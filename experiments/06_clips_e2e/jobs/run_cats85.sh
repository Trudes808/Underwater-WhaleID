#!/bin/bash
# WP7: first run of `uwid scan` on a deployment nothing was built on (tag CC-09-85, 2025-01-02), unchanged defaults.
set -u
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
LOGD=/home/sat3737/NOAA_whales/underwater_cache/scan/logs
date
nice -n 10 .venv/bin/uwid scan "/home/sat3737/NOAA_whales/underwater_video/TestSubset 2025-1-2 CATS -85"
date
nice -n 10 .venv/bin/uwid scan "/home/sat3737/NOAA_whales/underwater_video/TestSubset 2025-1-2 CATS -85" --preset strict
date; touch $LOGD/run_cats85.DONE
