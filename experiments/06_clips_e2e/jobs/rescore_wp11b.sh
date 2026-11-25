#!/bin/bash
# WP11b CPU re-scores of one deployment from the cache (no GPU work left): the default and the comparison runs.
set -u
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
D="$1"
for args in "" "--preset strict" "--no-boxclf" "--guard 9" "--no-boxclf --guard 9"; do
  echo "$(date) $D $args"
  nice -n 10 .venv/bin/uwid scan "$D" $args 2>&1 | grep -E "kept|stretches by|Traceback|Error"
done
echo "$(date) done $D"
