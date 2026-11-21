#!/bin/bash
cd /home/sat3737/NOAA_whales/Underwater-WhaleID/experiments/04_whale_fish_detection
R=/home/sat3737/NOAA_whales/underwater_cache/reference
for d in $R/CC-11-96-20250203-*-0000[4-8]; do
  stem=$(basename $d)
  for i in 1 2 3 4 5; do
    out=$(nice -n 10 flock /var/tmp/sat3737-parts-cache/gpu_train.lock ../../.venv/bin/python m2f.py run --frames-dir $d --tag $stem 2>&1 | grep -v -i "warn\|loading\|torch_dtype")
    echo "$stem try $i: $out"
    echo "$out" | grep -q "^DONE" && break
  done
done
touch /home/sat3737/NOAA_whales/underwater_cache/detect/logs/m2f_test.DONE
