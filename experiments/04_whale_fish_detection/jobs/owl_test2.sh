#!/bin/bash
# OWLv2 owl_all over each 2025-02-03 recording (.m4v, the form WP3 labelled), stride 15 + WP3's labelled frames.
# Resumable: each locked call stops after ~12 min; loop until DONE.
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
VD="/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96/Video"
until [ -e /home/sat3737/NOAA_whales/underwater_cache/detect/logs/owl_test.DONE ]; do sleep 30; done
for n in 00009 00010; do
  V=$(ls "$VD"/CC-11-96-20250203-*-$n.m4v)
  stem=$(basename "$V" .m4v)
  for i in 1 2 3 4; do
    out=$(nice -n 10 flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04_whale_fish_detection/run_detect.py --video "$V" --model owl --prompts owl_all --stride 15 --extra-ref /home/sat3737/NOAA_whales/underwater_cache/reference/$stem --budget 720 2>&1 | grep -v -i "warn\|Loading weights")
    echo "$stem try $i: $out"
    echo "$out" | grep -q "^DONE" && break
  done
done
touch /home/sat3737/NOAA_whales/underwater_cache/detect/logs/owl_test2.DONE
