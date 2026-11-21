#!/bin/bash
# SAM 3 on the labelled frames only (single-frame scoring), prompts whale / fish / seafloor; resumable.
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
VD="/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96/Video"
LB=experiments/02_reference_labels/labels
for n in 00004 00005 00006 00007 00008 00009 00010; do
  V=$(ls "$VD"/CC-11-96-20250203-*-$n.m4v); stem=$(basename "$V" .m4v)
  for i in 1 2 3; do
    out=$(nice -n 10 flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/04_whale_fish_detection/run_detect.py --video "$V" --model sam3 --prompts sam3_test --stride 99999999 --labels $LB/$stem/frames.csv --budget 600 2>&1 | grep -v -i "warn\|Loading weights\|deprecated")
    echo "$stem try $i: $out"
    echo "$out" | grep -q "^DONE" && break
  done
done
touch /home/sat3737/NOAA_whales/underwater_cache/detect/logs/sam3_test.DONE
