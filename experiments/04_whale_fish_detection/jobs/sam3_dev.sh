#!/bin/bash
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
V="/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
nice -n 10 flock /var/tmp/sat3737-parts-cache/gpu_train.lock bash -c "
.venv/bin/python experiments/04_whale_fish_detection/run_detect.py --video \"$V\" --model sam3 --prompts sam3_all --stride 15 --budget 500 2>&1 | grep -v -i 'warn\|Loading weights\|deprecated';
cd experiments/04_whale_fish_detection && ../../.venv/bin/python sam3_masks.py run 2>&1 | grep -v -i 'warn\|Loading weights\|deprecated'"
touch /home/sat3737/NOAA_whales/underwater_cache/detect/logs/sam3_dev.DONE
