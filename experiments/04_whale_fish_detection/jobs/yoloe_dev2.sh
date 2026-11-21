#!/bin/bash
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
V="/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
nice -n 10 flock /var/tmp/sat3737-parts-cache/gpu_train.lock bash -c "
for P in yoloe_whale yoloe_whale_fish; do .venv/bin/python experiments/04_whale_fish_detection/run_detect.py --video \"$V\" --model yoloe --prompts \$P --stride 5 --budget 300 2>&1 | grep -v -i warn; done"
touch /home/sat3737/NOAA_whales/underwater_cache/detect/logs/yoloe_dev2.DONE
