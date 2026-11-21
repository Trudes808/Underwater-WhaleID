#!/bin/bash
# LLMDet A and B on the dev video at stride 10, inside one GPU lock (about 7 min)
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
V="/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
nice -n 10 flock /var/tmp/sat3737-parts-cache/gpu_train.lock bash -c "
for P in llmdet_A llmdet_B; do .venv/bin/python experiments/04_whale_fish_detection/run_detect.py --video \"$V\" --model llmdet --prompts \$P --stride 10 --budget 400 2>&1 | grep -v -i 'warn\|Loading weights'; done"
touch /home/sat3737/NOAA_whales/underwater_cache/detect/logs/llmdet_dev.DONE
