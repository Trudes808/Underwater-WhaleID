#!/bin/bash
# Second set of GPU steps of 04b (test mask frames, cost), each under the shared GPU lock.
#   nohup bash experiments/04b_patch_segmentation/run_gpu2.sh > underwater_cache/patchseg/run_gpu2.log 2>&1 &
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
P=../../.venv/bin/python
L="flock /var/tmp/sat3737-parts-cache/gpu_train.lock nice -n 10"
C=/home/sat3737/NOAA_whales/underwater_cache/patchseg
$L $P test_masks.py features
$L $P test_masks.py curve
$L $P test_masks.py adapt
[ -f results/cost.json ] || $L $P cost.py
touch $C/run_gpu2.DONE
