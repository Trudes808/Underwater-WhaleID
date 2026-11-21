#!/bin/bash
# GPU steps of 04b (each under the shared GPU lock, each < 15 min).
#   nohup bash experiments/04b_patch_segmentation/run_gpu.sh > underwater_cache/patchseg/run_gpu.log 2>&1 &
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
P=../../.venv/bin/python
L="flock /var/tmp/sat3737-parts-cache/gpu_train.lock nice -n 10"
C=/home/sat3737/NOAA_whales/underwater_cache/patchseg
[ -f results/cv_compare.json ] || $L $P cv_dev.py compare
[ -f results/cv_curve_w1280_h0.json ] || $L $P cv_dev.py curve --config w1280
[ -f $C/clf_w1280_h0_all.pt ] || $L $P cv_dev.py final --config w1280
$L $P unknown_calib.py
[ -f $C/pred_dev_w1280_h0.npz ] || $L $P predict_video.py dev
for d in /home/sat3737/NOAA_whales/underwater_cache/reference/CC-11-96-20250203-*; do
  s=$(basename $d)
  [ -f $d/index.json ] || continue
  [ -f $C/pred_${s}_w1280_h0.npz ] || $L $P predict_video.py test --stem $s
done
$L $P m2f_seafloor.py
touch $C/run_gpu.DONE
