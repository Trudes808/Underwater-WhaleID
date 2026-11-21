#!/bin/bash
# Leave-one-recording-out refit (FITTED ON THE TEST DEPLOYMENT): window predictions with clf_lomo_<stem>.pt.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
P=../../.venv/bin/python
L="flock /var/tmp/sat3737-parts-cache/gpu_train.lock nice -n 10"
C=/home/sat3737/NOAA_whales/underwater_cache/patchseg
$L $P test_masks.py lomo_save
for d in /home/sat3737/NOAA_whales/underwater_cache/reference/CC-11-96-20250203-*; do
  s=$(basename $d)
  [ -f $C/clf_lomo_$s.pt ] || continue
  [ -f $C/pred_${s}_w1280_h0_lomo.npz ] || $L $P predict_video.py test --stem $s --clf $C/clf_lomo_$s.pt --tag _lomo
done
touch $C/run_gpu3.DONE
