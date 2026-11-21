#!/bin/bash
# WP7 scan job: dev clip, then the 2025-02-03 deployment (the detector runs once; every later run re-scores the
# cache), then the clips of the deployment (sensitive preset, copy mode). Detached:
#   nohup bash experiments/06_clips_e2e/jobs/run_scan.sh > <cache>/scan/logs/run_scan.log 2>&1 &
set -u
cd /home/sat3737/NOAA_whales/Underwater-WhaleID
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
LOGD=/home/sat3737/NOAA_whales/underwater_cache/scan/logs
mkdir -p $LOGD
U="nice -n 10 .venv/bin/uwid scan"
DEV="/home/sat3737/NOAA_whales/underwater_video/2025-12-20 CG.m4v"
DEP="/home/sat3737/NOAA_whales/underwater_video/2025-2-3 Mini-mother 11-96"
date; echo "== dev"
$U "$DEV" --preset sensitive && $U "$DEV" --preset strict
date; echo "== deployment, sensitive (runs the models)"
$U "$DEP" --preset sensitive || exit 1
date; echo "== re-scoring"
for P in sensitive strict; do
  $U "$DEP" --preset $P > $LOGD/rescore_${P}.log 2>&1
  for PX in 200 450; do $U "$DEP" --preset $P --min-whale-px $PX > $LOGD/rescore_${P}_px$PX.log 2>&1; done
  $U "$DEP" --preset $P --no-near-whale > $LOGD/rescore_${P}_nonear.log 2>&1
  $U "$DEP" --preset $P --fish-veto > $LOGD/rescore_${P}_fishveto.log 2>&1
  $U "$DEV" --preset $P --no-near-whale > /dev/null 2>&1
done
date; echo "== cut (sensitive, copy)"
$U "$DEP" --preset sensitive --cut --mode copy
date; touch $LOGD/run_scan.DONE
