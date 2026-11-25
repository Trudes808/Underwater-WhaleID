#!/bin/bash
# WP15 phase 2: inputs (CPU) then TripoSG meshes in locked, resumable pieces (each stops starting new meshes after
# 740 s, so a piece holds the lock < 14 min; timeout 900 as CLAUDE.md requires). Lock given back 30 s between pieces.
# Done marker: underwater_cache/mesh3d/trial/meshes.DONE (or .FAILED). One copy only.
C=/home/sat3737/NOAA_whales/underwater_cache
R=/home/sat3737/NOAA_whales/Underwater-WhaleID
T=$C/mesh3d/trial; IN=$T/inputs; OUT=$T/meshes; LOG=$T/run.log
exec 9>/home/sat3737/NOAA_whales/underwater_cache/mesh3d/trial/job.lock; flock -n 9 || { echo "another copy runs"; exit 1; }
rm -f $T/meshes.DONE $T/meshes.FAILED
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TMPDIR=$C/mesh3d/tmp
mkdir -p $TMPDIR $IN $OUT
echo "$(date) job start" >> $LOG
nice -n 10 timeout 3600 $R/.venv/bin/python $R/experiments/10_mesh3d/prep_inputs.py $T/picks.csv $IN --crop >> $LOG 2>&1
echo "$(date) inputs exit $? ($(ls $IN/*.png | wc -l) png)" >> $LOG
for piece in $(seq 1 40); do
  [ -e $OUT/ALL.DONE ] && break
  echo "$(date) piece $piece waiting for lock; $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)" >> $LOG
  flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 nice -n 10 \
     $C/envs/triposg/bin/python $R/experiments/10_mesh3d/run_triposg.py --inputs $IN --out $OUT --budget-s 740 \
     --keep-faces 100000 >> $LOG 2>&1
  echo "$(date) piece $piece exit $?" >> $LOG
  [ -e $OUT/ALL.DONE ] && break
  sleep 30
done
if [ -e $OUT/ALL.DONE ]; then touch $T/meshes.DONE; else touch $T/meshes.FAILED; fi
echo "$(date) job end" >> $LOG
