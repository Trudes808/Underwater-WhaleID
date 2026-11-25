#!/bin/bash
# WP15 phase 2 arm B (whales cut by the frame): inputs (CPU), then TripoSG meshes in pick order (life_s, longest
# first) in locked pieces of < 14 min, same settings as arm A. Lock-held budget 45 min in total: each piece gets
# min(740 s, what is left - 60 s) to start new meshes. Done marker: trial_partial/meshes.DONE (all made) or
# meshes.BUDGET (budget used, some left out) or meshes.FAILED. One copy only.
C=/home/sat3737/NOAA_whales/underwater_cache
R=/home/sat3737/NOAA_whales/Underwater-WhaleID
T=$C/mesh3d/trial_partial; IN=$T/inputs; OUT=$T/meshes; LOG=$T/run.log
exec 9>/home/sat3737/NOAA_whales/underwater_cache/mesh3d/trial_partial/job.lock; flock -n 9 || { echo "another copy runs"; exit 1; }
rm -f $T/meshes.DONE $T/meshes.FAILED $T/meshes.BUDGET
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TMPDIR=$C/mesh3d/tmp
mkdir -p $TMPDIR $IN $OUT
echo "$(date) job start" >> $LOG
nice -n 10 timeout 3600 $R/.venv/bin/python $R/experiments/10_mesh3d/prep_inputs.py $T/picks_partial.csv $IN --crop >> $LOG 2>&1
echo "$(date) inputs exit $? ($(ls $IN/*.png | wc -l) png)" >> $LOG
BUDGET=2700; USED=0
for piece in $(seq 1 20); do
  [ -e $OUT/ALL.DONE ] && break
  LEFT=$((BUDGET - USED)); B=$((LEFT - 60)); [ $B -gt 740 ] && B=740
  if [ $B -lt 60 ]; then echo "$(date) lock budget used ($USED s)" >> $LOG; break; fi
  echo "$(date) piece $piece waiting for lock (budget $B s, used $USED s)" >> $LOG
  flock /var/tmp/sat3737-parts-cache/gpu_train.lock bash -c "echo \$(date +%s) > $T/lock_t0; \
     timeout 900 nice -n 10 $C/envs/triposg/bin/python $R/experiments/10_mesh3d/run_triposg.py --inputs $IN \
     --out $OUT --budget-s $B --keep-faces 100000 --order $T/picks_partial.csv >> $LOG 2>&1; rc=\$?; \
     echo \$(( \$(date +%s) - \$(cat $T/lock_t0) )) > $T/lock_dt; exit \$rc"
  RC=$?; DT=$(cat $T/lock_dt); USED=$((USED + DT))
  echo "$(date) piece $piece exit $RC, held $DT s, used $USED s" >> $LOG
  [ -e $OUT/ALL.DONE ] && break
  sleep 30
done
if [ -e $OUT/ALL.DONE ]; then touch $T/meshes.DONE; elif [ $USED -ge $((BUDGET - 120)) ]; then touch $T/meshes.BUDGET; else touch $T/meshes.FAILED; fi
echo "$(date) job end, lock held $USED s" >> $LOG
