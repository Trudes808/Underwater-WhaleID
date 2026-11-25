#!/bin/bash
# WP15 smoke: TripoSG meshes for underwater_cache/mesh3d/smoke/inputs, in locked pieces of <= ~4-5 min.
# Gives the lock back between pieces (sleep 30) so the tracker run (WP13) can take it. Done marker: smoke/meshes.DONE
C=/home/sat3737/NOAA_whales/underwater_cache
R=/home/sat3737/NOAA_whales/Underwater-WhaleID/experiments/10_mesh3d
IN=$C/mesh3d/smoke/inputs; OUT=$C/mesh3d/smoke/meshes; LOG=$C/mesh3d/smoke/meshes.log
rm -f $C/mesh3d/smoke/meshes.DONE $C/mesh3d/smoke/meshes.FAILED
echo "$(date) job start" >> $LOG
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TMPDIR=$C/mesh3d/tmp
mkdir -p $TMPDIR
for piece in $(seq 1 12); do
  [ -e $OUT/ALL.DONE ] && break
  echo "$(date) piece $piece waiting for lock; $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)" >> $LOG
  flock /var/tmp/sat3737-parts-cache/gpu_train.lock timeout 900 nice -n 10 \
     $C/envs/triposg/bin/python $R/run_triposg.py --inputs $IN --out $OUT --budget-s 220 >> $LOG 2>&1
  echo "$(date) piece $piece exit $?" >> $LOG
  [ -e $OUT/ALL.DONE ] && break
  sleep 30
done
if [ -e $OUT/ALL.DONE ]; then touch $C/mesh3d/smoke/meshes.DONE; else touch $C/mesh3d/smoke/meshes.FAILED; fi
echo "$(date) job end" >> $LOG
