#!/bin/bash
# Closing pass: run b4d_v2_planner.py every minute and keep its wanted units running on onboard
# (8 A100 shards per job, <=4 in flight, 59-min checkpointed jobs that resume). A unit that has
# been submitted MAXTRY times without finishing gets a FAILED marker so the planner moves on.
W=$HOME/hier-reassess-20261121; R=$W/runs/BS4D_scan; LOG=$W/logs/b4d_v2_manager.log
SB=$R/unit.sbatch; JF=$R/jobs_v2.txt; MAXTRY=5; MAXJ=4
declare -A TRIES
done_marker () { local u=$1 t=${1%%__*}; case "$u" in *__reeval) echo "$R/$t/reeval_n8000.json $R/$t/FAILED_reeval";; *) echo "$R/$t/result.json $R/$t/FAILED";; esac; }
is_done () { for f in $(done_marker $1); do [ -f $f ] && return 0; done; return 1; }
echo "$(date) V2 start" >> $LOG
while true; do
  $W/env/bin/python $R/b4d_v2_planner.py > $JF.tmp 2>> $LOG || { echo "$(date) planner error" >> $LOG; sleep 120; continue; }
  mv $JF.tmp $JF
  grep -q '^#COMPLETE' $JF && { echo "$(date) ALL DONE" >> $LOG; break; }
  q=$(squeue -M sporc -u $USER -h -o "%i %j %P %T" 2>/dev/null | grep -v CLUSTER)
  n=$(echo "$q" | awk '$3=="onboard"' | grep -c .)
  for U in $(grep -v '^#' $JF | cut -d' ' -f1); do
    is_done $U && continue
    echo "$q" | awk -v u=$U '$2==u' | grep -q . && continue
    t=${TRIES[$U]:-0}
    if [ $t -ge $MAXTRY ]; then
      T=${U%%__*}; mkdir -p $R/$T
      case "$U" in *__reeval) touch $R/$T/FAILED_reeval;; *) touch $R/$T/FAILED;; esac
      echo "$(date) FAILED $U after $t tries" >> $LOG; continue
    fi
    [ $n -ge $MAXJ ] && continue
    j=$(sbatch -M sporc -p onboard -t 00:59:00 --gres=shard:a100:8 --export=ALL,MEMFRAC=0.20 \
          -J $U -o $W/logs/b4dv2_${U}_%j.out $SB $U $JF 2>&1 | awk '{print $4}')
    TRIES[$U]=$((t+1)); n=$((n+1))
    echo "$(date) SUBMIT $U try $((t+1)) job $j" >> $LOG
  done
  sleep 60
done
