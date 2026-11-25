#!/bin/bash
# O4 follow-up manager: onboard shards, <=4 in flight, 59-min resumable jobs; GPU-less starts are not counted as tries.
W=$HOME/hier-reassess-20261121; R=$W/runs/BS4D_O4/scan_n8000; LOG=$W/logs/b4d_o4b_manager.log; JF=$R/jobs.txt; MAXTRY=8; MAXJ=4
declare -A TRIES
is_done () { local u=$1 t=${1%__lap}; case "$u" in *__lap) for k in 11 202 3003; do [ -f $W/runs/BS4D_O4/laplace_n8000/${t}_n8000_k$k.json ] || return 1; done; return 0;; *) [ -f $R/$t/result.json ];; esac; }
echo "$(date) O4b start" >> $LOG
while true; do
  $W/env/bin/python $R/b4d_o4b_planner.py > $JF.tmp 2>> $LOG || { echo "$(date) planner error" >> $LOG; sleep 120; continue; }
  mv $JF.tmp $JF
  grep -q '^#COMPLETE' $JF && { echo "$(date) ALL DONE" >> $LOG; break; }
  q=$(squeue -M sporc -u $USER -h -o "%i %j %P %T" 2>/dev/null | grep -v CLUSTER); n=$(echo "$q" | awk '$3=="onboard"' | grep -c .)
  for U in $(grep -v '^#' $JF | cut -d' ' -f1); do
    is_done $U && continue
    echo "$q" | awk -v u=$U '$2==u' | grep -q . && continue
    T=${U%__lap}; ng=$(ls $R/$T/nogpu_* 2>/dev/null | wc -l); t=$(( ${TRIES[$U]:-0} - ng ))
    if [ $t -ge $MAXTRY ]; then mkdir -p $R/$T; case "$U" in *__lap) touch $R/$T/FAILED_lap;; *) touch $R/$T/FAILED;; esac; echo "$(date) FAILED $U" >> $LOG; continue; fi
    [ $n -ge $MAXJ ] && continue
    [ "$(find $W/logs/no_gpu.log -mmin -5 2>/dev/null | wc -l)" -gt 0 ] && continue
    j=$(sbatch -M sporc -t 00:59:00 -J $U -o $W/logs/o4b_${U}_%j.out $R/unit.sbatch $U $JF 2>&1 | awk '{print $4}')
    TRIES[$U]=$(( ${TRIES[$U]:-0} + 1 )); n=$((n+1)); echo "$(date) SUBMIT $U try ${TRIES[$U]} job $j" >> $LOG
  done
  sleep 60
done
