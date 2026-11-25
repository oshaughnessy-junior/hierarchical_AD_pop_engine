#!/bin/bash
# Multi-start manager: onboard shards, <=4 in flight; 59-min resumable jobs; GPU-less starts are not counted as tries.
W=$HOME/hier-reassess-20261121; ROOT=$W/runs/BS4D_O4_ms; LOG=$W/logs/b4d_ms_manager.log; JF=$ROOT/jobs.txt; MAXTRY=8; MAXJ=4
declare -A TRIES
is_done () { case "$1" in screen__*) local x=${1#screen__}; [ -f $ROOT/screen/${x%__*}_${x##*__}_f0.3.done ];; *__eval) [ -f $ROOT/eval/${1%__eval}.done ];; *) [ -f $ROOT/fits/$1/result.json ];; esac; }
echo "$(date) MS start" >> $LOG
while true; do
  $W/env/bin/python $ROOT/b4d_ms_planner.py > $JF.tmp 2>> $LOG || { echo "$(date) planner error" >> $LOG; sleep 120; continue; }
  mv $JF.tmp $JF
  grep -q '^#COMPLETE' $JF && { echo "$(date) ALL DONE" >> $LOG; break; }
  q=$(squeue -M sporc -u $USER -h -o "%i %j %P %T" 2>/dev/null | grep -v CLUSTER); n=$(echo "$q" | awk '$3=="onboard"' | grep -c .)
  for U in $(grep -v '^#' $JF | cut -d' ' -f1); do
    is_done $U && continue
    echo "$q" | awk -v u=$U '$2==u' | grep -q . && continue
    ng=$(ls $ROOT/nogpu/${U}_* 2>/dev/null | wc -l); t=$(( ${TRIES[$U]:-0} - ng ))
    if [ $t -ge $MAXTRY ]; then echo "$(date) FAILED $U" >> $LOG; mkdir -p $ROOT/fits/$U; touch $ROOT/fits/$U/FAILED; case "$U" in *__eval) touch $ROOT/eval/${U%__eval}.done;; esac; continue; fi
    [ $n -ge $MAXJ ] && continue
    [ "$(find $W/logs/no_gpu.log -mmin -5 2>/dev/null | wc -l)" -gt 0 ] && continue
    j=$(sbatch -M sporc -t 00:59:00 -J $U -o $W/logs/ms_${U}_%j.out $ROOT/unit.sbatch $U $JF 2>&1 | awk '{print $4}')
    TRIES[$U]=$(( ${TRIES[$U]:-0} + 1 )); n=$((n+1)); echo "$(date) SUBMIT $U try ${TRIES[$U]} job $j" >> $LOG
  done
  sleep 60
done
