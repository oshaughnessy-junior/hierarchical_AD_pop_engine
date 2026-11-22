#!/bin/bash
# 4D B-series: phase 1 = 21 fits (jobs_b4d.txt); phase 2 = reevals + kappa profile (jobs_phase2.txt, built by mkphase2.py).
# Slots: onboard <=2 (59 min, checkpoint/resume), sporc-gpu <=2 (60 min); a sporc-gpu job pending >20 min is cancelled
# and that unit goes back to the pool (onboard only afterwards). Unit done when its marker file exists.
W=$HOME/hier-reassess-20261121; R=$W/runs/BS4D_scan; LOG=$W/logs/b4d_manager.log; SB=$R/unit.sbatch
MAXTRY=5
declare -A TRIES NOGPU
done_marker () { local u=$1 t=${1%%__*}; case "$u" in *__reeval) echo $R/$t/reeval_n8000.json;; *) echo $R/$t/result.json;; esac; }
run_phase () {
  local JF=$1
  while true; do
    q=$(squeue -M sporc -u $USER -h -o "%i %j %P %T %V" 2>/dev/null | grep -v CLUSTER)
    left=0
    for U in $(grep -v '^#' $JF | cut -d' ' -f1); do
      [ -f "$(done_marker $U)" ] && continue
      left=$((left+1))
      row=$(echo "$q" | awk -v u=$U '$2==u')
      if [ -n "$row" ]; then
        read jid jn part st sub <<<"$row"
        if [ "$part" = "sporc-gpu" ] && [ "$st" = "PENDING" ]; then
          age=$(( $(date +%s) - $(date -d "$sub" +%s) ))
          [ $age -gt 1200 ] && { scancel -M sporc $jid; NOGPU[$U]=1; echo "$(date) CANCEL $U $jid (sporc-gpu pending ${age}s)" >> $LOG; }
        fi
        continue
      fi
      t=${TRIES[$U]:-0}; [ $t -ge $MAXTRY ] && continue
      n_on=$(echo "$q" | awk '$3=="onboard"' | grep -c .); n_gpu=$(echo "$q" | awk '$3=="sporc-gpu"' | grep -c .)
      # onboard: 8 A100 shards (whole GPUs there are often blocked by other users' shard jobs)
      if [ $n_on -lt 4 ]; then P=onboard; T=00:59:00; G="--gres=shard:a100:8 --export=ALL,MEMFRAC=0.20"
      elif [ $n_gpu -lt 2 ] && [ -z "${NOGPU[$U]}" ]; then P=sporc-gpu; T=01:00:00; G="--gres=gpu:a100:1"
      else continue; fi
      j=$(sbatch -M sporc -p $P -t $T $G -J $U -o $W/logs/b4d_${U}_%j.out $SB $U $JF 2>&1 | awk '{print $4}')
      TRIES[$U]=$((t+1)); echo "$(date) SUBMIT $U try $((t+1)) $P job $j" >> $LOG
      q="$q
$j $U $P PENDING $(date +%Y-%m-%dT%H:%M:%S)"
    done
    [ $left -eq 0 ] && return 0
    stuck=1; for U in $(grep -v '^#' $JF | cut -d' ' -f1); do [ -f "$(done_marker $U)" ] && continue; [ ${TRIES[$U]:-0} -lt $MAXTRY ] && stuck=0; echo "$q" | awk -v u=$U '$2==u' | grep -q . && stuck=0; done
    [ $stuck -eq 1 ] && { echo "$(date) GIVING UP in $JF" >> $LOG; return 1; }
    sleep 60
  done
}
echo "$(date) PHASE1 start" >> $LOG
run_phase $R/jobs_b4d.txt || exit 1
echo "$(date) PHASE1 DONE" >> $LOG
$W/env/bin/python $R/mkphase2.py > $R/jobs_phase2.txt 2>> $LOG
echo "$(date) PHASE2 start ($(grep -vc '^#' $R/jobs_phase2.txt) units)" >> $LOG
run_phase $R/jobs_phase2.txt || exit 1
echo "$(date) ALL DONE" >> $LOG
