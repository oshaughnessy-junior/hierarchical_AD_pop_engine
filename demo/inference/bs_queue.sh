#!/usr/bin/env bash
# B-series exact-cascade refit queue (cardassia, reassessment 2026-11-20).
# 5 workers x 4 cores (8-11 ... 24-27). Resumable: a job with result.json is
# skipped; a dead claim is cleared and the fit resumes from state.npz.
# Order: the fits in jobs.txt; then f_dep 0.1 for any model still best at 0.2
# (jobs_extend.txt); then each model's best f_dep is re-evaluated at n_seed 8000 /
# n_b 6000 on keys 11 202 3003 (skipped where done); then the table is rebuilt.
#
#   setsid nohup bash bs_queue.sh > logs/BS_queue.log 2>&1 < /dev/null &
#
# Another scan: BS_SCAN=<dir> BS_SUFFIX=_x BS_JOB_EXTRA="--m-max-lo 20" BS_LOGP=BS20.
# Load gate (shared host): workers 1-3 (cores 8-19) start a job only while the
# 1-min load is below BS_LOAD_HI (26, so 4 more cores stay under ~30); workers 4-5
# (cores 20-27) only while it is below BS_LOAD_LO (16).
set -u
B=$HOME/20261120-hier-reassess
source $B/runs/BS_env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
SCAN=${BS_SCAN:-$B/runs/BS_scan}
LOGP=${BS_LOGP:-BS}
LOAD_HI=${BS_LOAD_HI:-26}
LOAD_LO=${BS_LOAD_LO:-16}
RARGS=(--scan "$SCAN" --suffix "${BS_SUFFIX:-}" --job-extra "${BS_JOB_EXTRA:-}")
DRV=$E/demo/inference/hier_map_laplace.py
RANK=$E/demo/inference/bseries_rank.py
JOBS=$SCAN/jobs.txt
LIM="prlimit --as=8589934592"
echo $$ > $SCAN/queue.pid

clear_stale() {
  for c in $SCAN/*/.claim; do
    [ -d "$c" ] || continue
    pid=$(cat "$c/pid" 2>/dev/null || echo 0)
    kill -0 "$pid" 2>/dev/null || { echo "$(date) clearing stale claim $c"; rm -rf "$c"; }
  done
}

load1() { awk '{print int($1)}' /proc/loadavg; }

worker() {   # CORES MODE JOBFILE LOADLIMIT
  # Repeated passes over the job file. The memory and load gates are checked
  # BEFORE a job is claimed, so a worker held back by load never blocks a job
  # that another worker could run. Exits when every job is done or failed.
  local cores=$1 mode=$2 jobs=$3 lim=$4
  while :; do
    local pending=0 ran=0
    while read -r tag rest; do
      [ -z "$tag" ] && continue
      case "$tag" in \#*) continue;; esac
      d=$SCAN/$tag; mkdir -p "$d"
      if [ "$mode" = fit ]; then done_f=$d/result.json; else done_f=$d/reeval_n8000.json; fi
      [ -f "$done_f" ] && continue
      [ -f "$d/.failed_$mode" ] && continue
      pending=1
      [ -d "$d/.claim" ] && continue
      [ "$(free -g | awk '/Mem:/{print $7}')" -lt 12 ] && continue
      [ "$(vmstat 5 2 | tail -n 1 | awk '{print ($7+$8>5000)?1:0}')" = 1 ] && continue
      [ "$(load1)" -ge "$lim" ] && continue
      mkdir "$d/.claim" 2>/dev/null || continue
      echo $BASHPID > "$d/.claim/pid"
      echo "$(date) START $mode $tag cores=$cores load=$(load1)"
      if [ "$mode" = fit ]; then
        $LIM taskset -c "$cores" $PY $DRV --mode fit --outdir "$d" --tag "$tag" $rest \
          >> $B/logs/${LOGP}_$tag.log 2>&1 < /dev/null
      else
        $LIM taskset -c "$cores" $PY $DRV --mode reeval --outdir "$d" --n-seed 8000 --n-b 6000 \
          --keys 11 202 3003 >> $B/logs/${LOGP}_${tag}_reeval.log 2>&1 < /dev/null
      fi
      rc=$?
      [ -f "$done_f" ] || { echo "$(date) FAIL $mode $tag rc=$rc"; touch "$d/.failed_$mode"; }
      echo "$(date) END $mode $tag rc=$rc"
      rm -rf "$d/.claim"
      ran=1
    done < "$jobs"
    [ "$pending" = 0 ] && break
    [ "$ran" = 0 ] && sleep 180
  done
}

run_pool() {   # MODE JOBFILE
  [ -s "$2" ] || return 0
  clear_stale
  local i=0
  for c in 8-11 12-15 16-19 20-23 24-27; do
    i=$((i + 1))
    if [ $i -le 3 ]; then worker $c "$1" "$2" "$LOAD_HI" & else worker $c "$1" "$2" "$LOAD_LO" & fi
    sleep 20
  done
  wait
}

run_pool fit $JOBS
$PY $RANK "${RARGS[@]}" --emit-extend > $SCAN/jobs_extend.txt
echo "$(date) extend jobs:"; cut -d" " -f1 $SCAN/jobs_extend.txt
run_pool fit $SCAN/jobs_extend.txt
$PY $RANK "${RARGS[@]}" --emit-reeval > $SCAN/jobs_reeval.txt
echo "$(date) reeval jobs:"; cat $SCAN/jobs_reeval.txt
run_pool reeval $SCAN/jobs_reeval.txt
$PY $RANK "${RARGS[@]}"
echo "$(date) QUEUE DONE"
