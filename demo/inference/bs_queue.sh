#!/usr/bin/env bash
# B-series exact-cascade refit queue (cardassia, reassessment 2026-11-20).
# 5 workers x 4 cores (8-11 ... 24-27). Resumable: a job with result.json is
# skipped; a dead claim is cleared and the fit resumes from state.npz.
# Order: the fits in jobs.txt; then f_dep 0.1 for any model still best at 0.2
# (jobs_extend.txt); then each model's best f_dep is re-evaluated at n_seed 8000 /
# n_b 6000 on keys 11 202 3003 (skipped where done); then the table is rebuilt.
#
#   setsid nohup bash bs_queue.sh > logs/BS_queue.log 2>&1 < /dev/null &
set -u
B=$HOME/20261120-hier-reassess
source $B/runs/BS_env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
SCAN=$B/runs/BS_scan
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

worker() {   # CORES MODE JOBFILE
  local cores=$1 mode=$2 jobs=$3
  while read -r tag rest; do
    [ -z "$tag" ] && continue
    case "$tag" in \#*) continue;; esac
    d=$SCAN/$tag; mkdir -p "$d"
    if [ "$mode" = fit ]; then done_f=$d/result.json; else done_f=$d/reeval_n8000.json; fi
    [ -f "$done_f" ] && continue
    [ -f "$d/.failed_$mode" ] && continue
    mkdir "$d/.claim" 2>/dev/null || continue
    echo $BASHPID > "$d/.claim/pid"
    while [ "$(free -g | awk '/Mem:/{print $7}')" -lt 12 ] || \
          [ "$(vmstat 5 2 | tail -n 1 | awk '{print ($7+$8>5000)?1:0}')" = 1 ]; do
      echo "$(date) $tag waiting for memory"; sleep 120; done
    echo "$(date) START $mode $tag cores=$cores"
    if [ "$mode" = fit ]; then
      $LIM taskset -c "$cores" $PY $DRV --mode fit --outdir "$d" --tag "$tag" $rest \
        >> $B/logs/BS_$tag.log 2>&1 < /dev/null
    else
      $LIM taskset -c "$cores" $PY $DRV --mode reeval --outdir "$d" --n-seed 8000 --n-b 6000 \
        --keys 11 202 3003 >> $B/logs/BS_${tag}_reeval.log 2>&1 < /dev/null
    fi
    rc=$?
    [ -f "$done_f" ] || { echo "$(date) FAIL $mode $tag rc=$rc"; touch "$d/.failed_$mode"; }
    echo "$(date) END $mode $tag rc=$rc"
    rm -rf "$d/.claim"
  done < "$jobs"
}

run_pool() {   # MODE JOBFILE
  [ -s "$2" ] || return 0
  clear_stale
  for c in 8-11 12-15 16-19 20-23 24-27; do worker $c "$1" "$2" & sleep 20; done
  wait
}

run_pool fit $JOBS
$PY $RANK --emit-extend > $SCAN/jobs_extend.txt
echo "$(date) extend jobs:"; cut -d" " -f1 $SCAN/jobs_extend.txt
run_pool fit $SCAN/jobs_extend.txt
$PY $RANK --emit-reeval > $SCAN/jobs_reeval.txt
echo "$(date) reeval jobs:"; cat $SCAN/jobs_reeval.txt
run_pool reeval $SCAN/jobs_reeval.txt
$PY $RANK
echo "$(date) QUEUE DONE"
