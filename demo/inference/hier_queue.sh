#!/usr/bin/env bash
# Workstream-B scan queue (cardassia). At most 3 jobs at once, each pinned to
# 4 cores (8-11, 12-15, 16-19). Resumable: a job with result.json is skipped, a
# job whose claim pid is dead is re-run (hier_map_laplace.py resumes from its
# state.npz). After the fit queue drains: postprocess, re-evaluate the top fits
# at n_seed 8000 with three sampler keys, postprocess again.
#
#   setsid nohup bash hier_queue.sh > logs/B_queue.log 2>&1 < /dev/null &
set -u
B=$HOME/20261120-hier-reassess
source $B/runs/B_env.sh
SCAN=$B/runs/B_scan
DRV=$E/demo/inference/hier_map_laplace.py
PP=$E/demo/inference/hier_postprocess.py
JOBS=$SCAN/jobs.txt
echo $$ > $SCAN/queue.pid

clear_stale() {
  for c in $SCAN/*/.claim; do
    [ -d "$c" ] || continue
    pid=$(cat "$c/pid" 2>/dev/null || echo 0)
    kill -0 "$pid" 2>/dev/null || { echo "$(date) clearing stale claim $c"; rm -rf "$c"; }
  done
}

# worker CORES MODE JOBFILE
worker() {
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
    # memory guard (shared host; other projects run multi-GB jobs here): wait
    # while < 12 GB available or the host is actively swapping (si+so > 5 MB/s)
    while [ "$(free -g | awk '/Mem:/{print $7}')" -lt 12 ] || \
          [ "$(vmstat 5 2 | tail -n 1 | awk '{print ($7+$8>5000)?1:0}')" = 1 ]; do
      echo "$(date) $tag waiting for memory"; sleep 120; done
    echo "$(date) START $mode $tag cores=$cores"
    if [ "$mode" = fit ]; then
      taskset -c "$cores" $PY $DRV --mode fit --outdir "$d" --tag "$tag" $rest \
        >> $B/logs/B_$tag.log 2>&1 < /dev/null
    else
      taskset -c "$cores" $PY $DRV --mode reeval --outdir "$d" --n-seed 8000 --n-b 6000 \
        --keys 11 202 3003 >> $B/logs/B_${tag}_reeval.log 2>&1 < /dev/null
    fi
    rc=$?
    [ -f "$done_f" ] || { echo "$(date) FAIL $mode $tag rc=$rc"; touch "$d/.failed_$mode"; }
    echo "$(date) END $mode $tag rc=$rc"
    rm -rf "$d/.claim"
  done < "$jobs"
}

run_pool() {   # MODE JOBFILE
  clear_stale
  worker 8-11 "$1" "$2" & sleep 20
  worker 12-15 "$1" "$2" & sleep 20
  worker 16-19 "$1" "$2" &
  wait
}

run_pool fit $JOBS
$PY $PP --scan $SCAN --emit-reeval 3 > $SCAN/jobs_reeval.txt
echo "$(date) reeval jobs:"; cat $SCAN/jobs_reeval.txt
run_pool reeval $SCAN/jobs_reeval.txt
$PY $PP --scan $SCAN
echo "$(date) QUEUE DONE"
