#!/bin/bash
# Submit one B-series run with the path-weighted cascade on SPORC (not yet run):
# 4 chains, the bias check after them, and one results sync after the bias check.
# Appends job IDs to jobs.txt, bias_jobs.txt and sync_jobs.txt. Run from runs/BS_nuts on SPORC.
# usage: submit_run.sh TAG PROPOSAL CAP     (CAP = m_max_A prior upper edge, or none)
set -e
cd /home/rossma/hier-reassess-20261121/runs/BS_nuts
TAG=$1; PP=$2; CAP=$3
ids=()
for C in 0 1 2 3; do
  mkdir -p $TAG/chain$C
  j=$(sbatch -M sporc --parsable -J bsnuts-$TAG-c$C --export=ALL,TAG=$TAG,CHAIN=$C,PP=$PP,CAP=$CAP -o $TAG/chain$C/slurm_%j.out chain.sbatch)
  ids+=(${j%%;*})
done
b=$(sbatch -M sporc --parsable -J bsbias-$TAG --dependency=afterany:$(IFS=:; echo "${ids[*]}") --export=ALL,TAG=$TAG,PP=$PP,CAP=$CAP -o $TAG/bias_%j.out bias.sbatch)
s=$(sbatch -M sporc --parsable -J bsnuts-sync-$TAG --dependency=afterany:${b%%;*} --export=ALL,MODE=once,TAG=$TAG -o sync_%j.out sync.sbatch)
echo "$TAG ${ids[*]} proposal $PP cap $CAP $(date +%F)" >> jobs.txt
echo "$TAG ${b%%;*} after ${ids[*]}" >> bias_jobs.txt
echo "$TAG after ${b%%;*} sync ${s%%;*}" >> sync_jobs.txt
echo "$TAG chains ${ids[*]} bias ${b%%;*} sync ${s%%;*}"
