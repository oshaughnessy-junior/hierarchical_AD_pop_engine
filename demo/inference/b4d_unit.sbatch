#!/bin/bash
#SBATCH -A grav-wave-rift
#SBATCH --qos=coll_college_of_science
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# usage: sbatch -p <part> -t <time> -J <unit> unit.sbatch <unit> <jobsfile>
W=$HOME/hier-reassess-20261121; R=$W/runs/BS4D_scan
source $W/env.sh
[ -n "$MEMFRAC" ] && export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC
U=$1; JF=$2
LINE=$(grep -E "^$U " $JF); REST=${LINE#"$U "}
TAG=${U%%__*}; D=$R/$TAG; mkdir -p $D
echo "$(date) START $U job $SLURM_JOB_ID part $SLURM_JOB_PARTITION node $(hostname) cuda=$CUDA_VISIBLE_DEVICES memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION" >> $D/sporc.log
# word-split REST on purpose: init-json has no spaces
$PY $E/demo/inference/hier_map_laplace.py --outdir $D --tag $TAG $REST >> $D/${U#$TAG}fit.log 2>&1
echo "$(date) END $U rc=$?" >> $D/sporc.log
