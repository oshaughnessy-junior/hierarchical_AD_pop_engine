#!/bin/bash
#SBATCH -A grav-wave-rift
#SBATCH --qos=coll_college_of_science
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# usage: sbatch -p <part> -t <time> -J <unit> unit.sbatch <unit> <jobsfile>
W=$HOME/hier-reassess-20261121; R=$W/runs/BS4D_O4/scan
source $W/env.sh
[ -n "$MEMFRAC" ] && export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMFRAC
# onboard shards share a physical A100: give up fast (exit 75) if no CUDA context can be created
U=$1; JF=$2
LINE=$(grep -E "^$U " $JF); REST=${LINE#"$U "}
TAG=${U%%__*}; D=$R/$TAG; mkdir -p $D
# onboard shards share a physical A100: give up fast (exit 75) if no CUDA context can be created
$PY -c "import jax; assert jax.devices()[0].platform == 'gpu'" >/dev/null 2>&1 || { echo "$(date) NO-GPU $U job $SLURM_JOB_ID $(hostname)" >> $W/logs/no_gpu.log; touch $D/nogpu_${U#$TAG}_$SLURM_JOB_ID; exit 75; }
echo "$(date) START $U job $SLURM_JOB_ID part $SLURM_JOB_PARTITION node $(hostname) cuda=$CUDA_VISIBLE_DEVICES memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION" >> $D/sporc.log
# word-split REST on purpose: init-json has no spaces
$PY $E/demo/inference/hier_map_laplace.py --outdir $D --tag $TAG $REST >> $D/${U#$TAG}fit.log 2>&1
echo "$(date) END $U rc=$?" >> $D/sporc.log
