#!/bin/bash
#SBATCH -A grav-wave-rift
#SBATCH -p onboard
#SBATCH --qos=coll_college_of_science
#SBATCH --gres=shard:a100:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# usage: sbatch -t 00:59:00 -J <unit> unit.sbatch <unit> <jobsfile>
W=$HOME/hier-reassess-20261121; R=$W/runs/BS4D_O4/scan_n8000; source $W/env.sh; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
U=$1; JF=$2; TAG=${U%__lap}; D=$R/$TAG; mkdir -p $D
$PY -c "import jax; assert jax.devices()[0].platform == 'gpu'" >/dev/null 2>&1 || { echo "$(date) NO-GPU $U $SLURM_JOB_ID $(hostname)" >> $W/logs/no_gpu.log; touch $D/nogpu_$SLURM_JOB_ID; exit 75; }
LINE=$(grep -E "^$U " $JF); REST=${LINE#"$U "}
echo "$(date) START $U job $SLURM_JOB_ID $(hostname)" >> $D/sporc.log
case "$U" in
  *__lap) $PY $E/demo/inference/bs4d_o4_laplace.py $W/runs/BS4D_O4/laplace_n8000 8000 6000 11,202,3003 $REST >> $D/lap.log 2>&1 ;;
  *) $PY $E/demo/inference/hier_map_laplace.py --mode fit --outdir $D --tag $TAG $REST >> $D/fit.log 2>&1 ;;
esac
echo "$(date) END $U rc=$?" >> $D/sporc.log
