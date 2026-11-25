#!/bin/bash
#SBATCH -A grav-wave-rift
#SBATCH -p onboard
#SBATCH --qos=coll_college_of_science
#SBATCH --gres=shard:a100:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# usage: sbatch -t 00:59:00 -J <unit> unit.sbatch <unit> <jobsfile>
W=$HOME/hier-reassess-20261121; ROOT=$W/runs/BS4D_O4_ms; source $W/env.sh; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
U=$1; JF=$2; LOG=$W/logs
$PY -c "import jax; assert jax.devices()[0].platform == 'gpu'" >/dev/null 2>&1 || { echo "$(date) NO-GPU $U $SLURM_JOB_ID $(hostname)" >> $LOG/no_gpu.log; mkdir -p $ROOT/nogpu; touch $ROOT/nogpu/${U}_$SLURM_JOB_ID; exit 75; }
REST=$(grep -E "^$U " $JF | cut -d' ' -f2-)
echo "$(date) START $U job $SLURM_JOB_ID $(hostname)" >> $ROOT/units.log
case "$U" in
  screen__*) M=$(echo $U | cut -d_ -f3); M=${U#screen__}; DS=${M##*__}; M=${M%__*}
    mkdir -p $ROOT/screen; $PY $E/demo/inference/bs4d_ms_screen.py $ROOT/screen $M $DS 0.3 >> $ROOT/screen/${M}_${DS}_f0.3.log 2>&1 ;;
  *__eval) T=${U%__eval}; set -- $REST; mkdir -p $ROOT/eval
    $PY $E/demo/inference/bs4d_ms_eval.py $ROOT/eval/$T.npz $2 $1 >> $ROOT/eval/$T.log 2>&1 && touch $ROOT/eval/$T.done ;;
  *) mkdir -p $ROOT/fits/$U; $PY $E/demo/inference/hier_map_laplace.py --mode fit --outdir $ROOT/fits/$U --tag $U $REST >> $ROOT/fits/$U/fit.log 2>&1 ;;
esac
echo "$(date) END $U rc=$?" >> $ROOT/units.log
