#!/bin/bash
# bias check for one run after its chains end: path-weighted lnL (the sampled one) against the
# recentred exact lnL at 20 posterior draws.
#   sbatch -M sporc -J bsbias-TAG --dependency=afterany:J0:J1:J2:J3 --export=ALL,TAG=TAG,PP=PROPOSAL,CAP=X bias.sbatch
#SBATCH -A grav-wave-rift
#SBATCH -p tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
source /home/rossma/hier-reassess-20261121/runs/BS_nuts/env.sh
export JAX_PLATFORMS=cpu
D=/home/rossma/hier-reassess-20261121/runs/BS_nuts/$TAG
CAPARG=""; [ -n "$CAP" ] && [ "$CAP" != none ] && CAPARG="--m-max-cap $CAP"
echo "START $(date) job $SLURM_JOB_ID tag $TAG commit $(git -C $E rev-parse --short HEAD) proposal $PP cap ${CAP:-none}"
$PY $E/demo/inference/hier_nuts.py --mode bias --map-json $MAPS/$TAG/result.json --ckpt-dirs "$D/chain*" --outdir $D $INP \
    --key 11 --n-seed 8000 --n-b 6000 --n-draws 20 --bias-seed 0 \
    --path-proposal $PP --path-steps-scale 1.2 $CAPARG
echo "END $(date) rc=$?"
