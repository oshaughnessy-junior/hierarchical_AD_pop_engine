#!/bin/bash
# bias check for one run after its chains end: sbatch -M sporc --dependency=afterany:J0:J1:J2:J3 --export=ALL,TAG=TAG bias.sbatch
#SBATCH -A grav-wave-rift
#SBATCH -p tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
source /home/rossma/hier-reassess-20261121/runs/BS_nuts/env.sh
export JAX_PLATFORMS=cpu
D=/home/rossma/hier-reassess-20261121/runs/BS_nuts/$TAG
$PY $E/demo/inference/hier_nuts.py --mode bias --map-json $MAPS/$TAG/result.json --ckpt-dirs "$D/chain*" --outdir $D $INP \
    --key 11 --n-seed 8000 --n-b 6000 --n-draws 20 --bias-seed 0
