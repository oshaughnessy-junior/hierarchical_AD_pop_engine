#!/bin/bash
#SBATCH -J nuts-probe-cpu
#SBATCH -A grav-wave-rift
#SBATCH -p tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH -o /home/rossma/hier-reassess-20261121/runs/BS_nuts/probe/%x_%j.out
source /home/rossma/hier-reassess-20261121/runs/BS_nuts/env.sh
export JAX_PLATFORMS=cpu
lscpu | grep "Model name"
$PY $E/demo/inference/hier_nuts.py --mode parity --map-json $MAPS/NULL_floor40/result.json --outdir /home/rossma/hier-reassess-20261121/runs/BS_nuts/probe/cpu $INP --n-time 3
