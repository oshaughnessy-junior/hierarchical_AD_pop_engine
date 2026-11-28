#!/bin/bash
# parity + timing of the path-weighted cascade on a tier3 CPU node: sbatch -M sporc --export=ALL,TAG=TAG,PP=P,CAP=X probe/parity_path.sbatch
#SBATCH -J bsnuts-probe-path
#SBATCH -A grav-wave-rift
#SBATCH -p tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH -o /home/rossma/hier-reassess-20261121/runs/BS_nuts/probe/%x_%j.out
source /home/rossma/hier-reassess-20261121/runs/BS_nuts/env.sh
export JAX_PLATFORMS=cpu
lscpu | grep "Model name"; /usr/bin/time -v true 2>/dev/null | head -0
CAPARG=""; [ -n "$CAP" ] && [ "$CAP" != none ] && CAPARG="--m-max-cap $CAP"
$PY $E/demo/inference/hier_nuts.py --mode parity --map-json $MAPS/$TAG/result.json --outdir /home/rossma/hier-reassess-20261121/runs/BS_nuts/probe/path_$TAG $INP --n-time 3 \
    --path-proposal $PP --path-steps-scale 1.2 $CAPARG
