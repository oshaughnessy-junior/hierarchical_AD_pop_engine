#!/bin/bash
# one NUTS chain: sbatch -M sporc -J bsnuts-TAG-cC --export=ALL,TAG=TAG,CHAIN=C -o TAG/chainC/slurm_%j.out chain.sbatch
#SBATCH -A grav-wave-rift
#SBATCH -p tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00
source /home/rossma/hier-reassess-20261121/runs/BS_nuts/env.sh
export JAX_PLATFORMS=cpu
D=/home/rossma/hier-reassess-20261121/runs/BS_nuts/$TAG/chain$CHAIN
echo "START $(date) job $SLURM_JOB_ID node $(hostname) tag $TAG chain $CHAIN commit $(git -C $E rev-parse --short HEAD)"
$PY $E/demo/inference/hier_nuts.py --mode nuts --map-json $MAPS/$TAG/result.json --outdir $D --chain-seed $CHAIN $INP \
    --key 11 --n-seed 8000 --n-b 6000 --num-warmup 500 --num-samples 1000 --chunk 25 \
    --max-tree-depth 6 --target-accept 0.8 --dense-mass 1 --progress 1
echo "END $(date) rc=$?"
