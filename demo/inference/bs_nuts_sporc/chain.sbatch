#!/bin/bash
# one NUTS chain, path-weighted cascade (runs/BS_weights on cardassia):
#   sbatch -M sporc -J bsnuts-TAG-cC --export=ALL,TAG=TAG,CHAIN=C,PP=PROPOSAL,CAP=X -o TAG/chainC/slurm_%j.out chain.sbatch
# PP: defensive (exchange off) or exact (exchange on, MAP-pinned); CAP: m_max_A prior upper edge, or "none".
#SBATCH -A grav-wave-rift
#SBATCH -p tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
source /home/rossma/hier-reassess-20261121/runs/BS_nuts/env.sh
export JAX_PLATFORMS=cpu
D=/home/rossma/hier-reassess-20261121/runs/BS_nuts/$TAG/chain$CHAIN
CAPARG=""; [ -n "$CAP" ] && [ "$CAP" != none ] && CAPARG="--m-max-cap $CAP"
echo "START $(date) job $SLURM_JOB_ID node $(hostname) tag $TAG chain $CHAIN commit $(git -C $E rev-parse --short HEAD) proposal $PP cap ${CAP:-none}"
$PY $E/demo/inference/hier_nuts.py --mode nuts --map-json $MAPS/$TAG/result.json --outdir $D --chain-seed $CHAIN $INP \
    --key 11 --n-seed 8000 --n-b 6000 --num-warmup 500 --num-samples 1000 --chunk 25 \
    --max-tree-depth 6 --target-accept 0.8 --dense-mass 1 --progress 1 \
    --path-proposal $PP --path-steps-scale 1.2 $CAPARG
echo "END $(date) rc=$?"
