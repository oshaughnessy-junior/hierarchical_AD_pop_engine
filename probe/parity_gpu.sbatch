#!/bin/bash
#SBATCH -J nuts-probe-gpu
#SBATCH -A grav-wave-rift
#SBATCH -p onboard
#SBATCH --qos=coll_college_of_science
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /home/rossma/hier-reassess-20261121/runs/BS_nuts/probe/%x_%j.out
source /home/rossma/hier-reassess-20261121/runs/BS_nuts/env.sh
nvidia-smi -L
$PY -c "import jax; print(jax.devices())"
$PY $E/demo/inference/hier_nuts.py --mode parity --map-json $MAPS/NULL_floor40/result.json --outdir /home/rossma/hier-reassess-20261121/runs/BS_nuts/probe/gpu $INP --n-time 5
