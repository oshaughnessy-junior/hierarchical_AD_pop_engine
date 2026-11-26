#!/bin/bash
#SBATCH -J s241-fitterm
#SBATCH -A grav-wave-rift
#SBATCH -p onboard
#SBATCH --qos=coll_college_of_science
#SBATCH --gres=shard:a100:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:59:00
#SBATCH --array=0-7%4
#SBATCH -o /home/rossma/hier-reassess-20261121/runs/BS4D_O4_ms/S241011k/fitterm_%a_%j.out
W=$HOME/hier-reassess-20261121; source $W/env.sh; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
$PY -c "import jax; assert jax.devices()[0].platform == 'gpu'" || exit 75
F=$W/runs/BS4D_O4_ms/fits
P=(BB_ALL168_b1_f0.3_xp BMIX_ALL168_b1_f0.3 BD_ALL168_b1_f0.3 BD_ALL168_b2_f0.4 REFRC_ALL168_b1_f0.3 BC_ALL168_b1_f0.4_xm BB_NO241011_b3_f0.3 BB_ALL168_b3_f0.3)
T=${P[$SLURM_ARRAY_TASK_ID]}
$PY $E/demo/inference/bs4d_s241_fitterm.py $W/runs/BS4D_O4_ms/S241011k/fitterm_$T.npz $F/$T/result.json 2>&1 | grep -v -E "Warning|warn\(|INJECTION|NATIVE"
