# BS_nuts SPORC environment (source me). Engine worktree engine-nuts, branch reassess-nuts-20261127.
W=/home/rossma/hier-reassess-20261121
E=$W/engine-nuts
export HIER_WS=$W
export PYTHONPATH=$E:$E/src:$E/demo/inference:$W/gwkokab-sampled/src:$W/stubs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
PY=$W/env/bin/python
MAPS=$E/demo/inference/bs_nuts_maps
INP="--catalog $W/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz --pdet-grid $W/inputs/mass_pdet_grid_kappa2.7.npz"
