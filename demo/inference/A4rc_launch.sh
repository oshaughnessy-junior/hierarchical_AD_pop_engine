#!/usr/bin/env bash
# A4rc: headline combined model (REFRC: exact, recentred cascade) refit with the revised
# 4D event likelihoods, kappa fixed 2.7.  Two depths: f_dep 0.5 (REFRC best on 3D NAL)
# and 0.7 (headline).  Start: B's REFRC MAP at that depth (runs/B_scan/REFRC_f*/result.json),
# with log_R shifted by -ln(T_obs) + ln(mu_old/mu_new) = -0.6365 into native4d rate units.
# MAP at n_seed 3000/3000 (Adam stages + recentring + L-BFGS + Laplace + 3 held-out keys),
# then reeval of the optimum at 8000/6000 on keys 11 202 3003.
# Resumable: rerun this script; each fit resumes from runs/A4rc_f*/state.npz and a
# finished fit (result.json present) skips straight to the reeval.
# Cores: f0.5 on 0-3, f0.7 on 4-7.  Memory cap 8 GB per process.
set -u
W=$HOME/20261120-hier-reassess
cd "$W"
source runs/A4rc_env.sh
DRV=$E/demo/inference/hier_map_laplace.py
COMMON="--model REFRC --likelihood native4d --kappa 2.7 --native4d-window --native4d-nodes 32"

init_for () {
  $PY - "$1" <<'EOF'
import json, sys
r = json.load(open(f"runs/B_scan/REFRC_f{sys.argv[1]}/result.json"))
th = dict(r["theta_map"])
for k in ("log_R_A", "log_R_B"):
    th[k] += -0.6365
print(json.dumps(th))
EOF
}

run_one () {
  local f=$1 cores=$2
  local out=runs/A4rc_f$f log=logs/A4rc_f$f.log
  mkdir -p "$out"
  (
    echo "$(date) START f_dep=$f cores=$cores"
    if [ ! -f "$out/result.json" ]; then
      prlimit --as=8589934592 taskset -c "$cores" $PY $DRV --mode fit $COMMON --f-dep "$f" \
        --n-seed 3000 --n-b 3000 --key 11 --heldout-keys 202 3003 40004 \
        --init-json "$(init_for "$f")" --tag "B_REFRC_f$f" --outdir "$out" || { echo "$(date) FIT FAILED rc=$?"; exit 1; }
    fi
    prlimit --as=8589934592 taskset -c "$cores" $PY $DRV --mode reeval $COMMON \
      --n-seed 8000 --n-b 6000 --keys 11 202 3003 --outdir "$out" || { echo "$(date) REEVAL FAILED rc=$?"; exit 1; }
    echo "$(date) END f_dep=$f"
  ) >> "$log" 2>&1
}

run_one 0.5 0-3 &
echo $! > logs/A4rc_f0.5.pid
run_one 0.7 4-7 &
echo $! > logs/A4rc_f0.7.pid
wait
