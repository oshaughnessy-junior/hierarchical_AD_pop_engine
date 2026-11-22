# After phase 1: for each model, the f_dep with the best held-out lnL -> reeval at n_seed 8000 (keys 11 202 3003).
# For the overall best model at its best f_dep: kappa profile fits at 1.0 2.0 3.5 4.5 (2.7 is the phase-1 fit).
import json, os, sys
R = os.path.expanduser("~/hier-reassess-20261121/runs/BS4D_scan")
FL = "--likelihood native4d --native4d-window --native4d-nodes 32 --m-max-lo 20"
lines = [l.split(" ", 1) for l in open(f"{R}/jobs_b4d.txt") if l.strip() and not l.startswith("#")]
best = {}
for tag, rest in lines:
    p = f"{R}/{tag}/result.json"
    if not os.path.exists(p): print(f"missing {tag}", file=sys.stderr); continue
    r = json.load(open(p)); m = r["model"]; L = r["lnL_heldout_mean"]
    if m not in best or L > best[m][0]: best[m] = (L, tag, r)
for m, (L, tag, r) in best.items():
    print(f"{tag}__reeval --mode reeval --n-seed 8000 --n-b 6000 --keys 11 202 3003 --kappa 2.7 {FL}")
    print(f"# best {m}: {tag} lnL_heldout={L:.2f}", file=sys.stderr)
mtop, (L, tag, r) = max(best.items(), key=lambda kv: kv[1][0])
th = dict(r["theta_map"])
for k in ("1.0", "2.0", "3.5", "4.5"):
    print(f"{tag}_k{k} --model {mtop} --f-dep {r['f_dep']} --init-json {json.dumps(th, separators=(',', ':'))} --kappa {k} {FL}")
print(f"# kappa profile on {tag}", file=sys.stderr)
