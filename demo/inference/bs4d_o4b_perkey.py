import csv, glob, json, os, math, numpy as np
rows = []
for p in sorted(glob.glob("lap/*_n8000_k*.json")):
    d = json.load(open(p)); A = -np.array(d["hessian"]); ev, V = np.linalg.eigh(A); v = V[:, 0]; o = np.argsort(-np.abs(v)); pd = bool(ev[0] > 0); n = len(v)
    rows.append(dict(tag=d["tag"], key=d["key"], lnL=d["lnL"], pd=pd, eig0=float(ev[0]), eig1=float(ev[1]), dominant=d["names"][o[0]], dom_weight=float(abs(v[o[0]])), second=d["names"][o[1]], neff=d["selection"]["neff"], spin_chi_weight=float(abs(v[d["names"].index("spin_chi")])), lnZ_bdy_if_pd=(d["lnZ_laplace_bdy"] if pd else None), grad_spin_chi=d["grad"][d["names"].index("spin_chi")]))
with open("laplace_per_key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); [w.writerow({k: (f"{x:.4f}" if isinstance(x, float) else x) for k, x in r.items()}) for r in rows]
from collections import Counter
print("non-PD keys:", sum(not r["pd"] for r in rows), "of", len(rows)); print("dominant param at non-PD:", Counter(r["dominant"] for r in rows if not r["pd"]))
for r in rows:
    if not r["pd"]: print(f"{r['tag']:16s} k{r['key']:<5d} eig0 {r['eig0']:11.1f} eig1 {r['eig1']:9.2f} dom {r['dominant']} ({r['dom_weight']:.2f}) neff {r['neff']:.0f}")
