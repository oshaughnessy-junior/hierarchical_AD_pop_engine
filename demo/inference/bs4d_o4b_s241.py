import numpy as np, csv
Z = {}
for p in ("data/scatter_n8000.npz", "data/scatter_n8000b.npz", "data/scatter_n24000.npz"):
    z = np.load(p, allow_pickle=True)
    for k in z.files: Z[k] = z[k]
rows = []
for t in ("BB_f0.3", "BB_f0.4", "BB_f0.3_sBC", "BMIX_f0.3_e1m", "BC_f0.3_e2m"):
    for ns in (8000, 24000):
        ks = sorted({int(k.split("|")[2]) for k in Z if k.startswith(f"{t}|{ns}|") and k.endswith("|lnL")})
        if len(ks) < 2: continue
        nm = [str(x) for x in Z[f"{t}|names"]]; j = nm.index("S241011k")
        I = np.array([Z[f"{t}|{ns}|{k}|lnI"] for k in ks]); E = np.array([Z[f"{t}|{ns}|{k}|ess"] for k in ks]); L = np.array([float(Z[f"{t}|{ns}|{k}|lnL"]) for k in ks])
        wo = L - I[:, j]; rows.append(dict(point=t, n_seed=ns, keys=len(ks), lnL=L.mean(), lnL_sd=L.std(ddof=1), S241011k_lnI=I[:, j].mean(), S241011k_sd=I[:, j].std(ddof=1), S241011k_ess_min=E[:, j].min(), S241011k_ess_max=E[:, j].max(), lnL_without=wo.mean(), lnL_without_sd=wo.std(ddof=1)))
with open("s241011k_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); [w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]
for r in rows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
b = {(r["point"], r["n_seed"]): r for r in rows}
for ns in (8000, 24000):
    for a, c in (("BB_f0.3_sBC", "BMIX_f0.3_e1m"), ("BB_f0.3_sBC", "BB_f0.4"), ("BB_f0.3_sBC", "BC_f0.3_e2m")):
        if (a, ns) in b and (c, ns) in b: print(ns, a, "-", c, "with %.2f (sd %.2f) without S241011k %.2f (sd %.2f)" % (b[(a, ns)]["lnL"] - b[(c, ns)]["lnL"], np.hypot(b[(a, ns)]["lnL_sd"], b[(c, ns)]["lnL_sd"]), b[(a, ns)]["lnL_without"] - b[(c, ns)]["lnL_without"], np.hypot(b[(a, ns)]["lnL_without_sd"], b[(c, ns)]["lnL_without_sd"])))
