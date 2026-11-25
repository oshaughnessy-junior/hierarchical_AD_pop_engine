"""Per-event sampler-key scatter of ln I at fixed points (bs4d_bb_scatter.py output).
Usage: bb_scatter_analyse.py OUT_CSV NPZ [NPZ ...]"""
import csv, sys
import numpy as np

OUT = sys.argv[1]; Z = {}
for p in sys.argv[2:]:
    z = np.load(p, allow_pickle=True)
    for k in z.files: Z[k] = z[k]
tags = sorted({k.split("|")[0] for k in Z if "|lnL" in k}); nss = sorted({int(k.split("|")[1]) for k in Z if "|lnL" in k})
rows = []
def get(t, ns, key, f): return Z[f"{t}|{ns}|{key}|{f}"]
def keys(t, ns): return sorted(int(k.split("|")[2]) for k in Z if k.startswith(f"{t}|{ns}|") and k.endswith("|lnL"))
tot = {}
for t in tags:
    for ns in nss:
        ks = keys(t, ns)
        if len(ks) < 2: continue
        L = np.array([float(get(t, ns, k, "lnL")) for k in ks]); I = np.array([get(t, ns, k, "lnI") for k in ks]); E = np.array([get(t, ns, k, "ess") for k in ks])
        sd = I.std(0, ddof=1); order = np.argsort(-sd); names = Z[f"{t}|names"]
        top = [dict(event=str(names[j]), sd_lnI=float(sd[j]), mean_lnI=float(I[:, j].mean()), min_ess=float(E[:, j].min()), max_ess=float(E[:, j].max())) for j in order[:5]]
        tot[(t, ns)] = dict(mean=L.mean(), sd=L.std(ddof=1), se=L.std(ddof=1) / np.sqrt(len(L)), keys=len(L), sumI_sd_quad=float(np.sqrt((sd ** 2).sum())), top=top, n_ess_lt10=int(np.sum(E.min(0) < 10)), I=I.mean(0), sdI=sd, names=names)
        print(f"{t:16s} n{ns:<6d} keys {len(L)} lnL mean {L.mean():.2f} sd {L.std(ddof=1):.2f} (se {L.std(ddof=1)/np.sqrt(len(L)):.2f}); events with ESS<10 in some key: {int(np.sum(E.min(0) < 10))}; quad-sum of per-event sd {np.sqrt((sd**2).sum()):.2f}; top5 share of variance {(np.sort(sd**2)[::-1][:5].sum()/(sd**2).sum()):.2f}")
        for r in top: print("      ", r["event"], "sd lnI %.2f mean %.2f ESS %.1f-%.1f" % (r["sd_lnI"], r["mean_lnI"], r["min_ess"], r["max_ess"]))
        for r in top: rows.append(dict(point=t, n_seed=ns, n_keys=len(L), lnL_mean=float(L.mean()), lnL_sd=float(L.std(ddof=1)), **r))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); [w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]
def cmp(a, b, ns):
    if (a, ns) not in tot or (b, ns) not in tot: return
    d = tot[(a, ns)]["I"] - tot[(b, ns)]["I"]; o = np.argsort(-np.abs(d))
    print(f"n{ns}: sum lnI {a} - {b} = {d.sum():+.2f}; top events:", [(str(tot[(a, ns)]["names"][j]), round(float(d[j]), 2)) for j in o[:8]], "events with |d|>2:", int((np.abs(d) > 2).sum()))
for ns in nss:
    cmp("BB_f0.3_sBC", "BB_f0.4", ns); cmp("BB_f0.3_sBC", "BMIX_f0.3_e1m", ns); cmp("BB_f0.3_sBC", "BC_f0.3_e2m", ns); cmp("BB_f0.4", "BC_f0.3_e2m", ns)
