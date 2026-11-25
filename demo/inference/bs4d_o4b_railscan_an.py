import numpy as np, csv, json, math
z = np.load("data/railscan.npz", allow_pickle=True); keys = sorted({k.split("|")[0] + "|" + k.split("|")[1] for k in z.files if k.count("|") == 2 and k.split("|")[2].isdigit()}); rows = []
for tp in keys:
    tag, par = tp.split("|"); g = z[f"{tp}|grid"]; A = z[f"{tp}|A"]; lo, hi = z[f"{tp}|lohi"]; W = hi - lo
    ip = None; import json as J
    d = J.load(open(f"lap/{tag}_n8000_k11.json")); ip = d["names"].index(par); C = np.linalg.inv(A) if np.all(np.linalg.eigvalsh(A) > 0) else None
    pm = float(z[f"{tp}|theta"][ip]); rest = [i for i in range(len(A)) if i != ip]
    lam = A[ip, ip] - A[ip, rest] @ np.linalg.solve(A[np.ix_(rest, rest)], A[rest, ip])     # Schur complement: curvature along p after profiling
    sg = 1 / math.sqrt(lam) if lam > 0 else float("nan"); half = math.log(0.5 * math.sqrt(2 * math.pi) * sg / W) if lam > 0 else float("nan")
    for k in (11, 202, 3003):
        y = z[f"{tp}|{k}"]; m = np.isfinite(y) & (y > -1e9); yy = y[m]; gg = g[m]
        y0 = float(np.interp(pm, gg, yy)); fine = np.linspace(gg[0], gg[-1], 4001); Y = np.interp(fine, gg, yy) - y0
        ln_int = math.log(np.trapezoid(np.exp(np.clip(Y, -700, 50)), fine) / W)
        rows.append(dict(tag=tag, param=par, key=k, p_map=pm, lo=lo, hi=hi, lnL_drop_at_far_end=float(yy.max() - yy[-1]), argmax=float(gg[np.argmax(yy)]), scan_lnVol=ln_int, halfgauss_lnVol=half, sigma_profile=sg, excess=ln_int - half))
with open("railscan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); [w.writerow({a: (f"{b:.4f}" if isinstance(b, float) else b) for a, b in r.items()}) for r in rows]
for r in rows: print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})
