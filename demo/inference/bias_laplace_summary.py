"""Combine bias_laplace.py outputs over keys: bias = mean_k(pinned - exact) at each point."""
import glob, json, os, sys
import numpy as np
d = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(d, "*_k*.json"))):
    r = json.load(open(f))
    for x in r["rows"]:
        rows.setdefault((r["tag"], x["label"]), []).append((r["key"], x))
out = []
print(f"{'tag':14s} {'point':7s} {'d':>5s} {'spin':>6s} {'mmax':>6s} {'bias':>7s} {'se':>5s} {'sd_pin':>6s} {'sd_exa':>6s} diffs(k11,k202,k3003)")
for (tag, lab), v in sorted(rows.items(), key=lambda t: (t[0][0], t[1][0][1]["mahal"], t[0][1])):
    v = sorted(v, key=lambda t: t[0])
    P = np.array([x["lnL_pinned"] for _, x in v]); E = np.array([x["lnL_exact"] for _, x in v])
    D = P - E; n = len(D)
    x0 = v[0][1]
    rec = dict(tag=tag, label=lab, mahal=x0["mahal"], keys=[k for k, _ in v], spin_chi=x0["theta"]["spin_chi"],
               m_max_A=x0["theta"]["m_max_A"], pinned=P.tolist(), exact=E.tolist(), diff=D.tolist(),
               bias=float(D.mean()), bias_se=float(D.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
               sd_pinned=float(P.std(ddof=1)) if n > 1 else None, sd_exact=float(E.std(ddof=1)) if n > 1 else None)
    out.append(rec)
    print(f"{tag:14s} {lab:7s} {rec['mahal']:5.2f} {rec['spin_chi']:6.3f} {rec['m_max_A']:6.2f} {rec['bias']:+7.2f} "
          f"{rec['bias_se'] or 0:5.2f} {rec['sd_pinned'] or 0:6.2f} {rec['sd_exact'] or 0:6.2f} " + " ".join(f"{x:+.2f}" for x in D))
json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
