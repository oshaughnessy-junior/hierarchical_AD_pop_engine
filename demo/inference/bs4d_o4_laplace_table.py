"""n_seed 8000 Laplace table for the O4-only fits, from the per-key JSON of bs4d_o4_laplace.py.

Per point and key (volume terms only where -H is positive definite at that key):
  lnZ_bdy  = lnL + d/2 ln 2pi - 1/2 ln det(-H) - sum ln widths - n_rail ln 2   (the driver's lnZ_laplace_bdy;
             rails = parameters within 1% of a bound; each is treated as a half Gaussian with its mode at the bound);
  lnZ_box  = same Gaussian without the ln 2, multiplied by its probability inside the prior box (mode at theta);
  lnZ_boxg = as lnZ_box, but the quadratic uses the gradient at theta too (mode at theta + (-H)^-1 g);
  lnZ_fix  = railed parameters held at their values: one Gaussian dimension and one width fewer per rail.
lnL is the mean over keys. Non-PD keys give no volume: the point is marked and lnZ is not reported as a number.
Usage: bs4d_o4_laplace_table.py LAPDIR SCANDIR[,SCANDIR2] OUT_CSV TAG [TAG ...]"""
import csv, glob, json, math, os, sys
import numpy as np
from scipy.stats import norm

LAP, SCANS, OUT, TAGS = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4:]
KEYS = (11, 202, 3003)


def ghk(mu, C, lo, hi, n, seed):
    """GHK estimate of ln P(lo < x < hi), x ~ N(mu, C): sequential truncated-normal importance sampling."""
    d = len(mu); L = np.linalg.cholesky(C); rng = np.random.default_rng(seed); u = rng.random((n, d)); z = np.zeros((n, d)); lw = np.zeros(n)
    for i in range(d):
        m = (mu[i] + z[:, :i] @ L[i, :i]); a_, b_ = (lo[i] - m) / L[i, i], (hi[i] - m) / L[i, i]
        pa, pb = norm.cdf(a_), norm.cdf(b_); pr = np.maximum(pb - pa, 1e-300); lw += np.log(pr)
        z[:, i] = norm.ppf(np.clip(pa + u[:, i] * pr, 1e-300, 1 - 1e-16))
    return float(np.log(np.mean(np.exp(lw - lw.max()))) + lw.max())


def box_lnP(mu, C, lo, hi):
    v = [ghk(mu, C, lo, hi, 40000, s) for s in (1, 2)]; return float(np.log(np.mean(np.exp(v)))) , float(abs(v[0] - v[1]))


def result(tag):
    for s in SCANS:
        if os.path.exists(f"{s}/{tag}/result.json"): return json.load(open(f"{s}/{tag}/result.json"))


rows = []
for tag in TAGS:
    R = result(tag); per = []
    for k in KEYS:
        p = f"{LAP}/{tag}_n8000_k{k}.json"
        if not os.path.exists(p): continue
        d = json.load(open(p)); names = d["names"]; n = len(names); th = np.array(d["theta"]); lo = np.array([d["bounds"][x][0] for x in names]); hi = np.array([d["bounds"][x][1] for x in names])
        w = hi - lo; A = -np.array(d["hessian"]); g = np.array(d["grad"]); ev = np.linalg.eigvalsh(A); pd = bool(np.all(ev > 0)); rails = [x for i, x in enumerate(names) if min(th[i] - lo[i], hi[i] - th[i]) < 0.01 * w[i]]
        e = dict(key=k, lnL=d["lnL"], pd=pd, eig0=float(ev[0]), neff=d["selection"]["neff"], rails=rails)
        if pd:
            lv = 0.5 * n * math.log(2 * math.pi) - 0.5 * float(np.sum(np.log(ev))) - float(np.sum(np.log(w))); Ci = np.linalg.inv(A)
            free = [i for i, x in enumerate(names) if x not in rails]
            Af = A[np.ix_(free, free)]; e.update(vol=lv, vol_bdy=lv - len(rails) * math.log(2))
            e["vol_fix"] = 0.5 * len(free) * math.log(2 * math.pi) - 0.5 * np.linalg.slogdet(Af)[1] - float(np.sum(np.log(w[free])))
            lp0, er0 = box_lnP(th, Ci, lo, hi); step = Ci @ g; dec = 0.5 * float(g @ step); lpg, erg = box_lnP(th + step, Ci, lo, hi)
            e.update(vol_box=lv + lp0, vol_boxg=lv + dec + lpg, newton_decrement=dec, box_err=max(er0, erg))
        per.append(e)
    if len(per) < len(KEYS): print("incomplete", tag, len(per)); continue
    L = np.array([e["lnL"] for e in per]); allpd = all(e["pd"] for e in per)
    m = lambda x: float(np.mean(x)); s = lambda x: float(np.std(x, ddof=1))
    row = dict(tag=tag, model=R["model"], f_dep=R["f_dep"], rails="/".join(per[0]["rails"]), n_pd=sum(e["pd"] for e in per), n_keys=len(per),
               eig0_min=min(e["eig0"] for e in per), eig0_max=max(e["eig0"] for e in per), lnL_n8000=m(L), lnL_sd_keys=s(L),
               lnL_n3000_k11=R["lnL"], lnZ_bdy_n3000_k11=R["lnZ_laplace_bdy"], dlnL_part=m(L) - R["lnL"], neff_min=min(e["neff"] for e in per))
    if allpd:
        for c in ("vol_bdy", "vol_box", "vol_boxg", "vol_fix", "newton_decrement"):
            row[c] = m([e[c] for e in per])
        z = np.array([e["lnL"] + e["vol_bdy"] for e in per]); row["lnZ_bdy_sd_keys"] = s(z); row["dvol_part"] = row["vol_bdy"] - (R["lnZ_laplace_bdy"] - R["lnL"])
        for c in ("bdy", "box", "boxg", "fix"): row[f"lnZ_{c}"] = row["lnL_n8000"] + row[f"vol_{c}"]
        row["box_err"] = max(e["box_err"] for e in per)
    else:
        row["note"] = "non-PD at " + ",".join(str(e["key"]) for e in per if not e["pd"]) + ": lnZ not reported"
    rows.append(row)
cols = ["tag", "model", "f_dep", "rails", "n_pd", "n_keys", "eig0_min", "eig0_max", "lnL_n8000", "lnL_sd_keys", "lnZ_bdy", "lnZ_bdy_sd_keys", "lnZ_box", "lnZ_boxg", "lnZ_fix",
        "newton_decrement", "box_err", "dlnL_part", "dvol_part", "lnZ_bdy_n3000_k11", "lnL_n3000_k11", "vol_bdy", "vol_box", "vol_boxg", "vol_fix", "neff_min", "note"]
with open(OUT, "w", newline="") as f:
    w_ = csv.DictWriter(f, cols, extrasaction="ignore"); w_.writeheader(); [w_.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]
for r in rows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in cols})
