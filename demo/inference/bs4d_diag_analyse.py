"""Analyse the fixed-parameter evaluations of bs4d_diag_eval.py (keys 11/202/3003, n_seed 8000).
4D values use the exact per-event integrals (no Mc windows); window_bias.csv compares with the windowed
values used by the fits.

Writes runs/BS4D_diag/{decomposition,events_all,events_top_d4,events_top_d4m3,robustness,
param_vs_likelihood,detected_summary}.csv, detected_distributions.png, delta_vs_properties.png, README.md.
Event partition (fixed by the 4D tables, applied to both likelihoods): "4D pass" = primary-tier 4D fit with
validation pass_all_three_declared; "4D fail" = primary-tier 4D fit failing a declared check; "3D fallback" =
no primary 4D fit (3D NAL in both likelihoods). The rate/selection term is N ln T - mu (4D) or -mu (3D).
"""
import csv, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

W = os.path.expanduser("~/20261120-hier-reassess"); OUT = f"{W}/runs/BS4D_diag"
KEYS = [11, 202, 3003]
ZW = {k: np.load(f"{OUT}/diag_key{k}.npz", allow_pickle=True) for k in KEYS}            # windowed (as in the fits)
ZN = {k: np.load(f"{OUT}/diag_key{k}_nowin.npz", allow_pickle=True) for k in KEYS}      # exact, no Mc windows
Z = ZN
cat = np.load(f"{W}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz", allow_pickle=True)
tab = np.load(f"{W}/runs/A_tables/native4d_events.npz", allow_pickle=True)
names = [str(n) for n in cat["names"]]; NI = {n: i for i, n in enumerate(names)}; NE = len(names)
prim = tab["tier"] == "primary"
val = {str(n): str(v) for n, v in zip(tab["name"][prim], tab["validation"][prim])}
zmed = {str(n): float(q[1]) for n, q in zip(tab["name"][prim], tab["z_q"][prim])}
group = np.array(["3D fallback"] * NE, dtype=object)
for n, v in val.items(): group[NI[n]] = "4D pass" if v == "pass_all_three_declared" else "4D fail"
GROUPS = ["4D pass", "4D fail", "3D fallback"]


def g(k, tag, f, win=False):
    src = ZW if (win or tag.startswith("3D")) else ZN
    return src[k][f"{tag}|{f}"]


def ev4(tag):
    """per key: lnL, rate term, per-event logI/ESS in catalog order, NAL logI at same draws."""
    rows = []
    for k in KEYS:
        lI = np.full(NE, np.nan); es = np.full(NE, np.nan)
        for nm, li, e in zip(g(k, tag, "names4"), g(k, tag, "logI4"), g(k, tag, "ess4")): lI[NI[str(nm)]] = li; es[NI[str(nm)]] = e
        for nm, li, e in zip(g(k, tag, "names3"), g(k, tag, "logI3"), g(k, tag, "ess3")): lI[NI[str(nm)]] = li; es[NI[str(nm)]] = e
        n, T, mu = int(g(k, tag, "n")), float(g(k, tag, "T")), float(g(k, tag, "mu"))
        rows.append(dict(lnL=float(g(k, tag, "lnL")), lnL_parts=float(g(k, tag, "lnL_parts")), rate=n * np.log(T) - mu,
                         mu=mu, logI=lI, ess=es, nal=np.asarray(g(k, tag, "nal_logI")), lnT=np.log(T),
                         overflow=int(g(k, tag, "overflow"))))
    return rows


def ev3(tag):
    return [dict(lnL=float(g(k, tag, "lnL")), lnL_parts=float(g(k, tag, "lnL_parts")), rate=-float(g(k, tag, "mu")),
                 mu=float(g(k, tag, "mu")), logI=np.asarray(g(k, tag, "logI")), ess=np.asarray(g(k, tag, "ess")))
            for k in KEYS]


R = {t: ev4(t) for t in ("4D_BD", "4D_REFRC", "4D_BMIX", "X_BD", "X_REFRC")}
R.update({t: ev3(t) for t in ("3D_BD", "3D_REFRC", "3D_BMIX")})
chk = {t: max(abs(r["lnL"] - r["lnL_parts"]) for r in rows) for t, rows in R.items()}
chk_sum = {t: max(abs(r["rate"] + np.sum(r["logI"]) - r["lnL"]) for r in rows) for t, rows in R.items()}
mean = lambda xs: float(np.mean(xs)); sd = lambda xs: float(np.std(xs, ddof=1))


def parts(t):
    """per-key component values: rate, per group sums, total."""
    out = []
    for r in R[t]:
        d = dict(rate=r["rate"], total=r["lnL"])
        for gname in GROUPS: d[gname] = float(np.sum(r["logI"][group == gname]))
        out.append(d)
    return out


def diff(a, b):
    pa, pb = parts(a), parts(b)
    return {c: [x[c] - y[c] for x, y in zip(pa, pb)] for c in ["rate"] + GROUPS + ["total"]}


DEC = [("4D (v3 optima)", "4D_BD", "4D_REFRC"), ("3D (union optima)", "3D_BD", "3D_REFRC"),
       ("4D at 3D optima", "X_BD", "X_REFRC"), ("4D Bmix-REFRC", "4D_BMIX", "4D_REFRC"), ("3D Bmix-REFRC", "3D_BMIX", "3D_REFRC")]
rows = []
for lab, a, b in DEC:
    d = diff(a, b); comp = sum(np.array(d[c]) for c in ["rate"] + GROUPS)
    rows.append(dict(comparison=lab, **{f"d_{c}": mean(d[c]) for c in d}, **{f"sd_{c}": sd(d[c]) for c in d},
                     sum_check=float(np.max(np.abs(comp - np.array(d["total"]))))))
cols = list(rows[0])
with open(f"{OUT}/decomposition.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); [w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]

# per event
mI = lambda t: np.mean([r["logI"] for r in R[t]], 0)
mE = lambda t: np.min([r["ess"] for r in R[t]], 0)
d4 = mI("4D_BD") - mI("4D_REFRC"); d3 = mI("3D_BD") - mI("3D_REFRC"); dd = d4 - d3
ess4 = np.minimum(mE("4D_BD"), mE("4D_REFRC")); ess3 = np.minimum(mE("3D_BD"), mE("3D_REFRC"))
d4sd = np.std([a["logI"] - b["logI"] for a, b in zip(R["4D_BD"], R["4D_REFRC"])], 0, ddof=1)
C = np.asarray(cat["centers"])
ev = []
for i, n in enumerate(names):
    ev.append(dict(event=n, source="4D" if group[i] != "3D fallback" else "3D", group=group[i],
                   validation=val.get(n, "no primary 4D fit"), Mc=C[i, 0], eta=C[i, 1], chi_eff=C[i, 2],
                   z=zmed.get(n, np.nan), d4=d4[i], d4_sd_keys=d4sd[i], d3=d3[i], d4_minus_d3=dd[i],
                   ess_4D=ess4[i], ess_3D=ess3[i], low_ess=bool(min(ess4[i], ess3[i]) < 10)))
ecols = list(ev[0])
def wcsv(path, rs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, ecols); w.writeheader()
        for r in rs: w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
wcsv(f"{OUT}/events_all.csv", ev)
top4 = sorted(ev, key=lambda r: -abs(r["d4"]))[:15]; topd = sorted(ev, key=lambda r: -abs(r["d4_minus_d3"]))[:15]
wcsv(f"{OUT}/events_top_d4.csv", top4); wcsv(f"{OUT}/events_top_d4m3.csv", topd)

# robustness at the fixed 4D optima
fail = group == "4D fail"; top5 = np.argsort(-np.abs(d4))[:5]
rob = []
for lab, a, b in (("B_d - REFRC", "4D_BD", "4D_REFRC"), ("Bmix - REFRC", "4D_BMIX", "4D_REFRC")):
    base = [x["lnL"] - y["lnL"] for x, y in zip(R[a], R[b])]
    swap = [(x["lnL"] - np.sum(x["logI"][fail]) + np.sum(x["nal"][fail])) - (y["lnL"] - np.sum(y["logI"][fail]) + np.sum(y["nal"][fail]))
            for x, y in zip(R[a], R[b])]
    drop = [(x["lnL"] - np.sum(x["logI"][top5])) - (y["lnL"] - np.sum(y["logI"][top5])) for x, y in zip(R[a], R[b])]
    rob.append(dict(comparison=lab, baseline=mean(base), baseline_sd=sd(base), failing_on_NAL=mean(swap), failing_on_NAL_sd=sd(swap),
                    drop_top5=mean(drop), drop_top5_sd=sd(drop), n_failing=int(fail.sum()), top5=";".join(names[i] for i in top5)))
with open(f"{OUT}/robustness.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rob[0])); w.writeheader(); [w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in rob]

# parameter vs likelihood
pvl = dict(gap_4D_at_4D_optima=mean(diff("4D_BD", "4D_REFRC")["total"]), gap_4D_at_3D_optima=mean(diff("X_BD", "X_REFRC")["total"]),
           gap_3D_at_3D_optima=mean(diff("3D_BD", "3D_REFRC")["total"]),
           lnL_4D_BD_at_3D_opt=mean([r["lnL"] for r in R["X_BD"]]), lnL_4D_BD_at_4D_opt=mean([r["lnL"] for r in R["4D_BD"]]),
           lnL_4D_REFRC_at_3D_opt=mean([r["lnL"] for r in R["X_REFRC"]]), lnL_4D_REFRC_at_4D_opt=mean([r["lnL"] for r in R["4D_REFRC"]]))
with open(f"{OUT}/param_vs_likelihood.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["quantity", "value"]); [w.writerow([k, f"{v:.3f}"]) for k, v in pvl.items()]

# predicted detected distributions (key 11)
def det(t):
    k = 11; w = np.asarray(Z[k][f"{t}|det_w"]); x = np.asarray(Z[k][f"{t}|det_x"])
    zg = np.asarray(Z[k][f"{t}|det_z_grid"]); zw = np.asarray(Z[k][f"{t}|det_z_w"])
    return w / w.sum(), x, zg, zw / zw.sum()
def wq(v, w, q): o = np.argsort(v); c = np.cumsum(w[o]); return float(np.interp(q, c / c[-1], v[o]))
cz = np.array([zmed[n] for n in names if n in zmed])
summ = []
for t, lab in (("4D_BD", "B_d"), ("4D_REFRC", "REFRC"), ("4D_BMIX", "Bmix")):
    w, x, zg, zw = det(t)
    mchi = float(np.sum(w * x[:, 2])); schi = float(np.sqrt(np.sum(w * (x[:, 2] - mchi) ** 2)))
    summ.append(dict(model=lab, Mc_q16=wq(x[:, 0], w, .16), Mc_q50=wq(x[:, 0], w, .5), Mc_q84=wq(x[:, 0], w, .84),
                     frac_Mc_lt_10=float(np.sum(w[x[:, 0] < 10])), eta_q50=wq(x[:, 1], w, .5), frac_eta_gt_0p24=float(np.sum(w[x[:, 1] > 0.24])),
                     chi_mean=mchi, chi_sd=schi, z_q50=wq(zg, zw, .5), z_q90=wq(zg, zw, .9)))
summ.append(dict(model="catalog medians", Mc_q16=float(np.quantile(C[:, 0], .16)), Mc_q50=float(np.median(C[:, 0])), Mc_q84=float(np.quantile(C[:, 0], .84)),
                 frac_Mc_lt_10=float(np.mean(C[:, 0] < 10)), eta_q50=float(np.median(C[:, 1])), frac_eta_gt_0p24=float(np.mean(C[:, 1] > 0.24)),
                 chi_mean=float(np.mean(C[:, 2])), chi_sd=float(np.std(C[:, 2])), z_q50=float(np.median(cz)), z_q90=float(np.quantile(cz, .9))))
with open(f"{OUT}/detected_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(summ[0])); w.writeheader(); [w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()}) for r in summ]

fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
sty = {"4D_BD": ("B_d", "C0"), "4D_REFRC": ("REFRC", "C3")}
for t, (lab, col) in sty.items():
    w, x, zg, zw = det(t)
    ax[0, 0].hist(np.log10(x[:, 0]), bins=60, weights=w, density=True, histtype="step", color=col, label=lab)
    ax[0, 1].hist(x[:, 1], bins=np.linspace(0.05, 0.25, 41), weights=w, density=True, histtype="step", color=col, label=lab)
    ax[1, 0].hist(x[:, 2], bins=np.linspace(-0.5, 0.8, 66), weights=w, density=True, histtype="step", color=col, label=lab)
    ax[1, 1].plot(zg, np.cumsum(zw), color=col, label=lab)
ax[0, 0].hist(np.log10(C[:, 0]), bins=30, density=True, histtype="stepfilled", color="0.8", label="catalog medians", zorder=0)
ax[0, 1].hist(C[:, 1], bins=np.linspace(0.05, 0.25, 41), density=True, histtype="stepfilled", color="0.8", label="catalog medians", zorder=0)
ax[1, 0].hist(C[:, 2], bins=np.linspace(-0.5, 0.8, 66), density=True, histtype="stepfilled", color="0.8", label="catalog medians", zorder=0)
ax[1, 1].plot(np.sort(cz), np.arange(1, len(cz) + 1) / len(cz), color="0.5", ls="--", label="catalog z medians (4D-fit events)")
for a_, xl in zip(ax.flat, [r"$\log_{10}\mathcal{M}_c\,[M_\odot]$ (source)", r"$\eta$", r"$\chi_{\rm eff}$", r"$z$ (CDF)"]):
    a_.set_xlabel(xl); a_.legend(fontsize=8)
ax[1, 1].set_xlim(0, 2.0)
fig.suptitle("Predicted detected distributions at the 4D kappa 2.7 optima (key 11, n_seed 8000)")
fig.tight_layout(); fig.savefig(f"{OUT}/detected_distributions.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
colg = {"4D pass": "C0", "4D fail": "C1", "3D fallback": "C2"}
zz = np.array([zmed.get(n, np.nan) for n in names])
for a_, v, xl in zip(ax, [zz, C[:, 0], C[:, 2]], ["median z (4D-fit events)", r"$\mathcal{M}_c$ centre", r"$\chi_{\rm eff}$ centre"]):
    for gname, col in colg.items():
        m = group == gname
        a_.scatter(v[m], d4[m], s=12, color=col, label=gname)
    a_.axhline(0, color="k", lw=0.5); a_.set_xlabel(xl)
ax[1].set_xscale("log"); ax[0].set_ylabel(r"$\Delta_{4D} = \ln I(B_d) - \ln I({\rm REFRC})$"); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/delta_vs_properties.png", dpi=130); plt.close(fig)

wb = []
for t in ("4D_BD", "4D_REFRC", "4D_BMIX", "X_BD", "X_REFRC"):
    lw_ = [float(g(k, t, "lnL", win=True)) for k in KEYS]; le_ = [float(g(k, t, "lnL")) for k in KEYS]
    wb.append(dict(point=t, lnL_windowed=mean(lw_), lnL_exact=mean(le_), bias=mean(le_) - mean(lw_),
                   overflow_windowed=max(int(g(k, t, "overflow", win=True)) for k in KEYS)))
with open(f"{OUT}/window_bias.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(wb[0])); w.writeheader(); [w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in wb]
print("window bias:", wb)
print("checks lnL vs parts():", {k: f"{v:.2e}" for k, v in chk.items()})
print("checks rate+sum logI vs lnL:", {k: f"{v:.2e}" for k, v in chk_sum.items()})
print("overflow:", {t: [r["overflow"] for r in R[t]] for t in R if t[:2] in ("4D", "X_")})
for r in rows: print(r["comparison"], {c: round(r[f"d_{c}"], 2) for c in ["rate"] + GROUPS + ["total"]}, "sumcheck", f"{r['sum_check']:.1e}")
print(pvl); print(rob); print(summ)
print("group sizes:", {gname: int((group == gname).sum()) for gname in GROUPS})
print("top d4:"); [print(f"  {r['event']:28s} {r['group']:11s} d4={r['d4']:+.2f} d3={r['d3']:+.2f} Mc={r['Mc']:.1f} eta={r['eta']:.3f} chi={r['chi_eff']:+.2f} z={r['z']:.2f} ess4={r['ess_4D']:.0f} ess3={r['ess_3D']:.0f}") for r in top4]
print("top d4-d3:"); [print(f"  {r['event']:28s} {r['group']:11s} d4={r['d4']:+.2f} d3={r['d3']:+.2f} dd={r['d4_minus_d3']:+.2f} Mc={r['Mc']:.1f} chi={r['chi_eff']:+.2f} z={r['z']:.2f} ess4={r['ess_4D']:.0f}") for r in topd]
print("low ESS (<10):", [(r["event"], round(r["ess_4D"], 1), round(r["ess_3D"], 1)) for r in ev if r["low_ess"]])
