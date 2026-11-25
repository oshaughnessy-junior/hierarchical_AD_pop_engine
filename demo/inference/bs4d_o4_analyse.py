"""Tables and figures for the O4 checks (bs4d_o4_check.py output): detected z / Mc / chi_eff vs the catalog,
kappa profiles, and the 188-event fixed-parameter comparison. Usage: bs4d_o4_analyse.py NPZ ZMED_JSON OUTDIR"""
import csv, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

z = np.load(sys.argv[1], allow_pickle=True); zm = json.load(open(sys.argv[2])); OUT = sys.argv[3]
tags = sorted({k.split("|")[0] for k in z.files}); ZB = np.linspace(0, 2.5, 51); ZC = 0.5 * (ZB[1:] + ZB[:-1])
wq = lambda v, w, p: float(np.interp(p, np.cumsum(w[np.argsort(v)]) / w.sum(), v[np.argsort(v)]))
cz = np.array(list(zm.values())); rows, prows = [], []
for t in tags:
    g = lambda k: z[f"{t}|{k}"]
    x, w = g("det_x"), g("det_w"); h = g("det_z_hist"); st = g("stack_z"); C = g("cat_centres")
    prof = json.loads(str(g("kappa_profile")))
    rows.append(dict(tag=t, kappa_fit=float(g("kappa_fit")),
        det_z_q50=wq(ZC, h, .5), det_z_q90=wq(ZC, h, .9), stack_z_q50=wq(ZC, st, .5), stack_z_q90=wq(ZC, st, .9),
        pe_z_med_q50=float(np.median(cz)), pe_z_med_q90=float(np.quantile(cz, .9)),
        det_Mc_q16=wq(x[:, 0], w, .16), det_Mc_q50=wq(x[:, 0], w, .5), det_Mc_q84=wq(x[:, 0], w, .84),
        cat_Mc_q16=float(np.quantile(C[:, 0], .16)), cat_Mc_q50=float(np.median(C[:, 0])), cat_Mc_q84=float(np.quantile(C[:, 0], .84)),
        det_chi_mean=float(np.sum(w * x[:, 2]) / w.sum()), det_chi_sd=float(np.sqrt(np.sum(w * x[:, 2] ** 2) / w.sum() - (np.sum(w * x[:, 2]) / w.sum()) ** 2)),
        cat_chi_mean=float(np.mean(C[:, 2])), cat_chi_sd=float(np.std(C[:, 2])),
        lnL_o4=float(g("o4_lnL")), lnL_o4all=float(g("o4all_lnL")), neff_min_profile=min(p["neff"] for p in prof)))
    for p in prof: prows.append(dict(tag=t, **p))
for name, rs in (("checks_detected", rows), ("checks_kappa_profile", prows)):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, list(rs[0])); wr.writeheader()
        for r in rs: wr.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
d = {t: (float(z[f"{t}|o4_lnL"]), float(z[f"{t}|o4all_lnL"])) for t in tags}
vs = [dict(comparison=f"{tags[0]} - {tags[1]}", o4_thinned=d[tags[0]][0] - d[tags[1]][0], o4all_no_thinning=d[tags[0]][1] - d[tags[1]][1])]
with open(f"{OUT}/variant_fixed_params.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, list(vs[0])); wr.writeheader(); [wr.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in vs]
fig, ax = plt.subplots(2, 2, figsize=(11, 8)); cols = {t: c for t, c in zip(tags, ("C0", "C3"))}
for t in tags:
    g = lambda k: z[f"{t}|{k}"]; x, w = g("det_x"), g("det_w"); w = w / w.sum(); h = g("det_z_hist")
    ax[0, 0].plot(ZB[1:], np.cumsum(h) / h.sum(), color=cols[t], label=f"{t}: predicted detected")
    ax[0, 0].plot(ZB[1:], np.cumsum(g("stack_z")) / g("stack_z").sum(), color=cols[t], ls=":", label=f"{t}: stacked event posteriors")
    ax[0, 1].hist(np.log10(x[:, 0]), bins=60, weights=w, density=True, histtype="step", color=cols[t], label=t)
    ax[1, 0].hist(x[:, 2], bins=np.linspace(-0.5, 0.8, 66), weights=w, density=True, histtype="step", color=cols[t], label=t)
    p = json.loads(str(g("kappa_profile"))); k = [q["kappa"] for q in p]; lp = np.array([q["lnLprof"] for q in p])
    ax[1, 1].plot(k, lp - lp.max(), marker="o", color=cols[t], label=f"{t} (fit kappa {float(g('kappa_fit')):.2f})")
ax[0, 0].plot(np.sort(cz), np.arange(1, len(cz) + 1) / len(cz), color="0.4", lw=2, label="PE z medians (168 events)")
ax[0, 1].hist(np.log10(z[f"{tags[0]}|cat_centres"][:, 0]), bins=30, density=True, histtype="stepfilled", color="0.85", label="catalog centres", zorder=0)
ax[1, 0].hist(z[f"{tags[0]}|cat_centres"][:, 2], bins=np.linspace(-0.5, 0.8, 66), density=True, histtype="stepfilled", color="0.85", label="catalog centres", zorder=0)
ax[0, 0].set_xlabel("z (CDF)"); ax[0, 0].set_xlim(0, 2); ax[0, 1].set_xlabel(r"$\log_{10}\mathcal{M}_c$ (source)"); ax[1, 0].set_xlabel(r"$\chi_{\rm eff}$")
ax[1, 1].set_xlabel(r"$\kappa$"); ax[1, 1].set_ylabel(r"rate-profiled $\Delta\ln L$ (fixed shape)")
for a in ax.flat: a.legend(fontsize=7)
fig.suptitle("O4-only fits (injection selection): predicted detected distributions, key 11, n_seed 8000"); fig.tight_layout()
fig.savefig(f"{OUT}/checks.png", dpi=130)
for r in rows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
print(vs)
