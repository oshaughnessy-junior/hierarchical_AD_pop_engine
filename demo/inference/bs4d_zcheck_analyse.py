"""Tables and figures for the redshift checks (bs4d_zcheck_eval.py output, key 11, n_seed 8000).
Usage: bs4d_zcheck_analyse.py ZCHECK_NPZ PZ_CHECK_JSON CATALOG_ZMED_JSON OUTDIR"""
import csv, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

z = np.load(sys.argv[1], allow_pickle=True); pzc = json.load(open(sys.argv[2])); czm = np.array(json.load(open(sys.argv[3])))
OUT = sys.argv[4]; N, N4, N3 = 236, 189, 47
ZB = np.linspace(0, 2.5, 51); ZC = 0.5 * (ZB[1:] + ZB[:-1])
q = lambda zz, ww, p: float(np.interp(p, np.cumsum(ww) / np.sum(ww), zz))

prof_rows, det_rows = [], []
for M in ("BD", "REFRC"):
    kp = json.loads(str(z[f"{M}|kappa_profile"]))
    S27 = next(p["sum_lnI"] for p in kp if p["kappa"] == 2.7)
    for p in kp:
        corr = N3 * (p["sum_lnI"] - S27) / N4          # fallback events given the mean 4D-event kappa response
        prof_rows.append(dict(model=M, kappa=p["kappa"], sum_lnI=p["sum_lnI"], mu_grid=p["mu_grid"], mu_inj=p["mu_inj"],
                              lnLprof_grid=p["lnLprof_grid"], lnLprof_inj=p["lnLprof_inj"], fallback_term=corr,
                              lnLprof_grid_fbcorr=p["lnLprof_grid"] + corr, lnLprof_inj_fbcorr=p["lnLprof_inj"] + corr))
    zg, dg, zi, di = z[f"{M}|z_grid"], z[f"{M}|det_grid"], z[f"{M}|z_inj"], z[f"{M}|det_inj"]
    st, md = z[f"{M}|stack"], z[f"{M}|pop_medians"]
    det_rows.append(dict(model=M, det_grid_q50=q(zg, dg, .5), det_grid_q90=q(zg, dg, .9), det_grid_frac_gt1=float(dg[zg > 1].sum() / dg.sum()),
                         det_inj_q50=q(zi, di, .5), det_inj_q90=q(zi, di, .9), det_inj_frac_gt1=float(di[zi > 1].sum() / di.sum()),
                         mu_grid=float(z[f"{M}|mu_grid"]), mu_inj=float(z[f"{M}|mu_inj"]),
                         stack_q50=q(ZC, st, .5), stack_q90=q(ZC, st, .9), popmed_q50=float(np.median(md)), popmed_q90=float(np.quantile(md, .9)),
                         pe_median_q50=float(np.median(czm)), pe_median_q90=float(np.quantile(czm, .9))))
for name, rows in (("kappa_profile", prof_rows), ("detected_z", det_rows)):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader()
        for r in rows: w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
hz = [dict(model=M, **d) for M in ("BD", "REFRC") for d in json.loads(str(z[f"{M}|hiz_by_m1"]))]
with open(f"{OUT}/hiz_by_m1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(hz[0])); w.writeheader(); [w.writerow(r) for r in hz]
json.dump(pzc, open(f"{OUT}/pz_vs_astropy.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for M, col in (("BD", "C0"), ("REFRC", "C3")):
    zg, dg, zi, di = z[f"{M}|z_grid"], z[f"{M}|det_grid"], z[f"{M}|z_inj"], z[f"{M}|det_inj"]
    ax[0].plot(zg, np.cumsum(dg) / dg.sum(), color=col, label=f"{M}: predicted, NN grid")
    ax[0].plot(zi, np.cumsum(di) / di.sum(), color=col, ls="--", label=f"{M}: predicted, injection histogram")
    st = z[f"{M}|stack"]; ax[0].plot(ZB[1:], np.cumsum(st) / st.sum(), color=col, ls=":", label=f"{M}: stacked event z posteriors (population prior)")
ax[0].plot(np.sort(czm), np.arange(1, len(czm) + 1) / len(czm), color="0.4", lw=2, label="PE z medians (189 4D events)")
ax[0].set_xlim(0, 2.0); ax[0].set_xlabel("z"); ax[0].set_ylabel("CDF"); ax[0].legend(fontsize=7)
for M, col in (("BD", "C0"), ("REFRC", "C3")):
    r = [p for p in prof_rows if p["model"] == M]; k = [p["kappa"] for p in r]
    for key, ls, lab in (("lnLprof_grid", "-", "NN grid"), ("lnLprof_inj", "--", "inj. histogram"),
                         ("lnLprof_grid_fbcorr", "-.", "NN grid, fallback corr."), ("lnLprof_inj_fbcorr", ":", "inj., fallback corr.")):
        v = np.array([p[key] for p in r]); ax[1].plot(k, v - v.max(), ls=ls, color=col, marker="o", ms=3, label=f"{M}: {lab}")
ax[1].set_ylim(-60, 2); ax[1].set_xlabel(r"$\kappa$"); ax[1].set_ylabel(r"rate-profiled $\Delta\ln L$ (fixed shape)"); ax[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/zcheck.png", dpi=130)
for r in prof_rows: print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
for r in det_rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
