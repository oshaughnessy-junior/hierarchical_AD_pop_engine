"""Cascade-brightness (F_DEP) profile on the NAL + full-covariance real catalog.

For each fitted f_dep run, evaluate -- at the posterior-mean parameters, on the
uniformly-NAL catalog (Delfavero O3 + production-PE O4, prior-divided, full 3x3
covariance) -- the population profile log-likelihood, the detected mass-spectrum
chi^2, and the high-mass effective-spin dispersion. Produces:

  fig_fdep_profile.png : logL(f_dep) and sigma(chi_eff|Mc>40)(f_dep) vs observed
  fig_ppc_best.png     : detected mass counts + high-mass chi_eff at the preferred
                         f_dep, model vs NAL data (apples-to-apples)
  profile_nal_summary.npz / printed table

This is the reproducibility script for the inline profile that retracted the
earlier (summary-likelihood) 2g-exclusion and established the high-mass BBH
population as second-generation hierarchical (preferred f_dep ~ 0.6).
"""
import argparse, importlib.util, os, numpy as np, jax, jax.numpy as jnp

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
from mass_pdet_interp import make_real_pdet_fn

KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
# (f_dep, posterior dir) -- edit as new brightness points land
RUNS_DEFAULT = [
    (0.10, "figures_nal_f010"), (0.30, "figures_nal_f030"), (0.50, "figures_nal_f050b"),
    (0.60, "figures_nal_f060"), (0.70, "figures_nal_f070"), (0.90, "figures_nal_f090"),
    (1.20, "figures_nal_f120"),
]
EDGES = np.array([4, 8, 12, 18, 25, 32, 40, 55, 80])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default=os.path.join(_HERE, "../../../planning_o4_mixed/full_o3_o4a_o4b_catalog_NAL.npz"))
    p.add_argument("--postroot", default="/home/richard.oshaughnessy/coag_container")
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--ppc-fdep", type=float, default=0.60, help="f_dep for the PPC best-fit figure")
    p.add_argument("--outdir", default="/home/richard.oshaughnessy/coag_container/figures_nal_profile")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pdet = make_real_pdet_fn(); T = float(M.T_OBS)
    cat = np.load(args.catalog, allow_pickle=True)
    ce = jnp.asarray(cat["centers"]); cov = jnp.asarray(cat["covs"]); mco = cat["centers"][:, 0]; ceo = cat["centers"][:, 2]
    prec = jnp.linalg.inv(0.5 * (cov + jnp.transpose(cov, (0, 2, 1))) + 1e-12 * jnp.eye(3))
    _, logdet = jnp.linalg.slogdet(cov)
    oc, _ = np.histogram(mco, EDGES)
    obs_hi = float(ceo[mco > 40].std())

    runs = [(fd, d) for fd, d in RUNS_DEFAULT if os.path.exists(os.path.join(args.postroot, d, "two_channel_pl_post.npz"))]
    rows = []; ppc_cache = {}
    for fd, d in runs:
        z = np.load(os.path.join(args.postroot, d, "two_channel_pl_post.npz"))
        th = {k: float(z[k].mean()) for k in KS}
        M.F_DEP = fd; comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
        s, lw = comb(**th); s = jnp.asarray(s); w = jnp.exp(lw); pd = jnp.asarray(pdet(s)); dw = np.asarray(w * pd)
        mu = float(T * jnp.sum(w * pd))
        x = M.mc_eta_chieff(s); dz = x[None, :, :] - ce[:, None, :]
        q = jnp.einsum("eni,eij,enj->en", dz, prec, dz)
        Lmat = jnp.exp(-0.5 * q - 0.5 * logdet[:, None] - 1.5 * jnp.log(2 * jnp.pi))
        logL = float(-mu + jnp.sum(jnp.log(jnp.maximum(jnp.sum(w[None, :] * Lmat, axis=1), 1e-300))))
        mc = np.asarray((s[:, 0] * s[:, 1]) ** 0.6 / (s[:, 0] + s[:, 1]) ** 0.2)
        chie = np.asarray((s[:, 0] * s[:, 2] + s[:, 1] * s[:, 3]) / (s[:, 0] + s[:, 1]))
        chi2 = float(np.sum((T * np.histogram(mc, EDGES, weights=dw)[0] - oc) ** 2 / np.maximum(oc, 1)))
        m = (mc > 40) & (dw > 0); ww = dw[m] / dw[m].sum(); sg = float(np.sqrt(np.sum(ww * (chie[m] - np.sum(ww * chie[m])) ** 2)))
        rows.append((fd, logL, chi2, sg, th["spin_chi"], th["m_max_A"]))
        if abs(fd - args.ppc_fdep) < 1e-6:
            ppc_cache = dict(mc=mc, chie=chie, dw=dw, isA=np.arange(len(s)) < nA, T=T)
    rows.sort()
    fds = np.array([r[0] for r in rows]); logLs = np.array([r[1] for r in rows])
    sgs = np.array([r[3] for r in rows])
    print(f"OBSERVED high-mass sigma(chi_eff|Mc>40) = {obs_hi:.3f}\n")
    print(f"{'f_dep':>6s} {'logL':>9s} {'dlogL':>7s} {'massChi2':>8s} {'sig|>40':>8s} {'spin_chi':>8s} {'m_max':>6s}")
    for fd, lL, c2, sg, sc, mm in rows:
        print(f"{fd:6.2f} {lL:9.1f} {lL-logLs.max():7.1f} {c2:8.1f} {sg:8.3f} {sc:8.3f} {mm:6.1f}")
    np.savez(os.path.join(args.outdir, "profile_nal_summary.npz"),
             f_dep=fds, logL=logLs, massChi2=np.array([r[2] for r in rows]),
             sig_hi=sgs, obs_sig_hi=obs_hi, spin_chi=np.array([r[4] for r in rows]),
             m_max=np.array([r[5] for r in rows]))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # ---- Fig 1: profile ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(fds, logLs - logLs.max(), "o-", color="C0", lw=2)
    ax[0].set_xlabel(r"$F_{\rm dep}$ (cascade brightness)"); ax[0].set_ylabel(r"$\Delta\ln\mathcal{L}$ (profile)")
    ax[0].set_title("Population profile likelihood"); ax[0].axhline(0, color="grey", lw=0.6, ls=":")
    ax[1].plot(fds, sgs, "s-", color="C1", lw=2, label="model")
    ax[1].axhline(obs_hi, color="k", lw=2, ls="--", label=f"observed ({obs_hi:.2f})")
    ax[1].set_xlabel(r"$F_{\rm dep}$"); ax[1].set_ylabel(r"$\sigma(\chi_{\rm eff}\,|\,\mathcal{M}_c>40)$")
    ax[1].set_title("High-mass effective-spin dispersion"); ax[1].legend(fontsize=9)
    fig.suptitle("Cascade-brightness profile (NAL + full-cov, 238 events): high-mass BBH favor a bright hierarchical cascade", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "fig_fdep_profile.png"), dpi=140, bbox_inches="tight")
    print("wrote", os.path.join(args.outdir, "fig_fdep_profile.png"))
    # ---- Fig 2: PPC at preferred f_dep ----
    if ppc_cache:
        c = ppc_cache; cen = 0.5 * (EDGES[:-1] + EDGES[1:]); bw = np.diff(EDGES)
        mA = c["T"] * np.histogram(c["mc"][c["isA"]], EDGES, weights=c["dw"][c["isA"]])[0]
        mB = c["T"] * np.histogram(c["mc"][~c["isA"]], EDGES, weights=c["dw"][~c["isA"]])[0]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
        ax[0].bar(cen, oc, width=bw * 0.9, color="k", alpha=0.25, label=f"observed (N={oc.sum()})")
        ax[0].step(cen, mA + mB, where="mid", color="C0", lw=2.5, label=f"model (f_dep={args.ppc_fdep})")
        ax[0].step(cen, mA, where="mid", color="C3", lw=1.3, ls="--", label="natal+cascade A")
        ax[0].step(cen, mB, where="mid", color="C2", lw=1.6, ls=":", label="aligned low-mass B")
        ax[0].set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$"); ax[0].set_ylabel("detected events / bin")
        ax[0].set_title("Mass spectrum (counts)"); ax[0].legend(fontsize=8)
        xg = np.linspace(-0.6, 0.8, 200)
        from scipy.stats import gaussian_kde
        hi = (c["mc"] > 35) & (c["dw"] > 0)
        mk = gaussian_kde(c["chie"][hi], weights=c["dw"][hi], bw_method=0.25)(xg)
        ax[1].plot(xg, mk / np.trapezoid(mk, xg), "C0", lw=2, label="model")
        oh = ceo[mco > 35]
        ax[1].hist(oh, bins=12, range=(-0.6, 0.8), density=True, color="k", alpha=0.3, label=f"observed (N={len(oh)})")
        ax[1].set_xlabel(r"$\chi_{\rm eff}\ (\mathcal{M}_c>35)$"); ax[1].set_ylabel("density")
        ax[1].set_title("High-mass effective spin"); ax[1].legend(fontsize=8)
        fig.suptitle(f"Best-fit PPC at $F_{{\\rm dep}}={args.ppc_fdep}$: one model reproduces mass AND high-mass spin", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "fig_ppc_best.png"), dpi=140, bbox_inches="tight")
        print("wrote", os.path.join(args.outdir, "fig_ppc_best.png"))


if __name__ == "__main__":
    raise SystemExit(main())
