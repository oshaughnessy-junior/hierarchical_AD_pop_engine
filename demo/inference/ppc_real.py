"""LVK-quality posterior-predictive check for the real-data deep-cascade fit.

Two coupled signatures, smoothed (no bare histograms), with posterior bands:
  (1) detected chirp-mass distribution dN/dMc -- does the model reproduce the
      observed mass spectrum incl. the ~35 Msun feature?
  (2) effective-spin DISPERSION sigma(chi_eff | Mc) -- the hierarchical
      signature: does the spin width widen at the high-mass (2g-remnant)
      feature as in the data?

Predicted bands: draw N hyper-posterior samples, generate the DETECTED population
(sampler x real pdet), KDE/binned-dispersion each, take the 5/50/95% envelope.
Observed: per-event Gaussians (mass) and measurement-error-deconvolved dispersion
(spin). Run at the profile point F_DEP (default 0.30), isotropic spins.

  python ppc_real.py --f-dep 0.30 --post <real_post.npz> --n-draw 40
"""
import argparse, importlib.util, os, numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import logsumexp

_HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "plpeak", os.path.join(_HERE, "production_plpeak_sampled_recovery.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
from mass_pdet_interp import make_real_pdet_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--f-dep", type=float, default=0.30)
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_fdep030/real_post.npz")
    p.add_argument("--catalog", default=os.path.join(_HERE, "../../../planning_o4_mixed/full_o3_o4a_o4b_catalog.npz"))
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-draw", type=int, default=40)
    p.add_argument("--out", default="/home/richard.oshaughnessy/coag_container/figures_fdep030/ppc_real.png")
    args = p.parse_args()

    M.ISOTROPIC_SPIN = True
    M.F_DEP = args.f_dep
    sampler = M.make_sampler_fn(args.n_seed)
    pdet = make_real_pdet_fn()
    post = np.load(args.post)
    keys = ("alpha", "lambda_peak", "mu_peak", "sigma_peak", "spin_scale")
    nd = min(args.n_draw, len(post["alpha"]))
    idx = np.linspace(0, len(post["alpha"]) - 1, nd).astype(int)

    mc_grid = np.linspace(4, 70, 80)
    mc_bins = np.array([5, 12, 18, 25, 32, 40, 55])
    mc_cen = 0.5 * (mc_bins[:-1] + mc_bins[1:])
    dens, disp = [], []
    from scipy.stats import gaussian_kde
    for i in idx:
        th = {k: float(post[k][i]) for k in keys}
        s, lw = sampler(th["alpha"], th["lambda_peak"], th["mu_peak"],
                        th["sigma_peak"], th["spin_scale"], 0.0)
        s = np.asarray(s); w = np.exp(np.asarray(lw - logsumexp(lw)))
        dw = w * np.asarray(pdet(s)); dw = dw / dw.sum()
        m1, m2, c1, c2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        chie = (m1 * c1 + m2 * c2) / (m1 + m2)
        ok = np.isfinite(mc) & (dw > 0)
        kde = gaussian_kde(mc[ok], weights=dw[ok], bw_method=0.15)
        dens.append(kde(mc_grid))
        d = []
        for lo, hi in zip(mc_bins[:-1], mc_bins[1:]):
            m = ok & (mc >= lo) & (mc < hi); ww = dw[m]
            if ww.sum() > 1e-6:
                ww = ww / ww.sum(); mu = np.sum(ww * chie[m])
                d.append(np.sum(ww * (chie[m] - mu) ** 2))   # store VARIANCE
            else:
                d.append(np.nan)
        disp.append(d)
    dens = np.array(dens); disp = np.array(disp)

    cat = np.load(args.catalog, allow_pickle=True)
    mco = cat["centers"][:, 0]; ceo = cat["centers"][:, 2]; sco = cat["sigs"][:, 2]
    smco = cat["sigs"][:, 0]
    obs_dens = np.zeros_like(mc_grid)
    for mu, sg in zip(mco, np.clip(smco, 0.3, None)):
        obs_dens += np.exp(-0.5 * ((mc_grid - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
    obs_dens /= len(mco)
    # RAW observed chi_eff dispersion (includes measurement scatter) + the bin's
    # mean measurement variance, so we forward-convolve the MODEL to data space
    # (deconvolving the data fails: meas err ~ intrinsic scatter).
    obs_disp, meas_var = [], []
    for lo, hi in zip(mc_bins[:-1], mc_bins[1:]):
        m = (mco >= lo) & (mco < hi)
        if m.sum() > 3:
            obs_disp.append(ceo[m].std())
            meas_var.append(np.mean(sco[m] ** 2))
        else:
            obs_disp.append(np.nan); meas_var.append(np.nan)
    obs_disp = np.array(obs_disp); meas_var = np.array(meas_var)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    dn = dens / np.trapezoid(dens, mc_grid, axis=1)[:, None]
    lo, md, hi = np.percentile(dn, [5, 50, 95], axis=0)
    ax[0].fill_between(mc_grid, lo, hi, alpha=0.3, color="C0", label="model 90%")
    ax[0].plot(mc_grid, md, color="C0", lw=2)
    ax[0].plot(mc_grid, obs_dens / np.trapezoid(obs_dens, mc_grid), color="k", lw=2, label="observed")
    ax[0].axvspan(33, 38, color="grey", alpha=0.12)
    ax[0].set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$"); ax[0].set_ylabel(r"detected $dN/d\mathcal{M}_c$ (norm.)")
    ax[0].set_title("Mass spectrum"); ax[0].legend(fontsize=8)
    # model VARIANCE + bin measurement variance -> predicted OBSERVED dispersion
    disp_obs = np.sqrt(disp + meas_var[None, :])
    loD, mdD, hiD = np.nanpercentile(disp_obs, [5, 50, 95], axis=0)
    ax[1].fill_between(mc_cen, loD, hiD, alpha=0.3, color="C1", label="model+meas 90%")
    ax[1].plot(mc_cen, mdD, color="C1", lw=2, marker="o", ms=3)
    ax[1].plot(mc_cen, obs_disp, color="k", lw=2, marker="s", ms=5, label="observed (raw)")
    ax[1].axvspan(33, 38, color="grey", alpha=0.12)
    ax[1].set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$"); ax[1].set_ylabel(r"$\sigma(\chi_{\rm eff})$")
    ax[1].set_title("Effective-spin dispersion (hierarchical signature)"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Deep-cascade PPC (F_DEP={args.f_dep}, oriented spins, 238 events)", fontsize=11)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("wrote", args.out)
    print("\n  Mc bin   model sigma(chi_eff) [5,50,95]      observed(deconv)")
    for j, mc in enumerate(mc_cen):
        print(f"  {mc:5.0f}    [{loD[j]:.3f}, {mdD[j]:.3f}, {hiD[j]:.3f}]            {obs_disp[j]:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
