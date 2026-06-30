"""Quantitative 2g exclusion: the observed high-mass chi_eff scatter is too NARROW
for an isotropic hierarchical (2g) origin. Compares, as-observed (measurement-
convolved), the high-mass chi_eff under two isolated hypotheses --
  HIERARCHICAL 2g: low natal cutoff (m_max=35) + bright cascade -> high mass IS
                   2g remnants -> broad chi_eff.
  NATAL (fit):     high natal cutoff (m_max=58) + faint cascade -> high mass is
                   natal -> narrow chi_eff.
-- against the observed O3+O4 high-mass events.
"""
import argparse, importlib.util, os, numpy as np, jax, jax.numpy as jnp
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
from mass_pdet_interp import make_real_pdet_fn


def hm(m_max, fdep, mc_cut, pdet, alpha=1.64, spin=0.168, n=12000):
    M.F_DEP = fdep; M.ISOTROPIC_SPIN = True
    samp = pl.make_sampler_A_mmax(n)
    s, lw = samp(alpha, m_max, spin, 1.0, 2.0, 0.0); s = np.asarray(s); w = np.exp(np.asarray(lw))
    dw = np.clip(w * np.asarray(pdet(s)), 0, None)
    mc = (s[:, 0] * s[:, 1]) ** 0.6 / (s[:, 0] + s[:, 1]) ** 0.2
    chie = (s[:, 0] * s[:, 2] + s[:, 1] * s[:, 3]) / (s[:, 0] + s[:, 1])
    hi = (mc > mc_cut) & (dw > 0)
    return chie[hi], dw[hi] / dw[hi].sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default=os.path.join(_HERE, "../../../planning_o4_mixed/full_o3_o4a_o4b_catalog.npz"))
    p.add_argument("--out", default="/home/richard.oshaughnessy/coag_container/figures_two_channel_plmmax/twog_exclusion.png")
    args = p.parse_args()
    pdet = make_real_pdet_fn(); rng = np.random.default_rng(0)
    cat = np.load(args.catalog, allow_pickle=True)
    mco, ceo, sceo = cat["centers"][:, 0], cat["centers"][:, 2], cat["sigs"][:, 2]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6)); xg = np.linspace(-0.8, 0.9, 220)
    for k, cut in enumerate([35, 40]):
        a = ax[k]; om = mco > cut; obs = ceo[om]; meas = np.sqrt(np.mean(sceo[om] ** 2)); Nobs = om.sum()
        sd_obs = obs.std(); sd_err = sd_obs / np.sqrt(2 * Nobs)
        def asobs(m_max, fdep):
            c, wn = hm(m_max, fdep, cut, pdet)
            idx = rng.choice(len(c), 6000, p=wn); xo = c[idx] + meas * rng.standard_normal(6000)
            return xo, gaussian_kde(xo, bw_method=0.18)(xg)
        x2, k2 = asobs(35., 0.35); xn, kn = asobs(58., 0.10)
        sd2, sdn = x2.std(), xn.std()
        a.fill_between(xg, k2, color="C3", alpha=0.25); a.plot(xg, k2, "C3", lw=2,
            label=f"Hierarchical 2g: $\\sigma$={sd2:.2f}")
        a.fill_between(xg, kn, color="C0", alpha=0.20); a.plot(xg, kn, "C0", lw=2,
            label=f"Natal fit: $\\sigma$={sdn:.2f}")
        a.hist(obs, bins=10, range=(-0.8, 0.9), density=True, color="k", alpha=0.30,
               label=f"Observed (N={Nobs}): $\\sigma$={sd_obs:.2f}")
        a.plot(xg, gaussian_kde(obs, bw_method=0.3)(xg), "k", lw=2)
        nsig = (sd2 - sd_obs) / sd_err
        a.set_title(f"$\\mathcal{{M}}_c>{cut}\\,M_\\odot$:  2g excluded at ${nsig:.1f}\\sigma$", fontsize=11)
        a.set_xlabel(r"$\chi_{\rm eff}$ (as observed)"); a.set_ylabel("density"); a.legend(fontsize=8.5)
        print(f"Mc>{cut}: obs sd={sd_obs:.3f}+/-{sd_err:.3f}  2g sd={sd2:.3f} ({nsig:.1f}sigma)  natal sd={sdn:.3f}")
    fig.suptitle("High-mass effective spin is too narrow for isotropic hierarchical (2g) formation", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight"); print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
