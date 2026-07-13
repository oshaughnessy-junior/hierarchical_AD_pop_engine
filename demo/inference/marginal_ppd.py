"""Marginalized (over F_DEP grid AND m_max posterior) population predictions:
  (1) posterior-predictive spin distributions p(chi_eff), p(chi), p(cos tilt)
      CONDITIONED ON PRIMARY-MASS REGIONS (low/mid/high), with the GWTC-5.0
      marginal overlaid -- shows the aligned low-mass chi_eff peak near zero and
      the broad, isotropic high-mass distributions.
  (2) dR/dm1 (detected merger-rate density) marginalized over F_DEP and m_max,
      vs GWTC-5.0.

Marginalization: pool the per-F_DEP populations weighted by the profile
likelihood (Table; concentrated near the peak F_DEP~0.9 -- NOT 0.6, which is
below the max-likelihood point), each over several posterior draws (marginalizing
m_max and the other parameters).

  python marginal_ppd.py            # both figures
"""
import argparse, importlib.util, os, glob, numpy as np, h5py
import jax, jax.numpy as jnp
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
M.ISOTROPIC_SPIN = False
from mass_pdet_interp import make_real_pdet_fn
KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
ROOT = "/home/richard.oshaughnessy/coag_container"
RUNS = {0.60: "figures_nal_f060_fa0w", 0.70: "figures_nal_f070_fa0w", 0.80: "figures_nal_f080_fa0w",
        0.90: "figures_nal_f090_fa0w", 1.00: "figures_nal_f100_fa0w", 1.20: "figures_nal_f120_fa0w"}
PDIR = "gwk-hierarchical-initial/paper/figures"
OUT = f"/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/{PDIR}"


def gw(param):
    base = os.path.expanduser("~/coag_container/vt_data/gwtc5_pop/popsummary_files")
    if param == "chi_eff":
        path = glob.glob(f"{base}/*ChiEff*/*with2Drates.h5")[0]
    else:
        path = glob.glob(f"{base}/**/*default_mmax*popsummary_result.h5", recursive=True)[0]
    with h5py.File(path, "r") as h:
        g = h[f"posterior/rates_on_grids/{param}"]
        x = np.asarray(g["positions"]).squeeze(); r = np.asarray(g["rates"])
    lo, med, hi = np.nanpercentile(r, [5, 50, 95], axis=0)
    return x, lo, med, hi


def gw_dm2(m2grid, rng, ndraw=400, nsamp=20000):
    """Reconstruct the GWTC-5.0 secondary-mass rate dR/dm2 (median + 90%) from its
    published mass_1 and mass_ratio marginals, assuming p(m1,q)=p(m1)p(q). This
    is approximate (ignores the m1-q correlation) but adequate for comparing the
    high-mass falloff; noted as such in the text."""
    base = os.path.expanduser("~/coag_container/vt_data/gwtc5_pop/popsummary_files")
    path = glob.glob(f"{base}/**/*default_mmax*popsummary_result.h5", recursive=True)[0]
    with h5py.File(path, "r") as h:
        x1 = np.asarray(h["posterior/rates_on_grids/mass_1/positions"]).squeeze()
        r1 = np.asarray(h["posterior/rates_on_grids/mass_1/rates"])
        xq = np.asarray(h["posterior/rates_on_grids/mass_ratio/positions"]).squeeze()
        rq = np.asarray(h["posterior/rates_on_grids/mass_ratio/rates"])
    def isample(x, pdf, n):
        pdf = np.clip(pdf, 0, None); cdf = np.cumsum(pdf); cdf /= cdf[-1]
        return np.interp(rng.random(n), cdf, x)
    draws = rng.choice(len(r1), size=min(ndraw, len(r1)), replace=False)
    dm2 = np.empty((len(draws), len(m2grid)))
    dm = m2grid[1] - m2grid[0]; edges = np.concatenate([m2grid - dm / 2, [m2grid[-1] + dm / 2]])
    for j, d in enumerate(draws):
        # dR/dm2 must integrate to the SAME total rate as dR/dm1, so scale the
        # (shape-only) histogram by this draw's total rate. Previously density=True
        # returned a PDF (int=1), leaving GWTC-5 dR/dm2 ~18x below its own dR/dm1.
        Rtot = np.trapezoid(np.clip(r1[d], 0, None), x1)
        m1s = isample(x1, r1[d], nsamp); qs = np.clip(isample(xq, rq[d], nsamp), 1e-3, 1.0)
        h2, _ = np.histogram(qs * m1s, edges, density=True); dm2[j] = h2 * Rtot
    return np.nanpercentile(dm2, [5, 50, 95], axis=0)


def build_pool(n_seed, n_b, n_draw, rng):
    summ = np.load(os.path.join(ROOT, "figures_nal_profile", "profile_nal_summary.npz"))
    Wf = {round(float(f), 2): float(np.exp(l - summ["logL"].max())) for f, l in zip(summ["f_dep"], summ["logL"])}
    M1, M2, CE, CH, CT, WI, PD = [], [], [], [], [], [], []
    pdet = make_real_pdet_fn()
    base_key = jax.random.PRNGKey(0); ctr = 0   # rotate the MC key per draw (decorrelate realizations)
    for fdep, d in RUNS.items():
        wfd = Wf.get(round(fdep, 2), 0.0)
        if wfd < 1e-3 or not os.path.exists(os.path.join(ROOT, d, "two_channel_pl_post.npz")):
            continue
        post = np.load(os.path.join(ROOT, d, "two_channel_pl_post.npz"))
        n = len(post["m_max_A"]); idx = np.linspace(0, n - 1, min(n_draw, n)).astype(int)
        M.F_DEP = fdep
        for i in idx:
            # rotate BOTH the pairing key and the natal PROPOSAL_KEY (heavy-tail
            # seed set), rebuilding the sampler, so the rare high-mass tail is not
            # pinned to one fixed seed set shared across draws (cheap: no recompile).
            M.PROPOSAL_KEY = jax.random.fold_in(base_key, ctr)
            M.SAMPLER_KEY = jax.random.fold_in(base_key, 100000 + ctr); ctr += 1
            comb, nA = pl.make_combined_pl(n_seed, n_b)
            th = {k: float(post[k][i]) for k in KS}
            s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))
            m0, m1c, c0, c1 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            isA = np.zeros(len(s), bool); isA[:nA] = True
            prim = m0 >= m1c; m1 = np.where(prim, m0, m1c); m2 = np.where(prim, m1c, m0)
            a_prim = np.abs(np.where(prim, c0, c1))
            ct = rng.uniform(-1, 1, (len(s), 2))
            cos0 = np.where(isA, ct[:, 0], 1.0); cos1 = np.where(isA, ct[:, 1], 1.0)
            cprim = np.where(prim, cos0, cos1)
            chieff = (m0 * np.abs(c0) * cos0 + m1c * np.abs(c1) * cos1) / (m0 + m1c)
            pd = np.asarray(pdet(s))
            ok = np.isfinite(m1) & (w > 0)
            wi = w[ok] / w[ok].sum() * wfd / len(idx)
            M1.append(m1[ok]); M2.append(m2[ok]); CE.append(chieff[ok]); CH.append(a_prim[ok]); CT.append(cprim[ok]); WI.append(wi); PD.append(pd[ok])
        print(f"  pooled f_dep={fdep} (weight {wfd:.3f})")
    return (np.concatenate(M1), np.concatenate(M2), np.concatenate(CE), np.concatenate(CH),
            np.concatenate(CT), np.concatenate(WI), np.concatenate(PD))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seed", type=int, default=3000)
    p.add_argument("--n-b", type=int, default=2500)
    p.add_argument("--n-draw", type=int, default=32)
    args = p.parse_args()
    rng = np.random.default_rng(11)
    m1, m2, ce, ch, ct, wi, pd = build_pool(args.n_seed, args.n_b, args.n_draw, rng)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    # ---- Figure 1: spin PPDs conditioned on mass regions ----
    regions = [(5, 15, "C0", r"$m_1<15$"), (15, 35, "C2", r"$15<m_1<35$"), (35, 100, "C3", r"$m_1>35$")]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ceg = np.linspace(-1, 1, 200); chg = np.linspace(0, 1, 200); ctg = np.linspace(-1, 1, 200)
    def kde_region(v, g, mask, bw):
        w = wi[mask]
        if w.sum() < 1e-9: return np.zeros_like(g)
        k = gaussian_kde(v[mask], weights=w, bw_method=bw)(g); return k / np.trapezoid(k, g)
    for lo, hi, col, lab in regions:
        m = (m1 >= lo) & (m1 < hi)
        ax[0].plot(ceg, kde_region(ce, ceg, m, 0.12), col, lw=2, label=lab)
        ax[1].plot(chg, kde_region(ch, chg, m, 0.12), col, lw=2, label=lab)
        ax[2].plot(ctg, kde_region(ct, ctg, m, 0.10), col, lw=2, label=lab)
    # GWTC-5.0 marginal overlays (grey)
    for axx, param, grid in [(ax[0], "chi_eff", ceg), (ax[1], "a_1", chg), (ax[2], "cos_tilt_1", ctg)]:
        gx, _, gmed, _ = gw(param); gr = np.interp(grid, gx, gmed); nn = np.trapezoid(gr, grid)
        if nn > 0: axx.fill_between(grid, gr / nn, color="grey", alpha=0.25, label="GWTC-5.0 (marginal)")
    ax[0].set_xlabel(r"$\chi_{\rm eff}$"); ax[0].set_title("Effective spin by mass region")
    ax[1].set_xlabel(r"$\chi$ (primary magnitude)"); ax[1].set_title("Spin magnitude by mass region")
    ax[2].set_xlabel(r"$\cos\theta$ (primary tilt)"); ax[2].set_title("Spin tilt by mass region")
    for a in ax: a.set_ylabel("norm. density"); a.legend(fontsize=8)
    fig.suptitle("Model spin PPDs conditioned on primary-mass region (marginalized over $F_{\\rm dep}$, $m_{\\max}$) vs GWTC-5.0 marginal", fontsize=12)
    fig.tight_layout(); f1 = os.path.join(OUT, "real_spin_ppd_massregions.png"); fig.savefig(f1, dpi=140, bbox_inches="tight"); print("wrote", f1)

    # ---- Figure 2: intrinsic dR/dm1 AND dR/dm2 marginalized vs GWTC-5.0 ----
    # ABSOLUTE rate: NO GWTC-5 anchor. The model's own z=0 rate R0 is self-calibrated
    # from the O3+O4a+O4b injection VT (vt_calibration, validated to ~20% on a GWTC-5
    # self-test), then dR/dm = R0 * (normalized mass marginal). GWTC-5 is overlaid only
    # to compare SHAPE -- if the curves also agree in HEIGHT that is an independent
    # rate cross-check, not an imposed normalization.
    import vt_calibration as V
    mg = np.linspace(3, 100, 120)
    R0 = V.rate_R0(V.mass_density_from_samples(m1, m2, wi))
    print(f"  [vt] self-calibrated R0 = {R0:.4g} Gpc^-3 yr^-1  (kappa={V.KAPPA_LVK}, n_obs={V.N_OBS})")
    def dRdm(x):
        p = np.clip(gaussian_kde(x, weights=wi, bw_method=0.08)(mg), 0, None)
        return R0 * p / np.trapezoid(p, mg)
    dR1, dR2 = dRdm(m1), dRdm(m2)
    gx1, glo1, gmed1, ghi1 = gw("mass_1")
    g2lo, g2med, g2hi = gw_dm2(mg, rng)

    def logslope(g, y, a=30, b=55):
        m = (g >= a) & (g <= b) & (y > 0)
        return np.polyfit(g[m], np.log(y[m]), 1)[0] if m.sum() > 3 else np.nan
    print(f"  high-mass (30-55 Msun) log-slope: model dR/dm1={logslope(mg, dR1):+.3f}  dR/dm2={logslope(mg, dR2):+.3f}")
    print(f"                                    GWTC-5 dR/dm1={logslope(gx1, gmed1):+.3f}  dR/dm2={logslope(mg, g2med):+.3f}")

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.8))
    ax2[0].fill_between(gx1, glo1, ghi1, color="grey", alpha=0.3, label="GWTC-5.0 90%"); ax2[0].plot(gx1, gmed1, "k", lw=1.5, label="GWTC-5.0 median")
    ax2[0].plot(mg, dR1, "C0", lw=2.2, label=r"model")
    ax2[1].fill_between(mg, g2lo, g2hi, color="grey", alpha=0.3, label="GWTC-5.0 90%"); ax2[1].plot(mg, g2med, "k", lw=1.5, label="GWTC-5.0 median")
    ax2[1].plot(mg, dR2, "C0", lw=2.2, label=r"model")
    for a, lab, ymax in [(ax2[0], r"$dR/dm_1$", gmed1.max()), (ax2[1], r"$dR/dm_2$", g2med.max())]:
        a.set_yscale("log"); a.set_xlim(4, 90); a.set_ylim(ymax * 3e-4, ymax * 3)
        a.set_xlabel(r"$m\ [M_\odot]$"); a.set_ylabel(lab + r" $[{\rm Gpc}^{-3}\,{\rm yr}^{-1}\,M_\odot^{-1}]$"); a.legend(fontsize=8)
    ax2[0].set_title(r"Primary mass $m_1$"); ax2[1].set_title(r"Secondary mass $m_2$ (GWTC-5 reconstructed)")
    fig2.suptitle(r"Intrinsic mass functions marginalized over $F_{\rm dep}$, $m_{\max}$ vs GWTC-5.0 (median + 90\%)", fontsize=12)
    fig2.tight_layout(); f2 = os.path.join(OUT, "real_dRdm1_marginal.png"); fig2.savefig(f2, dpi=140, bbox_inches="tight"); print("wrote", f2)


if __name__ == "__main__":
    raise SystemExit(main())
