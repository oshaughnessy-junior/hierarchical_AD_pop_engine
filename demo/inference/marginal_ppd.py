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
import jax.numpy as jnp
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
M.ISOTROPIC_SPIN = False
from mass_pdet_interp import make_real_pdet_fn
KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
ROOT = "/home/richard.oshaughnessy/coag_container"
RUNS = {0.60: "figures_nal_f060", 0.70: "figures_nal_f070", 0.80: "figures_nal_f080",
        0.90: "figures_nal_f090", 1.00: "figures_nal_f100", 1.20: "figures_nal_f120"}
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
    return x, np.nanmedian(r, axis=0)


def build_pool(n_seed, n_b, n_draw, rng):
    summ = np.load(os.path.join(ROOT, "figures_nal_profile", "profile_nal_summary.npz"))
    Wf = {round(float(f), 2): float(np.exp(l - summ["logL"].max())) for f, l in zip(summ["f_dep"], summ["logL"])}
    M1, CE, CH, CT, WI, PD = [], [], [], [], [], []
    pdet = make_real_pdet_fn()
    for fdep, d in RUNS.items():
        wfd = Wf.get(round(fdep, 2), 0.0)
        if wfd < 1e-3 or not os.path.exists(os.path.join(ROOT, d, "two_channel_pl_post.npz")):
            continue
        post = np.load(os.path.join(ROOT, d, "two_channel_pl_post.npz"))
        n = len(post["m_max_A"]); idx = np.linspace(0, n - 1, min(n_draw, n)).astype(int)
        M.F_DEP = fdep; comb, nA = pl.make_combined_pl(n_seed, n_b)
        for i in idx:
            th = {k: float(post[k][i]) for k in KS}
            s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))
            m0, m1c, c0, c1 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            isA = np.zeros(len(s), bool); isA[:nA] = True
            prim = m0 >= m1c; m1 = np.where(prim, m0, m1c)
            a_prim = np.abs(np.where(prim, c0, c1))
            ct = rng.uniform(-1, 1, (len(s), 2))
            cos0 = np.where(isA, ct[:, 0], 1.0); cos1 = np.where(isA, ct[:, 1], 1.0)
            cprim = np.where(prim, cos0, cos1)
            chieff = (m0 * np.abs(c0) * cos0 + m1c * np.abs(c1) * cos1) / (m0 + m1c)
            pd = np.asarray(pdet(s))
            ok = np.isfinite(m1) & (w > 0)
            wi = w[ok] / w[ok].sum() * wfd / len(idx)
            M1.append(m1[ok]); CE.append(chieff[ok]); CH.append(a_prim[ok]); CT.append(cprim[ok]); WI.append(wi); PD.append(pd[ok])
        print(f"  pooled f_dep={fdep} (weight {wfd:.3f})")
    return (np.concatenate(M1), np.concatenate(CE), np.concatenate(CH),
            np.concatenate(CT), np.concatenate(WI), np.concatenate(PD))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seed", type=int, default=3000)
    p.add_argument("--n-b", type=int, default=2500)
    p.add_argument("--n-draw", type=int, default=12)
    args = p.parse_args()
    rng = np.random.default_rng(11)
    m1, ce, ch, ct, wi, pd = build_pool(args.n_seed, args.n_b, args.n_draw, rng)
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
        gx, gr = gw(param); gr = np.interp(grid, gx, gr); nn = np.trapezoid(gr, grid)
        if nn > 0: axx.fill_between(grid, gr / nn, color="grey", alpha=0.25, label="GWTC-5.0 (marginal)")
    ax[0].set_xlabel(r"$\chi_{\rm eff}$"); ax[0].set_title("Effective spin by mass region")
    ax[1].set_xlabel(r"$\chi$ (primary magnitude)"); ax[1].set_title("Spin magnitude by mass region")
    ax[2].set_xlabel(r"$\cos\theta$ (primary tilt)"); ax[2].set_title("Spin tilt by mass region")
    for a in ax: a.set_ylabel("norm. density"); a.legend(fontsize=8)
    fig.suptitle("Model spin PPDs conditioned on primary-mass region (marginalized over $F_{\\rm dep}$, $m_{\\max}$) vs GWTC-5.0 marginal", fontsize=12)
    fig.tight_layout(); f1 = os.path.join(OUT, "real_spin_ppd_massregions.png"); fig.savefig(f1, dpi=140, bbox_inches="tight"); print("wrote", f1)

    # ---- Figure 2: intrinsic dR/dm1 marginalized vs GWTC-5.0 (both intrinsic) ----
    m1g = np.linspace(3, 100, 120)
    dR = gaussian_kde(m1, weights=wi, bw_method=0.08)(m1g)   # intrinsic (rate-weighted, NO pdet)
    gx, gr = gw("mass_1")                                     # GWTC-5.0 intrinsic dR/dm1
    sel = (gx >= 5) & (gx <= 80); selm = (m1g >= 5) & (m1g <= 80)
    dR *= np.trapezoid(gr[sel], gx[sel]) / np.trapezoid(dR[selm], m1g[selm])
    fig2, a2 = plt.subplots(figsize=(7, 5))
    a2.fill_between(gx, gr, color="grey", alpha=0.3, label="GWTC-5.0 (median)")
    a2.plot(m1g, dR, "C0", lw=2.2, label=r"model (marginalized $F_{\rm dep},m_{\max}$)")
    a2.set_yscale("log"); a2.set_xlim(4, 90); a2.set_ylim(gr.max() * 3e-4, gr.max() * 3)
    a2.set_xlabel(r"$m_1\ [M_\odot]$"); a2.set_ylabel(r"$dR/dm_1$ (intrinsic, arb. norm.)")
    a2.set_title(r"Intrinsic mass function, marginalized over $F_{\rm dep}$ and $m_{\max}$"); a2.legend()
    fig2.tight_layout(); f2 = os.path.join(OUT, "real_dRdm1_marginal.png"); fig2.savefig(f2, dpi=140, bbox_inches="tight"); print("wrote", f2)


if __name__ == "__main__":
    raise SystemExit(main())
