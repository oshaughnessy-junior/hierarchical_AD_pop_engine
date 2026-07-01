"""Spin-vs-mass trends of the recovered coagulation population, in the style of
the phenomenological companion analysis (Zeeshan & O'Shaughnessy): the primary
spin magnitude chi and the effective spin chi_eff conditioned on primary mass m1.

The physical two-channel model reproduces the empirical trend the companion
extracts: low mass = low, aligned spin (channel B); high mass = large,
isotropic spin (2g cascade remnants, chi~0.7); intermediate = a mix.

  python spin_mass_trend.py   # f_dep=0.6 NAL fit
"""
import argparse, importlib.util, os, numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
M.ISOTROPIC_SPIN = False        # magnitudes; tilt in numpy
KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]


def wquant(x, w, qs):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w
    c /= w.sum()
    return np.interp(qs, c, x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_nal_f060/two_channel_pl_post.npz")
    p.add_argument("--f-dep", type=float, default=0.60)
    p.add_argument("--n-seed", type=int, default=6000)
    p.add_argument("--n-b", type=int, default=5000)
    p.add_argument("--out", default="/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/gwk-hierarchical-initial/paper/figures/real_spin_vs_mass.png")
    args = p.parse_args()
    M.F_DEP = args.f_dep
    comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
    post = np.load(args.post)
    th = {k: float(post[k].mean()) for k in KS}
    s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))
    m0, m1c, c0, c1 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    isA = np.zeros(len(s), bool); isA[:nA] = True
    prim = m0 >= m1c
    m1 = np.where(prim, m0, m1c)
    a_prim = np.abs(np.where(prim, c0, c1))               # primary spin magnitude
    rng = np.random.default_rng(5)
    ct = rng.uniform(-1, 1, (len(s), 2))
    cos0 = np.where(isA, ct[:, 0], 1.0); cos1 = np.where(isA, ct[:, 1], 1.0)   # B aligned
    chieff = (m0 * np.abs(c0) * cos0 + m1c * np.abs(c1) * cos1) / (m0 + m1c)
    ok = np.isfinite(m1) & (w > 0)

    edges = np.linspace(5, 100, 26); cen = 0.5 * (edges[:-1] + edges[1:])
    aq = np.full((len(cen), 3), np.nan); eq = np.full((len(cen), 3), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = ok & (m1 >= lo) & (m1 < hi)
        if w[m].sum() > 1e-9:
            aq[i] = wquant(a_prim[m], w[m], [0.05, 0.5, 0.95])
            eq[i] = wquant(chieff[m], w[m], [0.05, 0.5, 0.95])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].fill_between(cen, aq[:, 0], aq[:, 2], color="C0", alpha=0.25, label="model 90%")
    ax[0].plot(cen, aq[:, 1], "C0", lw=2.2, label="model median")
    ax[0].axhline(0.69, color="grey", ls=":", lw=1.2); ax[0].text(88, 0.71, r"$\chi_{\rm 2g}\!\approx\!0.7$", fontsize=8, color="grey")
    ax[0].set_xlim(5, 100); ax[0].set_ylim(0, 1); ax[0].set_xlabel(r"$m_1\ [M_\odot]$")
    ax[0].set_ylabel(r"$\chi$ (primary spin magnitude)"); ax[0].set_title("Spin magnitude vs mass"); ax[0].legend(fontsize=9)
    ax[1].fill_between(cen, eq[:, 0], eq[:, 2], color="C1", alpha=0.25, label="model 90%")
    ax[1].plot(cen, eq[:, 1], "C1", lw=2.2, label="model median")
    ax[1].axhline(0, color="grey", lw=0.6, ls="--")
    ax[1].set_xlim(5, 100); ax[1].set_ylim(-1, 1); ax[1].set_xlabel(r"$m_1\ [M_\odot]$")
    ax[1].set_ylabel(r"$\chi_{\rm eff}$"); ax[1].set_title("Effective spin vs mass"); ax[1].legend(fontsize=9)
    fig.suptitle(r"Recovered coagulation population: spin trends vs mass ($F_{\rm dep}=0.6$ NAL fit)", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight"); print("wrote", args.out)
    # numeric trend summary
    for lab, q in [("chi|m1", aq), ("chi_eff|m1", eq)]:
        lo = np.nanmean(q[cen < 15, 1]); hi = np.nanmean(q[cen > 60, 1])
        loW = np.nanmean(q[cen < 15, 2] - q[cen < 15, 0]); hiW = np.nanmean(q[cen > 60, 2] - q[cen > 60, 0])
        print(f"  {lab}: median low-mass={lo:+.2f} high-mass={hi:+.2f} | 90%width low={loW:.2f} high={hiW:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
