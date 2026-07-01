"""Spin-vs-mass trends MARGINALIZED over the cascade brightness F_DEP (over the
grid, weighted by the profile likelihood) and, separately, over the natal cut-off
m_max (over the posterior draws at a fixed F_DEP). Companion to spin_mass_trend.py
(which fixes F_DEP and the posterior mean); these show the trends with those
uncertainties integrated out.

  python spin_mass_marginal.py --mode fdep
  python spin_mass_marginal.py --mode mmax
"""
import argparse, importlib.util, os, numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
M.ISOTROPIC_SPIN = False
KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
ROOT = "/home/richard.oshaughnessy/coag_container"
RUNS = {0.60: "figures_nal_f060", 0.70: "figures_nal_f070", 0.90: "figures_nal_f090", 1.20: "figures_nal_f120"}


def wquant(x, w, qs):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) - 0.5 * w; c /= w.sum()
    return np.interp(qs, c, x)


def population(th, fdep, comb, nA, rng):
    s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))
    m0, m1c, c0, c1 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    isA = np.zeros(len(s), bool); isA[:nA] = True
    prim = m0 >= m1c
    m1 = np.where(prim, m0, m1c)
    a_prim = np.abs(np.where(prim, c0, c1))
    ct = rng.uniform(-1, 1, (len(s), 2))
    cos0 = np.where(isA, ct[:, 0], 1.0); cos1 = np.where(isA, ct[:, 1], 1.0)
    chieff = (m0 * np.abs(c0) * cos0 + m1c * np.abs(c1) * cos1) / (m0 + m1c)
    ok = np.isfinite(m1) & (w > 0)
    return m1[ok], a_prim[ok], chieff[ok], w[ok]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fdep", "mmax"], default="fdep")
    p.add_argument("--mmax-fdep", type=float, default=0.60, help="F_DEP for the m_max marginalization")
    p.add_argument("--n-seed", type=int, default=3000)
    p.add_argument("--n-b", type=int, default=2500)
    p.add_argument("--n-draw", type=int, default=30, help="posterior draws for m_max marginalization")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rng = np.random.default_rng(7)
    summ = np.load(os.path.join(ROOT, "figures_nal_profile", "profile_nal_summary.npz"))
    Wf = {round(float(f), 2): float(np.exp(l - summ["logL"].max())) for f, l in zip(summ["f_dep"], summ["logL"])}

    M1, A, E, W = [], [], [], []
    if args.mode == "fdep":
        for fdep, d in RUNS.items():
            wfd = Wf.get(round(fdep, 2), 0.0)
            if wfd < 1e-3:
                continue
            post = np.load(os.path.join(ROOT, d, "two_channel_pl_post.npz"))
            th = {k: float(post[k].mean()) for k in KS}
            M.F_DEP = fdep; comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
            m1, a, e, w = population(th, fdep, comb, nA, rng)
            M1.append(m1); A.append(a); E.append(e); W.append(w / w.sum() * wfd)  # profile x rate
            print(f"  pooled f_dep={fdep} (weight {wfd:.3f}), {len(m1)} samples")
        title = r"marginalized over $F_{\rm dep}$ (profile-likelihood weighted)"
        tag = "fdep"
    else:
        fdep = args.mmax_fdep
        post = np.load(os.path.join(ROOT, RUNS[round(fdep, 2)], "two_channel_pl_post.npz"))
        n = len(post["m_max_A"]); idx = np.linspace(0, n - 1, min(args.n_draw, n)).astype(int)
        M.F_DEP = fdep; comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
        for i in idx:
            th = {k: float(post[k][i]) for k in KS}
            m1, a, e, w = population(th, fdep, comb, nA, rng)
            M1.append(m1); A.append(a); E.append(e); W.append(w / w.sum() / len(idx))  # equal-weight draws
        print(f"  m_max marginalized at f_dep={fdep} over {len(idx)} posterior draws "
              f"(m_max range {post['m_max_A'][idx].min():.0f}-{post['m_max_A'][idx].max():.0f})")
        title = rf"marginalized over $m_{{\max}}$ (posterior, $F_{{\rm dep}}={fdep}$)"
        tag = "mmax"
    m1 = np.concatenate(M1); a = np.concatenate(A); e = np.concatenate(E); w = np.concatenate(W)

    edges = np.linspace(5, 100, 26); cen = 0.5 * (edges[:-1] + edges[1:])
    aq = np.full((len(cen), 3), np.nan); eq = np.full((len(cen), 3), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (m1 >= lo) & (m1 < hi)
        if w[m].sum() > 1e-9:
            aq[i] = wquant(a[m], w[m], [0.05, 0.5, 0.95]); eq[i] = wquant(e[m], w[m], [0.05, 0.5, 0.95])
    # fixed f_dep=0.6 posterior-mean reference median
    pref = np.load(os.path.join(ROOT, "figures_nal_f060", "two_channel_pl_post.npz"))
    thr = {k: float(pref[k].mean()) for k in KS}; M.F_DEP = 0.6; cr, nAr = pl.make_combined_pl(args.n_seed, args.n_b)
    m1r, ar, er, wr = population(thr, 0.6, cr, nAr, rng)
    amr = np.array([wquant(ar[(m1r >= lo) & (m1r < hi)], wr[(m1r >= lo) & (m1r < hi)], [0.5])[0]
                    if wr[(m1r >= lo) & (m1r < hi)].sum() > 1e-9 else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
    emr = np.array([wquant(er[(m1r >= lo) & (m1r < hi)], wr[(m1r >= lo) & (m1r < hi)], [0.5])[0]
                    if wr[(m1r >= lo) & (m1r < hi)].sum() > 1e-9 else np.nan for lo, hi in zip(edges[:-1], edges[1:])])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].fill_between(cen, aq[:, 0], aq[:, 2], color="C0", alpha=0.25, label="marginalized 90%")
    ax[0].plot(cen, aq[:, 1], "C0", lw=2.2, label="marginalized median")
    ax[0].plot(cen, amr, "k--", lw=1.2, alpha=0.7, label=r"fixed $F_{\rm dep}=0.6$ median")
    ax[0].axhline(0.69, color="grey", ls=":", lw=1.0)
    ax[0].set_xlim(5, 100); ax[0].set_ylim(0, 1); ax[0].set_xlabel(r"$m_1\ [M_\odot]$")
    ax[0].set_ylabel(r"$\chi$ (primary spin magnitude)"); ax[0].set_title("Spin magnitude vs mass"); ax[0].legend(fontsize=8)
    ax[1].fill_between(cen, eq[:, 0], eq[:, 2], color="C1", alpha=0.25, label="marginalized 90%")
    ax[1].plot(cen, eq[:, 1], "C1", lw=2.2, label="marginalized median")
    ax[1].plot(cen, emr, "k--", lw=1.2, alpha=0.7, label=r"fixed $F_{\rm dep}=0.6$ median")
    ax[1].axhline(0, color="grey", lw=0.6, ls="--")
    ax[1].set_xlim(5, 100); ax[1].set_ylim(-1, 1); ax[1].set_xlabel(r"$m_1\ [M_\odot]$")
    ax[1].set_ylabel(r"$\chi_{\rm eff}$"); ax[1].set_title("Effective spin vs mass"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Recovered coagulation population: spin trends vs mass, {title}", fontsize=12)
    out = args.out or f"/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/gwk-hierarchical-initial/paper/figures/real_spin_vs_mass_marg_{tag}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    raise SystemExit(main())
