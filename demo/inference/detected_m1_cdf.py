"""Stress test: predicted DETECTED primary-mass survival CDF  P(M_1 > m_1)  vs the
empirical catalog CDF with a binomial error band (GWTC-3 style consistency check).

No Gaussian-summary chi^2 -- this compares the shape of the *detected* primary-mass
distribution the model predicts (forward population x P_det, normalized) against the
observed catalog, event-count (binomial) errors on the empirical CDF. If a bright
cascade over-produces high-mass mergers, its predicted P(M_1>m_1) sits ABOVE the
empirical binomial band at high mass.

  python detected_m1_cdf.py
"""
import argparse, importlib.util, os, numpy as np
import jax

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
from mass_pdet_interp import make_real_pdet_fn
KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
ROOT = "/home/richard.oshaughnessy/coag_container"
CAT = os.path.join(_HERE, "../../../planning_o4_mixed/full_o3_o4a_o4b_catalog_NAL.npz")
SHOW = [(0.30, "figures_nal_f030_fa0w", "C0"), (0.60, "figures_nal_f060_fa0w", "C2"),
        (0.90, "figures_nal_f090_fa0w", "C1"), (1.20, "figures_nal_f120_fa0w", "C3")]
OUT = "/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/gwk-hierarchical-initial/paper/figures/real_detected_m1_cdf.png"


def m1_from_mc_eta(mc, eta):
    eta = np.clip(eta, 1e-6, 0.25)
    mtot = mc / eta ** 0.6
    return mtot / 2 * (1 + np.sqrt(np.clip(1 - 4 * eta, 0, 1)))


def survival(m1, w, grid):
    """S(m) = sum_{m1>m} w / sum w, on grid (weighted survival function)."""
    order = np.argsort(m1); m1s = m1[order]; ws = w[order]
    csum = np.cumsum(ws); tot = csum[-1]
    # fraction with m1 > grid point = 1 - CDF
    idx = np.searchsorted(m1s, grid, side="right")
    below = np.where(idx > 0, csum[np.clip(idx - 1, 0, len(csum) - 1)], 0.0)
    return 1.0 - below / tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seed", type=int, default=3000)
    p.add_argument("--n-b", type=int, default=2500)
    p.add_argument("--n-draw", type=int, default=12)
    args = p.parse_args()
    pdet = make_real_pdet_fn()
    grid = np.linspace(5, 100, 400)

    # ---- empirical detected CDF + binomial band ----
    cat = np.load(CAT, allow_pickle=True); cen = cat["centers"]
    m1o = m1_from_mc_eta(cen[:, 0], cen[:, 1]); N = len(m1o)
    Semp = np.array([(m1o > g).sum() / N for g in grid])
    Serr = np.sqrt(np.clip(Semp * (1 - Semp), 0, None) / N)   # binomial 1-sigma
    print(f"empirical: N={N}  S(40)={ (m1o>40).sum()/N:.3f}  S(60)={(m1o>60).sum()/N:.3f}  S(80)={(m1o>80).sum()/N:.3f}")

    # ---- predicted detected CDF per f_dep ----
    base = jax.random.PRNGKey(0)
    curves = []
    for fi, (fdep, d, col) in enumerate(SHOW):
        post = np.load(os.path.join(ROOT, d, "two_channel_pl_post.npz"))
        n = len(post["m_max_A"]); didx = np.linspace(0, n - 1, min(args.n_draw, n)).astype(int)
        M.F_DEP = fdep
        m1L, wL = [], []
        for dj, di in enumerate(didx):
            M.PROPOSAL_KEY = jax.random.fold_in(jax.random.fold_in(base, fi), int(dj))
            M.SAMPLER_KEY = jax.random.fold_in(jax.random.fold_in(base, 100 + fi), int(dj))
            comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
            th = {k: float(post[k][di]) for k in KS}
            s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))
            m1 = np.maximum(s[:, 0], s[:, 1])
            dw = w * np.asarray(pdet(s))              # DETECTED weight = rate x P_det
            ok = np.isfinite(m1) & (dw > 0)
            m1L.append(m1[ok]); wL.append(dw[ok])
        m1 = np.concatenate(m1L); dw = np.concatenate(wL)
        Smod = survival(m1, dw, grid)
        # detected high-mass fractions for the log
        f40 = survival(m1, dw, np.array([40.]))[0]; f60 = survival(m1, dw, np.array([60.]))[0]
        print(f"  f_dep={fdep}: predicted detected S(40)={f40:.3f} S(60)={f60:.3f}  "
              f"(obs {Semp[np.searchsorted(grid,40)]:.3f}, {Semp[np.searchsorted(grid,60)]:.3f})")
        curves.append((fdep, col, Smod))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a in ax:
        a.fill_between(grid, np.clip(Semp - Serr, 0, 1), np.clip(Semp + Serr, 0, 1),
                       color="0.6", alpha=0.4, label="empirical $\\pm1\\sigma$ (binomial)", step="mid")
        a.step(grid, Semp, where="mid", color="k", lw=2, label=f"empirical ($N={N}$)")
        for fdep, col, Smod in curves:
            a.plot(grid, Smod, color=col, lw=2, label=rf"model $F_{{\rm dep}}={fdep}$")
        a.set_xlabel(r"$m_1\ [M_\odot]$"); a.set_ylabel(r"$P(M_1 > m_1)$ (detected)")
    ax[0].set_xlim(5, 100); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    ax[0].set_title("Detected primary-mass survival CDF")
    ax[1].set_yscale("log"); ax[1].set_xlim(20, 100); ax[1].set_ylim(3e-3, 1)
    ax[1].set_title("High-mass tail (log scale)")
    fig.suptitle(r"Predicted vs observed detected primary-mass distribution "
                 rf"($n_{{\rm seed}}={args.n_seed}$, {args.n_draw} draws pooled)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT, dpi=140, bbox_inches="tight"); print("wrote", OUT)


if __name__ == "__main__":
    raise SystemExit(main())
