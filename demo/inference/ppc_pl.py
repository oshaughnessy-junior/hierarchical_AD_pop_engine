"""Coupled mass+spin PPC for the POWER-LAW two-channel model (two_channel_pl_demo).
Mirrors ppc_two_channel.py but with the power-law Channel A + the PL param set.
"""
import argparse, importlib.util, os, numpy as np, jax, jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
from mass_pdet_interp import make_real_pdet_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_two_channel_pl/two_channel_pl_post.npz")
    p.add_argument("--catalog", default=os.path.join(_HERE, "../../../planning_o4_mixed/full_o3_o4a_o4b_catalog.npz"))
    p.add_argument("--f-dep", type=float, default=0.15)
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--n-draw", type=int, default=25)
    p.add_argument("--out", default="/home/richard.oshaughnessy/coag_container/figures_two_channel_pl/ppc_pl.png")
    args = p.parse_args()
    M.F_DEP = args.f_dep
    comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
    pdet = make_real_pdet_fn()
    post = np.load(args.post)
    keys = ["alpha_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
    idx = np.linspace(0, len(post["alpha_A"]) - 1, min(args.n_draw, len(post["alpha_A"]))).astype(int)
    mc_grid = np.linspace(4, 70, 80); mc_bins = np.array([5, 12, 18, 25, 32, 40, 55]); mc_cen = 0.5 * (mc_bins[:-1] + mc_bins[1:])
    dens, densA, densB, dvar = [], [], [], []
    for i in idx:
        th = {k: float(post[k][i]) for k in keys}
        s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw - logsumexp(lw)))
        dw = w * np.asarray(pdet(s)); dw /= dw.sum()
        m1, m2, c1, c2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2; chie = (m1 * c1 + m2 * c2) / (m1 + m2)
        ok = np.isfinite(mc) & (dw > 0); isA = np.zeros(len(s), bool); isA[:nA] = True
        dens.append(gaussian_kde(mc[ok], weights=dw[ok], bw_method=0.12)(mc_grid))
        for sub, st in [(isA, densA), (~isA, densB)]:
            mm = ok & sub
            st.append(gaussian_kde(mc[mm], weights=dw[mm] + 1e-12, bw_method=0.12)(mc_grid) * dw[mm].sum() if dw[mm].sum() > 1e-6 else np.zeros_like(mc_grid))
        d = []
        for lo, hi in zip(mc_bins[:-1], mc_bins[1:]):
            mm = ok & (mc >= lo) & (mc < hi); ww = dw[mm]
            d.append(np.sum((ww / ww.sum()) * (chie[mm] - np.sum((ww / ww.sum()) * chie[mm])) ** 2) if ww.sum() > 1e-6 else np.nan)
        dvar.append(d)
    dens, dvar = np.array(dens), np.array(dvar); densA, densB = np.array(densA), np.array(densB)
    cat = np.load(args.catalog, allow_pickle=True)
    mco, ceo, sco, smco = cat["centers"][:, 0], cat["centers"][:, 2], cat["sigs"][:, 2], cat["sigs"][:, 0]
    obs = np.zeros_like(mc_grid)
    for mu, sg in zip(mco, np.clip(smco, 0.3, None)):
        obs += np.exp(-0.5 * ((mc_grid - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
    obs /= len(mco)
    od, mv = [], []
    for lo, hi in zip(mc_bins[:-1], mc_bins[1:]):
        m = (mco >= lo) & (mco < hi)
        od.append(ceo[m].std() if m.sum() > 3 else np.nan); mv.append(np.mean(sco[m] ** 2) if m.sum() > 3 else np.nan)
    od, mv = np.array(od), np.array(mv)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    dn = dens / np.trapezoid(dens, mc_grid, axis=1)[:, None]; lo, md, hi = np.percentile(dn, [5, 50, 95], 0)
    ax[0].fill_between(mc_grid, lo, hi, alpha=0.3, color="C0", label="mixture 90%"); ax[0].plot(mc_grid, md, "C0", lw=2)
    nrm = np.trapezoid(dens, mc_grid, axis=1).mean()
    ax[0].plot(mc_grid, np.median(densA, 0) / nrm, "C3--", lw=1.5, label="power-law+cascade (A)")
    ax[0].plot(mc_grid, np.median(densB, 0) / nrm, "C2:", lw=1.8, label="aligned (B)")
    ax[0].plot(mc_grid, obs / np.trapezoid(obs, mc_grid), "k", lw=2, label="observed")
    ax[0].set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$"); ax[0].set_ylabel("detected (norm.)"); ax[0].set_title("Mass spectrum + decomposition"); ax[0].legend(fontsize=8)
    disp = np.sqrt(dvar + mv[None, :]); lo, md, hi = np.nanpercentile(disp, [5, 50, 95], 0)
    ax[1].fill_between(mc_cen, lo, hi, alpha=0.3, color="C1", label="model+meas 90%"); ax[1].plot(mc_cen, md, "C1", lw=2, marker="o", ms=4)
    ax[1].plot(mc_cen, od, "k", lw=2, marker="s", ms=5, label="observed (raw)")
    ax[1].set_xlabel(r"$\mathcal{M}_c\ [M_\odot]$"); ax[1].set_ylabel(r"$\sigma(\chi_{\rm eff})$"); ax[1].set_title("Effective-spin dispersion"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Power-law two-channel PPC (F_DEP={args.f_dep}, 238 events)", fontsize=11)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight"); print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
