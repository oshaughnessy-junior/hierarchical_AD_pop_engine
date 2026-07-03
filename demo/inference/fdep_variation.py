"""Appendix systematic: how the key observables change across cascade-brightness
(F_DEP) realizations. Overlays, for several F_DEP grid points, the detected
merger-rate density dR/dm1 and the primary spin-magnitude trend chi(m1) --
showing how a fainter/brighter cascade shifts the high-mass tail and the onset of
the second-generation (chi~0.7) spin population.

  python fdep_variation.py
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
from spin_mass_trend import kernel_quantiles
KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
ROOT = "/home/richard.oshaughnessy/coag_container"
SHOW = [(0.30, "figures_nal_f030"), (0.60, "figures_nal_f060"), (0.90, "figures_nal_f090"), (1.20, "figures_nal_f120")]
OUT = "/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/gwk-hierarchical-initial/paper/figures/real_fdep_variation.png"


def gwmass():
    base = os.path.expanduser("~/coag_container/vt_data/gwtc5_pop/popsummary_files")
    path = glob.glob(f"{base}/**/*default_mmax*popsummary_result.h5", recursive=True)[0]
    with h5py.File(path, "r") as h:
        g = h["posterior/rates_on_grids/mass_1"]
        x = np.asarray(g["positions"]).squeeze(); r = np.asarray(g["rates"])
    lo, med, hi = np.nanpercentile(r, [5, 50, 95], axis=0)
    return x, lo, med, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seed", type=int, default=3000)
    p.add_argument("--n-b", type=int, default=2500)
    p.add_argument("--n-draw", type=int, default=24, help="posterior draws pooled per F_DEP (each a distinct params+key realization; decorrelates high-mass tail)")
    args = p.parse_args()
    rng = np.random.default_rng(4); pdet = make_real_pdet_fn()
    m1g = np.linspace(3, 100, 120); grid = np.linspace(6, 99, 120)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    gx, glo, gr, ghi = gwmass()
    ax[0].fill_between(gx, glo, ghi, color="grey", alpha=0.25, label="GWTC-5.0 90%")
    ax[0].plot(gx, gr, "k", lw=1.2, label="GWTC-5.0 median")
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(SHOW)))
    base_key = jax.random.PRNGKey(0)
    for fi, ((fdep, d), col) in enumerate(zip(SHOW, cols)):
        if not os.path.exists(os.path.join(ROOT, d, "two_channel_pl_post.npz")):
            continue
        post = np.load(os.path.join(ROOT, d, "two_channel_pl_post.npz"))
        M.F_DEP = fdep
        n = len(post["m_max_A"]); didx = np.linspace(0, n - 1, min(args.n_draw, n)).astype(int)
        m1L, aL, wL = [], [], []
        for dj, di in enumerate(didx):
            # fresh natal PROPOSAL seeds AND pairing key per draw, then rebuild the
            # sampler: the heavy-tail seeds are otherwise ONE fixed finite set shared
            # by every draw (make_proposal uses a fixed PROPOSAL_KEY), which pins the
            # high-mass lumps -- reweighting by params or rotating only the MC key
            # cannot move them. Resampling the proposal decorrelates the rare tail.
            # Cheap: the compiled MC keys on array SHAPES, not values -> no recompile.
            M.PROPOSAL_KEY = jax.random.fold_in(jax.random.fold_in(base_key, fi), int(dj))
            M.SAMPLER_KEY = jax.random.fold_in(jax.random.fold_in(base_key, 100 + fi), int(dj))
            comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
            th = {k: float(post[k][di]) for k in KS}
            s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))
            m0, m1c, c0, c1 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            prim = m0 >= m1c; m1i = np.where(prim, m0, m1c); ai = np.abs(np.where(prim, c0, c1))
            ok = np.isfinite(m1i) & (w > 0)
            m1L.append(m1i[ok]); aL.append(ai[ok]); wL.append(w[ok] / len(didx))
        m1 = np.concatenate(m1L); a_prim = np.concatenate(aL); w = np.concatenate(wL)
        dR = gaussian_kde(m1, weights=w, bw_method=0.08)(m1g)   # intrinsic (match GWTC-5)
        sel = (gx >= 5) & (gx <= 80); selm = (m1g >= 5) & (m1g <= 80)
        dR *= np.trapezoid(gr[sel], gx[sel]) / np.trapezoid(dR[selm], m1g[selm])
        ax[0].plot(m1g, dR, color=col, lw=2, label=rf"$F_{{\rm dep}}={fdep}$")
        aq = kernel_quantiles(m1, a_prim, w, grid, 6.0, [0.5])
        ax[1].plot(grid, aq[:, 0], color=col, lw=2, label=rf"$F_{{\rm dep}}={fdep}$")
    ax[0].set_yscale("log"); ax[0].set_xlim(4, 90); ax[0].set_ylim(ghi.max() * 3e-4, ghi.max() * 3)
    ax[0].set_xlabel(r"$m_1\ [M_\odot]$"); ax[0].set_ylabel(r"$dR/dm_1$ $[{\rm Gpc}^{-3}\,{\rm yr}^{-1}\,M_\odot^{-1}]$")
    ax[0].set_title("Mass function vs cascade brightness"); ax[0].legend(fontsize=8)
    ax[1].axhline(0.69, color="grey", ls=":", lw=1.0)
    ax[1].set_xlim(5, 100); ax[1].set_ylim(0, 1); ax[1].set_xlabel(r"$m_1\ [M_\odot]$")
    ax[1].set_ylabel(r"$\chi$ (primary spin magnitude, median)")
    ax[1].set_title("Spin-magnitude trend vs cascade brightness"); ax[1].legend(fontsize=8)
    fig.suptitle(r"Dependence of the observables on cascade brightness $F_{\rm dep}$", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT, dpi=140, bbox_inches="tight"); print("wrote", OUT)


if __name__ == "__main__":
    raise SystemExit(main())
