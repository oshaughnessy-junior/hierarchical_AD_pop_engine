"""GWTC-3/4/5-style 2D merger-rate density dR/dm_1 dm_2 for the coagulation
f_dep=0.9 result (NO marginalization -- the single f_dep=0.9 posterior).

Forward-samples the two-channel model at the f_dep=0.9 posterior draws, builds a
rate-weighted 2D KDE in (m_1, m_2), and renders it in the LVK house style: log-log
axes, the m_2 <= m_1 physical region, a sequential filled density + a few credible
contours, m_1=m_2 diagonal, colorbar. Companion to the dR/dm_1 marginal
(real_dRdm1_marginal.png)."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(os.path.abspath(__file__))
KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
ROOT = "/home/richard.oshaughnessy/coag_container"
OUT = "/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/gwk-hierarchical-initial/paper/figures"


def sample_f090(n_seed=3000, n_b=2500, n_draw=12):
    """Return (m1, m2, w) pooled over f_dep=0.9 posterior draws (no F_dep marg).
    The heavy engine import is LAZY here so plotting from the cache is instant."""
    import importlib.util, jax
    PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
    pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
    M = pl.M; M.ISOTROPIC_SPIN = False
    post = np.load(os.path.join(ROOT, "figures_nal_f090", "two_channel_pl_post.npz"))
    n = len(post["m_max_A"]); idx = np.linspace(0, n - 1, min(n_draw, n)).astype(int)
    M.F_DEP = 0.90
    base = jax.random.PRNGKey(0); ctr = 0
    M1, M2, W = [], [], []
    for i in idx:
        # rotate pairing + natal-proposal keys per draw so the rare high-mass tail
        # is not pinned to one seed set (matches marginal_ppd.py).
        M.PROPOSAL_KEY = jax.random.fold_in(base, ctr)
        M.SAMPLER_KEY = jax.random.fold_in(base, 100000 + ctr); ctr += 1
        comb, nA = pl.make_combined_pl(n_seed, n_b)
        th = {k: float(post[k][i]) for k in KS}
        s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))
        m0, m1c = s[:, 0], s[:, 1]
        prim = m0 >= m1c
        m1 = np.where(prim, m0, m1c); m2 = np.where(prim, m1c, m0)
        ok = np.isfinite(m1) & np.isfinite(m2) & (w > 0) & (m2 > 0)
        M1.append(m1[ok]); M2.append(m2[ok]); W.append(w[ok] / w[ok].sum() / len(idx))
        print(f"  draw {ctr}/{len(idx)}  (+{int(ok.sum())} events)", flush=True)
    return np.concatenate(M1), np.concatenate(M2), np.concatenate(W)


def main():
    # Cache the (m1,m2,w) pool so restyling the plot needs no re-sampling.
    cache = os.path.join(_HERE, "_dRdm1dm2_f090_pool.npz")
    if os.path.exists(cache):
        z = np.load(cache); m1, m2, w = z["m1"], z["m2"], z["w"]
        print(f"[pool] loaded {m1.size} cached events")
    else:
        m1, m2, w = sample_f090()
        np.savez(cache, m1=m1, m2=m2, w=w)
        print(f"[pool] {m1.size} events (cached); m1 med {np.median(m1):.1f}, m2 med {np.median(m2):.1f}")

    # rate-weighted KDE in log10 mass, evaluated on a LINEAR (m1,m2) grid, then
    # converted to the density per (m1,m2). GWTC-5 Fig.1 style: linear axes, blue
    # lower triangle (m2<=m1), rate density colorbar.
    kde = gaussian_kde(np.vstack([np.log10(m1), np.log10(m2)]), weights=w, bw_method=0.11)
    gm = np.linspace(3.0, 95.0, 260)
    M1g, M2g = np.meshgrid(gm, gm)                 # X = m1, Y = m2
    plog = kde(np.vstack([np.log10(M1g).ravel(), np.log10(M2g).ravel()])).reshape(M1g.shape)
    dR = plog / (M1g * M2g * np.log(10.0) ** 2)    # density per (m1, m2)
    dR = np.where(M2g <= M1g, dR, np.nan)          # physical region m2 <= m1
    # mask the PE-sparse corner (very low density) like GWTC's hatching
    vmax = np.nanmax(dR); vmin = vmax * 1e-3

    fig, ax = plt.subplots(figsize=(6.6, 5.7))
    pcm = ax.pcolormesh(gm, gm, np.clip(dR, vmin, None), norm=LogNorm(vmin, vmax),
                        cmap="Blues", shading="gouraud", rasterized=True)
    ax.contour(gm, gm, dR, levels=vmax * np.array([1e-2, 1e-1, 0.5]),
               colors="0.25", linewidths=0.7, alpha=0.6)
    ax.plot([3, 95], [3, 95], ls="--", color="0.5", lw=0.9, alpha=0.7)
    ax.set_xlim(3, 95); ax.set_ylim(3, 95)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$m_1\ [M_\odot]$"); ax.set_ylabel(r"$m_2\ [M_\odot]$")
    ax.set_title(r"Coagulation merger-rate density $dR/dm_1\,dm_2$, $f_{\rm dep}=0.9$")
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$dR/dm_1\,dm_2$ (arb.\ norm.)")
    fig.tight_layout()
    out = os.path.join(OUT, "real_dRdm1dm2_f090.png")
    for f in (out, out.replace(".png", ".pdf")):
        fig.savefig(f, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
