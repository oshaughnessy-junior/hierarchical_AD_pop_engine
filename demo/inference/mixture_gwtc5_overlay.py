"""GWTC-5.0 overlay for the TWO-CHANNEL MIXTURE: log dR/dm1 with the cascade(A) +
aligned(B) component decomposition, and the spin marginals (chi_eff, magnitude a,
cos-tilt) -- model vs GWTC-5.0 inferred population.

Key points addressed:
  * dR/dm1 on a LOG axis (a rate, not a normalized probability), model normalized
    to the GWTC-5 integrated rate over the common range; A and B shown separately.
  * the spin panels make explicit that the model spins are largely PREDICTIONS,
    not free fits: Channel A's high-magnitude tail is the NR remnant spin (~0.7)
    of its 2g+ mergers (only the natal scale sigma_chi is free); Channel B's
    aligned spins are fit. The excess vs GWTC-5 is the cascade's forced 2g
    signature the data do not support.

  python mixture_gwtc5_overlay.py --post <two_channel_post.npz> --f-dep 0.30
"""
import argparse, importlib.util, os, glob, numpy as np, h5py
from jax.scipy.special import logsumexp
import jax.numpy as jnp
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(__file__)
TC = importlib.util.spec_from_file_location("tc", os.path.join(_HERE, "two_channel_demo.py"))
tc = importlib.util.module_from_spec(TC); TC.loader.exec_module(tc)
M = tc.M
M.ISOTROPIC_SPIN = False        # return spin MAGNITUDES; we tilt in numpy to keep both


def gw(path, param):
    with h5py.File(path, "r") as h:
        g = h[f"posterior/rates_on_grids/{param}"]
        x = np.asarray(g["positions"]).squeeze(); r = np.asarray(g["rates"])
    return x, np.nanpercentile(r, [5, 50, 95], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_two_channel_real/two_channel_post.npz")
    p.add_argument("--f-dep", type=float, default=0.30)
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--n-draw", type=int, default=25)
    p.add_argument("--out", default="/home/richard.oshaughnessy/coag_container/figures_two_channel_real/mixture_gwtc5_overlay.png")
    args = p.parse_args()
    M.F_DEP = args.f_dep
    comb, nA = tc.make_combined_sampler(args.n_seed, args.n_b)
    base = os.path.expanduser("~/coag_container/vt_data/gwtc5_pop/popsummary_files")
    gw_def = glob.glob(f"{base}/**/*default_mmax*popsummary_result.h5", recursive=True)[0]
    gw_chi = glob.glob(f"{base}/*ChiEff*/*with2Drates.h5")[0]
    post = np.load(args.post)
    keys = ["mu_A", "sig_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
    idx = np.linspace(0, len(post["mu_A"]) - 1, min(args.n_draw, len(post["mu_A"]))).astype(int)

    m1g = np.linspace(3, 100, 120); ceg = np.linspace(-1, 1, 80)
    ag = np.linspace(0, 1, 60); ctg = np.linspace(-1, 1, 60)
    rng = np.random.default_rng(3)
    dR_t, dR_A, dR_B, pce, pa, pct = [], [], [], [], [], []
    for i in idx:
        th = {k: float(post[k][i]) for k in keys}
        s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))     # rate weights
        isA = np.zeros(len(s), bool); isA[:nA] = True
        m1 = np.maximum(s[:, 0], s[:, 1]); a1 = np.abs(s[:, 2]); a2 = np.abs(s[:, 3])
        ct = rng.uniform(-1, 1, (len(s), 2))
        cos1 = np.where(isA, ct[:, 0], 1.0); cos2 = np.where(isA, ct[:, 1], 1.0)  # B aligned
        chie = (s[:, 0] * a1 * cos1 + s[:, 1] * a2 * cos2) / (s[:, 0] + s[:, 1])
        cost = np.where(isA, ct[:, 0], 1.0)                                       # per-object tilt
        ok = np.isfinite(m1) & (w > 0)
        def kde(x, g, wt, m):
            mm = ok & m
            if wt[mm].sum() < 1e-9: return np.zeros_like(g)
            return gaussian_kde(x[mm], weights=wt[mm], bw_method=0.10)(g) * wt[mm].sum()
        dR_A.append(kde(m1, m1g, w, isA)); dR_B.append(kde(m1, m1g, w, ~isA))
        dR_t.append(dR_A[-1] + dR_B[-1])
        wn = w[ok] / w[ok].sum()
        pce.append(gaussian_kde(chie[ok], weights=wn, bw_method=0.10)(ceg))
        aa = np.concatenate([a1[ok], a2[ok]]); wa = np.concatenate([wn, wn]); wa /= wa.sum()
        pa.append(gaussian_kde(np.clip(aa, 0, 1), weights=wa, bw_method=0.10)(ag))
        cc = np.concatenate([cost[ok], cost[ok]]); pct.append(gaussian_kde(np.clip(cc, -1, 1), weights=wa, bw_method=0.08)(ctg))
    dR_t, dR_A, dR_B = map(lambda z: np.array(z), (dR_t, dR_A, dR_B))

    # normalize model dR/dm1 to the GWTC-5 integrated rate over [5,80]
    x_m, (mlo, mmd, mhi) = gw(gw_def, "mass_1")
    sel = (x_m >= 5) & (x_m <= 80); gw_int = np.trapezoid(mmd[sel], x_m[sel])
    mod_int = np.trapezoid(np.median(dR_t, 0)[(m1g >= 5) & (m1g <= 80)], m1g[(m1g >= 5) & (m1g <= 80)])
    sc = gw_int / mod_int
    dR_t *= sc; dR_A *= sc; dR_B *= sc

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    # (1) log dR/dm1
    a0 = ax[0, 0]
    a0.fill_between(x_m, mlo, mhi, color="grey", alpha=0.3, label="GWTC-5 90%"); a0.plot(x_m, mmd, "k", lw=2)
    lo, md, hi = np.percentile(dR_t, [5, 50, 95], 0)
    a0.fill_between(m1g, lo, hi, color="C0", alpha=0.3, label="mixture (total)"); a0.plot(m1g, md, "C0", lw=2)
    a0.plot(m1g, np.median(dR_A, 0), "C3--", lw=1.6, label="cascade (A)")
    a0.plot(m1g, np.median(dR_B, 0), "C2:", lw=1.8, label="aligned (B)")
    a0.set_yscale("log"); a0.set_xlim(4, 90); a0.set_ylim(np.median(dR_t, 0).max() * 3e-4, np.median(dR_t, 0).max() * 3)
    a0.set_xlabel(r"$m_1\ [M_\odot]$"); a0.set_ylabel(r"$dR/dm_1$ $[{\rm Gpc}^{-3}\,{\rm yr}^{-1}\,M_\odot^{-1}]$")
    a0.set_title("Mass function (log scale) + channel decomposition"); a0.legend(fontsize=8)
    # (2) chi_eff
    def overlay(axx, gx, gband, modg, modband, xl, ttl):
        lo, md, hi = gband; gn = np.trapezoid(md, gx)            # normalize to unit integral (shape)
        axx.fill_between(gx, lo / gn, hi / gn, color="grey", alpha=0.3, label="GWTC-5 90%")
        axx.plot(gx, md / gn, "k", lw=2)
        l2, m2, h2 = np.percentile(modband, [5, 50, 95], 0); mn = np.trapezoid(m2, modg)
        axx.fill_between(modg, l2 / mn, h2 / mn, color="C0", alpha=0.3, label="mixture")
        axx.plot(modg, m2 / mn, "C0", lw=2)
        axx.set_xlabel(xl); axx.set_ylabel("norm. density"); axx.set_title(ttl); axx.legend(fontsize=8)
    overlay(ax[0, 1], *gw(gw_chi, "chi_eff"), ceg, np.array(pce), r"$\chi_{\rm eff}$", "Effective spin")
    overlay(ax[1, 0], *gw(gw_def, "a_1"), ag, np.array(pa), r"$a$ (spin magnitude)", "Spin magnitude (model: 2g forced ~0.7)")
    overlay(ax[1, 1], *gw(gw_def, "cos_tilt_1"), ctg, np.array(pct), r"$\cos\theta$", "Spin tilt (A isotropic + B aligned)")
    fig.suptitle("Two-channel mixture vs GWTC-5.0: mass (log dR/dm1) and spin", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight"); print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
