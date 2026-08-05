"""GWTC-5.0 overlay for the FREE-m_max two-channel NAL fit (preferred f_dep=0.6).

Recovered intrinsic coagulation population vs the GWTC-5.0 inferred population:
  * dR/dm1 on a LOG rate axis (model normalized to the GWTC-5 integrated rate),
    with the natal+cascade (A) and aligned low-mass (B) decomposition.
  * spin marginals chi_eff, magnitude a, cos-tilt. The model spins are largely
    PREDICTIONS: channel A's high-magnitude component is the NR remnant spin
    (~0.7) of its 2g+ mergers (only the natal scale sigma_chi is free); channel B
    is aligned and low. With the corrected NAL likelihoods the model's broad
    high-mass chi_eff is SUPPORTED by the per-event data (cf. the PPC); this
    panel additionally compares it to GWTC-5.0's own parametric population fit.

  python gwtc5_overlay_nal.py   # defaults to figures_nal_f090_fa0w, f_dep=0.9 (flat-natal headline)
"""
import argparse, importlib.util, os, glob, numpy as np, h5py
import jax, jax.numpy as jnp
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(__file__)
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M
M.ISOTROPIC_SPIN = False        # return spin MAGNITUDES; tilt in numpy to keep both

KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]


def gw(path, param):
    with h5py.File(path, "r") as h:
        g = h[f"posterior/rates_on_grids/{param}"]
        x = np.asarray(g["positions"]).squeeze(); r = np.asarray(g["rates"])
    return x, np.nanpercentile(r, [5, 50, 95], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_nal_f090_fa0w/two_channel_pl_post.npz")
    p.add_argument("--f-dep", type=float, default=0.90)
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-b", type=int, default=6000)
    p.add_argument("--n-draw", type=int, default=64)
    p.add_argument("--mc-key", type=int, default=0)
    p.add_argument("--out", default="/home/richard.oshaughnessy/unixhome/Projects/LIGO-ScienceMode/communications/20260516-Me-CoagulationEngine/gwk-hierarchical-initial/paper/figures/real_gwtc5_overlay_nal.png")
    args = p.parse_args()
    M.F_DEP = args.f_dep
    comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
    base = os.path.expanduser("~/coag_container/vt_data/gwtc5_pop/popsummary_files")
    gw_def = glob.glob(f"{base}/**/*default_mmax*popsummary_result.h5", recursive=True)[0]
    gw_chi = glob.glob(f"{base}/*ChiEff*/*with2Drates.h5")[0]
    post = np.load(args.post)
    idx = np.linspace(0, len(post["alpha_A"]) - 1, min(args.n_draw, len(post["alpha_A"]))).astype(int)

    m1g = np.linspace(3, 100, 120); ceg = np.linspace(-1, 1, 80)
    ag = np.linspace(0, 1, 60); ctg = np.linspace(-1, 1, 60)
    rng = np.random.default_rng(3)
    base_key = jax.random.PRNGKey(int(args.mc_key))
    dR_t, dR_A, dR_B, pce, pa, pct = [], [], [], [], [], []
    poolm1, poolm2, poolw = [], [], []          # pooled population for the injection-VT rate
    for j, i in enumerate(idx):
        # independent MC realization per draw, rotating BOTH keys: the pairing
        # SAMPLER_KEY (Monte-Carlo noise) and the natal PROPOSAL_KEY (the heavy-tail
        # seed set). make_proposal otherwise reuses one fixed seed set for every
        # draw, which pins the rare high-mass lumps; resampling it decorrelates the
        # tail. Rebuild is cheap -- the compiled MC keys on array shapes, not values.
        M.PROPOSAL_KEY = jax.random.fold_in(base_key, j)
        M.SAMPLER_KEY = jax.random.fold_in(base_key, 10000 + j)
        comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
        th = {k: float(post[k][i]) for k in KS}
        # STALENESS REPAIR (2026-08-04): make_combined_pl's `combined()` gained a
        # REQUIRED keyword-only `v_esc` after this script was last run, so it had
        # been failing outright. Take it per-draw from the posterior when present
        # else M.V_ESC (= 300 km/s).
        # CORRECTED 2026-08-04 (was 1000.0): this script's DEFAULT --post is
        # figures_nal_f090_fa0w, whose committed config (refit_exchange/NOTES.md
        # sec.4) does not pass --v-esc; --v-esc defaults to M.V_ESC, so that
        # posterior was fitted at 300 km/s. Evaluating it forward at 1000 raises
        # remnant retention well above what the fit assumed -- a physics change,
        # not an API repair. (Note 55 does evaluate this posterior forward at
        # v_esc=1000, but that is a deliberate labelled forward assessment, not
        # the self-consistent overlay this figure shows.)
        th["v_esc"] = float(np.asarray(post["v_esc"])[i]) if "v_esc" in post.files else float(M.V_ESC)
        s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))     # rate weights
        isA = np.zeros(len(s), bool); isA[:nA] = True
        m1 = np.maximum(s[:, 0], s[:, 1]); m2 = np.minimum(s[:, 0], s[:, 1])
        a1 = np.abs(s[:, 2]); a2 = np.abs(s[:, 3])
        _ok = np.isfinite(m1) & np.isfinite(m2) & (w > 0)
        poolm1.append(m1[_ok]); poolm2.append(m2[_ok]); poolw.append(w[_ok])
        ct = rng.uniform(-1, 1, (len(s), 2))
        cos1 = np.where(isA, ct[:, 0], 1.0); cos2 = np.where(isA, ct[:, 1], 1.0)  # B aligned
        chie = (s[:, 0] * a1 * cos1 + s[:, 1] * a2 * cos2) / (s[:, 0] + s[:, 1])
        cost = np.where(isA, ct[:, 0], 1.0)
        ok = np.isfinite(m1) & (w > 0)
        def kde(x, g, wt, m):
            mm = ok & m
            if wt[mm].sum() < 1e-9: return np.zeros_like(g)
            return gaussian_kde(x[mm], weights=wt[mm], bw_method=0.10)(g) * wt[mm].sum()
        dR_A.append(kde(m1, m1g, w, isA)); dR_B.append(kde(m1, m1g, w, ~isA))
        dR_t.append(dR_A[-1] + dR_B[-1])
        wn = w[ok] / w[ok].sum()
        pce.append(gaussian_kde(chie[ok], weights=wn, bw_method=0.12)(ceg))
        aa = np.concatenate([a1[ok], a2[ok]]); wa = np.concatenate([wn, wn]); wa /= wa.sum()
        pa.append(gaussian_kde(np.clip(aa, 0, 1), weights=wa, bw_method=0.10)(ag))
        cc = np.concatenate([cost[ok], cost[ok]]); pct.append(gaussian_kde(np.clip(cc, -1, 1), weights=wa, bw_method=0.08)(ctg))
    dR_t, dR_A, dR_B = map(lambda z: np.array(z), (dR_t, dR_A, dR_B))

    x_m, (mlo, mmd, mhi) = gw(gw_def, "mass_1")
    # ABSOLUTE rate: self-calibrated z=0 total rate R0 from the injection VT
    # (vt_calibration, validated), NOT a GWTC-5 anchor. Scale the channel curves so
    # the median TOTAL dR/dm1 integrates to R0; the A/B split keeps its rate share.
    import vt_calibration as V
    R0 = V.rate_R0(V.mass_density_from_samples(np.concatenate(poolm1),
                                               np.concatenate(poolm2), np.concatenate(poolw)))
    mod_int = np.trapezoid(np.median(dR_t, 0), m1g)
    sc = R0 / mod_int; dR_t *= sc; dR_A *= sc; dR_B *= sc
    print(f"[vt] self-calibrated R0={R0:.4g} Gpc^-3 yr^-1 (kappa={V.KAPPA_LVK})")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a0 = ax[0, 0]
    a0.fill_between(x_m, mlo, mhi, color="grey", alpha=0.3, label="GWTC-5.0 90%"); a0.plot(x_m, mmd, "k", lw=2)
    lo, md, hi = np.percentile(dR_t, [5, 50, 95], 0)
    a0.fill_between(m1g, lo, hi, color="C0", alpha=0.3, label="model (total)"); a0.plot(m1g, md, "C0", lw=2)
    a0.plot(m1g, np.median(dR_A, 0), "C3--", lw=1.6, label="natal+cascade (A)")
    a0.plot(m1g, np.median(dR_B, 0), "C2:", lw=1.8, label="aligned (B)")
    a0.set_yscale("log"); a0.set_xlim(4, 90); a0.set_ylim(np.median(dR_t, 0).max() * 3e-4, np.median(dR_t, 0).max() * 3)
    a0.set_xlabel(r"$m_1\ [M_\odot]$"); a0.set_ylabel(r"$dR/dm_1$ $[{\rm Gpc}^{-3}\,{\rm yr}^{-1}\,M_\odot^{-1}]$")
    a0.set_title("Mass function (log scale) + channel decomposition"); a0.legend(fontsize=8)

    def overlay(axx, gx, gband, modg, modband, xl, ttl):
        lo, md, hi = gband; gn = np.trapezoid(md, gx)
        axx.fill_between(gx, lo / gn, hi / gn, color="grey", alpha=0.3, label="GWTC-5.0 90%")
        axx.plot(gx, md / gn, "k", lw=2)
        l2, m2, h2 = np.percentile(modband, [5, 50, 95], 0); mn = np.trapezoid(m2, modg)
        axx.fill_between(modg, l2 / mn, h2 / mn, color="C0", alpha=0.3, label="model")
        axx.plot(modg, m2 / mn, "C0", lw=2)
        axx.set_xlabel(xl); axx.set_ylabel("norm. density"); axx.set_title(ttl); axx.legend(fontsize=8)
    overlay(ax[0, 1], *gw(gw_chi, "chi_eff"), ceg, np.array(pce), r"$\chi_{\rm eff}$", "Effective spin")
    overlay(ax[1, 0], *gw(gw_def, "a_1"), ag, np.array(pa), r"$a$ (spin magnitude)", "Spin magnitude (A: natal + 2g $\\sim$0.7)")
    overlay(ax[1, 1], *gw(gw_def, "cos_tilt_1"), ctg, np.array(pct), r"$\cos\theta$", "Spin tilt (A isotropic + B aligned)")
    fig.suptitle(r"Free-$m_{\max}$ NAL fit ($F_{\rm dep}=0.9$, flat natal $\alpha_A=0$) vs GWTC-5.0: mass ($\log\,dR/dm_1$) and spin", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight"); print("wrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
