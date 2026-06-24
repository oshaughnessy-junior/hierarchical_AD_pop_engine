"""Model-side INTRINSIC population distributions for the GWTC-5 overlay.

Generates the f_dep deep-cascade oriented-spin population (NO selection) at a set
of hyper-posterior draws and computes the population distributions GWTC-5 reports:
  p(m1) / dR/dm1, p(chi_eff), p(chi) [component spin magnitude], and the
  spin-vs-mass diagnostics <a|m1>, sigma(chi_eff|m1). Saves grids + 5/50/95 bands
  to npz for plotting against the GWTC-5 popsummary.

Spins: the cascade gives magnitudes a_i (1g ~ sigma_chi, 2g ~ 0.7); isotropic
tilts -> signed chi_eff. Both the magnitude p(chi) and the signed chi_eff are kept.

  python model_intrinsic.py --f-dep 0.30 --post <real_post.npz> --n-draw 40
"""
import argparse, importlib.util, os, numpy as np
from jax.scipy.special import logsumexp
from scipy.stats import gaussian_kde

_HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "plpeak", os.path.join(_HERE, "production_plpeak_sampled_recovery.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--f-dep", type=float, default=0.30)
    p.add_argument("--post", default="/home/richard.oshaughnessy/coag_container/figures_fdep030/real_post.npz")
    p.add_argument("--n-seed", type=int, default=8000)
    p.add_argument("--n-draw", type=int, default=40)
    p.add_argument("--out", default="/home/richard.oshaughnessy/coag_container/figures_fdep030/model_intrinsic.npz")
    args = p.parse_args()

    M.ISOTROPIC_SPIN = False     # keep magnitudes; we tilt in numpy to keep both
    M.F_DEP = args.f_dep
    sampler = M.make_sampler_fn(args.n_seed)
    post = np.load(args.post); keys = ("alpha", "lambda_peak", "mu_peak", "sigma_peak", "spin_scale")
    idx = np.linspace(0, len(post["alpha"]) - 1, min(args.n_draw, len(post["alpha"]))).astype(int)

    m1_grid = np.linspace(3, 100, 100)
    ce_grid = np.linspace(-1, 1, 80)
    a_grid = np.linspace(0, 1, 60)
    m1_bins = np.array([5, 12, 18, 25, 32, 45, 70]); m1c = 0.5 * (m1_bins[:-1] + m1_bins[1:])
    rng = np.random.default_rng(7)
    pm1, pce, pa, sig_ce_m, mean_a_m = [], [], [], [], []
    for i in idx:
        th = {k: float(post[k][i]) for k in keys}
        s, lw = sampler(th["alpha"], th["lambda_peak"], th["mu_peak"], th["sigma_peak"], th["spin_scale"], 0.0)
        s = np.asarray(s); w = np.exp(np.asarray(lw - logsumexp(lw)))
        m1 = np.maximum(s[:, 0], s[:, 1]); m2 = np.minimum(s[:, 0], s[:, 1])
        a1, a2 = np.abs(s[:, 2]), np.abs(s[:, 3])
        ct = rng.uniform(-1, 1, size=(len(s), 2))
        chie = (s[:, 0] * a1 * ct[:, 0] + s[:, 1] * a2 * ct[:, 1]) / (s[:, 0] + s[:, 1])
        ok = np.isfinite(m1) & np.isfinite(chie) & (w > 0)
        W = w[ok]; W = W / W.sum()
        pm1.append(gaussian_kde(m1[ok], weights=W, bw_method=0.12)(m1_grid))
        pce.append(gaussian_kde(chie[ok], weights=W, bw_method=0.12)(ce_grid))
        aa = np.concatenate([a1[ok], a2[ok]]); WW = np.concatenate([W, W]); WW /= WW.sum()
        pa.append(gaussian_kde(np.clip(aa, 0, 1), weights=WW, bw_method=0.12)(a_grid))
        sc, ma = [], []
        for lo, hi in zip(m1_bins[:-1], m1_bins[1:]):
            mb = ok & (m1 >= lo) & (m1 < hi); wb = w[mb]
            if wb.sum() > 1e-6:
                wb = wb / wb.sum(); mu = np.sum(wb * chie[mb])
                sc.append(np.sqrt(np.sum(wb * (chie[mb] - mu) ** 2)))
                ma.append(np.sum(wb * a1[mb]))
            else:
                sc.append(np.nan); ma.append(np.nan)
        sig_ce_m.append(sc); mean_a_m.append(ma)
    def band(x): return np.nanpercentile(np.array(x), [5, 50, 95], axis=0)
    np.savez(args.out, m1_grid=m1_grid, ce_grid=ce_grid, a_grid=a_grid, m1c=m1c,
             pm1=band(pm1), pce=band(pce), pa=band(pa),
             sig_ce_m=band(sig_ce_m), mean_a_m=band(mean_a_m), f_dep=args.f_dep)
    print("wrote", args.out)
    pm = band(pm1)
    print(f"p(m1) peak at m1~{m1_grid[np.argmax(pm[1])]:.0f}; "
          f"p(chi_eff) median width ~{np.sqrt(np.sum(band(pce)[1]*ce_grid**2)/np.sum(band(pce)[1])):.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
