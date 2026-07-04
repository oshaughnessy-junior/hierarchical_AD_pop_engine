"""Post-hoc correction for the missing (1+z)^kappa in the fit's SELECTION.

The two-channel fits used a constant-comoving (kappa=0) mass_pdet_grid, so the
expected-count mu(Lambda) omitted the redshift evolution (Eq. 4). Rather than
re-fit, we importance-reweight the EXISTING posterior from the kappa=0 selection
to a kappa=2.7 selection. Since only the selection changes (the source-frame NAL
per-event terms are z-independent and unchanged), the joint importance weight is

    w_i = exp( -(mu_new(Lambda_i) - mu_old(Lambda_i)) ),
    mu(Lambda) = T_obs * sum_s w_s(Lambda) P_det(theta_s),

with P_det from the kappa=2.7 (new) vs kappa=0 (old) grids. mu ~ N_obs, so a
UNIFORM shift normalizes out; only the sample-to-sample VARIATION of
(mu_new - mu_old) drives the effective sample size. CRITICAL: the selection
enters as xi^-N with N=238, so if xi_new/xi_old varies even ~1% across the
posterior the ESS collapses and a real re-fit is required -- this script MEASURES
that before trusting the reweight.

  python reweight_kappa.py --f-dep 0.90 --n-draw 200
"""
import argparse, importlib.util, os, numpy as np, jax
from mass_pdet_interp import make_real_pdet_fn

_HERE = os.path.dirname(os.path.abspath(__file__))
PL = importlib.util.spec_from_file_location("pl", os.path.join(_HERE, "two_channel_pl_demo.py"))
pl = importlib.util.module_from_spec(PL); PL.loader.exec_module(pl)
M = pl.M; M.ISOTROPIC_SPIN = False
ROOT = "/home/richard.oshaughnessy/coag_container"
KS = ["alpha_A", "m_max_A", "spin_chi", "mu_B", "sig_B", "mu_chiB", "sig_chiB", "log_R_A", "log_R_B"]
GRID_OLD = "/home/richard.oshaughnessy/coag_container/vt_data/mass_pdet_grid.npz"          # kappa=0
GRID_NEW = "/home/richard.oshaughnessy/coag_container/vt_data/mass_pdet_grid_kappa2.7.npz"  # kappa=2.7


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--f-dep", type=float, default=0.90)
    p.add_argument("--n-draw", type=int, default=200)
    p.add_argument("--n-seed", type=int, default=3000)
    p.add_argument("--n-b", type=int, default=2500)
    args = p.parse_args()
    M.F_DEP = args.f_dep
    rundir = os.path.join(ROOT, f"figures_nal_f{int(round(args.f_dep*100)):03d}")
    post = np.load(os.path.join(rundir, "two_channel_pl_post.npz"))
    n = len(post["m_max_A"]); idx = np.linspace(0, n - 1, min(args.n_draw, n)).astype(int)
    pdet_old = make_real_pdet_fn(GRID_OLD); pdet_new = make_real_pdet_fn(GRID_NEW)
    base = jax.random.PRNGKey(0)

    mu_old, mu_new, xi_ratio = [], [], []
    for row, i in enumerate(idx):
        M.PROPOSAL_KEY = jax.random.fold_in(base, row)
        M.SAMPLER_KEY = jax.random.fold_in(base, 100000 + row)
        comb, nA = pl.make_combined_pl(args.n_seed, args.n_b)
        th = {k: float(post[k][i]) for k in KS}
        s, lw = comb(**th); s = np.asarray(s); w = np.exp(np.asarray(lw))   # sum w = R_A+R_B
        po = np.asarray(pdet_old(s)); pn = np.asarray(pdet_new(s))
        mo = float(np.sum(w * po)); mn = float(np.sum(w * pn))              # T_obs=1
        mu_old.append(mo); mu_new.append(mn)
        xi_ratio.append((np.sum(w * pn) / np.sum(w)) / max(np.sum(w * po) / np.sum(w), 1e-30))
        if row < 5 or row % 25 == 0:
            print(f"  {row+1}/{len(idx)}  mu_old={mo:.1f} mu_new={mn:.1f}  d_mu={mn-mo:+.1f}  "
                  f"xi_new/xi_old={xi_ratio[-1]:.3f}", flush=True)
    mu_old = np.array(mu_old); mu_new = np.array(mu_new); xi_ratio = np.array(xi_ratio)

    dmu = mu_new - mu_old
    logw = -(dmu - dmu.min())                 # subtract min for numerical stability
    w = np.exp(logw); w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    print(f"\n=== kappa reweight, f_dep={args.f_dep}, {len(idx)} samples ===")
    print(f"xi_new/xi_old: mean={xi_ratio.mean():.3f} std={xi_ratio.std():.3f} "
          f"[{xi_ratio.min():.3f},{xi_ratio.max():.3f}]")
    print(f"d_mu = mu_new-mu_old: mean={dmu.mean():+.1f} std={dmu.std():.1f}")
    print(f"ESS = {ess:.1f} / {len(idx)}  ({100*ess/len(idx):.1f}%)  "
          f"-> {'OK, reweight viable' if ess > 0.1*len(idx) else 'TOO LOW: re-fit needed'}")
    # shape shift under reweight (high-mass params)
    for k in ["m_max_A", "alpha_A", "log_R_A", "log_R_B"]:
        v = post[k][idx]
        print(f"  {k:9s}: orig {v.mean():.3f} -> reweighted {np.sum(w*v):.3f}")
    out = os.path.join(rundir, "kappa_reweight.npz")
    np.savez(out, idx=idx, weight=w, mu_old=mu_old, mu_new=mu_new, xi_ratio=xi_ratio, ess=ess)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
