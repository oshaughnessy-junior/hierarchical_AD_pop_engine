r"""Diagnose the n_obs=300 natal-spin-location (mu_chi,natal) recovery bias.

Question (ROS): is the likelihood higher at TRUTH than at the recovered mode?
 - If logL(truth) > logL(recovered): the fit is SUBOPTIMAL -> sampling/inference
   problem (better init / guide / more steps should recover the nonzero mean).
 - If logL(recovered) >= logL(truth): the forward model genuinely PREFERS
   mu_chi->0 -> a density-estimation / MC-size bias (spin density is hard right
   at the chi=0 edge, where the natal pulse mu_chi=0.05, sigma=0.10 piles up).

Also: profile mu_chi,natal holding the OTHER params at (a) truth vs (b) the
recovered medians — separates a pure spin-axis miss from a joint degeneracy that
drags spin low once the other (biased) params move. And sweep n_seed (MC size)
to test whether more particles sharpen the spin preference toward truth.

Reproduces posterior_stage1_gwkokab._setup EXACTLY (same seed split, sampler,
scatter) so the catalogue is the one the recovery fit saw. Pure evaluation.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

_DEMO = Path(__file__).resolve().parent
for p in (str(_DEMO), str(_DEMO.parent), str(_DEMO.parent / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_gwkokab_fit_stage1 as demo
import fit_stage1_gwkokab as fit
from mock_catalog import scatter_to_posteriors
from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.spin_orientation import Isotropic
import os as _os
if _os.environ.get("COAG_LEAN_GAIN"):
    import hierarchical_ad_pop_engine.orientation_precompute as _op
    import hierarchical_ad_pop_engine.response.implicit as _imp
    _op.R_CHI_OUT_DEFAULT = None; _imp.R_CHI_OUT_DEFAULT = None
from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (
    coagulation_poisson_mean_estimator, make_coagulation_dist_fn)

PN = fit.PARAM_NAMES  # mu_natal, sigma_natal, natal_spin_loc, natal_spin_scale, kernel_b, log_rate
_SAMPLER = _os.environ.get("DIAG_SAMPLER", "grid")  # "grid" (default) | "particles"


def setup(n_seed, n_obs=300, n_samples=5, k_max=3, n_quad=32, seed=20260530,
          delta=False, spin_loc_truth=0.05, hybrid_1g=None):
    fixed = demo.FIXED
    fit_grid = BinaryGrid.default(m_min=fixed["m_min"], m_max=fixed["m_max"],
                                  n_m=demo.N_M_FIT, n_chi=demo.N_CHI_FIT)
    demo.build_truth_model_and_calibrate_rate(n_obs, grid=fit_grid)
    demo.THETA_STAR["natal_spin_loc"] = spin_loc_truth
    theta_star = dict(demo.THETA_STAR)
    orientation = Isotropic.default(n_theta=2, n_dphi=4)
    key = jax.random.PRNGKey(seed)
    k_mc, k_cat = jax.random.split(key, 2)
    dist_fn = make_coagulation_dist_fn(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=k_max, n_seed=n_seed,
        depletion_fraction=0.10, gate="soft", grid=fit_grid, mc_jax_key=k_mc,
        solver="mc_jax", orientation=orientation, hybrid_1g=hybrid_1g)
    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = coagulation_poisson_mean_estimator(probe, T_obs=1.0, n_quad=n_quad)
    theta_star["log_rate"] = float(jnp.log(n_obs / mu_unit))
    truth_model = dist_fn(**theta_star)
    k_true, k_scat = jax.random.split(k_cat, 2)
    m_hi = fixed["m_max"] * 2.5
    if _SAMPLER == "particles":
        # exact particle spins (no coarse chi-grid smear) — draws binary pairs
        # straight from the mc_jax pool.
        true_pts = fit.sample_pairs_from_particles(truth_model, n_obs, k_true)
    else:
        true_pts = fit.sample_from_model(truth_model, n_obs, k_true,
                                         m_lo=fixed["m_min"], m_hi=m_hi)
    if delta:
        events = jnp.asarray(true_pts)[:, None, :]
        ln_off = jnp.zeros((n_obs, 1))
    else:
        events = scatter_to_posteriors(true_pts, n_samples, k_scat, m_hi=m_hi)
        ln_off = jnp.zeros((n_obs, n_samples))
    logL = fit.build_logL(events, ln_off, dist_fn,
                          coagulation_poisson_mean_estimator,
                          T_obs=1.0, n_quad=n_quad)
    return jax.jit(logL), theta_star


def vec(theta_star, **overrides):
    d = dict(theta_star); d.update(overrides)
    return jnp.asarray([float(d[p]) for p in PN])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, nargs="+", default=[1000, 4000, 8000])
    ap.add_argument("--recovered", type=str,
                    default="figures/stage1_post_svi_iso_mcjax_meas_n300_wide.npz")
    ap.add_argument("--edge-sweep", type=float, nargs="*", default=None,
                    help="if set, inject each of these natal_spin_loc TRUTH values "
                         "and report where the spin_loc profile (others@truth) peaks "
                         "— tests whether a mean farther from the chi=0 edge is "
                         "recoverable. Uses the first --n-seeds value.")
    ap.add_argument("--hybrid-1g", action="store_true",
                    help="evaluate the gen==1 natal population ANALYTICALLY "
                         "(no chi=0 KDE boundary bias) — the edge-density fix")
    args = ap.parse_args()
    hyb = True if args.hybrid_1g else None
    print(f"[cfg] hybrid_1g={hyb}")

    if args.edge_sweep is not None:
        ns = args.n_seeds[0]
        sl_grid = np.linspace(0.0, 0.40, 41)
        print(f"[edge-sweep] n_seed={ns}; profile peak of natal_spin_loc "
              f"(others held at truth) vs injected truth:")
        print("  injected_truth   profile_peak   peak-at-truth?   logL(truth)-logL(0)")
        for slt in args.edge_sweep:
            logL, th = setup(ns, spin_loc_truth=float(slt), hybrid_1g=hyb)
            i_sl = PN.index("natal_spin_loc")
            v = vec(th)
            prof = np.array([float(logL(v.at[i_sl].set(x))) for x in sl_grid])
            pk = sl_grid[int(np.argmax(prof))]
            ll_truth = float(logL(v))
            ll_zero = float(logL(v.at[i_sl].set(0.0)))
            ok = "YES" if abs(pk - slt) <= 0.03 else "no"
            print(f"   {slt:7.3f}        {pk:7.3f}         {ok:>5s}          "
                  f"{ll_truth - ll_zero:+8.2f}")
        return

    zz = np.load(_DEMO / args.recovered, allow_pickle=True)
    rec = {p: float(np.median(np.asarray(zz[p]).ravel())) for p in PN}
    print("[recovered medians]", {p: round(rec[p], 4) for p in PN})

    sl_grid = np.linspace(0.0, 0.18, 19)
    for ns in args.n_seeds:
        logL, th = setup(ns, hybrid_1g=hyb)
        i_sl = PN.index("natal_spin_loc")
        v_truth = vec(th)                        # all truth
        v_rec = jnp.asarray([rec[p] for p in PN])
        ll_truth = float(logL(v_truth))
        ll_rec = float(logL(v_rec))
        # profiles of spin_loc holding others at truth / at recovered
        prof_t = np.array([float(logL(v_truth.at[i_sl].set(x))) for x in sl_grid])
        prof_r = np.array([float(logL(v_rec.at[i_sl].set(x))) for x in sl_grid])
        pk_t = sl_grid[int(np.argmax(prof_t))]
        pk_r = sl_grid[int(np.argmax(prof_r))]
        print(f"\n===== n_seed={ns} =====")
        print(f"  logL(TRUTH full)      = {ll_truth:.3f}")
        print(f"  logL(RECOVERED full)  = {ll_rec:.3f}   Δ(truth-rec) = {ll_truth-ll_rec:+.3f}")
        print(f"  spin_loc profile PEAK  others@truth: {pk_t:.3f} (truth 0.05)   "
              f"others@recovered: {pk_r:.3f}")
        print("  spin_loc  logL(others@truth)  logL(others@recovered)")
        for x, a, b in zip(sl_grid, prof_t, prof_r):
            mark = " <-truth" if abs(x - 0.05) < 0.006 else ""
            print(f"   {x:5.3f}   {a:12.3f}       {b:12.3f}{mark}")


if __name__ == "__main__":
    main()
