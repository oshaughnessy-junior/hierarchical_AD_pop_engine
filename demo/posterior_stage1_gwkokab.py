r"""Stage-1 single-pulse POSTERIOR through the gwkokab path — SVI + NUTS.

Follow-up to ``fit_stage1_gwkokab.py`` (which showed all 6 params are
*identified* with a unique origin): here we get the full **posterior** — not
just the MAP point — over the 6-parameter single-pulse Stage-1 toy, routed
through the REAL gwkokab ``numpyro_analytical_poisson_likelihood`` via the new
``inference/stage1_wiring`` (priors + variables + variables_index + likelihood
factory).  Two inferences on the same self-consistent mock catalogue:

* **SVI** — ``AutoNormal`` mean-field guide + Adam (fast, Gaussian posterior);
* **NUTS** — full HMC posterior (the gold standard; ``--nuts``).

For each we report posterior median + 90 % credible interval per parameter and
whether the truth sits inside it (coverage), plus a pairs/corner figure to read
off degeneracies (notably ``kernel_b`` ↔ the mass params).

Deliberately SIMPLE: single natal Gaussian, flat selection, self-consistent
data (the fit model generates its own catalogue, isolating identifiability from
the implicit-vs-mc_jax forward-model gap).  No two-pulse, no pdet here.

Run (full stack; in the Cowork VM use the RIFT stub — HANDOFF_PORTABLE §4.2):

    python3 demo/posterior_stage1_gwkokab.py --svi-steps 1500 --nuts \
        --n-obs 40 --n-samples 5 --n-seed 300 --k-max 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_DEMO = Path(__file__).resolve().parent
_REPO = _DEMO.parent
for p in (str(_DEMO), str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_gwkokab_fit_stage1 as demo  # noqa: E402
import fit_stage1_gwkokab as fit  # noqa: E402
from mock_catalog import scatter_to_posteriors  # noqa: E402
from hierarchical_ad_pop_engine.grids import BinaryGrid  # noqa: E402
from hierarchical_ad_pop_engine.spin_orientation import Isotropic, Fisher  # noqa: E402

# COAG_LEAN_GAIN: disable the F2 gain-side chi-out cache (orientation_precompute
# R_CHI_OUT_DEFAULT) in BOTH the forward solve and the implicit-function-theorem
# BACKWARD pass. Without this the backward re-runs _f2_combined_scan and OOMs the
# isotropic gradient (the F2 cache ~ n_chi^4); the lean per-iter orientation scan
# is bit-equivalent. Essential for isotropic implicit SVI/NUTS. See the fmobs
# "efficient high-resolution settings" docstring.
import os as _os  # noqa: E402
if _os.environ.get("COAG_LEAN_GAIN"):
    import hierarchical_ad_pop_engine.orientation_precompute as _op  # noqa: E402
    import hierarchical_ad_pop_engine.response.implicit as _imp  # noqa: E402
    _op.R_CHI_OUT_DEFAULT = None
    _imp.R_CHI_OUT_DEFAULT = None

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)
from hierarchical_ad_pop_engine.inference.stage1_wiring import (  # noqa: E402
    build_coagulation_likelihood,
    build_stage1_priors,
    build_stage1_variables,
)

PARAM_NAMES = fit.PARAM_NAMES
_CI = (5.0, 95.0)  # 90% credible interval


def _build_orientation(spec, n_theta=None, n_dphi=None):
    # Reduced orientation quadrature keeps the reverse-mode gradient (which
    # stacks the per-quadrature-point residuals) inside GPU memory; the natal
    # parameters the fit recovers are insensitive to the orientation-quad
    # resolution, so a coarse quad is fine for FITTING (full quad is only needed
    # for the fractal forward figures).
    extra = {}
    if n_theta is not None:
        extra["n_theta"] = int(n_theta)
    if n_dphi is not None:
        extra["n_dphi"] = int(n_dphi)
    s = str(spec).strip().lower()
    if s in ("aligned", "a"):
        return None  # None -> engine Aligned default (v1 scalar path)
    if s in ("isotropic", "iso", "i"):
        return Isotropic.default(**extra)
    if s.startswith("fisher:"):
        return Fisher.default(kappa=float(s.split(":", 1)[1]), **extra)
    raise ValueError(f"bad --orientation {spec!r}")


def _setup(args):
    fixed = demo.FIXED
    n_m = args.n_m_fit or demo.N_M_FIT
    n_chi = args.n_chi_fit or demo.N_CHI_FIT
    fit_grid = BinaryGrid.default(m_min=fixed["m_min"], m_max=fixed["m_max"],
                                  n_m=n_m, n_chi=n_chi)
    demo.build_truth_model_and_calibrate_rate(args.n_obs, grid=fit_grid)
    demo.THETA_STAR["natal_spin_loc"] = 0.05
    theta_star = dict(demo.THETA_STAR)
    orientation = _build_orientation(args.orientation, args.n_theta_fit, args.n_dphi_fit)

    key = jax.random.PRNGKey(args.seed)
    k_mc, k_cat = jax.random.split(key, 2)
    dist_fn = make_coagulation_dist_fn(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
        depletion_fraction=0.10, gate="soft", grid=fit_grid, mc_jax_key=k_mc,
        solver=args.solver, orientation=orientation)

    # self-consistent log_rate calibration against the fit's own μ metric
    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = coagulation_poisson_mean_estimator(probe, T_obs=1.0, n_quad=args.n_quad)
    theta_star["log_rate"] = float(jnp.log(args.n_obs / mu_unit))

    # self-consistent mock catalogue
    truth_model = dist_fn(**theta_star)
    k_true, k_scat = jax.random.split(k_cat, 2)
    m_hi = fixed["m_max"] * 2.5
    true_pts = fit.sample_from_model(truth_model, args.n_obs, k_true,
                                     m_lo=fixed["m_min"], m_hi=m_hi)
    if args.delta:
        # exact-injection (delta PE): one delta sample per event, flat weights
        # — the density-mode cross-check catalogue.
        events = jnp.asarray(true_pts)[:, None, :]
        ln_offsets = jnp.zeros((args.n_obs, 1))
    else:
        events = scatter_to_posteriors(true_pts, args.n_samples, k_scat, m_hi=m_hi)
        ln_offsets = jnp.zeros((args.n_obs, args.n_samples))

    # the REAL gwkokab numpyro model (priors + analytical-Poisson factor)
    variables, variables_index = build_stage1_variables()
    priors = build_stage1_priors(variables)
    log_lik_fn = build_coagulation_likelihood(
        dist_fn=dist_fn, priors=priors, variables=variables,
        variables_index=variables_index,
        poisson_mean_estimator=coagulation_poisson_mean_estimator)

    pmean = {"T_obs": 1.0, "n_quad": args.n_quad}

    def model():
        log_lik_fn(events, ln_offsets, pmean)

    return theta_star, model, events


def _report(name, samples):
    print(f"\n=== {name} posterior (90% CI) ===")
    print(f"  {'param':18s} {'truth':>8s} {'median':>8s} "
          f"{'CI_lo':>8s} {'CI_hi':>8s}  cover")
    cov = {}
    for p in PARAM_NAMES:
        s = np.asarray(samples[p]).ravel()
        s = s[np.isfinite(s)]
        if s.size == 0:
            print(f"  {p:18s} {'?':>8s}  (all non-finite)")
            cov[p] = False
            continue
        lo, hi = np.percentile(s, _CI)
        med = float(np.median(s))
        truth = float(_REPORT_TRUTH[p])
        inside = lo <= truth <= hi
        cov[p] = inside
        print(f"  {p:18s} {truth:+8.3f} {med:+8.3f} {lo:+8.3f} {hi:+8.3f}"
              f"   {'Y' if inside else '.'}")
    n_in = sum(cov.values())
    print(f"  coverage: {n_in}/{len(PARAM_NAMES)} truths in their 90% CI")
    return cov


def _pairs_figure(samples, theta_star, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[fig] skipped ({exc})")
        return
    P = PARAM_NAMES
    n = len(P)
    arr = {p: np.asarray(samples[p]).ravel() for p in P}
    finite = np.all([np.isfinite(arr[p]) for p in P], axis=0)
    if finite.sum() == 0:
        print("[fig] skipped (no finite posterior draws)")
        return
    arr = {p: arr[p][finite] for p in P}
    fig, axes = plt.subplots(n, n, figsize=(2.0 * n, 2.0 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(arr[P[i]], bins=30, color="C0", alpha=0.8)
                ax.axvline(float(theta_star[P[i]]), color="C3", ls="--")
            elif j < i:
                ax.plot(arr[P[j]], arr[P[i]], ".", ms=1.5, alpha=0.25, color="C0")
                ax.axvline(float(theta_star[P[j]]), color="C3", ls="--", lw=0.8)
                ax.axhline(float(theta_star[P[i]]), color="C3", ls="--", lw=0.8)
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(P[j], fontsize=8)
            if j == 0 and i > 0:
                ax.set_ylabel(P[i], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Stage-1 single-pulse posterior (red = truth)")
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"[fig] {out}")


_REPORT_TRUTH = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-seed", type=int, default=300)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-quad", type=int, default=32)
    ap.add_argument("--svi-steps", type=int, default=2500)
    ap.add_argument("--svi-lr", type=float, default=0.01)
    ap.add_argument("--svi-draws", type=int, default=2000)
    ap.add_argument("--svi-init-scale", type=float, default=0.1)
    ap.add_argument("--svi-clip", type=float, default=10.0)
    ap.add_argument("--nuts", action="store_true")
    ap.add_argument("--nuts-warmup", type=int, default=150)
    ap.add_argument("--nuts-samples", type=int, default=300)
    ap.add_argument("--nuts-tree-depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--orientation", type=str, default="aligned",
                    help="aligned | isotropic | fisher:<kappa>")
    ap.add_argument("--solver", type=str, default="mc_jax",
                    choices=("mc_jax", "implicit"))
    ap.add_argument("--n-chi-fit", type=int, default=None,
                    help="override the fit-grid n_chi (default demo.N_CHI_FIT)")
    ap.add_argument("--n-m-fit", type=int, default=None,
                    help="override the fit-grid n_m (default demo.N_M_FIT)")
    ap.add_argument("--n-theta-fit", type=int, default=2,
                    help="isotropic/Fisher orientation-quad n_theta for the fit "
                         "(reduced default keeps the AD gradient in GPU memory)")
    ap.add_argument("--n-dphi-fit", type=int, default=4,
                    help="isotropic/Fisher orientation-quad n_dphi for the fit")
    ap.add_argument("--delta", action="store_true",
                    help="exact-injection (delta-PE) catalogue: 1 sample/event, "
                         "no measurement scatter — the density-mode cross-check")
    ap.add_argument("--out-tag", type=str, default="",
                    help="suffix appended to output figure names")
    args = ap.parse_args()

    print(f"[cfg] n_obs={args.n_obs} n_samples={args.n_samples} "
          f"n_seed={args.n_seed} k_max={args.k_max} svi_steps={args.svi_steps} "
          f"nuts={args.nuts}")
    theta_star, model, events = _setup(args)
    _REPORT_TRUTH.update(theta_star)
    print(f"[cat] events {tuple(events.shape)};  "
          f"log_rate* = {theta_star['log_rate']:.3f}")

    import optax
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import optax_to_numpyro

    # --- SVI -----------------------------------------------------------------
    # Robust optimiser: gradient clipping + modest lr + a tight guide init keep
    # the mean-field guide out of the param regions where the soft-gate cascade
    # NaNs in the BACKWARD pass (the forward is nan_to_num-guarded inside
    # gwkokab; the gradient is not — [[feedback-where-guards-forward-not-backward]]).
    print(f"\n[SVI] AutoNormal(init_scale={args.svi_init_scale}) + clipped Adam, "
          f"{args.svi_steps} steps")
    guide = AutoNormal(model, init_scale=args.svi_init_scale)
    optimizer = optax_to_numpyro(
        optax.chain(optax.clip_by_global_norm(args.svi_clip),
                    optax.adam(args.svi_lr)))
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    k_svi, k_post, k_nuts = jax.random.split(jax.random.PRNGKey(args.seed + 1), 3)
    svi_res = svi.run(k_svi, args.svi_steps, progress_bar=False)
    losses = np.asarray(svi_res.losses)
    print(f"[SVI] -ELBO {float(losses[0]):.2f} → {float(losses[-1]):.2f} "
          f"(Δ {float(losses[0] - losses[-1]):.2f} nats)")
    svi_post = guide.sample_posterior(
        k_post, svi_res.params, sample_shape=(args.svi_draws,))
    _report("SVI", svi_post)
    _pairs_figure(svi_post, theta_star,
                  _DEMO / "figures" / f"stage1_posterior_svi{args.out_tag}.png")

    # --- NUTS (optional) -----------------------------------------------------
    if args.nuts:
        from numpyro.infer import MCMC, NUTS
        print(f"\n[NUTS] warmup={args.nuts_warmup} samples={args.nuts_samples} "
              f"max_tree_depth={args.nuts_tree_depth}")
        kernel = NUTS(model, target_accept_prob=0.8,
                      max_tree_depth=args.nuts_tree_depth,
                      init_strategy=__import__("numpyro").infer.init_to_median())
        mcmc = MCMC(kernel, num_warmup=args.nuts_warmup,
                    num_samples=args.nuts_samples, num_chains=1,
                    progress_bar=False)
        mcmc.run(k_nuts)
        nuts_post = mcmc.get_samples()
        # r-hat-ish: just report; single chain
        _report("NUTS", nuts_post)
        _pairs_figure(nuts_post, theta_star,
                      _DEMO / "figures" / f"stage1_posterior_nuts{args.out_tag}.png")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
