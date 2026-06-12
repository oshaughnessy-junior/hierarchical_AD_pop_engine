"""Two equivalent inference calculations, one catalog: MC vs implicit.

The paper's "Figure 5" upgrade (ROS 2026-06-04): recover the single-pulse
coagulation hyperparameters ``(mu_natal, sigma_natal, log_rate)`` from ONE
synthetic catalog by BOTH dual integrand-evaluation routes of methods
2.A/2.C, and overlay the posteriors:

* **MC arm (sample mode)** — weighted MC merger draws feed the gwkokab
  fork's ``numpyro_sampled_poisson_likelihood`` (population supplies the
  sample points; per-event Gaussian genie likelihoods evaluated at the
  draws). Shown in blue.
* **Implicit arm (density mode)** — the implicit Picard solver supplies
  the pointwise density to the standard
  ``numpyro_analytical_poisson_likelihood`` path (events supply the
  sample points: per-event PE samples drawn from the same Gaussian
  likelihoods under a flat reference prior, ``ln_offsets = 0``). Shown
  as black shading and contours.

Both arms share the catalog, the measurement model (10% Gaussian error
in component masses, +/-0.2 in component spins — the toy's genie
widths), flat selection, and the physics primitives; the implicit arm
runs at the depletion-calibrated coupling ``eps*`` root-found so its
converged ``f_natal_depleted`` equals the MC's ``F_DEP`` (the H2
matching of methods 2.C — without it the two arms would answer
different physical questions).

Each arm's posterior is cached as npz, so the corner can be assembled
incrementally (sandbox: MC arm + implicit wiring smoke; workstation:
both at production scale).

Run::

    # MC arm end-to-end (sandbox ~40 s at reduced knobs)
    python demo/inference/compare_recovery_two_methods.py --arm mc \
        --n-seed 1000 --n-obs 20 --num-warmup 40 --num-samples 80
    # implicit arm (workstation; sandbox: --smoke for a 2-sample wiring check)
    python demo/inference/compare_recovery_two_methods.py --arm implicit \
        --n-pe 256 --num-warmup 300 --num-samples 600
    # assemble the overlay from whatever is cached
    python demo/inference/compare_recovery_two_methods.py --arm plot

Production (workstation)::

    python demo/inference/compare_recovery_two_methods.py --arm both \
        --n-seed 20000 --n-obs 60 --n-pe 512 \
        --num-warmup 500 --num-samples 1000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from toy_single_pulse_sampled_likelihood import (  # noqa: E402
    make_sampler_fn,
    draw_catalog,
    MU_TRUE, SIGMA_TRUE, T_OBS,
    M_MIN, M_MAX, SPIN_LOC, SPIN_SCALE, KERNEL_A, KERNEL_B, V_ESC, F_DEP,
    CATALOG_KEY,
)

VAR_NAMES = ("log_rate", "mu_natal", "sigma_natal")  # sorted order
LABELS = {"mu_natal": r"$\mu_{\rm natal}$",
          "sigma_natal": r"$\sigma_{\rm natal}$",
          "log_rate": r"$\log R$"}


def _variables():
    import numpyro.distributions as dist
    return {
        "mu_natal": dist.Uniform(5.0, 12.0),
        "sigma_natal": dist.Uniform(0.3, 3.0),
        "log_rate": dist.Uniform(0.0, 7.0),
    }


def _run_nuts(model, run_args, num_warmup, num_samples, key, robust=False):
    """Run NUTS. ``robust=True`` (the sample-mode MC arm only) hardens the
    warmup against the rough importance-weighted likelihood; the implicit
    (density) arm uses numpyro defaults — init_to_median there fails with
    "cannot find valid initial parameters" because the engine density is
    nan outside a sub-region of the prior."""
    from numpyro.infer import MCMC, NUTS, init_to_median
    if robust:
        # init_to_median (not default init_to_uniform): the single-pulse
        # population gives near-zero per-event ESS whenever mu_natal is drawn
        # far from the events, so a uniform init can land NUTS where the
        # likelihood is numerically degenerate -> step size collapses to
        # ~1e-9 and warmup never recovers.  Plus max_tree_depth +
        # target_accept_prob: the IS-weighted MC likelihood is deterministic
        # but ROUGH, so the default target 0.8 drives the step size down,
        # deepening the tree to its 2^10 cap (~10 min/iter).  Capping depth
        # bounds per-iteration cost; a relaxed target skims the roughness.
        nuts = NUTS(model, init_strategy=init_to_median,
                    max_tree_depth=8, target_accept_prob=0.7)
    else:
        nuts = NUTS(model)
    mcmc = MCMC(nuts, num_warmup=num_warmup,
                num_samples=num_samples, progress_bar=True)
    mcmc.run(key, *run_args)
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


# ── MC arm (sample mode) ──────────────────────────────────────────────
def run_mc_arm(args, Ls, logr_true):
    import numpyro.distributions as dist  # noqa: F401
    from gwkokab.inference.numpyro_sampled_poisson_likelihood import (
        numpyro_sampled_poisson_likelihood,
    )
    from gwkokab.models.utils import JointDistribution

    sampler_fn = make_sampler_fn(args.n_seed)
    variables = _variables()
    names = sorted(variables)
    priors = JointDistribution(*[variables[k] for k in names])
    model = numpyro_sampled_poisson_likelihood(
        sampler_fn, priors, variables, {},
        {k: i for i, k in enumerate(names)},
        Ls, None, None,
    )
    return _run_nuts(model, (jnp.asarray(T_OBS),),
                     args.num_warmup, args.num_samples,
                     jax.random.PRNGKey(31), robust=True)


# ── Implicit arm (density mode) ───────────────────────────────────────
def _calibrate_epsilon(grid, target_f_dep, lo=1e-3, hi=3.0, iters=14):
    """Geometric bisection: eps* with f_natal_depleted(eps*) = target.

    The H2 depletion matching of methods 2.C — both arms must sit at the
    same physical cascade depth.
    """
    from hierarchical_ad_pop_engine.response.coagulation import (
        build_gaussian_coagulation,
    )
    from hierarchical_ad_pop_engine.response.implicit import (
        solve_implicit_fixed_point,
    )

    def f_dep(eps):
        resp = build_gaussian_coagulation(
            mu=MU_TRUE, sigma=SIGMA_TRUE, m_min=M_MIN, m_max=M_MAX,
            natal_spin_loc=SPIN_LOC, natal_spin_scale=SPIN_SCALE,
            kernel_a=KERNEL_A, kernel_b=KERNEL_B, grid=grid,
            solver="implicit",
        )
        res = solve_implicit_fixed_point(
            resp, jnp.asarray(V_ESC), epsilon=eps,
            max_iter=80, tol=1e-7, damping=0.5,
        )
        return float(res.f_natal_depleted)

    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if f_dep(mid) < target_f_dep:
            lo = mid
        else:
            hi = mid
    eps_star = float(np.sqrt(lo * hi))
    print(f"H2 calibration: eps* = {eps_star:.4f} "
          f"(f_dep = {f_dep(eps_star):.4f}, target {target_f_dep})")
    return eps_star


def run_implicit_arm(args, observed, widths, logr_true):
    """Standard density-mode path: PE samples from the Gaussian event
    likelihoods (flat reference prior => ln_offsets = 0)."""
    import numpyro.distributions as dist  # noqa: F401
    from gwkokab.models.utils import JointDistribution
    from gwkokab.inference.numpyro_analytical_poisson_likelihood import (
        numpyro_analytical_poisson_likelihood,
    )
    from hierarchical_ad_pop_engine.grids import BinaryGrid
    from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (
        make_coagulation_dist_fn,
        coagulation_poisson_mean_estimator,
    )

    grid = BinaryGrid.clustered_chi(
        m_min=M_MIN, m_max=M_MAX, n_m=args.n_m, n_chi=args.n_chi,
    )
    eps_star = _calibrate_epsilon(grid, F_DEP)

    # PE samples per event ~ N(center, diag widths^2), flat ref prior.
    rng = np.random.default_rng(CATALOG_KEY + 1)
    n_obs = observed.shape[0]
    pe = (observed[:, None, :]
          + widths[:, None, :] * rng.standard_normal(
              (n_obs, args.n_pe, observed.shape[1])))
    # keep PE draws inside the population support (mask convention drops
    # the rest; theta-independent normalisation difference only)
    pe[..., 2:] = np.clip(pe[..., 2:], 1e-4, 1.0 - 1e-4)
    pe[..., :2] = np.clip(pe[..., :2], M_MIN * 1.01, M_MAX * 0.99)
    samples_stack = jnp.asarray(pe)
    ln_offsets = jnp.zeros(pe.shape[:2])

    dist_fn = make_coagulation_dist_fn(
        v_esc=V_ESC, m_min=M_MIN, m_max=M_MAX, kernel_a=KERNEL_A,
        grid=grid, solver="implicit", epsilon=eps_star,
    )
    variables = _variables()
    names = sorted(variables)
    priors = JointDistribution(*[variables[k] for k in names])
    model = numpyro_analytical_poisson_likelihood(
        dist_fn, priors, variables,
        {"natal_spin_loc": SPIN_LOC, "natal_spin_scale": SPIN_SCALE,
         "kernel_b": KERNEL_B},
        {k: i for i, k in enumerate(names)},
        coagulation_poisson_mean_estimator,
        None,
    )
    return _run_nuts(model, (samples_stack, ln_offsets, {"T_obs": T_OBS}),
                     args.num_warmup, args.num_samples,
                     jax.random.PRNGKey(37))


# ── Corner overlay ────────────────────────────────────────────────────
def make_corner(post_mc, post_impl, truths, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import corner

    def _stack(post):
        return np.stack([post[k] for k in ("mu_natal", "sigma_natal",
                                           "log_rate")], axis=1)

    labels = [LABELS["mu_natal"], LABELS["sigma_natal"], LABELS["log_rate"]]
    fig = None
    if post_impl is not None:
        fig = corner.corner(
            _stack(post_impl), labels=labels, color="k",
            truths=truths, truth_color="C3",
            fill_contours=True, plot_datapoints=False,
            hist_kwargs=dict(density=True),
        )
    if post_mc is not None:
        fig = corner.corner(
            _stack(post_mc), labels=labels, color="C0",
            truths=truths, truth_color="C3",
            fig=fig, plot_datapoints=False,
            hist_kwargs=dict(density=True),
        )
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("mc", "implicit", "both", "plot"),
                   default="mc")
    p.add_argument("--n-seed", type=int, default=1000)
    p.add_argument("--n-obs", type=int, default=20)
    p.add_argument("--n-pe", type=int, default=256)
    p.add_argument("--n-m", type=int, default=16)
    p.add_argument("--n-chi", type=int, default=12)
    p.add_argument("--num-warmup", type=int, default=40)
    p.add_argument("--num-samples", type=int, default=80)
    p.add_argument("--smoke", action="store_true",
                   help="2-warmup/2-sample wiring check (implicit arm)")
    p.add_argument("--cache-dir", type=str,
                   default=os.path.join(_HERE, ".cache_two_methods"))
    p.add_argument("--outdir", type=str,
                   default=os.path.join(_HERE, "figures_two_methods"))
    args = p.parse_args()
    if args.smoke:
        args.num_warmup, args.num_samples = 2, 2

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    # shared catalog (same key/widths as the toy)
    sampler_fn = make_sampler_fn(args.n_seed)
    Ls, truths4, observed = draw_catalog(sampler_fn, args.n_obs,
                                         narrow=False)
    logr_true = float(np.log(args.n_obs / T_OBS))
    widths = np.stack([0.10 * truths4[:, 0], 0.10 * truths4[:, 1],
                       np.full(args.n_obs, 0.2),
                       np.full(args.n_obs, 0.2)], axis=1)
    truths = [MU_TRUE, SIGMA_TRUE, logr_true]
    print(f"catalog: {args.n_obs} events; implied log_rate truth "
          f"{logr_true:.3f}")

    mc_npz = os.path.join(args.cache_dir, "post_mc.npz")
    impl_npz = os.path.join(args.cache_dir, "post_implicit.npz")

    if args.arm in ("mc", "both"):
        post = run_mc_arm(args, Ls, logr_true)
        np.savez(mc_npz, **post)
        print("MC-arm posterior:")
        for k in ("mu_natal", "sigma_natal", "log_rate"):
            print(f"  {k:12s}: {post[k].mean():.3f} +/- {post[k].std():.3f}")

    if args.arm in ("implicit", "both"):
        post = run_implicit_arm(args, observed, widths, logr_true)
        np.savez(impl_npz, **post)
        print("implicit-arm posterior:")
        for k in ("mu_natal", "sigma_natal", "log_rate"):
            print(f"  {k:12s}: {post[k].mean():.3f} +/- {post[k].std():.3f}")

    post_mc = (dict(np.load(mc_npz)) if os.path.exists(mc_npz) else None)
    post_impl = (dict(np.load(impl_npz))
                 if os.path.exists(impl_npz) else None)
    if post_mc is None and post_impl is None:
        print("nothing cached yet — run an arm first")
        return 1
    make_corner(post_mc, post_impl, truths,
                os.path.join(args.outdir, "two_method_corner.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
