r"""Two-pulse natal-IMF identifiability + recovery through the gwkokab path.

The methods-paper "two-component, small initial spins" experiment
(INSTRUCTIONS_gwkokab_stage1 §6, easiest tier): inject two near-zero-spin
natal pulses at **10 and 30 M⊙**, run the hierarchical coagulation cascade
(``solver="mc_jax"``, AD-traceable) through the **real**
``gwkokab.inference.poissonlikelihood_utils.analytical_poisson_likelihood_fn``,
and ask whether the 10-parameter two-pulse population is identified and whether
the ``μ₁ ↔ μ₂`` label-switch degeneracy is broken.

This is the single-pulse :mod:`fit_stage1_gwkokab` lifted to the two-pulse
model: same likelihood contract, same grid-free / self-consistent mock, just
the ``natal_family="two_gaussian"`` ``dist_fn`` (with ``kernel_a`` floated →
10 free parameters) and the matching 10-param reparametrisation.

The 10 parameters (model order)::

    mu_natal  sigma_natal  mu_natal_2  sigma_natal_2  lambda_natal
    natal_spin_loc  natal_spin_scale  kernel_a  kernel_b  log_rate

Label switching is broken **structurally** by disjoint mass windows in the
reparametrisation (μ₁ ∈ (5,20), μ₂ ∈ (20,50)) — matching the fit-side priors
in ``inference/stage1_wiring.two_pulse_prior_factories`` — which enclose the
10 / 30 M⊙ injection cleanly.  The ``--label-switch-probe`` flag additionally
starts the MAP from a *swapped* (μ₁=30, μ₂=10) point and confirms it recovers
the correctly-ordered truth rather than the mirror image.

Run (needs the full engine+gwkokab stack; in the Cowork VM use the RIFT stub —
see HANDOFF_PORTABLE §4.2)::

    python3 demo/fit_two_pulse_gwkokab.py --n-obs 60 --n-samples 5 \
        --n-seed 600 --k-max 3 --map-steps 200 --label-switch-probe

    → demo/figures/two_pulse_identifiability_profiles.png
    → demo/figures/two_pulse_natal_recovery.png

Breadcrumb for the methods paper: the two recovered pulses + the broken
label-switch is the headline two-component figure; promote the
``two_pulse_natal_recovery.png`` reconstruction panel.
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
from fit_stage1_gwkokab import (  # noqa: E402  (shared cross-method machinery)
    sample_from_model,            # grid sampler (--gen self)
    pair_sample_from_mc,          # exact-particle MC pairing (--gen mc)
    _match_epsilon_to_fdep,       # H2 ε↔depletion matching
    cross_method_js,              # H1 inject-θ JS gate
    _H1_JS_WARN,
    fisher_1d_sigma,              # conditional 1σ from profile curvature
)
from mock_catalog import scatter_to_posteriors  # noqa: E402

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)

# Model-order 10-param names (NOT the alphabetical gwkokab sample order;
# the dist_fn consumes keywords, so model order is what we unpack to).
PARAM_NAMES = [
    "mu_natal", "sigma_natal", "mu_natal_2", "sigma_natal_2", "lambda_natal",
    "natal_spin_loc", "natal_spin_scale", "kernel_a", "kernel_b", "log_rate",
]

# Truth for the inject-at-10&30 small-spin experiment.
THETA_STAR_2P = dict(
    mu_natal=10.0,
    sigma_natal=1.0,
    mu_natal_2=30.0,
    sigma_natal_2=2.0,
    lambda_natal=0.4,
    natal_spin_loc=0.0,
    natal_spin_scale=0.05,
    kernel_a=0.0,
    kernel_b=1.0,
    log_rate=None,  # calibrated against the fit μ-metric below
)


def _theta_vec(d):
    return jnp.asarray([float(d[k]) for k in PARAM_NAMES])


def _unpack(theta):
    return {k: theta[i] for i, k in enumerate(PARAM_NAMES)}


def build_logL(events, ln_offsets, dist_fn, estimator, *, T_obs, n_quad):
    r"""θ-vector → scalar gwkokab analytical-Poisson log-likelihood (jit-able)."""
    from gwkokab.inference.poissonlikelihood_utils import (
        analytical_poisson_likelihood_fn,
    )
    pmean = {"T_obs": float(T_obs), "n_quad": int(n_quad)}

    def logL(theta):
        model = dist_fn(**_unpack(theta))
        return analytical_poisson_likelihood_fn(
            model, estimator, events, ln_offsets, pmean, None)

    return logL


# --- 10-param reparametrisation: unconstrained u -> physical θ ---------------
# Disjoint mass windows (μ₁∈(5,20), μ₂∈(20,50)) break the label switch.
def _to_physical(u):
    mu1 = 5.0 + 15.0 * jax.nn.sigmoid(u[0])           # (5, 20)
    sigma1 = jnp.clip(jnp.exp(u[1]), 0.1, 10.0)        # > 0
    mu2 = 20.0 + 30.0 * jax.nn.sigmoid(u[2])           # (20, 50)
    sigma2 = jnp.clip(jnp.exp(u[3]), 0.1, 15.0)        # > 0
    lam = 0.05 + 0.90 * jax.nn.sigmoid(u[4])           # (0.05, 0.95)
    spin_loc = 0.3 * jax.nn.sigmoid(u[5])              # (0, 0.3)
    spin_scale = jnp.clip(jnp.exp(u[6]), 0.01, 0.5)    # > 0
    kernel_a = u[7]                                    # free
    kernel_b = u[8]                                    # free
    log_rate = u[9]                                    # free
    return jnp.stack([mu1, sigma1, mu2, sigma2, lam,
                      spin_loc, spin_scale, kernel_a, kernel_b, log_rate])


def _from_physical(theta):
    r"""Invert :func:`_to_physical` (so the MAP can start near a physical θ)."""
    (mu1, s1, mu2, s2, lam, sl, ss, ka, kb, lr) = [float(theta[i]) for i in range(10)]
    logit = lambda x: float(np.log(x / (1.0 - x)))
    return jnp.asarray([
        logit((mu1 - 5.0) / 15.0), float(np.log(s1)),
        logit((mu2 - 20.0) / 30.0), float(np.log(s2)),
        logit((lam - 0.05) / 0.90),
        logit(max(sl, 1e-4) / 0.3), float(np.log(ss)), ka, kb, lr,
    ])


def _profile_ranges(theta_star):
    (mu1, s1, mu2, s2, lam, sl, ss, ka, kb, lr) = [float(theta_star[i]) for i in range(10)]
    return {
        "mu_natal": np.linspace(6.0, 18.0, 13),
        "sigma_natal": np.linspace(0.5, 3.0, 13),
        "mu_natal_2": np.linspace(22.0, 40.0, 13),
        "sigma_natal_2": np.linspace(0.8, 5.0, 13),
        "lambda_natal": np.linspace(0.1, 0.8, 13),
        "natal_spin_loc": np.linspace(0.0, 0.25, 13),
        "natal_spin_scale": np.linspace(0.03, 0.25, 13),
        "kernel_a": np.linspace(ka - 2.0, ka + 2.0, 13),
        "kernel_b": np.linspace(kb - 3.0, kb + 3.0, 13),
        "log_rate": np.linspace(lr - 2.0, lr + 2.0, 13),
    }


def _run_map(logL, theta0, u_init, *, steps, lr, tag, clip_norm=10.0):
    r"""Adam MAP with gradient clipping + a non-finite-step guard.

    The likelihood has ``log(max(N, 1e-300))`` cliffs: when an event lands in
    a near-empty region of the (KDE+analytic) density and a width parameter
    shrinks, the ``1/N`` gradient spikes and plain Adam takes a divergent step
    → ``-logL`` goes ``nan`` (observed ~step 40).  Two robustness measures,
    both standard and additive:

    * ``optax.clip_by_global_norm(clip_norm)`` ahead of Adam caps the step;
    * any iterate whose value or grad is non-finite is SKIPPED (keep the last
      good ``u`` / optimiser state) rather than propagated, and a one-time
      warning is printed.  This turns a hard ``nan`` crash into a logged stall
      so the run still returns the best finite point.
    """
    import optax
    neg = lambda u: -logL(_to_physical(u))
    negj = jax.jit(jax.value_and_grad(neg))
    opt = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr))
    st = opt.init(u_init)
    u = u_init
    u_best, val_best = u, np.inf
    n_skipped = 0
    for step in range(steps):
        val, g = negj(u)
        val_f = float(val)
        g_finite = bool(jnp.all(jnp.isfinite(g)))
        if not (np.isfinite(val_f) and g_finite):
            n_skipped += 1
            if n_skipped == 1:
                print(f"   [{tag}] step {step:4d}  non-finite val/grad — "
                      f"skipping (keeping last good point)")
            continue
        if val_f < val_best:
            u_best, val_best = u, val_f
        upd, st = opt.update(g, st)
        u = optax.apply_updates(u, upd)
        if step % 40 == 0 or step == steps - 1:
            print(f"   [{tag}] step {step:4d}  -logL = {val_f:.3f}")
    if n_skipped:
        print(f"   [{tag}] skipped {n_skipped}/{steps} non-finite steps; "
              f"returning best finite (-logL={val_best:.3f})")
    return np.asarray(_to_physical(u_best))


def _two_pulse_density(theta, m):
    """Reconstructed natal mass marginal (1-G IMF) for a θ vector, on grid m."""
    from hierarchical_ad_pop_engine.natal.mass import log_natal_mass_two_gaussian
    d = _unpack(theta)
    log_f = log_natal_mass_two_gaussian(
        m, d["mu_natal"], d["sigma_natal"], d["mu_natal_2"],
        d["sigma_natal_2"], d["lambda_natal"], 4.0, 80.0)
    return np.asarray(jnp.exp(log_f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=60)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-seed", type=int, default=600)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-quad", type=int, default=32)
    ap.add_argument("--map-steps", type=int, default=200)
    ap.add_argument("--map-lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--label-switch-probe", action="store_true",
                    help="also MAP from a swapped (μ₁=30, μ₂=10) start and "
                         "confirm recovery of the correctly-ordered truth.")
    ap.add_argument("--kde-adaptive", action="store_true",
                    help="use the Abramson variable-bandwidth gen≥2 KDE "
                         "(sharpens the cascade-remnant peaks the global "
                         "Silverman bandwidth oversmooths — the σ₂/λ pulse-2 "
                         "recovery fix). Default off = fixed-bandwidth KDE.")
    ap.add_argument("--solver", choices=["mc_jax", "implicit"], default="mc_jax",
                    help="recovery density backend.")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="implicit coupling (ignored by mc_jax / overridden by "
                         "the ε-match under --gen mc).")
    ap.add_argument("--gen", choices=["self", "mc"], default="self",
                    help="'self' = grid sampler from the fit model; 'mc' = EXACT "
                         "particle-MC injection (cross-method). Two-pulse gives "
                         "the q spread (1×1, 1×2, 2×2 pairings) that actually "
                         "pins kernel_b — pair with --solver implicit.")
    ap.add_argument("--n-m", type=int, default=None)
    ap.add_argument("--n-chi", type=int, default=None)
    ap.add_argument("--sigma-log-m", type=float, default=0.1)
    ap.add_argument("--sigma-chi", type=float, default=0.1)
    args = ap.parse_args()

    fixed = demo.FIXED
    print(f"[cfg] TWO-PULSE  n_obs={args.n_obs} n_samples={args.n_samples} "
          f"n_seed={args.n_seed} k_max={args.k_max}")

    if args.n_m or args.n_chi:
        from hierarchical_ad_pop_engine.grids import BinaryGrid
        fit_grid = BinaryGrid.default(
            m_min=fixed["m_min"], m_max=fixed["m_max"],
            n_m=int(args.n_m or demo.N_M_FIT),
            n_chi=int(args.n_chi or demo.N_CHI_FIT))
    else:
        fit_grid = demo._grid_for_fit()
    theta_star = dict(THETA_STAR_2P)

    key = jax.random.PRNGKey(args.seed)
    k_mc, k_cat, k_map = jax.random.split(key, 3)
    _INJECT_DEPLETION = 0.10

    _common = dict(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
        gate="soft", grid=fit_grid, natal_family="two_gaussian",
        float_kernel_a=True)

    # Cross-method ε-matching (PLAN_cross_method Step 2 / H2): build the MC
    # injection up-front, match the implicit recovery ε to its depletion.
    recovery_epsilon = args.epsilon
    mc_inject_ev = _impl_resp = None
    if args.gen == "mc" and args.solver == "implicit":
        mc_resp = make_coagulation_dist_fn(
            **_common, depletion_fraction=_INJECT_DEPLETION, mc_jax_key=k_mc,
            solver="mc_jax")(**{**theta_star, "log_rate": 0.0})._response.mc_jax_response
        mc_inject_ev = mc_resp.evaluator
        n_init = int(getattr(mc_resp, "n_init", args.n_seed))
        W_total = float(jnp.sum(jnp.asarray(mc_inject_ev.weights)))
        f_target = max(0.0, min(0.999, 1.0 - W_total / max(n_init, 1)))
        from hierarchical_ad_pop_engine.response.implicit import (
            solve_implicit_fixed_point,
        )
        _impl_resp = make_coagulation_dist_fn(
            **_common, solver="implicit", epsilon=1.0)(
                **{**theta_star, "log_rate": 0.0})._response
        _v = jnp.asarray(fixed["v_esc"])
        recovery_epsilon, f_impl, _st = _match_epsilon_to_fdep(
            lambda e: float(solve_implicit_fixed_point(
                _impl_resp, _v, epsilon=float(e)).f_natal_depleted),
            f_target, lo=1e-3, hi=50.0)
        print(f"[ε-match] MC f_dep={f_target:.4f} → implicit ε*="
              f"{recovery_epsilon:.4g} (f={f_impl:.4f}, |Δf|="
              f"{abs(f_impl - f_target):.4f}, {_st})")
        _jm, _jc, _j2 = cross_method_js(
            _impl_resp, mc_inject_ev, fit_grid, fixed["v_esc"],
            epsilon=recovery_epsilon)
        print(f"[H1-JS ] inject-θ JS(implicit‖MC): dN/dm={_jm:.4f}  "
              f"dN/dχ={_jc:.4f}  N(m,χ)2D={_j2:.4f} nats")
        if max(_jm, _jc, _j2) > _H1_JS_WARN:
            print(f"[H1-JS ] WARNING: JS > {_H1_JS_WARN} — cross-method shape "
                  f"mismatch at θ*; recovery carries it as a systematic.")

    # dist_fn: two-pulse family, kernel_a floated (10 params).
    dist_fn = make_coagulation_dist_fn(
        **_common, depletion_fraction=_INJECT_DEPLETION, mc_jax_key=k_mc,
        solver=args.solver, epsilon=recovery_epsilon,
        kde_adaptive=args.kde_adaptive)
    if args.kde_adaptive:
        print("[cfg] gen≥2 KDE: Abramson adaptive (variable-bandwidth)")
    estimator = coagulation_poisson_mean_estimator

    # self-consistent log_rate calibration against the FIT μ metric.
    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = estimator(probe, T_obs=1.0, n_quad=args.n_quad)
    log_rate_star = float(jnp.log(args.n_obs / mu_unit))
    theta_star["log_rate"] = log_rate_star
    print(f"[cal] μ(logR=0)={float(mu_unit):.4g} → log_rate* = {log_rate_star:.3f}")

    m_hi = fixed["m_max"] * 2.5
    k_true, k_scat = jax.random.split(k_cat, 2)
    if args.gen == "mc":
        # Cross-method: EXACT particle-MC injection (no grid/jitter).
        if mc_inject_ev is None:  # solver != implicit → build injection here
            mc_inject_ev = make_coagulation_dist_fn(
                **_common, depletion_fraction=_INJECT_DEPLETION, mc_jax_key=k_mc,
                solver="mc_jax")(**theta_star)._response.mc_jax_response.evaluator
        true_pts = pair_sample_from_mc(mc_inject_ev, args.n_obs, k_true)
        events = scatter_to_posteriors(
            true_pts, args.n_samples, k_scat, m_hi=m_hi,
            sigma_log_m=args.sigma_log_m, sigma_chi=args.sigma_chi)
        gm, gc = fit_grid.m, fit_grid.chi
        events = jnp.stack([
            jnp.clip(events[..., 0], float(gm[0]), float(gm[-1])),
            jnp.clip(events[..., 1], float(gm[0]), float(gm[-1])),
            jnp.clip(events[..., 2], float(gc[0]), float(gc[-1])),
            jnp.clip(events[..., 3], float(gc[0]), float(gc[-1])),
        ], axis=-1)
        print(f"[mock] MC injection: {int(mc_inject_ev.m_active.shape[0])} "
              f"particle slots; exact-particle pairing ∝ w·w·K")
    else:
        # self-consistent grid sampler (faithful: 2× fit grid).
        truth_model = dist_fn(**theta_star)
        n_sm, n_sc = 2 * int(fit_grid.m.shape[0]), 2 * int(fit_grid.chi.shape[0])
        true_pts = sample_from_model(
            truth_model, args.n_obs, k_true, m_lo=fixed["m_min"], m_hi=m_hi,
            n_sm=n_sm, n_sc=n_sc)
        events = scatter_to_posteriors(
            true_pts, args.n_samples, k_scat, m_hi=m_hi,
            sigma_log_m=args.sigma_log_m, sigma_chi=args.sigma_chi)
    ln_offsets = jnp.zeros((args.n_obs, args.n_samples))
    print(f"[cat] gen={args.gen} events {tuple(events.shape)}  "
          f"ln_offsets {tuple(ln_offsets.shape)}")

    logL = build_logL(events, ln_offsets, dist_fn, estimator,
                      T_obs=1.0, n_quad=args.n_quad)
    logL_j = jax.jit(logL)
    theta0 = _theta_vec(theta_star)
    print(f"[ll ] logL(θ*) = {float(logL_j(theta0)):.3f}")

    # --- 1-D profiles --------------------------------------------------------
    print("\n=== 1-D LOG-LIKELIHOOD PROFILES (through θ*) ===")
    ranges = _profile_ranges(theta0)
    profiles, ident, sig1d = {}, {}, {}
    for i, name in enumerate(PARAM_NAMES):
        xs = ranges[name]
        ys = np.asarray([float(logL_j(theta0.at[i].set(float(x)))) for x in xs])
        profiles[name] = (xs, ys)
        peak = float(xs[int(np.argmax(ys))])
        truth = float(theta0[i])
        span = float(ys.max() - ys.min())
        ident[name] = (peak, truth, span)
        sig1d[name] = fisher_1d_sigma(xs, ys)
        flag = "IDENTIFIED" if span > 2.0 else "weak/flat"
        sig = sig1d[name]
        sig_s = f"σ₁d={sig:.3g}" if sig else "σ₁d=flat"
        print(f"  {name:18s} truth={truth:+.3f}  peak={peak:+.3f}  "
              f"Δ(logL)={span:8.2f}  {sig_s:12s}  [{flag}]")

    # --- MAP from a perturbed (correctly-ordered) start ----------------------
    map_recovered = None
    map_swapped = None
    if not args.no_map:
        print(f"\n=== MAP (gradient ascent, {args.map_steps} steps) ===")
        try:
            import optax  # noqa: F401
        except Exception as exc:  # pragma: no cover
            print(f"  [skip MAP] optax unavailable: {exc}")
        else:
            u0 = _from_physical(theta0) + jnp.asarray(
                [0.3, 0.2, 0.3, 0.2, 0.3, 0.3, 0.2, 0.4, 0.5, 0.4])
            map_recovered = _run_map(
                logL, theta0, u0, steps=args.map_steps, lr=args.map_lr, tag="ord")
            print("\n   param              truth      MAP       Δ      Δ/σ₁d")
            for i, name in enumerate(PARAM_NAMES):
                t, m = float(theta0[i]), float(map_recovered[i])
                sig = sig1d.get(name)
                # Δ/σ₁d: significance of the bias vs the CONDITIONAL profile
                # width.  ">N σ" on kernel_a/b/log_rate = degenerate direction
                # (conditional σ ≪ marginal σ), NOT a clean bias — see the
                # fisher_1d_sigma docstring.
                z = f"{(m - t) / sig:+6.1f}σ" if sig else "   flat"
                print(f"   {name:18s} {t:+8.3f} {m:+8.3f} {m - t:+8.3f}   {z}")

            if args.label_switch_probe:
                print("\n=== LABEL-SWITCH PROBE (start μ₁=30, μ₂=10) ===")
                swapped = dict(theta_star)
                swapped.update(mu_natal=28.0, sigma_natal=2.0,
                               mu_natal_2=12.0, sigma_natal_2=1.0,
                               lambda_natal=0.6)
                # clip the swapped start into the disjoint windows so the
                # reparametrisation is well-defined (this is the point:
                # the windows make μ₁=30 inexpressible → forced back to truth).
                swapped["mu_natal"] = min(swapped["mu_natal"], 19.0)
                swapped["mu_natal_2"] = max(swapped["mu_natal_2"], 21.0)
                u_sw = _from_physical(_theta_vec(swapped))
                map_swapped = _run_map(
                    logL, theta0, u_sw, steps=args.map_steps, lr=args.map_lr,
                    tag="swap")
                d_ord = abs(map_swapped[0] - 10.0) + abs(map_swapped[2] - 30.0)
                d_mir = abs(map_swapped[0] - 30.0) + abs(map_swapped[2] - 10.0)
                verdict = ("recovered TRUTH ordering" if d_ord < d_mir
                           else "FELL INTO MIRROR (label switch NOT broken)")
                print(f"   swapped-start MAP: μ₁={map_swapped[0]:.2f} "
                      f"μ₂={map_swapped[2]:.2f}  → {verdict}")

    # --- figures -------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # (a) 10-panel identifiability profiles.
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for ax, name in zip(axes.ravel(), PARAM_NAMES):
            xs, ys = profiles[name]
            ax.plot(xs, ys - ys.max(), "o-", color="C0")
            ax.axvline(float(theta0[PARAM_NAMES.index(name)]), color="C3",
                       ls="--", label="truth")
            ax.set_title(name)
            ax.set_ylabel(r"$\log L - \max$")
            ax.set_ylim(-15, 1)
            ax.grid(True, ls=":", alpha=0.4)
        axes.ravel()[0].legend()
        fig.suptitle("Two-pulse identifiability — gwkokab analytical-Poisson "
                     f"(inject 10 & 30 M⊙, N={args.n_obs}, S={args.n_samples})")
        fig.tight_layout()
        out1 = _DEMO / "figures" / "two_pulse_identifiability_profiles.png"
        out1.parent.mkdir(exist_ok=True)
        fig.savefig(out1, dpi=140)
        print(f"\n[fig] {out1}")

        # (b) natal-IMF reconstruction: truth vs MAP two-pulse density.
        m = jnp.linspace(4.0, 80.0, 600)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(np.asarray(m), _two_pulse_density(theta0, m), color="C3",
                 lw=2.5, label="truth (10 & 30 M⊙)")
        if map_recovered is not None:
            ax2.plot(np.asarray(m), _two_pulse_density(jnp.asarray(map_recovered), m),
                     color="C0", lw=2.0, ls="--", label="MAP recovery")
        if map_swapped is not None:
            ax2.plot(np.asarray(m), _two_pulse_density(jnp.asarray(map_swapped), m),
                     color="C2", lw=1.5, ls=":", label="swapped-start MAP")
        ax2.set_xlabel(r"$m\ [M_\odot]$")
        ax2.set_ylabel(r"natal $f_{\rm IMF}(m)$")
        ax2.set_title("Two-pulse natal-IMF recovery")
        ax2.legend()
        ax2.grid(True, ls=":", alpha=0.4)
        fig2.tight_layout()
        out2 = _DEMO / "figures" / "two_pulse_natal_recovery.png"
        fig2.savefig(out2, dpi=140)
        print(f"[fig] {out2}")
    except Exception as exc:  # pragma: no cover
        print(f"[fig] skipped ({exc})")

    print("\n=== SUMMARY ===")
    strong = [n for n in PARAM_NAMES if ident[n][2] > 2.0]
    weak = [n for n in PARAM_NAMES if ident[n][2] <= 2.0]
    print(f"  identified (peaked logL): {', '.join(strong) or '—'}")
    print(f"  weak / flat:              {', '.join(weak) or '—'}")
    return profiles, ident, map_recovered, map_swapped


if __name__ == "__main__":
    main()
