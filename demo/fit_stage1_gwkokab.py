r"""Stage-1 single-pulse identifiability + recovery through the gwkokab path.

The "lowest-hanging fruit" check: with a multi-sample mock catalogue drawn from
a known truth ``θ*``, can the **gwkokab analytical-Poisson likelihood** — fed by
the coagulation engine via the new ``inference/stage1_wiring`` + ``selection`` —
recover the natal-population parameters and the interaction (pairing) model, and
is the origin *unique* (a single likelihood peak at truth) or degenerate?

Rather than a full SVI/NUTS run (whose ``AutoNormal`` width underestimate +
Gaussian-KDE χ-oversmoothing make the M4 6-param coverage a known xfail), this
script answers identifiability *directly*:

1. **Profile** the log-likelihood along each parameter axis through ``θ*``
   (others held at truth).  A clear peak at the truth value ⇒ that parameter is
   locally identified; a flat profile ⇒ unconstrained by the data.
2. **MAP**: gradient-ascend the log-likelihood from a perturbed start (the AD
   payoff — gradients flow through the whole cascade) and compare the recovered
   point to ``θ*``.

Everything routes through the REAL ``gwkokab.inference.poissonlikelihood_utils.
analytical_poisson_likelihood_fn`` (so it exercises the actual fit contract),
with the engine built via ``make_coagulation_dist_fn`` (``solver="mc_jax"``) and
the flat-selection ``coagulation_poisson_mean_estimator`` by default
(``--selection vt`` swaps in the ``M_chirp^{15/6}`` VT estimator).

Run (needs the full engine+gwkokab stack; in the Cowork VM use the RIFT stub —
see HANDOFF_PORTABLE §4.2):

    python3 demo/fit_stage1_gwkokab.py --n-obs 40 --n-samples 5 \
        --n-seed 400 --k-max 3 --map-steps 150

NB the absolute ``log_rate`` rides on ``MCResponseJax.total_rate``, whose
metric bias at small ``n_seed`` is the open §4.3 thread; we calibrate ``θ*``'s
``log_rate`` against that SAME metric so its identifiability is self-consistent
(the rate scale cancels), and focus the headline on the SHAPE parameters.
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
from mock_catalog import make_mock_posteriors  # noqa: E402

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)
from hierarchical_ad_pop_engine.inference.selection import (  # noqa: E402
    make_vt_poisson_mean_estimator,
    vt_powerlaw,
)

PARAM_NAMES = ["mu_natal", "sigma_natal", "natal_spin_loc",
               "natal_spin_scale", "kernel_b", "log_rate"]


def _theta_vec(d):
    return jnp.asarray([float(d[k]) for k in PARAM_NAMES])


def _unpack(theta):
    return {k: theta[i] for i, k in enumerate(PARAM_NAMES)}


def sample_pairs_from_particles(model, n_events, key, *, vt_fn=None):
    r"""Grid-free MC draw of ``(N,4)`` binary events by **resampling the engine's
    own particles** — the correct Monte-Carlo way (no grid, no KDE).

    The mc_jax cascade leaves a pool of single-BH particles
    ``(m_active, chi_active, weights)`` (the surviving remnants across all
    generations; zero weight = inactive slot).  A merging binary is a pair
    ``(i, j)`` drawn with probability ``∝ K(m_i,m_j)·w_i·w_j·VT(m_i,m_j)``
    (the pair rate density evaluated on the particles), and the event is just
    the two particles' actual masses and spins — so the discrete high-χ 2G+
    remnant spins are represented EXACTLY, with none of the resolution loss a
    density grid or KDE would impose.

    ``vt_fn(m1,m2,χ1,χ2)`` (optional) detection-weights the pair draw.
    Self-pairs (``i==j``) are excluded.  jit-safe.
    """
    from hierarchical_ad_pop_engine.evaluation.mc_jax import _log_pairing_kernel_jax

    ev = model._response.mc_jax_response.evaluator
    m = ev.m_active
    chi = ev.chi_active
    w = ev.weights
    n = m.shape[0]

    logK = _log_pairing_kernel_jax(m[:, None], m[None, :], ev.kernel_a, ev.kernel_b)
    P = jnp.exp(logK) * w[:, None] * w[None, :]
    if vt_fn is not None:
        P = P * vt_fn(m[:, None], m[None, :], jnp.zeros(()), jnp.zeros(()))
    P = P * (1.0 - jnp.eye(n))          # no self-pairs
    p = (P / jnp.sum(P)).reshape(-1)

    idx = jax.random.choice(key, n * n, shape=(n_events,), p=p)
    i, j = idx // n, idx % n
    m1, m2, c1, c2 = m[i], m[j], chi[i], chi[j]
    hi = m1 >= m2                        # order m1 >= m2, carry spins
    return jnp.stack([jnp.where(hi, m1, m2), jnp.where(hi, m2, m1),
                      jnp.where(hi, c1, c2), jnp.where(hi, c2, c1)], axis=-1)


def sample_from_model(model, n_events, key, *, m_lo, m_hi, n_sm=20, n_sc=8,
                      vt_fn=None):
    r"""Self-consistent mock: inverse-CDF draw of ``(N,4)`` true events from the
    FIT model's own pair intensity ``exp(model.log_prob)`` on a coarse
    ``m₁×m₂×χ₁×χ₂`` grid (geomspaced mass, linear χ), jittered within cells.

    Generating the data from the SAME model we fit removes the
    implicit-vs-mc_jax solver mismatch, so a profile peak away from θ* is then a
    genuine identifiability statement, not a forward-model discrepancy.

    ``vt_fn(m1,m2,chi1,chi2)`` (optional): a selection weight applied to the
    sampling density BEFORE normalising, so draws are **detection-weighted**
    (the predicted *observed* catalogue under that ``VT``).  ``None`` → the
    intrinsic distribution.
    """
    log_m = jnp.linspace(jnp.log(m_lo), jnp.log(m_hi), n_sm)
    m_grid = jnp.exp(log_m)
    chi = jnp.linspace(0.0, 1.0, n_sc)
    M1, M2, C1, C2 = jnp.meshgrid(m_grid, m_grid, chi, chi, indexing="ij")
    nodes = jnp.stack([M1, M2, C1, C2], axis=-1)
    logp = model.log_prob(nodes)
    p = jnp.exp(logp - jnp.max(logp))
    if vt_fn is not None:
        p = p * vt_fn(M1, M2, C1, C2)
    p = p.reshape(-1)
    p = p / jnp.sum(p)

    k_idx, k_jm1, k_jm2, k_jc1, k_jc2 = jax.random.split(key, 5)
    idx = jax.random.choice(k_idx, p.size, shape=(n_events,), p=p)
    i1, i2, j1, j2 = jnp.unravel_index(idx, (n_sm, n_sm, n_sc, n_sc))
    dlog = log_m[1] - log_m[0]   # keep as jax scalars (jit-safe; float() breaks under jit)
    dchi = chi[1] - chi[0]
    # jitter: uniform in log-m within the mass cell, linear within the χ cell.
    m1 = jnp.exp(log_m[i1] + dlog * (jax.random.uniform(k_jm1, (n_events,)) - 0.5))
    m2 = jnp.exp(log_m[i2] + dlog * (jax.random.uniform(k_jm2, (n_events,)) - 0.5))
    c1 = jnp.clip(chi[j1] + dchi * (jax.random.uniform(k_jc1, (n_events,)) - 0.5), 0.0, 1.0)
    c2 = jnp.clip(chi[j2] + dchi * (jax.random.uniform(k_jc2, (n_events,)) - 0.5), 0.0, 1.0)
    hi = m1 >= m2
    return jnp.stack([jnp.where(hi, m1, m2), jnp.where(hi, m2, m1),
                      jnp.where(hi, c1, c2), jnp.where(hi, c2, c1)], axis=-1)


def pair_sample_from_mc(evaluator, n_events, key):
    r"""EXACT-particle MC injection (PLAN_cross_method_injection_recovery Step 1).

    Draw ``(N,4)`` true merger events ``(m₁,m₂,χ₁,χ₂)`` by pairing the surviving
    MC particle pool with the pairing kernel — i.e. directly from the discrete
    forward intensity ``½ K(m_i,m_j)·w_i·w_j`` over particles ``i≠j``.  Unlike
    :func:`sample_from_model`, there is **no grid and no within-cell jitter**:
    each event's masses and spins are exact particle values straight out of the
    cascade.  This is the injection-side strength of the cross-method scheme —
    no grid-resolution bias on generation.

    Sampling factorises without materialising an ``O(N²)`` pair distribution per
    draw: pick ``i ∝ w_i·Z_i`` (the joint's i-marginal, ``Z_i=Σ_{j≠i} w_j
    K(m_i,m_j)``), then ``j ∝ w_j K(m_i,m_j)`` — joint ``∝ w_i w_j K``.  Builds
    the ``(N,N)`` kernel once (fine for demo ``n_seed``; chunk if ``n_seed`` is
    large).  Self-pairs (``i=j``) are excluded (a BH cannot merge with itself).
    """
    from hierarchical_ad_pop_engine.evaluation.mc_jax import (
        _log_pairing_kernel_jax,
    )

    m = jnp.asarray(evaluator.m_active)
    chi = jnp.asarray(evaluator.chi_active)
    w = jnp.asarray(evaluator.weights)
    a = float(jnp.asarray(evaluator.kernel_a))
    b = float(jnp.asarray(evaluator.kernel_b))
    n = int(m.shape[0])

    logK = _log_pairing_kernel_jax(m[:, None], m[None, :], a, b)  # (N, N)
    Kw = jnp.exp(logK) * w[None, :]                              # col j weighted w_j
    Kw = Kw.at[jnp.diag_indices(n)].set(0.0)                     # no self-pairs
    Z = jnp.sum(Kw, axis=1)                                      # Σ_{j≠i} w_j K
    Wi = w * Z
    Wi = jnp.where(jnp.isfinite(Wi) & (Wi > 0.0), Wi, 0.0)

    k_i, k_j = jax.random.split(key)
    i = jax.random.choice(k_i, n, shape=(n_events,), p=Wi / jnp.sum(Wi))
    rows = Kw[i]                                                 # (n_events, N) ∝ w_j K(m_i,·)
    rows = rows / jnp.sum(rows, axis=1, keepdims=True)
    j = jax.vmap(lambda kk, p: jax.random.choice(kk, n, p=p))(
        jax.random.split(k_j, n_events), rows)

    m1, m2, c1, c2 = m[i], m[j], chi[i], chi[j]
    hi = m1 >= m2
    return jnp.stack([jnp.where(hi, m1, m2), jnp.where(hi, m2, m1),
                      jnp.where(hi, c1, c2), jnp.where(hi, c2, c1)], axis=-1)


def _match_epsilon_to_fdep(f_of_eps, target, *, lo=1e-3, hi=50.0, n_scan=11,
                           tol=3e-3):
    r"""Find ε with ``f_of_eps(ε) ≈ target`` ROBUSTLY (PLAN_cross_method Step 2).

    ``f_of_eps`` is ``implicit f_natal_depleted(ε)``, monotone-increasing 0→1
    in the regime where the Picard solve CONVERGES — but it diverges at large ε
    and returns non-finite / non-monotone garbage there.  So we cannot trust a
    fixed [lo, hi] bracket (the naive version returned the lo endpoint when
    ``f_of_eps(1e3)`` diverged → ε*=0.001, no cascade, −inf recovery).

    Strategy: scan a log-ε grid in the converging range, drop non-finite
    points, find the first finite crossing ``f ≥ target``, then geometric-
    bisect within that bracket.  Returns ``(eps, f_at_eps, status)`` where
    status ∈ {"ok", "UNREACHED" (target above every converging point),
    "BELOW_GRID" (already ≥ target at the smallest ε)}.
    """
    grid = np.geomspace(lo, hi, n_scan)
    lo_e = lo_f = hi_e = hi_f = None
    for e in grid:
        v = f_of_eps(float(e))
        if not np.isfinite(v):
            continue                       # diverged solve — skip
        if v >= target:
            hi_e, hi_f = float(e), float(v)
            break
        lo_e, lo_f = float(e), float(v)
    if hi_e is None:
        return (lo_e if lo_e is not None else float(grid[0])), \
               (lo_f if lo_f is not None else float("nan")), "UNREACHED"
    if lo_e is None:
        return hi_e, hi_f, "BELOW_GRID"
    e_star, f_star = _bisect_geometric(
        f_of_eps, target, lo_e, hi_e, max_iter=20, tol=tol)
    return e_star, f_star, "ok"


# Cross-method JS warn level (nats).  ~0.05 is the red_team/compare_all
# same-physics agreement scale (engine_v2 ↔ particle MC ≈ 0.007 at ε=1); a
# cross-method inject/recover above this is a SHAPE systematic worth chasing.
_H1_JS_WARN = 0.05


def cross_method_js(impl_response, mc_evaluator, grid, v_esc, *, epsilon):
    r"""H1 cross-method gate (PLAN_cross_method Step 3): Jensen–Shannon
    divergence between the IMPLICIT and the MC ``dN/dm`` & ``dN/dχ`` at a fixed
    θ, on a common grid.

    Both single-BH densities are marginalised through the SAME
    :func:`observables.marginal_1d` forward-rate quadrature, and the JS is
    shape-only (each marginal is renormalised), so the density-convention
    constants between the two solvers cancel — the divergence measures the
    genuine method-to-method SHAPE disagreement (e.g. the kernel_b residual),
    not a normalisation offset.  The implicit density is taken at the
    depletion-matched ``epsilon`` so this is NOT contaminated by an
    ε↔depletion (H2) mismatch.

    Returns ``(js_dN_dm, js_dN_dchi)`` in nats (bounded by ln 2 ≈ 0.693).
    """
    from hierarchical_ad_pop_engine.response.implicit import (
        solve_implicit_fixed_point,
    )
    from hierarchical_ad_pop_engine import observables as _obs
    from red_team.phase1_overlay import js_divergence

    res = solve_implicit_fixed_point(
        impl_response, jnp.asarray(v_esc), epsilon=float(epsilon))
    N_impl = jnp.exp(res.log_N)
    M, C = jnp.meshgrid(grid.m, grid.chi, indexing="ij")
    N_mc = mc_evaluator.density(M, C)

    kp = impl_response.kernel_params
    m_ax, dm_i = _obs.marginal_1d(N_impl, grid, kp, axis="m1")
    _, dm_m = _obs.marginal_1d(N_mc, grid, kp, axis="m1")
    c_ax, dc_i = _obs.marginal_1d(N_impl, grid, kp, axis="chi1")
    _, dc_m = _obs.marginal_1d(N_mc, grid, kp, axis="chi1")

    js_m = js_divergence(np.asarray(dm_i), np.asarray(dm_m), np.asarray(m_ax))
    js_c = js_divergence(np.asarray(dc_i), np.asarray(dc_m), np.asarray(c_ax))
    return float(js_m), float(js_c)


def _bisect_geometric(fn, target, lo, hi, *, max_iter=30, tol=1e-3):
    r"""Geometric bisection for a MONOTONE-INCREASING ``fn`` (ε spans orders of
    magnitude, so bisect in log-ε).  Returns ``(x, fn(x))`` with
    ``|fn(x)-target| < tol`` or the last midpoint after ``max_iter``.

    Used to match the implicit recovery ε to the MC injection's depletion
    fraction (PLAN_cross_method Step 2 / H2): ``f_natal_depleted(ε)`` is
    monotone increasing 0→1, so a unique ε reproduces the MC's cascade depth.
    No closed-form K_eff needed — the dimensionless depletion fraction is
    matched directly, sidestepping all weight-normalisation conventions.
    """
    flo, fhi = fn(lo), fn(hi)
    if not (flo <= target <= fhi):
        # target outside the bracket — return the nearer endpoint.
        if abs(flo - target) <= abs(fhi - target):
            return lo, flo
        return hi, fhi
    x, fx = (lo * hi) ** 0.5, None
    for _ in range(max_iter):
        x = (lo * hi) ** 0.5
        fx = fn(x)
        if abs(fx - target) < tol:
            return x, fx
        if fx < target:
            lo = x
        else:
            hi = x
    return x, fx


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


# --- MAP reparametrisation: unconstrained u -> physical θ (engine-safe) ------
def _to_physical(u):
    mu = 5.0 + 45.0 * jax.nn.sigmoid(u[0])           # (5, 50)
    sigma = jnp.clip(jnp.exp(u[1]), 0.1, 20.0)        # > 0
    spin_loc = 0.3 * jax.nn.sigmoid(u[2])             # (0, 0.3)
    spin_scale = jnp.clip(jnp.exp(u[3]), 0.01, 0.5)   # > 0
    kernel_b = u[4]                                   # free
    log_rate = u[5]                                   # free
    return jnp.stack([mu, sigma, spin_loc, spin_scale, kernel_b, log_rate])


def _from_physical(theta):
    r"""Invert :func:`_to_physical` (so the MAP can start near a physical θ)."""
    mu, sigma, sl, ss, kb, lr = [float(theta[i]) for i in range(6)]
    logit = lambda x: float(np.log(x / (1.0 - x)))
    return jnp.asarray([
        logit((mu - 5.0) / 45.0), float(np.log(sigma)),
        logit(sl / 0.3), float(np.log(ss)), kb, lr,
    ])


def _profile_ranges(theta_star):
    mu, sg, sl, ss, kb, lr = [float(theta_star[i]) for i in range(6)]
    return {
        "mu_natal": np.linspace(max(5.0, mu - 4), mu + 4, 13),
        "sigma_natal": np.linspace(0.5, 3.0, 13),
        "natal_spin_loc": np.linspace(0.0, 0.25, 13),
        "natal_spin_scale": np.linspace(0.03, 0.25, 13),
        "kernel_b": np.linspace(kb - 3, kb + 3, 13),
        "log_rate": np.linspace(lr - 2.0, lr + 2.0, 13),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-seed", type=int, default=400)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-quad", type=int, default=32)
    ap.add_argument("--map-steps", type=int, default=150)
    ap.add_argument("--map-lr", type=float, default=0.05)
    ap.add_argument("--selection", choices=["flat", "vt"], default="flat")
    ap.add_argument("--gen", choices=["self", "implicit", "mc"], default="self",
                    help="mock-event generator: 'self' = the fit model's own "
                         "grid sampler (clean identifiability); 'implicit' = the "
                         "implicit oracle; 'mc' = EXACT particle-MC injection "
                         "(PLAN_cross_method Step 1) — pair the surviving MC "
                         "particle pool ∝ w_i·w_j·K(m_i,m_j), no grid/jitter. "
                         "Pair with --solver implicit for cross-method "
                         "inject-MC / recover-implicit.")
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--solver", choices=["mc_jax", "implicit", "perturbative"],
                    default="mc_jax",
                    help="density backend the fit reads through. 'implicit' is "
                         "the single-pulse self-consistency check: with --gen "
                         "self the mock is drawn from the SAME implicit model, "
                         "so the MLE must recover (mu_natal, sigma_natal) — "
                         "isolates the implicit forward model from the "
                         "two-pulse λ machinery.")
    ap.add_argument("--epsilon", type=float, default=1.0,
                    help="coupling×time for the implicit solver (ignored by mc_jax).")
    ap.add_argument("--n-m", type=int, default=None,
                    help="override fit-grid N_m (default demo's 64).")
    ap.add_argument("--n-chi", type=int, default=None,
                    help="override fit-grid N_chi (default demo's 20).")
    ap.add_argument("--sigma-log-m", type=float, default=0.1,
                    help="mock mass measurement scatter (default 0.1). Pass 0 "
                         "for the DELTA control: with --sigma-chi 0 the events "
                         "are unscattered, so any residual width/spin bias is "
                         "the forward model, NOT measurement deconvolution.")
    ap.add_argument("--sigma-chi", type=float, default=0.1,
                    help="mock spin measurement scatter (default 0.1 — note this "
                         "EQUALS the truth natal_spin_scale, so spin is barely "
                         "identifiable). Pass 0 with --sigma-log-m 0 for delta.")
    ap.add_argument("--n-sm", type=int, default=None,
                    help="self-consistent SAMPLER mass-grid resolution. Default "
                         "(None) tracks 2× the fit n_m so the mock is FAITHFUL "
                         "to the density the likelihood reads — the coarse old "
                         "default (20 over geomspace [m_min,2.5·m_max], Δlog_m≈"
                         "0.2 with ~0.5 M⊙ within-cell jitter) was fine for the "
                         "smooth mc_jax KDE but inflated recovered σ against the "
                         "SHARP implicit grid density. Set explicitly to probe.")
    ap.add_argument("--n-sc", type=int, default=None,
                    help="self-consistent SAMPLER spin-grid resolution. Default "
                         "(None) tracks 2× the fit n_chi (the old default 8 gave "
                         "Δχ≈0.14, far coarser than a 0.1-wide natal spin).")
    args = ap.parse_args()

    fixed = demo.FIXED
    print(f"[cfg] n_obs={args.n_obs} n_samples={args.n_samples} "
          f"n_seed={args.n_seed} k_max={args.k_max} selection={args.selection}")

    # --- 1. truth (implicit oracle for event positions) + θ* shape -----------
    if args.n_m or args.n_chi:
        from hierarchical_ad_pop_engine.grids import BinaryGrid
        fit_grid = BinaryGrid.default(
            m_min=demo.FIXED["m_min"], m_max=demo.FIXED["m_max"],
            n_m=int(args.n_m or demo.N_M_FIT),
            n_chi=int(args.n_chi or demo.N_CHI_FIT))
    else:
        fit_grid = demo._grid_for_fit()
    truth_implicit = demo.build_truth_model_and_calibrate_rate(
        args.n_obs, grid=fit_grid)
    demo.THETA_STAR["natal_spin_loc"] = 0.05  # boundary-corrected (M4 rationale)
    theta_star = dict(demo.THETA_STAR)

    # --- 2. dist_fn + estimator (the new wiring) -----------------------------
    key = jax.random.PRNGKey(args.seed)
    k_mc, k_cat, k_map = jax.random.split(key, 3)
    _INJECT_DEPLETION = 0.10  # MC injection cascade depth (the budget the soft
                              # gate pins); the recovery ε is matched to it.

    # Cross-method ε-matching (PLAN_cross_method Step 2 / H2).  For --gen mc we
    # inject with MC at depletion f and MUST recover with implicit at the ε that
    # reproduces the SAME cascade depth — otherwise the two methods are at
    # different "times" and recovery is wildly biased (observed 2026-06-01).
    # Build the MC injection up-front, read its achieved depletion, then
    # root-find the implicit ε that matches it.
    recovery_epsilon = args.epsilon
    mc_inject_ev = None
    k_inject = k_cat
    if args.gen == "mc" and args.solver == "implicit":
        mc_inject_dist_fn = make_coagulation_dist_fn(
            v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
            kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
            depletion_fraction=_INJECT_DEPLETION, gate="soft", grid=fit_grid,
            mc_jax_key=k_mc, solver="mc_jax")
        mc_inject_model = mc_inject_dist_fn(**theta_star)
        mc_resp = mc_inject_model._response.mc_jax_response
        mc_inject_ev = mc_resp.evaluator
        n_init = int(getattr(mc_resp, "n_init", args.n_seed))
        W_total = float(jnp.sum(jnp.asarray(mc_inject_ev.weights)))
        f_target = max(0.0, min(0.999, 1.0 - W_total / max(n_init, 1)))

        # f_of_eps: implicit f_natal_depleted at coupling ε (build response once;
        # vary ε in the solve only — ε does not change natal/kernel).
        from hierarchical_ad_pop_engine.response.implicit import (
            solve_implicit_fixed_point,
        )
        _impl_dist_fn = make_coagulation_dist_fn(
            v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
            kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
            grid=fit_grid, solver="implicit", epsilon=1.0)
        _impl_resp = _impl_dist_fn(**theta_star)._response
        _v = jnp.asarray(fixed["v_esc"])

        def _f_of_eps(eps):
            res = solve_implicit_fixed_point(_impl_resp, _v, epsilon=float(eps))
            return float(res.f_natal_depleted)

        recovery_epsilon, f_impl, _status = _match_epsilon_to_fdep(
            _f_of_eps, f_target, lo=1e-3, hi=50.0)
        print(f"[ε-match] MC injection f_dep={f_target:.4f}  →  implicit "
              f"ε*={recovery_epsilon:.4g} (f_natal_depleted={f_impl:.4f}, "
              f"|Δf|={abs(f_impl - f_target):.4f}, {_status})  [H2 parity gate]")
        if _status != "ok" or abs(f_impl - f_target) > 0.02:
            print("[ε-match] WARNING: depletion NOT matched in the converging ε "
                  "range — recovery cascade depth differs from injection; the "
                  "cross-method fit is unreliable until this is resolved "
                  "(raise hi / check implicit convergence at this grid).")

        # H1 gate (PLAN Step 3): cross-method JS(implicit ↔ MC) at the injection
        # θ, BEFORE the fit.  Quantifies the genuine method-to-method shape
        # disagreement (the kernel_b / σ residuals) at the matched ε — if this
        # is large the recovered posterior is a cross-method systematic, not a
        # measurement.
        _js_m, _js_c = cross_method_js(
            _impl_resp, mc_inject_ev, fit_grid, fixed["v_esc"],
            epsilon=recovery_epsilon)
        print(f"[H1-JS ] inject-θ JS(implicit‖MC): dN/dm={_js_m:.4f}  "
              f"dN/dχ={_js_c:.4f} nats (ln2≈0.693 max)  [cross-method gate]")
        if max(_js_m, _js_c) > _H1_JS_WARN:
            print(f"[H1-JS ] WARNING: JS > {_H1_JS_WARN} — implicit and MC "
                  f"disagree on the population SHAPE at θ*; the cross-method "
                  f"recovery carries this as a systematic (chase kernel_b / "
                  f"resolution / H3 primitive parity before trusting the fit).")

    dist_fn = make_coagulation_dist_fn(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
        depletion_fraction=_INJECT_DEPLETION, gate="soft", grid=fit_grid,
        mc_jax_key=k_mc, solver=args.solver, epsilon=recovery_epsilon)
    if args.selection == "vt":
        estimator = make_vt_poisson_mean_estimator(
            vt_powerlaw(), m_min=fixed["m_min"], m_max=fixed["m_max"])
    else:
        estimator = coagulation_poisson_mean_estimator

    # --- 3. self-consistent log_rate calibration against the FIT's μ metric --
    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = estimator(probe, T_obs=1.0, n_quad=args.n_quad)
    log_rate_star = float(jnp.log(args.n_obs / mu_unit))
    theta_star["log_rate"] = log_rate_star
    print(f"[cal] μ(logR=0)={float(mu_unit):.4g} → log_rate* = {log_rate_star:.3f} "
          f"(targets {args.n_obs} detected)")

    # --- 4. multi-sample mock catalogue --------------------------------------
    m_hi = fixed["m_max"] * 2.5
    if args.gen == "self":
        from mock_catalog import scatter_to_posteriors
        truth_model = dist_fn(**theta_star)
        k_true, k_scat = jax.random.split(k_cat, 2)
        # Faithful self-consistent sampler: default the SAMPLER grid to 2× the
        # fit grid so the drawn mock is sharper than the density the likelihood
        # reads (coarse-cell jitter was the implicit σ-inflation root cause,
        # 2026-06-01). Override via --n-sm/--n-sc to probe the dependence.
        n_m_fit, n_chi_fit = int(fit_grid.m.shape[0]), int(fit_grid.chi.shape[0])
        n_sm = int(args.n_sm) if args.n_sm is not None else 2 * n_m_fit
        n_sc = int(args.n_sc) if args.n_sc is not None else 2 * n_chi_fit
        print(f"[mock] self-consistent sampler grid = ({n_sm},{n_sc})  "
              f"(fit grid ({n_m_fit},{n_chi_fit}); faithful default 2×)")
        true_pts = sample_from_model(
            truth_model, args.n_obs, k_true, m_lo=fixed["m_min"], m_hi=m_hi,
            n_sm=n_sm, n_sc=n_sc)
        events = scatter_to_posteriors(
            true_pts, args.n_samples, k_scat, m_hi=m_hi,
            sigma_log_m=args.sigma_log_m, sigma_chi=args.sigma_chi)
        ln_offsets = jnp.zeros((args.n_obs, args.n_samples))
    elif args.gen == "mc":
        # Cross-method injection (PLAN_cross_method Step 1+2): draw EXACT pair
        # events from the MC particle pool; recover with --solver implicit at
        # the ε matched to the injection depletion (computed in §2 above).
        from mock_catalog import scatter_to_posteriors
        k_true, k_scat = jax.random.split(k_inject, 2)
        if mc_inject_ev is None:
            # solver != implicit (e.g. mc_jax recovery): build injection here.
            mc_inject_dist_fn = make_coagulation_dist_fn(
                v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
                kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
                depletion_fraction=_INJECT_DEPLETION, gate="soft", grid=fit_grid,
                mc_jax_key=k_mc, solver="mc_jax")
            mc_inject_ev = mc_inject_dist_fn(
                **theta_star)._response.mc_jax_response.evaluator
        n_part = int(mc_inject_ev.m_active.shape[0])
        n_merg = int(jnp.asarray(mc_inject_ev.mergers).shape[0])
        print(f"[mock] MC injection: {n_part} particle slots, {n_merg} merger "
              f"rows; exact-particle pairing ∝ w·w·K (no grid/jitter)")
        true_pts = pair_sample_from_mc(mc_inject_ev, args.n_obs, k_true)
        events = scatter_to_posteriors(
            true_pts, args.n_samples, k_scat, m_hi=m_hi,
            sigma_log_m=args.sigma_log_m, sigma_chi=args.sigma_chi)
        # Clip scattered samples to the RECOVERY grid box.  The implicit grid
        # density is only defined on [m₀,m_N]×[χ₀,χ_N]; scatter (esp. spin:
        # natal χ≈0.05 blurs below χ₀, cascade χ≈0.69 toward 1) can push a
        # sample off-grid → log_density_at = −inf, and one fully-off-grid event
        # makes logL = −inf.  Clipping keeps the cross-method PE samples on the
        # support the recovery model can represent (delta case is unaffected —
        # particles are already interior).
        gm, gc = fit_grid.m, fit_grid.chi
        m_clip = (float(gm[0]), float(gm[-1]))
        c_clip = (float(gc[0]), float(gc[-1]))
        events = jnp.stack([
            jnp.clip(events[..., 0], *m_clip),
            jnp.clip(events[..., 1], *m_clip),
            jnp.clip(events[..., 2], *c_clip),
            jnp.clip(events[..., 3], *c_clip),
        ], axis=-1)
        ln_offsets = jnp.zeros((args.n_obs, args.n_samples))
    else:
        events, ln_offsets = make_mock_posteriors(
            truth_implicit, args.n_obs, args.n_samples, k_cat, m_hi=m_hi)
    print(f"[cat] gen={args.gen}  events {tuple(events.shape)}  "
          f"ln_offsets {tuple(ln_offsets.shape)}")

    # --- 5. jitted log-likelihood -------------------------------------------
    logL = build_logL(events, ln_offsets, dist_fn, estimator,
                      T_obs=1.0, n_quad=args.n_quad)
    logL_j = jax.jit(logL)
    theta0 = _theta_vec(theta_star)
    ll_star = float(logL_j(theta0))
    print(f"[ll ] logL(θ*) = {ll_star:.3f}")

    # --- 6. profiles ---------------------------------------------------------
    print("\n=== 1-D LOG-LIKELIHOOD PROFILES (through θ*) ===")
    ranges = _profile_ranges(theta0)
    profiles = {}
    ident = {}
    for i, name in enumerate(PARAM_NAMES):
        xs = ranges[name]
        ys = []
        for x in xs:
            th = theta0.at[i].set(float(x))
            ys.append(float(logL_j(th)))
        ys = np.asarray(ys)
        profiles[name] = (xs, ys)
        peak = float(xs[int(np.argmax(ys))])
        truth = float(theta0[i])
        span = float(ys.max() - ys.min())
        # curvature proxy: drop from the peak across the scanned window
        ident[name] = (peak, truth, span)
        flag = "IDENTIFIED" if span > 2.0 else "weak/flat"
        print(f"  {name:18s} truth={truth:+.3f}  peak={peak:+.3f}  "
              f"Δ(logL) over scan={span:6.2f}  [{flag}]")

    # --- 7. MAP (gradient ascent from a perturbed start) ---------------------
    map_recovered = None
    if not args.no_map:
        print(f"\n=== MAP (gradient ascent, {args.map_steps} steps) ===")
        try:
            import optax
        except Exception as exc:  # pragma: no cover
            print(f"  [skip MAP] optax unavailable: {exc}")
        else:
            # init at a MILD perturbation of θ* (a far start lands in a
            # −inf region — the intensity underflows where no data sit).
            u0 = _from_physical(theta0) + jnp.asarray(
                [0.4, 0.25, 0.4, 0.25, 0.6, 0.4])
            neg = lambda u: -logL(_to_physical(u))
            negj = jax.jit(jax.value_and_grad(neg))
            opt = optax.adam(args.map_lr)
            st = opt.init(u0)
            u = u0
            for step in range(args.map_steps):
                val, g = negj(u)
                upd, st = opt.update(g, st)
                u = optax.apply_updates(u, upd)
                if step % 25 == 0 or step == args.map_steps - 1:
                    print(f"   step {step:4d}  -logL = {float(val):.3f}")
            map_recovered = np.asarray(_to_physical(u))
            print("\n   param              truth      MAP       Δ")
            for i, name in enumerate(PARAM_NAMES):
                t = float(theta0[i]); m = float(map_recovered[i])
                print(f"   {name:18s} {t:+8.3f} {m:+8.3f} {m - t:+8.3f}")

    # --- 8. figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
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
        fig.suptitle("Stage-1 single-pulse identifiability — gwkokab "
                     f"analytical-Poisson ({args.selection} selection, "
                     f"N={args.n_obs}, S={args.n_samples})")
        fig.tight_layout()
        out = _DEMO / "figures" / "stage1_identifiability_profiles.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=140)
        print(f"\n[fig] {out}")
    except Exception as exc:  # pragma: no cover
        print(f"[fig] skipped ({exc})")

    print("\n=== SUMMARY ===")
    strong = [n for n in PARAM_NAMES if ident[n][2] > 2.0]
    weak = [n for n in PARAM_NAMES if ident[n][2] <= 2.0]
    print(f"  identified (peaked logL): {', '.join(strong) or '—'}")
    print(f"  weak / flat:              {', '.join(weak) or '—'}")
    return profiles, ident, map_recovered


if __name__ == "__main__":
    main()
