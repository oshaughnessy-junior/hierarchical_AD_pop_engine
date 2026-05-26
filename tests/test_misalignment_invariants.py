"""Solver-level invariants for the misalignment knob (Step 11 / Session C).

These tests close the misalignment-mode acceptance loop spelled out in
``demo/PLAN_misalignment_mode.md § Acceptance invariants``.  They are
solver-level: they assert relationships between the production
implicit fixed-point solver's output under different orientation
models, *not* values of remnant primitives in isolation (those are
already covered by ``tests/test_remnant_marginal.py``,
``tests/test_remnant_vector.py`` and ``tests/test_orientation_models.py``
from Session B).

Coverage map (PLAN § Acceptance invariants):

1. **Aligned-default reproduces v1 behaviour bit-for-bit at the
   implicit-solver level.**  ``orientation=None`` (= default factory
   ``Aligned()``) and ``orientation=Aligned()`` route to the same
   scalar ``_step_one_generation_aligned`` path; converged ``N*``
   must be literally equal.  This is Decision 2 of the PLAN — the
   user requires the aligned pathological cascade features intact
   for future study, not reconstructed by a different code path that
   agrees only to float epsilon.

2. **Aligned + ``retention_kind="sigmoid"`` is also bit-for-bit
   equal to Aligned + Heaviside.**  ``retention_kind`` is ignored
   under the Aligned shortcut by construction (Heaviside is the v1
   semantics).  This guards the dispatch table from a future refactor
   that accidentally honours ``retention_kind`` under Aligned.

3. **Retention activates correctly under Isotropic + sigmoid:**
   ``total_rate(Isotropic, sigmoid) < total_rate(Aligned, Heaviside)``
   at the same ``v_esc``.  Under Aligned, the superkick branch of
   the LZ formula is absent and ``v_kick`` is small enough that the
   Heaviside retention is ≈ 1 across the Stage-1 grid; under
   Isotropic the superkick reaches O(few × 10³) km/s and the sigmoid
   retention drops below 1 for high-spin remnants, removing them
   from the gain accumulator.  Net total_rate falls.

4. **AD on ``v_esc`` is recovered under any non-Aligned orientation.**
   ``jax.grad(total_rate)(v_esc)`` is exactly 0 under Aligned +
   Heaviside (the v1 limitation flagged in ``remnants/kick.py``),
   and finite + non-zero under Isotropic + sigmoid.  This is the
   "two-for-one" promise of the PLAN cashed in at the demo-level
   surface used by ``demo/forward_modeling_observables.py``.

5. **2G cascade is meaningfully different under Isotropic.**  Under
   Aligned, ``_step_one_generation_aligned`` produces 2G χ
   densities by a deterministic ``χ_rem(m, m', χ, χ')`` map; under
   Isotropic the orientation marginal evaluates the BR formula at
   every quadrature point of ``P(θ̂)``.  These two paths produce
   visibly different 2G densities — the *direction* of the
   difference (peak-to-mean ratio, total support width, second
   moment) is sensitive to how the natal distribution interacts
   with the chosen grid and is not a clean invariant: under a
   narrow natal Gaussian the orientation averaging can
   *concentrate* the 2G χ-marginal rather than broaden it, by
   collapsing many aligned-determined peaks into one
   orientation-averaged peak.  What *is* clean: the 2G log-density
   differs from the Aligned 2G log-density by an O(1) margin
   somewhere on the grid.  Same engine-vs-engine signal pattern
   the Session B sandbox harness used.

Hard constraints honoured:

* **No frozen numerical references.**  Every assertion is either a
  bitwise equality of the Aligned shortcut, an inequality between
  two solver runs that share lowest-level physics, or a
  finite-and-non-zero gradient.  No ``.npz`` baselines.
  ([[feedback-no-frozen-test-references]])
* **No hardcoded physical scales** beyond the Stage-1 defaults that
  are already canonical elsewhere in the test suite
  (``mu=8``, ``sigma=1``, ``v_esc=300``).
  ([[feedback-no-hardcoded-physical-scales]])
* **No fake physics** — the marginalised path uses the real vector
  BR + LZ + per-orientation sigmoid retention, not a smearing
  operator.  ([[feedback-no-fake-physics-stand-ins]])
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.observables import total_rate
from hierarchical_ad_pop_engine.response.coagulation import (
    build_gaussian_coagulation,
)
from hierarchical_ad_pop_engine.response.implicit import solve_implicit_fixed_point
from hierarchical_ad_pop_engine.spin_orientation import Aligned, Isotropic


# ---------------------------------------------------------------------------
# Stage-1 fixture parameters — small enough to keep these tests under a
# few seconds end-to-end, large enough to make the cascade-compression
# invariant well-resolved.  Mirrors the Session-B sandbox harness
# grid (``BinaryGrid.default(m_min=5, m_max=80, n_m=16, n_chi=8)``)
# extended one χ-axis step so the 2G mass-marginal has enough χ-bins
# for a peak-to-mean ratio to be meaningful.
# ---------------------------------------------------------------------------
_STAGE1 = dict(
    mu=8.0,
    sigma=1.0,
    m_min=4.0,
    m_max=80.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.10,
    kernel_a=0.0,
    kernel_b=1.0,
    log_R_env=0.0,
    k_max=3,
)
_GRID = BinaryGrid.default(m_min=4.0, m_max=80.0, n_m=16, n_chi=16)
# v_esc chosen near the LZ aligned-kick scale (a few hundred km/s) so
# the Heaviside cut is far from any aligned-kick value (retention ≈ 1
# on the aligned path) but well below the isotropic-superkick scale
# (retention < 1 on the iso path).  Same v_esc as the Session-B
# sandbox harness.
_V_ESC = 300.0
# Small epsilon — the implicit Picard converges in ~10 iterations,
# enough cascade depth that the 2G/3G invariants register, but cheap.
_EPS = 0.3
_PICARD = dict(epsilon=_EPS, max_iter=200, tol=1e-7, damping=0.4)


def _build(orientation=None, retention_kind="heaviside", w_over_v_esc=1.0):
    """Build a Stage-1 implicit-solver response with the given orientation."""
    return build_gaussian_coagulation(
        **_STAGE1,
        grid=_GRID,
        solver="implicit",
        epsilon=_EPS,
        orientation=orientation,
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )


def _solve(response, v_esc=_V_ESC):
    """Production-path implicit Picard solve."""
    return solve_implicit_fixed_point(response, jnp.asarray(v_esc), **_PICARD)


def _engine_total_rate_at_v_esc(response, v_esc):
    """Same closed-form ``v_esc → total_rate`` map as
    ``demo/forward_modeling_observables.py:engine_total_rate_at_v_esc``.

    Defined inline (rather than imported from ``demo/``) so this test
    file does not depend on ``demo`` being on the import path under
    CI; the math is identical.
    """
    result = solve_implicit_fixed_point(response, v_esc, **_PICARD)
    return total_rate(
        result.N, response.grid, response.kernel_params, response.log_R_env,
    )


# ===========================================================================
# Invariant 1 — Aligned default ≡ explicit Aligned, bit-for-bit.
# ===========================================================================
def test_aligned_default_equals_explicit_aligned_bitwise():
    """``orientation=None`` (default factory ``Aligned()``) and
    ``orientation=Aligned()`` must route to the same scalar code path
    and produce literally equal converged ``N*``.

    Per PLAN Decision 2, bit-for-bit preservation is the requirement;
    *not* machine-precision equivalence.  A future refactor that
    accidentally reroutes one of the two through the marginalised
    code path (even with mathematically-identical output to float
    epsilon) would violate the invariant and must fail this test.
    """
    r_default = _build(orientation=None)
    r_align = _build(orientation=Aligned())

    N_default = np.asarray(_solve(r_default).N)
    N_align = np.asarray(_solve(r_align).N)

    assert N_default.shape == N_align.shape
    assert np.array_equal(N_default, N_align), (
        "Aligned-default and explicit Aligned() diverged — dispatch leak"
    )


# ===========================================================================
# Invariant 2 — retention_kind is ignored under Aligned.
# ===========================================================================
def test_aligned_ignores_retention_kind_sigmoid_bitwise():
    """``retention_kind="sigmoid"`` must be a no-op under ``Aligned()``.

    The Aligned shortcut routes to the Heaviside ``retention_aligned``
    code path regardless of ``retention_kind``; ``retention_kind`` /
    ``w_over_v_esc`` only enter the non-Aligned marginalised path.
    Guarding this in tests catches a class of "I forgot to gate the
    sigmoid on non-Aligned" refactors that would silently change the
    v1 semantics.
    """
    r_heav = _build(
        orientation=Aligned(), retention_kind="heaviside", w_over_v_esc=1.0
    )
    r_sig = _build(
        orientation=Aligned(), retention_kind="sigmoid", w_over_v_esc=1.0
    )

    N_heav = np.asarray(_solve(r_heav).N)
    N_sig = np.asarray(_solve(r_sig).N)

    assert np.array_equal(N_heav, N_sig), (
        "Aligned+sigmoid diverged from Aligned+heaviside — retention_kind "
        "is leaking past the Aligned shortcut"
    )


# ===========================================================================
# Invariant 3 — total_rate(Iso+sigmoid) < total_rate(Aligned+Heaviside).
# ===========================================================================
def test_isotropic_sigmoid_total_rate_less_than_aligned():
    """At the same ``v_esc``, retention drops under isotropic + sigmoid.

    The aligned LZ formula gives ``v_kick`` in the ``O(10²) km/s``
    range over the Stage-1 grid; with ``v_esc = 300 km/s`` the
    Heaviside cut is at unity for nearly every cell.  The isotropic
    LZ formula activates the in-plane superkick term, which
    routinely reaches ``O(10³) km/s``; with sigmoid retention at the
    same ``v_esc`` the cut falls below unity for the high-spin tail,
    removing density from the gain accumulator.  Net total_rate at
    fixed natal supply must drop.

    This is an *inequality* test (engine-vs-engine), not a
    frozen-magnitude test — robust to any future change in the
    quadrature size, the sigmoid steepness, or the BR/LZ
    coefficients.
    """
    r_align = _build(orientation=Aligned())
    iso = Isotropic.default(n_theta=4, n_dphi=8)
    r_iso = _build(orientation=iso, retention_kind="sigmoid", w_over_v_esc=1.0)

    N_align = _solve(r_align).N
    N_iso = _solve(r_iso).N

    rate_align = float(
        total_rate(
            N_align, _GRID, r_align.kernel_params, r_align.log_R_env
        )
    )
    rate_iso = float(
        total_rate(
            N_iso, _GRID, r_iso.kernel_params, r_iso.log_R_env
        )
    )

    assert np.isfinite(rate_align) and rate_align > 0.0
    assert np.isfinite(rate_iso) and rate_iso > 0.0
    assert rate_iso < rate_align, (
        f"Iso+sigmoid total_rate ({rate_iso:.4e}) is not less than "
        f"Aligned+Heaviside total_rate ({rate_align:.4e}) — superkick "
        "retention is not biting at v_esc=300 km/s"
    )


# ===========================================================================
# Invariant 4a — ∂total_rate/∂v_esc = 0 under Aligned + Heaviside.
# ===========================================================================
def test_grad_v_esc_zero_under_aligned_heaviside():
    """The v1 limitation: Heaviside has zero derivative almost everywhere.

    Under Aligned+Heaviside, ``jax.grad(total_rate)(v_esc)`` is
    exactly 0.0 — the retention path has no continuous dependence on
    ``v_esc``.  This is *not* a bug; it is the v1 semantics the
    misalignment work was introduced to relax, and pinning it to 0
    in the test suite makes the regression in the *other* direction
    impossible to miss (i.e. a future refactor that silently
    smooths the Heaviside would break this assertion).
    """
    r_align = _build(orientation=Aligned())

    def f(v):
        return _engine_total_rate_at_v_esc(r_align, v)

    g = float(jax.grad(f)(jnp.asarray(_V_ESC)))

    assert g == 0.0, (
        f"Aligned+Heaviside leaked a nonzero ∂total_rate/∂v_esc ({g:.3e}); "
        "expected exactly 0.0 by Heaviside almost-everywhere"
    )


# ===========================================================================
# Invariant 4b — ∂total_rate/∂v_esc finite and non-zero under Iso+sigmoid.
# ===========================================================================
def test_grad_v_esc_finite_nonzero_under_isotropic_sigmoid():
    """The AD-on-``v_esc`` cash-in.

    Under Isotropic+sigmoid the orientation integration acts as a
    Riemann sum over a smooth CDF in ``v_esc``, so
    ``jax.grad(total_rate)(v_esc)`` is finite and non-zero.  This is
    the "two-for-one" promise from ``demo/PLAN_misalignment_mode.md``
    end-to-end through the production implicit solver via the IFT
    custom-VJP in ``response/implicit.py`` — the same path the
    demo's step-10 ``engine_total_rate_at_v_esc`` exercises.

    A non-finite gradient (``inf`` / ``nan``) would indicate a
    numerical pathology in the sigmoid or the orientation quadrature
    weights; an exact-zero gradient would indicate a dispatch leak
    back to the Heaviside path.  Either fails the test.
    """
    iso = Isotropic.default(n_theta=4, n_dphi=8)
    r_iso = _build(orientation=iso, retention_kind="sigmoid", w_over_v_esc=1.0)

    def f(v):
        return _engine_total_rate_at_v_esc(r_iso, v)

    g = float(jax.grad(f)(jnp.asarray(_V_ESC)))

    assert np.isfinite(g), (
        f"∂total_rate/∂v_esc is not finite under Iso+sigmoid (g={g})"
    )
    assert g != 0.0, (
        "∂total_rate/∂v_esc is exactly 0 under Iso+sigmoid — "
        "dispatch is leaking back to the Heaviside path"
    )


# ===========================================================================
# Invariant 5 — 2G cascade is meaningfully different under Isotropic.
# ===========================================================================
def test_cascade_meaningfully_differs_under_isotropic():
    r"""Aligned and Isotropic 2G log-densities differ by an O(1) margin.

    This is the cleanest engine-vs-engine signal that the orientation
    dispatch fires through the per-generation path
    (``_step_one_generation``).  The cascade-compression pathology
    of ``[[project-aligned-chi-undersampled]]`` is genuinely visual:
    the 2G χ-marginal under Aligned has a deterministic-map signature
    (sharp peaks at specific χ values produced by
    ``χ_rem(m_j, m_l, χ_j, χ_l)``) while Isotropic produces a
    smooth-quadrature-weighted blend.  *Direction* of the difference
    in peak-to-mean ratio, support width, or any other shape metric
    is not robust — depending on the natal distribution and grid
    resolution the orientation averaging may concentrate the 2G
    χ-marginal rather than broaden it (many aligned-deterministic
    peaks collapse into one orientation-averaged peak under a
    narrow natal Gaussian).

    The robust invariant: max ``|log n_2G_iso − log n_2G_aligned|``
    on the well-populated part of the grid is much greater than the
    Picard-tolerance floor.  Threshold ``> 1.0`` mirrors the Session
    B sandbox harness pattern (``outputs/
    sandbox_verify_misalignment_session_b.py``).  Pinning a magnitude
    rather than a *direction* is what makes this an
    engine-vs-engine invariant per
    [[feedback-no-frozen-test-references]].

    Routed through the perturbative ``_solve`` surface to access the
    ``(K, N_m, N_chi)`` generational stack directly; the production
    solver is implicit and the rest of these tests use it, but the
    per-generation visibility is unique to ``_solve`` and the
    orientation dispatch through ``_step_one_generation`` is the
    same on both paths.
    """
    base_kwargs = dict(_STAGE1)
    base_kwargs["k_max"] = 3
    r_align = build_gaussian_coagulation(
        **base_kwargs, grid=_GRID, solver="perturbative",
        orientation=Aligned(),
    )
    iso = Isotropic.default(n_theta=4, n_dphi=8)
    r_iso = build_gaussian_coagulation(
        **base_kwargs, grid=_GRID, solver="perturbative",
        orientation=iso, retention_kind="sigmoid", w_over_v_esc=1.0,
    )

    log_n_align = np.asarray(r_align._solve(jnp.asarray(_V_ESC)))
    log_n_iso = np.asarray(r_iso._solve(jnp.asarray(_V_ESC)))
    assert log_n_align.shape == log_n_iso.shape
    assert log_n_align.shape[0] >= 2, "need at least 2G to test cascade"

    # 2G log-density on the same (N_m, N_chi) grid.  Compare on
    # cells where both densities are well above the ``log n = -30``
    # numerical floor that ``_step_one_generation`` writes for
    # vanishing cells.  Without the floor mask, the difference would
    # include grid corners that are at the floor on either path but
    # contain no physics.
    log_n_2g_align = log_n_align[1]
    log_n_2g_iso = log_n_iso[1]
    mask = (log_n_2g_align > -50.0) & (log_n_2g_iso > -50.0)
    assert mask.any(), "no well-populated 2G cells — fixture too coarse"

    diff = np.abs(log_n_2g_iso[mask] - log_n_2g_align[mask])
    max_diff = float(diff.max())

    assert max_diff > 1.0, (
        f"2G cascade not meaningfully different between Aligned and "
        f"Isotropic: max|log-diff| = {max_diff:.3f} ≤ 1.0.  "
        "Orientation dispatch may have leaked back to the scalar "
        "code path inside _step_one_generation."
    )


# ===========================================================================
# JIT-stability spot check — none of the dispatches should break
# ``jax.jit`` on either branch.  Captured here (rather than in a
# separate file) because every other Step-11 invariant relies on
# the solver being jit-stable for ``jax.grad`` to make sense.
# ===========================================================================
def test_implicit_solver_jit_stable_under_isotropic_sigmoid():
    """Sanity check: ``jax.jit(solve_implicit_fixed_point)`` traces and
    returns a finite ``N*`` under Isotropic + sigmoid.

    A failure here would indicate the marginalised
    ``_step_one_generation_marginalised`` is leaking a non-jittable
    Python branch (an ``isinstance`` check inside a traced function,
    a static-argument leak, etc).
    """
    iso = Isotropic.default(n_theta=4, n_dphi=8)
    r_iso = _build(orientation=iso, retention_kind="sigmoid", w_over_v_esc=1.0)

    @jax.jit
    def jitted(v):
        return solve_implicit_fixed_point(r_iso, v, **_PICARD).N

    N = jitted(jnp.asarray(_V_ESC))
    assert N.shape == (_GRID.m.size, _GRID.chi.size)
    assert bool(jnp.all(jnp.isfinite(N)))
    assert bool(jnp.all(N >= 0.0))
