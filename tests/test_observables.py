r"""Tests for the marginal-merger-rate observables module.

What this suite pins
--------------------
1. **Self-consistency under integration.**  Each ``marginal_1d`` and
   ``marginal_2d`` reduces back to :func:`observables.total_rate` when
   the kept axis is itself integrated out — the four 1-D marginals and
   the six 2-D marginals all must agree to ``≲ 1e-12`` relative.  This
   is the headline correctness test: it catches einsum/axis bugs and
   any drift between the 2-D and 1-D code paths.

2. **m_1 ↔ m_2 / χ_1 ↔ χ_2 symmetries.**  The forward-rate integrand
   is symmetric under each swap, so the corresponding marginals must
   be bit-equal (modulo float roundoff).

3. **AD pipeline integrity.**  ``jax.grad`` of a marginal-derived
   scalar w.r.t. an upstream physics parameter must agree with a
   finite-difference reference to ``≤ 1e-4`` (the FD reference's own
   accuracy limit).

4. **Independent re-implementation.**  An explicit numpy 4-D quadrature
   (built from the lowest-level primitives — the kernel formula, the
   trapezoid weights, the density grid) must reproduce the JAX
   marginals to ``≤ 1e-10`` relative.  Per
   ``feedback_red_team_validation`` — internal self-consistency tests
   alone are insufficient; we need a fundamentally different code path
   confirming the same answer.

5. **Ordered-mass fold conserves total rate.**  ``fold_ordered_m``
   moves the unordered ``½ K N N`` density to the ``m_1 ≥ m_2``
   triangle.  Integrating the folded density over the triangle must
   equal :func:`total_rate` exactly.

6. **Sanity:** non-negativity (within float floor), expected shapes,
   axis values returned in ``BinaryGrid`` order, etc.

Why not pin to a closed-form analytic case?  The forward rate
``½ R_env K(m_1, m_2) N(m_1, χ_1) N(m_2, χ_2)`` admits no clean
analytic marginal for the production kernel + Talbot-Thrane PL+Peak
combination; building a contrived case would require fixtures that
diverge from the engine's production path, defeating the purpose.  The
4-D numpy re-implementation gives an independent reference without
introducing fake physics (``feedback_no_fake_physics_stand_ins``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine import observables as obs
from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.kernels import log_pairing_kernel
from hierarchical_ad_pop_engine.response.coagulation import (
    CoagulationResponse,
    build_default_coagulation,
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_response() -> CoagulationResponse:
    r"""Small (m, χ) grid for fast quadrature tests.

    We deliberately use ``solver='perturbative'`` and ``k_max=2`` here —
    we only need a non-degenerate single-BH density ``N(m, χ)`` to feed
    the marginal routines; this is the fastest way to get one without
    running the full implicit Picard loop (which is exercised by
    ``test_implicit_parity.py``).  The density is the same shape and
    units regardless of which solver produced it; the marginal code is
    solver-agnostic.
    """
    return build_default_coagulation(
        k_max=2,
        kernel_a=0.3,
        kernel_b=1.0,
        log_R_env=0.5,
        grid=BinaryGrid.default(m_min=5.0, m_max=80.0, n_m=20, n_chi=8),
    )


@pytest.fixture(scope="module")
def N_mchi(small_response: CoagulationResponse) -> jnp.ndarray:
    r"""Single-BH density ``N = Σ_k n_k`` on the small grid."""
    log_N = small_response._log_total_density(jnp.asarray(300.0))
    return jnp.exp(log_N)


# ---------------------------------------------------------------------------
# Headline: total-rate consistency.
# ---------------------------------------------------------------------------
def test_marginal_1d_integrates_to_total_rate(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""Each 1-D marginal must integrate back to :func:`total_rate`.

    Catches einsum / axis / weight bugs.  Tolerance ``1e-12`` reflects
    the fact that both paths use the same trapezoid weights and only
    differ in einsum contraction order — round-off accumulation is
    bounded by ``O(N_m · N_chi) · eps``.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env

    C_total = obs.total_rate(N_mchi, grid, kp, log_R)
    w_m, w_chi = obs.trapezoid_weights(grid)

    for axis in obs.VALID_AXES:
        x, dC_dx = obs.marginal_1d(N_mchi, grid, kp, log_R, axis=axis)
        w = w_m if axis in (obs.AXIS_M1, obs.AXIS_M2) else w_chi
        C_via_1d = jnp.sum(dC_dx * w)
        rel = abs(C_via_1d / C_total - 1.0)
        assert rel < 1e-12, (
            f"axis {axis!r}: ∫ dC/dx dx = {float(C_via_1d):.6e} vs "
            f"total_rate {float(C_total):.6e}, rel {float(rel):.2e}"
        )


def test_marginal_2d_integrates_to_total_rate(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""Each 2-D marginal must integrate back to :func:`total_rate`.

    Same logic as the 1-D case, just over the kept pair.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env

    C_total = obs.total_rate(N_mchi, grid, kp, log_R)
    w_m, w_chi = obs.trapezoid_weights(grid)

    pairs = [
        (obs.AXIS_M1, obs.AXIS_M2),
        (obs.AXIS_M1, obs.AXIS_CHI1),
        (obs.AXIS_M1, obs.AXIS_CHI2),
        (obs.AXIS_M2, obs.AXIS_CHI1),
        (obs.AXIS_M2, obs.AXIS_CHI2),
        (obs.AXIS_CHI1, obs.AXIS_CHI2),
    ]
    for a, b in pairs:
        x, y, dens = obs.marginal_2d(
            N_mchi, grid, kp, log_R, axes=(a, b)
        )
        w_a = w_m if a in (obs.AXIS_M1, obs.AXIS_M2) else w_chi
        w_b = w_m if b in (obs.AXIS_M1, obs.AXIS_M2) else w_chi
        C_via_2d = jnp.einsum("ij,i,j->", dens, w_a, w_b)
        rel = abs(C_via_2d / C_total - 1.0)
        assert rel < 1e-12, (
            f"axes ({a!r}, {b!r}): ∫∫ dC dx dy = {float(C_via_2d):.6e} vs "
            f"total_rate {float(C_total):.6e}, rel {float(rel):.2e}"
        )


def test_marginal_2d_reduces_to_marginal_1d(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""Integrating one axis of a 2-D marginal reproduces the 1-D marginal.

    E.g. ``∫ dC/(dm_1 dχ_1) dχ_1 == dC/dm_1`` pointwise on the m_1
    grid.  Pins consistency between the 1-D and 2-D code paths.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env
    w_m, w_chi = obs.trapezoid_weights(grid)

    # Test (m1, χ1) → m1 marginalisation against marginal_1d(m1)
    _, _, dens = obs.marginal_2d(
        N_mchi, grid, kp, log_R, axes=(obs.AXIS_M1, obs.AXIS_CHI1)
    )
    dC_dm1_via_2d = jnp.einsum("ij,j->i", dens, w_chi)
    _, dC_dm1_direct = obs.marginal_1d(N_mchi, grid, kp, log_R, axis=obs.AXIS_M1)
    rel = jnp.max(
        jnp.abs(dC_dm1_via_2d - dC_dm1_direct)
        / jnp.maximum(jnp.abs(dC_dm1_direct), 1e-30)
    )
    assert float(rel) < 1e-12

    # Test (m1, m2) → m1 marginalisation against marginal_1d(m1)
    _, _, dens_mm = obs.marginal_2d(
        N_mchi, grid, kp, log_R, axes=(obs.AXIS_M1, obs.AXIS_M2)
    )
    dC_dm1_via_mm = jnp.einsum("ij,j->i", dens_mm, w_m)
    rel = jnp.max(
        jnp.abs(dC_dm1_via_mm - dC_dm1_direct)
        / jnp.maximum(jnp.abs(dC_dm1_direct), 1e-30)
    )
    assert float(rel) < 1e-12


# ---------------------------------------------------------------------------
# Symmetries.
# ---------------------------------------------------------------------------
def test_m1_m2_symmetry(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``dC/dm_1`` and ``dC/dm_2`` are identical (forward rate is symmetric)."""
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env

    _, dC_dm1 = obs.marginal_1d(N_mchi, grid, kp, log_R, axis=obs.AXIS_M1)
    _, dC_dm2 = obs.marginal_1d(N_mchi, grid, kp, log_R, axis=obs.AXIS_M2)
    np.testing.assert_allclose(np.asarray(dC_dm1), np.asarray(dC_dm2), rtol=1e-13)


def test_chi1_chi2_symmetry(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``dC/dχ_1`` and ``dC/dχ_2`` are identical."""
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env

    _, dC_dchi1 = obs.marginal_1d(N_mchi, grid, kp, log_R, axis=obs.AXIS_CHI1)
    _, dC_dchi2 = obs.marginal_1d(N_mchi, grid, kp, log_R, axis=obs.AXIS_CHI2)
    np.testing.assert_allclose(
        np.asarray(dC_dchi1), np.asarray(dC_dchi2), rtol=1e-13
    )


def test_marginal_2d_m1_m2_is_symmetric(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``dC/(dm_1 dm_2)`` is a symmetric matrix in ``(m_1, m_2)``."""
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env

    _, _, dens = obs.marginal_2d(
        N_mchi, grid, kp, log_R, axes=(obs.AXIS_M1, obs.AXIS_M2)
    )
    np.testing.assert_allclose(np.asarray(dens), np.asarray(dens).T, rtol=1e-13)


def test_marginal_2d_axis_order_transposes(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``marginal_2d(axes=(a,b))`` and ``axes=(b,a)`` are transposes."""
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env

    _, _, ab = obs.marginal_2d(
        N_mchi, grid, kp, log_R, axes=(obs.AXIS_M1, obs.AXIS_CHI2)
    )
    _, _, ba = obs.marginal_2d(
        N_mchi, grid, kp, log_R, axes=(obs.AXIS_CHI2, obs.AXIS_M1)
    )
    np.testing.assert_allclose(np.asarray(ab), np.asarray(ba).T, rtol=1e-13)


# ---------------------------------------------------------------------------
# Independent re-implementation — the gold-standard validation.
# ---------------------------------------------------------------------------
def _numpy_rate_density_4d(
    N_mchi: np.ndarray,
    m: np.ndarray,
    chi: np.ndarray,
    kernel_a: float,
    kernel_b: float,
    log_R_env: float,
) -> np.ndarray:
    r"""Independent pure-numpy 4-D forward-rate density.

    Shares **only** the kernel functional form (``(m1+m2)^a q^b``) and
    the (m, χ) grid + density values with the JAX path — no shared
    helpers, no shared einsum.  This is the red-team backstop per
    ``feedback_red_team_validation``.
    """
    # K(m_i, m_j) on the cross-product mass grid.
    m1 = m[:, None]
    m2 = m[None, :]
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    K = (m1 + m2) ** kernel_a * q ** kernel_b  # (N_m, N_m)

    R_env = np.exp(log_R_env)
    # ½ R_env K(m1, m2) N(m1, χ1) N(m2, χ2)
    rate = 0.5 * R_env * (
        K[:, None, :, None]
        * N_mchi[:, :, None, None]
        * N_mchi[None, None, :, :]
    )
    return rate  # (N_m, N_chi, N_m, N_chi)


def _numpy_trapezoid_weights(
    m: np.ndarray, log_m: np.ndarray, chi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Same symmetric-trapezoid weights as :func:`trapezoid_weights`, in numpy."""
    dlog_m = np.diff(log_m)
    dlog_m_full = np.concatenate(
        [dlog_m[:1] * 0.5, 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:] * 0.5]
    )
    w_m = m * dlog_m_full
    d = np.diff(chi)
    w_chi = np.concatenate([d[:1] * 0.5, 0.5 * (d[:-1] + d[1:]), d[-1:] * 0.5])
    return w_m, w_chi


def test_independent_numpy_marginals_match_jax(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""Red-team backstop: explicit numpy quadrature matches the JAX path.

    Builds the 4-D rate tensor in numpy from scratch and computes 1-D
    and 2-D marginals by raw axis sums.  Agreement to ``≤ 1e-12``
    relative confirms the JAX einsum contractions implement the
    intended integrals — not just self-consistent, but **correct**.
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = float(small_response.log_R_env)

    m = np.asarray(grid.m)
    log_m = np.asarray(grid.log_m)
    chi = np.asarray(grid.chi)
    N_np = np.asarray(N_mchi)
    a = float(kp["a"])
    b = float(kp["b"])

    R_4d_np = _numpy_rate_density_4d(N_np, m, chi, a, b, log_R)
    w_m_np, w_chi_np = _numpy_trapezoid_weights(m, log_m, chi)

    # Total rate.
    C_np = np.einsum(
        "ijkl,i,j,k,l->", R_4d_np, w_m_np, w_chi_np, w_m_np, w_chi_np
    )
    C_jax = float(obs.total_rate(N_mchi, grid, kp, small_response.log_R_env))
    assert abs(C_np / C_jax - 1.0) < 1e-12

    # 1-D marginals: contract the 4-D tensor over three axes.
    # axis layout: (m1, χ1, m2, χ2) — indices (0, 1, 2, 3).
    contractions_1d = {
        obs.AXIS_M1: ("j,k,l", w_chi_np, w_m_np, w_chi_np),
        obs.AXIS_CHI1: ("i,k,l", w_m_np, w_m_np, w_chi_np),
        obs.AXIS_M2: ("i,j,l", w_m_np, w_chi_np, w_chi_np),
        obs.AXIS_CHI2: ("i,j,k", w_m_np, w_chi_np, w_m_np),
    }
    keep_axis_indices = {
        obs.AXIS_M1: 0,
        obs.AXIS_CHI1: 1,
        obs.AXIS_M2: 2,
        obs.AXIS_CHI2: 3,
    }
    axis_chars = "ijkl"
    for axis, (kept_chars, w1, w2, w3) in contractions_1d.items():
        keep_idx = keep_axis_indices[axis]
        keep_char = axis_chars[keep_idx]
        einsum_str = f"ijkl,{kept_chars[0]},{kept_chars[2]},{kept_chars[4]}->{keep_char}"
        dC_np = np.einsum(einsum_str, R_4d_np, w1, w2, w3)
        _, dC_jax = obs.marginal_1d(
            N_mchi, grid, kp, small_response.log_R_env, axis=axis
        )
        dC_jax_np = np.asarray(dC_jax)
        rel = np.max(np.abs(dC_np - dC_jax_np) / np.maximum(np.abs(dC_np), 1e-30))
        assert rel < 1e-12, (
            f"1-D marginal axis {axis!r}: numpy vs JAX max rel = {rel:.2e}"
        )

    # 2-D (m1, m2) marginal — contract over χ axes.
    dC_mm_np = np.einsum("ijkl,j,l->ik", R_4d_np, w_chi_np, w_chi_np)
    _, _, dC_mm_jax = obs.marginal_2d(
        N_mchi, grid, kp, small_response.log_R_env,
        axes=(obs.AXIS_M1, obs.AXIS_M2),
    )
    rel = np.max(
        np.abs(dC_mm_np - np.asarray(dC_mm_jax))
        / np.maximum(np.abs(dC_mm_np), 1e-30)
    )
    assert rel < 1e-12, f"(m1, m2) marginal: numpy vs JAX max rel = {rel:.2e}"

    # 2-D (m1, χ1) marginal — contract over (m2, χ2).
    dC_mc_np = np.einsum("ijkl,k,l->ij", R_4d_np, w_m_np, w_chi_np)
    _, _, dC_mc_jax = obs.marginal_2d(
        N_mchi, grid, kp, small_response.log_R_env,
        axes=(obs.AXIS_M1, obs.AXIS_CHI1),
    )
    rel = np.max(
        np.abs(dC_mc_np - np.asarray(dC_mc_jax))
        / np.maximum(np.abs(dC_mc_np), 1e-30)
    )
    assert rel < 1e-12, f"(m1, χ1) marginal: numpy vs JAX max rel = {rel:.2e}"

    # 2-D (m1, χ2) cross-pair marginal — contract over (χ1, m2).
    dC_m1c2_np = np.einsum("ijkl,j,k->il", R_4d_np, w_chi_np, w_m_np)
    _, _, dC_m1c2_jax = obs.marginal_2d(
        N_mchi, grid, kp, small_response.log_R_env,
        axes=(obs.AXIS_M1, obs.AXIS_CHI2),
    )
    rel = np.max(
        np.abs(dC_m1c2_np - np.asarray(dC_m1c2_jax))
        / np.maximum(np.abs(dC_m1c2_np), 1e-30)
    )
    assert rel < 1e-12, (
        f"(m1, χ2) marginal: numpy vs JAX max rel = {rel:.2e}"
    )


# ---------------------------------------------------------------------------
# Ordered-mass fold.
# ---------------------------------------------------------------------------
def test_fold_ordered_m_conserves_total_rate(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``∫∫_{m1≥m2} fold(dC/dm1 dm2) dm1 dm2 == total_rate``."""
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env

    _, _, dens = obs.marginal_2d(
        N_mchi, grid, kp, log_R, axes=(obs.AXIS_M1, obs.AXIS_M2)
    )
    folded = obs.fold_ordered_m(grid.m, dens)
    w_m, _ = obs.trapezoid_weights(grid)

    C_folded = jnp.einsum("ij,i,j->", folded, w_m, w_m)
    C_total = obs.total_rate(N_mchi, grid, kp, log_R)
    rel = abs(float(C_folded) / float(C_total) - 1.0)
    assert rel < 1e-12, (
        f"folded ordered-mass integral {float(C_folded):.6e} vs "
        f"total_rate {float(C_total):.6e}, rel {rel:.2e}"
    )


def test_fold_ordered_m_is_zero_below_diagonal(small_response):
    r"""Lower triangle (``m_1 < m_2``) must be exactly 0 after folding."""
    grid = small_response.grid
    n = grid.m.size
    # Synthetic constant density; only the mask matters here.
    dens = jnp.ones((n, n))
    folded = obs.fold_ordered_m(grid.m, dens)
    folded_np = np.asarray(folded)
    for i in range(n):
        for j in range(n):
            if grid.m[i] < grid.m[j]:
                assert folded_np[i, j] == 0.0


# ---------------------------------------------------------------------------
# AD: finite-difference vs jax.grad.
# ---------------------------------------------------------------------------
def test_total_rate_grad_finite_difference(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""``jax.grad(total_rate)`` w.r.t. ``log_R_env`` matches FD.

    Since ``total_rate ∝ exp(log_R_env)``, the analytic gradient is
    the total rate itself.  We check this to ``< 1e-6`` relative —
    well inside the float64 FD floor.
    """
    grid = small_response.grid
    kp = small_response.kernel_params

    def f(log_R):
        return obs.total_rate(N_mchi, grid, kp, log_R)

    log_R0 = jnp.asarray(0.5)
    grad_ad = jax.grad(f)(log_R0)
    h = 1e-5
    grad_fd = (f(log_R0 + h) - f(log_R0 - h)) / (2 * h)
    rel = abs(float(grad_ad) / float(grad_fd) - 1.0)
    assert rel < 1e-6, (
        f"jax.grad {float(grad_ad):.6e} vs FD {float(grad_fd):.6e}, rel {rel:.2e}"
    )


def test_marginal_1d_grad_through_N(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""Gradient through the marginals w.r.t. the input density ``N`` is finite.

    ``∂(marginal sum) / ∂N`` should be analytically computable: the
    marginal is bilinear in ``N``, so the gradient is linear in ``N``
    — far from any pathology.  We pin that the AD path returns a
    finite, non-zero gradient (the kind of basic invariant the
    population-fit loop relies on).
    """
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env

    def loss(N):
        _, dC_dm1 = obs.marginal_1d(N, grid, kp, log_R, axis=obs.AXIS_M1)
        return jnp.sum(dC_dm1)

    grad = jax.grad(loss)(N_mchi)
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.sum(jnp.abs(grad))) > 0.0


def test_marginal_2d_grad_through_N(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""Same AD sanity for a 2-D marginal."""
    grid = small_response.grid
    kp = small_response.kernel_params
    log_R = small_response.log_R_env

    def loss(N):
        _, _, dens = obs.marginal_2d(
            N, grid, kp, log_R, axes=(obs.AXIS_M1, obs.AXIS_CHI2)
        )
        return jnp.sum(dens)

    grad = jax.grad(loss)(N_mchi)
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.sum(jnp.abs(grad))) > 0.0


# ---------------------------------------------------------------------------
# Adapter sanity.
# ---------------------------------------------------------------------------
def test_marginals_from_implicit_bundle_layout(
    small_response: CoagulationResponse, N_mchi: jnp.ndarray
):
    r"""Convenience bundle returns the expected dict layout.

    We don't run the implicit solver here (it's tested elsewhere); we
    build a stub ``ImplicitSolveResult``-shaped object carrying just
    the ``N`` field.  The bundle adapter doesn't read any other field.
    """
    from hierarchical_ad_pop_engine.response.implicit import ImplicitSolveResult

    stub = ImplicitSolveResult(
        N=N_mchi,
        log_N=jnp.log(jnp.maximum(N_mchi, 1e-30)),
        epsilon=jnp.asarray(1.0),
        n_iter=jnp.asarray(1.0),
        final_diff=jnp.asarray(0.0),
        loss_integral=jnp.asarray(0.0),
        natal_integral=jnp.asarray(1.0),
        f_natal_depleted=jnp.asarray(0.0),
    )
    bundle = obs.marginals_from_implicit(stub, small_response)
    assert set(bundle["1d"].keys()) == set(obs.VALID_AXES)
    assert len(bundle["2d"]) == 6
    # Spot-check: 1-D m1 marginal integrates to total_rate.
    x, dC = bundle["1d"][obs.AXIS_M1]
    w_m, _ = obs.trapezoid_weights(small_response.grid)
    rel = abs(float(jnp.sum(dC * w_m)) / float(bundle["total_rate"]) - 1.0)
    assert rel < 1e-12
