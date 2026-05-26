"""Tests for the orientation-distribution abstraction.

Invariant-based (no frozen numerical references — see
``feedback-no-frozen-test-references``).  Every assertion is either
algebraic identity (Σ w = 1, scalar-reduction equality), a JAX
property (``jax.grad`` finite, jit-trace stable), or a quadrature-
exactness statement (Gauss-Legendre on a polynomial integrand).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# Match the rest of the test suite — matches production AD precision.
jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.spin_orientation import (  # noqa: E402
    Aligned,
    Fisher,
    Isotropic,
)


# ---------------------------------------------------------------------------
# Aligned
# ---------------------------------------------------------------------------
def test_aligned_quadrature_is_single_delta_point():
    """Aligned has one quad point at (cos θ_1, cos θ_2, Δφ, w) = (1, 1, 0, 1)."""
    q = Aligned().quadrature_3d()
    assert q.shape == (1, 4)
    assert float(q[0, 0]) == pytest.approx(1.0)
    assert float(q[0, 1]) == pytest.approx(1.0)
    assert float(q[0, 2]) == pytest.approx(0.0)
    assert float(q[0, 3]) == pytest.approx(1.0)


def test_aligned_sample_is_zero():
    """All samples lie along ẑ — θ_i = 0, φ_i = 0."""
    key = jax.random.PRNGKey(0)
    t1, p1, t2, p2 = Aligned().sample(key, 17)
    for a in (t1, p1, t2, p2):
        assert a.shape == (17,)
        assert jnp.allclose(a, 0.0)


# ---------------------------------------------------------------------------
# Isotropic
# ---------------------------------------------------------------------------
def test_isotropic_weights_sum_to_one():
    """Σ w_k = 1: ∫ 1 · P_iso dΩ_1 dΩ_2 = 1 (and quadrature is exact for
    constants since GL with any n integrates degree-0 exactly)."""
    iso = Isotropic.default()
    q = iso.quadrature_3d()
    total = float(jnp.sum(q[:, 3]))
    assert total == pytest.approx(1.0, rel=1e-12)


def test_isotropic_first_moment_of_cos_theta_is_zero():
    """⟨cos θ⟩_iso = 0 — odd function over [-1, 1] with uniform measure.
    GL nodes for any n ≥ 1 integrate degree-1 polynomials exactly."""
    iso = Isotropic.default()
    q = iso.quadrature_3d()
    cos_t1 = q[:, 0]
    weights = q[:, 3]
    mean_ct1 = float(jnp.sum(weights * cos_t1))
    assert abs(mean_ct1) < 1e-12


def test_isotropic_second_moment_of_cos_theta_is_one_third():
    """⟨cos² θ⟩_iso = 1/3 — exact for GL with n ≥ 2 (degree-2 polynomial)."""
    iso = Isotropic.default()
    q = iso.quadrature_3d()
    cos_t1 = q[:, 0]
    weights = q[:, 3]
    second = float(jnp.sum(weights * jnp.square(cos_t1)))
    assert second == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_isotropic_dphi_uniform_first_moment_zero():
    """⟨e^{i Δφ}⟩_iso = 0 — periodic trapezoidal exact for trig at any n."""
    iso = Isotropic.default()
    q = iso.quadrature_3d()
    weights = q[:, 3]
    dphi = q[:, 2]
    re = float(jnp.sum(weights * jnp.cos(dphi)))
    im = float(jnp.sum(weights * jnp.sin(dphi)))
    assert abs(re) < 1e-12
    assert abs(im) < 1e-12


def test_isotropic_sample_shapes_and_ranges():
    key = jax.random.PRNGKey(1)
    t1, p1, t2, p2 = Isotropic.default().sample(key, 1000)
    for a in (t1, p1, t2, p2):
        assert a.shape == (1000,)
        assert jnp.all(jnp.isfinite(a))
    assert jnp.all((0.0 <= t1) & (t1 <= jnp.pi))
    assert jnp.all((0.0 <= p1) & (p1 < 2.0 * jnp.pi + 1e-6))


# ---------------------------------------------------------------------------
# Fisher
# ---------------------------------------------------------------------------
def test_fisher_kappa_zero_limit_matches_isotropic_weights():
    """At κ = 0 Fisher must reduce to Isotropic, weight-for-weight.

    Same nodes by construction, weights agree via κ/sinh κ → 1.
    """
    iso = Isotropic.default(n_theta=8, n_dphi=16)
    fish = Fisher.default(kappa=0.0, n_theta=8, n_dphi=16)
    q_iso = iso.quadrature_3d()
    q_f = fish.quadrature_3d()
    # Same nodes — quadrature columns 0,1,2 identical.
    assert jnp.allclose(q_iso[:, :3], q_f[:, :3])
    # Same weights to high precision.
    assert jnp.allclose(q_iso[:, 3], q_f[:, 3], rtol=1e-10, atol=1e-12)


def test_fisher_weights_sum_to_one_at_finite_kappa():
    """Σ w_k(κ) = 1 — Fisher PDF on the sphere is normalised by construction.

    Holds to *quadrature accuracy* — for κ ≤ a few, GL(8) on cos θ
    resolves the Fisher peak well enough for relative error < 1e-6.
    """
    for kappa in (0.5, 1.0, 2.0, 4.0):
        fish = Fisher.default(kappa=kappa, n_theta=16, n_dphi=16)
        q = fish.quadrature_3d()
        total = float(jnp.sum(q[:, 3]))
        assert total == pytest.approx(1.0, rel=1e-6), f"κ = {kappa}, total = {total}"


def test_fisher_mean_cos_theta_matches_analytic():
    """⟨cos θ⟩_Fisher = coth(κ) − 1/κ (Langevin function), per-axis.

    Compared at large enough n_θ to resolve the Fisher peak.
    """
    for kappa in (0.5, 1.0, 2.0):
        fish = Fisher.default(kappa=kappa, n_theta=32, n_dphi=8)
        q = fish.quadrature_3d()
        # Marginal over spin-1: integrate out spin-2 and Δφ.
        weights_marg_1 = q[:, 3]  # joint already normalised → marginal
        mean_ct1 = float(jnp.sum(weights_marg_1 * q[:, 0]))
        analytic = 1.0 / jnp.tanh(kappa) - 1.0 / kappa
        assert mean_ct1 == pytest.approx(float(analytic), rel=1e-5)


def test_fisher_kappa_gradient_is_finite():
    """``jax.grad`` of any smooth functional of Fisher weights w.r.t. κ
    must be finite — this is PLAN Decision 4 (κ differentiable)."""

    def total_weight(kappa: jnp.ndarray) -> jnp.ndarray:
        q = Fisher.default(kappa=kappa, n_theta=16, n_dphi=16).quadrature_3d()
        return jnp.sum(q[:, 3])

    g = jax.grad(total_weight)(jnp.asarray(1.5))
    assert jnp.isfinite(g)
    # ⟨1⟩_Fisher = 1 for any κ ⇒ ∂/∂κ = 0 to quadrature accuracy.
    assert abs(float(g)) < 1e-5


def test_fisher_mean_costheta_gradient_is_finite_and_positive():
    """⟨cos θ⟩_Fisher = L(κ) is strictly increasing in κ for κ > 0.

    ``jax.grad`` w.r.t. κ must be finite and positive.
    """

    def mean_ct(kappa: jnp.ndarray) -> jnp.ndarray:
        q = Fisher.default(kappa=kappa, n_theta=32, n_dphi=8).quadrature_3d()
        return jnp.sum(q[:, 3] * q[:, 0])

    g = jax.grad(mean_ct)(jnp.asarray(1.5))
    assert jnp.isfinite(g)
    assert float(g) > 0.0


def test_fisher_high_kappa_concentrates_near_unity():
    """Increasing κ pushes Fisher mass towards cos θ = 1.  Sanity check —
    not a frozen numerical reference: the inequality is structural."""
    q_low = Fisher.default(kappa=0.5, n_theta=32, n_dphi=8).quadrature_3d()
    q_hi = Fisher.default(kappa=4.0, n_theta=32, n_dphi=8).quadrature_3d()
    mean_low = float(jnp.sum(q_low[:, 3] * q_low[:, 0]))
    mean_hi = float(jnp.sum(q_hi[:, 3] * q_hi[:, 0]))
    # Both ≥ 0, hi > low, hi closer to 1.
    assert 0.0 < mean_low < mean_hi < 1.0


def test_fisher_quadrature_is_jit_stable():
    """Building a Fisher quadrature inside ``jax.jit`` must trace
    without recompilation across κ values (κ is a traced leaf, nodes
    are not)."""

    @jax.jit
    def first_weight(kappa: jnp.ndarray) -> jnp.ndarray:
        return Fisher.default(kappa=kappa, n_theta=8, n_dphi=8).quadrature_3d()[0, 3]

    a = float(first_weight(jnp.asarray(1.0)))
    b = float(first_weight(jnp.asarray(2.0)))
    assert jnp.isfinite(a) and jnp.isfinite(b)
    assert a != b
