"""Tests for tilt-marginalised wrappers (Session B of misalignment work).

Invariant-based — no frozen numerical references (see
``feedback-no-frozen-test-references``).  Every assertion is one of:

* **Aligned shortcut equivalence**: every entry point with ``Aligned()``
  must return the existing scalar value bit-for-bit (the dispatch
  routes to the unchanged scalar code path; not "machine precision",
  *literal equality*).
* **Quadrature normalisation**: ``Σ w_k = 1`` for any concrete
  ``OrientationModel`` after :func:`*_distribution`.
* **Numpy-MC backstop**: marginalised distribution moments under
  ``Isotropic()`` match a brute-force numpy Monte-Carlo over the
  vector primitives at large ``n_mc``, to several standard errors.
* **Smoothness in v_esc**: ``∂P_retain/∂v_esc`` is finite and non-zero
  under any non-Aligned orientation (the AD-on-``v_esc`` two-for-one
  win from the PLAN).
* **Aligned-superkick ordering**: marginalised ``v_kick`` distribution
  under ``Isotropic()`` is dominated by configurations with much
  larger kick than the aligned point — the qualitative statement that
  motivates the retention-activates-correctly story.
* **Differentiability** in ``(m1, m2, chi_mag, v_esc)`` and in
  ``orientation.kappa`` for :class:`Fisher`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.remnants import (  # noqa: E402
    P_retain_marginalised,
    chi_rem,
    chi_rem_at_orientation,
    chi_rem_distribution,
    chi_rem_sample,
    chi_rem_vec,
    retention_aligned,
    v_kick_aligned,
    v_kick_at_orientation,
    v_kick_distribution,
    v_kick_sample,
    v_kick_vec,
)
from hierarchical_ad_pop_engine.spin_orientation import (  # noqa: E402
    Aligned,
    Fisher,
    Isotropic,
)


# ---------------------------------------------------------------------------
# Aligned shortcut bit-for-bit equivalence at every entry point.
#
# These are LITERAL EQUALITY assertions, not rtol/atol — the Aligned
# dispatch routes to the existing scalar code path with no
# arithmetic, so the result must be the same float to the bit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "m1, m2, chi1, chi2",
    [
        (30.0, 20.0, 0.0, 0.0),
        (40.0, 20.0, 0.3, 0.0),
        (50.0, 5.0, 0.7, -0.3),
    ],
)
def test_chi_rem_distribution_aligned_bitwise(m1, m2, chi1, chi2):
    x, w = chi_rem_distribution(
        jnp.asarray(m1), jnp.asarray(m2), jnp.asarray(chi1), jnp.asarray(chi2),
        Aligned(),
    )
    expected = chi_rem(
        jnp.asarray(m1), jnp.asarray(m2), jnp.asarray(chi1), jnp.asarray(chi2)
    )
    assert x.shape == (1,)
    assert w.shape == (1,)
    assert float(x[0]) == float(expected)
    assert float(w[0]) == 1.0


@pytest.mark.parametrize(
    "m1, m2, chi1, chi2",
    [
        (30.0, 20.0, 0.0, 0.0),
        (40.0, 20.0, 0.3, 0.0),
        (50.0, 5.0, 0.7, -0.3),
    ],
)
def test_v_kick_distribution_aligned_bitwise(m1, m2, chi1, chi2):
    v, w = v_kick_distribution(
        jnp.asarray(m1), jnp.asarray(m2), jnp.asarray(chi1), jnp.asarray(chi2),
        Aligned(),
    )
    expected = v_kick_aligned(
        jnp.asarray(m1), jnp.asarray(m2), jnp.asarray(chi1), jnp.asarray(chi2)
    )
    assert v.shape == (1,)
    assert float(v[0]) == float(expected)
    assert float(w[0]) == 1.0


@pytest.mark.parametrize("v_esc", [200.0, 300.0, 1000.0])
def test_P_retain_marginalised_aligned_bitwise(v_esc):
    """Aligned shortcut routes to Heaviside retention_aligned bit-for-bit."""
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1, chi2 = jnp.asarray(0.3), jnp.asarray(0.0)
    p_marg = P_retain_marginalised(m1, m2, chi1, chi2, jnp.asarray(v_esc), Aligned())
    p_align = retention_aligned(m1, m2, chi1, chi2, jnp.asarray(v_esc))
    assert float(p_marg) == float(p_align)


def test_chi_rem_sample_aligned_bitwise():
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1, chi2 = jnp.asarray(0.3), jnp.asarray(-0.2)
    x = chi_rem_sample(jax.random.PRNGKey(0), m1, m2, chi1, chi2, Aligned())
    expected = chi_rem(m1, m2, chi1, chi2)
    assert float(x) == float(expected)


def test_v_kick_sample_aligned_bitwise():
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1, chi2 = jnp.asarray(0.3), jnp.asarray(-0.2)
    v = v_kick_sample(jax.random.PRNGKey(0), m1, m2, chi1, chi2, Aligned())
    expected = v_kick_aligned(m1, m2, chi1, chi2)
    assert float(v) == float(expected)


# ---------------------------------------------------------------------------
# Quadrature normalisation — Σ w_k = 1 (to quadrature accuracy).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model",
    [
        Aligned(),
        Isotropic.default(n_theta=8, n_dphi=16),
        Fisher.default(kappa=0.0, n_theta=8, n_dphi=16),
        Fisher.default(kappa=1.5, n_theta=8, n_dphi=16),
        Fisher.default(kappa=5.0, n_theta=12, n_dphi=24),
    ],
)
def test_distribution_weights_normalised(model):
    """Σ w_k = 1 (to quadrature accuracy) for any orientation model."""
    _, w = chi_rem_distribution(
        jnp.asarray(40.0), jnp.asarray(20.0),
        jnp.asarray(0.5), jnp.asarray(0.3),
        model,
    )
    assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-2, abs=1e-2)


def test_v_kick_distribution_weights_match_chi_rem_distribution_weights():
    """chi_rem_distribution and v_kick_distribution must share the same
    orientation quadrature (they're the same physical integral over
    the same P(θ̂)) — the weight arrays are equal regardless of which
    quantity is being integrated."""
    model = Isotropic.default()
    _, w_chi = chi_rem_distribution(
        jnp.asarray(40.0), jnp.asarray(20.0),
        jnp.asarray(0.5), jnp.asarray(0.3),
        model,
    )
    _, w_v = v_kick_distribution(
        jnp.asarray(40.0), jnp.asarray(20.0),
        jnp.asarray(0.5), jnp.asarray(0.3),
        model,
    )
    assert jnp.allclose(w_chi, w_v, rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# Numpy-MC backstop — marginalised first/second moments under Isotropic()
# must match a brute-force MC over the same vector primitives at large
# n_mc, to within a few standard errors.
#
# Per ``[[feedback-red-team-validation]]``: the right validator is an
# independent re-implementation through the same lowest-level physics
# primitives (chi_rem_vec, v_kick_vec) but a different higher-level
# algorithm (random MC over uniform-on-sphere draws, vs deterministic
# Gauss-Legendre × trapezoidal quadrature).  Agreement to MC standard
# error validates the deterministic quadrature.
# ---------------------------------------------------------------------------
def _mc_uniform_on_sphere_pair(rng, n_mc):
    """Pair of uniform-on-sphere unit vectors via numpy."""
    z1 = rng.uniform(-1.0, 1.0, size=n_mc)
    z2 = rng.uniform(-1.0, 1.0, size=n_mc)
    phi1 = rng.uniform(0.0, 2.0 * np.pi, size=n_mc)
    phi2 = rng.uniform(0.0, 2.0 * np.pi, size=n_mc)
    s1 = np.sqrt(np.maximum(0.0, 1.0 - z1 * z1))
    s2 = np.sqrt(np.maximum(0.0, 1.0 - z2 * z2))
    e1 = np.stack(
        [s1 * np.cos(phi1), s1 * np.sin(phi1), z1], axis=-1
    )
    e2 = np.stack(
        [s2 * np.cos(phi2), s2 * np.sin(phi2), z2], axis=-1
    )
    return e1, e2


def test_chi_rem_distribution_isotropic_moments_match_numpy_mc():
    """Marginalised ⟨|χ_rem|⟩ and ⟨|χ_rem|²⟩ under Isotropic() match
    a numpy MC over chi_rem_vec at large n_mc."""
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1_mag, chi2_mag = jnp.asarray(0.7), jnp.asarray(0.5)
    model = Isotropic.default(n_theta=16, n_dphi=32)

    chi_q, w = chi_rem_distribution(m1, m2, chi1_mag, chi2_mag, model)
    quad_mean = float(jnp.sum(w * chi_q))
    quad_msq = float(jnp.sum(w * jnp.square(chi_q)))

    # Numpy MC over the same primitive.
    rng = np.random.default_rng(0)
    n_mc = 200_000
    e1, e2 = _mc_uniform_on_sphere_pair(rng, n_mc)
    chi1_v = float(chi1_mag) * e1
    chi2_v = float(chi2_mag) * e2
    chi_rem_mc_vec = chi_rem_vec(
        jnp.asarray(m1), jnp.asarray(m2),
        jnp.asarray(chi1_v), jnp.asarray(chi2_v),
    )
    mag = np.asarray(jnp.sqrt(jnp.sum(jnp.square(chi_rem_mc_vec), axis=-1)))
    mc_mean = float(mag.mean())
    mc_msq = float((mag * mag).mean())
    mc_se_mean = float(mag.std() / np.sqrt(n_mc))
    mc_se_msq = float((mag * mag).std() / np.sqrt(n_mc))

    # Quadrature should be much more accurate than MC standard error
    # at n_theta=16, n_dphi=32 — agreement to a few SE.
    assert abs(quad_mean - mc_mean) < 5.0 * mc_se_mean
    assert abs(quad_msq - mc_msq) < 5.0 * mc_se_msq


def test_v_kick_distribution_isotropic_mean_matches_numpy_mc():
    """⟨v_kick⟩ under Isotropic() matches numpy MC."""
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1_mag, chi2_mag = jnp.asarray(0.8), jnp.asarray(0.6)
    model = Isotropic.default(n_theta=16, n_dphi=32)

    v_q, w = v_kick_distribution(m1, m2, chi1_mag, chi2_mag, model)
    quad_mean = float(jnp.sum(w * v_q))

    rng = np.random.default_rng(1)
    n_mc = 200_000
    e1, e2 = _mc_uniform_on_sphere_pair(rng, n_mc)
    chi1_v = float(chi1_mag) * e1
    chi2_v = float(chi2_mag) * e2
    v_mc = np.asarray(
        v_kick_vec(
            jnp.asarray(m1), jnp.asarray(m2),
            jnp.asarray(chi1_v), jnp.asarray(chi2_v),
        )
    )
    mc_mean = float(v_mc.mean())
    mc_se_mean = float(v_mc.std() / np.sqrt(n_mc))

    assert abs(quad_mean - mc_mean) < 5.0 * mc_se_mean


def test_P_retain_marginalised_isotropic_matches_numpy_mc():
    """``P_retain_marginalised`` under Isotropic() with Heaviside
    retention matches numpy MC of ``mean[1[v_kick<v_esc]]``."""
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1_mag, chi2_mag = jnp.asarray(0.8), jnp.asarray(0.6)
    v_esc = jnp.asarray(500.0)
    model = Isotropic.default(n_theta=16, n_dphi=32)

    p_quad = float(P_retain_marginalised(m1, m2, chi1_mag, chi2_mag, v_esc, model))

    rng = np.random.default_rng(2)
    n_mc = 200_000
    e1, e2 = _mc_uniform_on_sphere_pair(rng, n_mc)
    v_mc = np.asarray(
        v_kick_vec(
            jnp.asarray(m1), jnp.asarray(m2),
            jnp.asarray(float(chi1_mag) * e1), jnp.asarray(float(chi2_mag) * e2),
        )
    )
    retain_mc = (v_mc < float(v_esc)).astype(np.float64)
    p_mc = float(retain_mc.mean())
    p_se = float(retain_mc.std() / np.sqrt(n_mc))

    # ~5 standard errors — Heaviside indicator has discrete jumps over
    # quadrature points, so quadrature accuracy is reduced; tolerate.
    assert abs(p_quad - p_mc) < max(5.0 * p_se, 5e-3)


# ---------------------------------------------------------------------------
# Smoothness in v_esc — the AD-on-v_esc two-for-one win.
# ---------------------------------------------------------------------------
def test_P_retain_marginalised_smooth_in_v_esc_under_isotropic():
    """Under Isotropic() with Heaviside retention, ``P_retain`` is a
    Riemann sum over a Heaviside applied at each orientation
    quadrature point — discontinuous in ``v_esc`` at finitely many
    breakpoints, finite-difference-smooth between them.

    The sigmoid retention is everywhere-smooth and is the right
    AD-clean path; verify ∂/∂v_esc is finite *and* non-zero under it.
    """
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1_mag, chi2_mag = jnp.asarray(0.8), jnp.asarray(0.6)
    model = Isotropic.default(n_theta=8, n_dphi=16)

    def f(v_esc):
        return P_retain_marginalised(
            m1, m2, chi1_mag, chi2_mag, v_esc, model,
            retention_kind="sigmoid", w_over_v_esc=1.0,
        )

    g = jax.grad(f)(jnp.asarray(500.0))
    assert jnp.isfinite(g)
    # Non-zero: under Isotropic the kick distribution has support
    # comparable to v_esc so the smoothed CDF has finite slope.
    assert abs(float(g)) > 1e-5


def test_P_retain_marginalised_aligned_has_zero_v_esc_gradient():
    """Aligned shortcut routes to the Heaviside path → ``∂/∂v_esc = 0``
    almost everywhere.  This is the failure mode the marginalised
    path *fixes*."""
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1, chi2 = jnp.asarray(0.3), jnp.asarray(0.0)

    def f(v_esc):
        return P_retain_marginalised(
            m1, m2, chi1, chi2, v_esc, Aligned()
        )

    g = jax.grad(f)(jnp.asarray(500.0))
    assert float(g) == 0.0


# ---------------------------------------------------------------------------
# Aligned-superkick ordering — marginalised v_kick exceeds aligned point.
# ---------------------------------------------------------------------------
def test_isotropic_v_kick_dominates_aligned_at_high_spin():
    """Equal-mass, high-spin: aligned v_kick ≈ 0; isotropic v_kick mean
    is dominated by superkick configurations and far exceeds it.  The
    statement that motivates the whole misalignment work."""
    m1, m2 = jnp.asarray(30.0), jnp.asarray(30.0)
    chi1_mag, chi2_mag = jnp.asarray(0.9), jnp.asarray(0.9)

    # Aligned: signed +chi each → v_kick_aligned ≈ 0 by q=1 symmetry.
    _, _ = v_kick_distribution(m1, m2, chi1_mag, chi2_mag, Aligned())
    v_align_scalar = float(v_kick_aligned(m1, m2, chi1_mag, chi2_mag))

    # Isotropic: vector spins; superkick mode contributes O(10³) km/s.
    v_q, w = v_kick_distribution(
        m1, m2, chi1_mag, chi2_mag,
        Isotropic.default(n_theta=12, n_dphi=24),
    )
    v_mean_iso = float(jnp.sum(w * v_q))

    assert v_align_scalar < 1.0
    assert v_mean_iso > 500.0
    assert v_mean_iso > 100.0 * max(v_align_scalar, 1.0)


# ---------------------------------------------------------------------------
# Differentiability across all inputs.
# ---------------------------------------------------------------------------
def test_chi_rem_distribution_differentiable_in_mass_and_spin():
    model = Isotropic.default(n_theta=6, n_dphi=12)

    def f(m1, m2, chi1, chi2):
        x, w = chi_rem_distribution(m1, m2, chi1, chi2, model)
        return jnp.sum(w * x)

    grads = jax.grad(f, argnums=(0, 1, 2, 3))(
        jnp.asarray(40.0),
        jnp.asarray(20.0),
        jnp.asarray(0.5),
        jnp.asarray(0.3),
    )
    for g in grads:
        assert jnp.isfinite(g)


def test_P_retain_marginalised_differentiable_in_kappa():
    """Fisher.kappa is a JAX leaf; ∂P_retain/∂κ flows."""
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1_mag, chi2_mag = jnp.asarray(0.7), jnp.asarray(0.5)
    v_esc = jnp.asarray(400.0)

    def f(kappa):
        model = Fisher(
            kappa=kappa, n_theta=8, n_dphi=16,
        )
        return P_retain_marginalised(
            m1, m2, chi1_mag, chi2_mag, v_esc, model,
            retention_kind="sigmoid", w_over_v_esc=1.0,
        )

    g = jax.grad(f)(jnp.asarray(2.0))
    assert jnp.isfinite(g)


def test_distribution_jit_stable():
    """``chi_rem_distribution`` survives ``jax.jit`` under Isotropic."""
    model = Isotropic.default(n_theta=6, n_dphi=12)

    @jax.jit
    def f(m1, m2, c1, c2):
        x, w = chi_rem_distribution(m1, m2, c1, c2, model)
        return jnp.sum(w * x)

    out = f(jnp.asarray(40.0), jnp.asarray(20.0), jnp.asarray(0.5), jnp.asarray(0.3))
    assert jnp.isfinite(out)


# ---------------------------------------------------------------------------
# Per-orientation primitives — broadcasting + boundary cases.
# ---------------------------------------------------------------------------
def test_chi_rem_at_orientation_aligned_node_matches_chi_rem_vec_aligned():
    """At the (cos θ_1, cos θ_2, Δφ) = (1, 1, 0) Aligned quadrature point,
    ``chi_rem_at_orientation`` returns ``|chi_rem_vec(χ_mag ẑ, χ_mag ẑ)|``.

    Specifically, with positive ``chi*_mag`` this equals the scalar
    ``chi_rem(m1, m2, chi1_mag, chi2_mag)`` to machine precision (the
    Session A invariant).
    """
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1_mag, chi2_mag = jnp.asarray(0.7), jnp.asarray(0.3)
    x = chi_rem_at_orientation(
        m1, m2, chi1_mag, chi2_mag,
        jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.0),
    )
    expected = chi_rem(m1, m2, chi1_mag, chi2_mag)
    assert float(x) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


def test_v_kick_at_orientation_aligned_node_matches_v_kick_aligned():
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1_mag, chi2_mag = jnp.asarray(0.7), jnp.asarray(0.3)
    v = v_kick_at_orientation(
        m1, m2, chi1_mag, chi2_mag,
        jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(0.0),
    )
    expected = v_kick_aligned(m1, m2, chi1_mag, chi2_mag)
    assert float(v) == pytest.approx(float(expected), rel=1e-12, abs=1e-9)


def test_chi_rem_at_orientation_broadcasts_over_progenitor_mesh():
    """Scalar orientation triple broadcasts against (..., ) progenitor inputs."""
    m1 = jnp.linspace(20.0, 100.0, 5)[:, None]
    m2 = jnp.linspace(10.0, 80.0, 4)[None, :]
    chi1_mag = jnp.full(m1.shape, 0.5)
    chi2_mag = jnp.full(m2.shape, 0.3)
    out = chi_rem_at_orientation(
        m1, m2, jnp.broadcast_to(chi1_mag, (5, 4)),
        jnp.broadcast_to(chi2_mag, (5, 4)),
        jnp.asarray(0.7), jnp.asarray(0.4), jnp.asarray(1.0),
    )
    assert out.shape == (5, 4)
    assert jnp.all(jnp.isfinite(out))


def test_sample_aligned_no_key_dependence():
    """``Aligned.sample`` is deterministic; chi_rem_sample / v_kick_sample
    at Aligned ignore the key entirely."""
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    c1, c2 = jnp.asarray(0.3), jnp.asarray(-0.2)
    k1 = jax.random.PRNGKey(0)
    k2 = jax.random.PRNGKey(99)
    x1 = chi_rem_sample(k1, m1, m2, c1, c2, Aligned())
    x2 = chi_rem_sample(k2, m1, m2, c1, c2, Aligned())
    assert float(x1) == float(x2)


def test_fisher_kappa_infinite_asymptote_chi_rem_to_aligned():
    """At very large ``κ`` the Fisher quadrature mean of ``|χ_rem|``
    asymptotes to ``|chi_rem(m1, m2, chi1_mag, chi2_mag)|`` for
    positive ``chi*_mag`` (since Aligned with positive signed χ
    is exactly the κ→∞ limit pole).

    Note: at moderate κ ≲ n_θ² Fisher is accurate; beyond that the
    tail-resolution warning in ``spin_orientation.Fisher`` docstring
    applies — use Aligned() instead.  At n_θ=20 we test κ up to ~10.
    """
    m1, m2 = jnp.asarray(40.0), jnp.asarray(20.0)
    chi1_mag, chi2_mag = jnp.asarray(0.7), jnp.asarray(0.3)
    target = float(chi_rem(m1, m2, chi1_mag, chi2_mag))

    means = []
    for kappa in [0.5, 2.0, 10.0]:
        model = Fisher(kappa=jnp.asarray(kappa), n_theta=20, n_dphi=24)
        x, w = chi_rem_distribution(m1, m2, chi1_mag, chi2_mag, model)
        means.append(float(jnp.sum(w * x)))

    # Monotone approach to the aligned target (|mean - target| shrinks
    # as κ grows in this regime).
    err_05 = abs(means[0] - target)
    err_2 = abs(means[1] - target)
    err_10 = abs(means[2] - target)
    assert err_10 < err_2 < err_05
