"""Tests for vector-spin remnant fits (Session A of misalignment work).

Invariant-based — no frozen numerical references (see
``feedback-no-frozen-test-references``).  Every assertion is one of:

* Scalar-reduction equality (vector formula on aligned input ==
  existing scalar formula, machine precision).
* Symmetry under ``(m1, χ_1) ↔ (m_2, χ_2)`` (canonicalisation).
* Magnitude bounds (``|χ_rem| ≤ 0.999``, ``v_kick ≥ 0``).
* Differentiability (``jax.grad`` finite).
* Structural ordering (misaligned superkick > aligned kick at the same
  spin magnitudes — the qualitative statement that motivates the
  whole misalignment-mode work).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# Match the rest of the test suite — matches production AD precision.
jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.remnants import (  # noqa: E402
    chi_rem,
    chi_rem_vec,
    v_kick_aligned,
    v_kick_vec,
)


# ---------------------------------------------------------------------------
# chi_rem_vec — scalar reduction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "m1, m2, chi1, chi2",
    [
        (30.0, 20.0, 0.0, 0.0),
        (30.0, 30.0, 0.0, 0.0),
        (50.0, 5.0, 0.7, 0.3),
        (30.0, 20.0, -0.9, -0.9),
        (40.0, 20.0, 0.9, -0.9),
    ],
)
def test_chi_rem_vec_reduces_to_scalar_chi_rem_at_aligned(m1, m2, chi1, chi2):
    """Vector BR evaluated at vectors along ẑ equals the scalar BR
    formula to machine precision in the z-component, and the x,y
    components are zero."""
    chi1_v = jnp.asarray([0.0, 0.0, chi1])
    chi2_v = jnp.asarray([0.0, 0.0, chi2])
    a_vec = chi_rem_vec(jnp.asarray(m1), jnp.asarray(m2), chi1_v, chi2_v)
    # Note: scalar chi_rem does NOT canonicalise; vector chi_rem_vec
    # DOES canonicalise.  Comparison is via the canonical input to
    # scalar chi_rem.
    if m1 >= m2:
        a_sc = chi_rem(
            jnp.asarray(m1), jnp.asarray(m2), jnp.asarray(chi1), jnp.asarray(chi2)
        )
    else:
        a_sc = chi_rem(
            jnp.asarray(m2), jnp.asarray(m1), jnp.asarray(chi2), jnp.asarray(chi1)
        )
    assert float(a_vec[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(a_vec[1]) == pytest.approx(0.0, abs=1e-12)
    assert float(a_vec[2]) == pytest.approx(float(a_sc), rel=1e-12, abs=1e-12)


def test_chi_rem_vec_canonicalises_internally():
    """``chi_rem_vec`` is symmetric under ``(m1, χ_1_vec) ↔ (m_2, χ_2_vec)``
    because it canonicalises m_1 ≥ m_2 internally (mirroring
    ``v_kick_aligned``'s pattern)."""
    m1, m2 = jnp.asarray(30.0), jnp.asarray(50.0)
    chi1 = jnp.asarray([0.3, 0.1, 0.5])
    chi2 = jnp.asarray([-0.2, 0.4, -0.1])
    a_fwd = chi_rem_vec(m1, m2, chi1, chi2)
    a_swap = chi_rem_vec(m2, m1, chi2, chi1)
    assert jnp.allclose(a_fwd, a_swap, rtol=1e-12, atol=1e-13)


def test_chi_rem_vec_magnitude_bounded_by_unity():
    """``|χ_rem_vec| ≤ 0.999`` for any physical inputs."""
    rng = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    chi1 = jax.random.normal(k1, (50, 3))
    chi2 = jax.random.normal(k2, (50, 3))
    # Renormalise to unit-magnitude spins (max physical).
    chi1 = chi1 / jnp.linalg.norm(chi1, axis=-1, keepdims=True) * 0.99
    chi2 = chi2 / jnp.linalg.norm(chi2, axis=-1, keepdims=True) * 0.99
    m1 = jax.random.uniform(k3, (50,), minval=5.0, maxval=100.0)
    m2 = jax.random.uniform(k4, (50,), minval=5.0, maxval=100.0)
    a_vec = chi_rem_vec(m1, m2, chi1, chi2)
    mag = jnp.linalg.norm(a_vec, axis=-1)
    assert jnp.all(mag <= 0.999 + 1e-9)


def test_chi_rem_vec_differentiable_in_spin_components():
    """``jax.grad`` w.r.t. each input component must be finite."""

    def f(chi1_x: jnp.ndarray) -> jnp.ndarray:
        chi1 = jnp.asarray([chi1_x, 0.1, 0.5])
        chi2 = jnp.asarray([-0.2, 0.0, 0.3])
        return jnp.linalg.norm(
            chi_rem_vec(jnp.asarray(40.0), jnp.asarray(20.0), chi1, chi2)
        )

    g = jax.grad(f)(jnp.asarray(0.2))
    assert jnp.isfinite(g)


def test_chi_rem_vec_zero_spin_q1_matches_kerr_isco():
    """Equal-mass non-spinning merger gives χ_rem in the Kerr/Schwarzschild
    ISCO band (~ 0.5 to 0.8)."""
    chi0 = jnp.zeros(3)
    a_vec = chi_rem_vec(jnp.asarray(30.0), jnp.asarray(30.0), chi0, chi0)
    mag = float(jnp.linalg.norm(a_vec))
    assert 0.5 < mag < 0.8


# ---------------------------------------------------------------------------
# v_kick_vec — scalar reduction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "m1, m2, chi1, chi2",
    [
        (40.0, 20.0, 0.0, 0.0),
        (30.0, 30.0, 0.5, 0.5),
        (40.0, 20.0, 0.3, 0.0),
        (50.0, 10.0, -0.6, 0.4),
    ],
)
def test_v_kick_vec_reduces_to_v_kick_aligned(m1, m2, chi1, chi2):
    """Vector LZ on aligned input (both spins along ẑ) matches the
    existing scalar v_kick_aligned to machine precision — by design,
    every coefficient and combinator was chosen to make this exact."""
    chi1_v = jnp.asarray([0.0, 0.0, chi1])
    chi2_v = jnp.asarray([0.0, 0.0, chi2])
    v_vec = v_kick_vec(jnp.asarray(m1), jnp.asarray(m2), chi1_v, chi2_v)
    v_sc = v_kick_aligned(
        jnp.asarray(m1), jnp.asarray(m2), jnp.asarray(chi1), jnp.asarray(chi2)
    )
    assert float(v_vec) == pytest.approx(float(v_sc), rel=1e-12, abs=1e-9)


def test_v_kick_vec_canonicalises_internally():
    """``v_kick_vec`` is symmetric under ``(m1, χ_1_vec) ↔ (m_2, χ_2_vec)``."""
    m1, m2 = jnp.asarray(30.0), jnp.asarray(50.0)
    chi1 = jnp.asarray([0.3, 0.1, 0.5])
    chi2 = jnp.asarray([-0.2, 0.4, -0.1])
    v_fwd = v_kick_vec(m1, m2, chi1, chi2)
    v_swap = v_kick_vec(m2, m1, chi2, chi1)
    assert float(v_fwd) == pytest.approx(float(v_swap), rel=1e-12, abs=1e-9)


def test_v_kick_vec_non_negative():
    rng = jax.random.PRNGKey(2)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    chi1 = jax.random.normal(k1, (50, 3))
    chi2 = jax.random.normal(k2, (50, 3))
    chi1 = chi1 / jnp.linalg.norm(chi1, axis=-1, keepdims=True) * 0.99
    chi2 = chi2 / jnp.linalg.norm(chi2, axis=-1, keepdims=True) * 0.99
    m1 = jax.random.uniform(k3, (50,), minval=5.0, maxval=100.0)
    m2 = jax.random.uniform(k4, (50,), minval=5.0, maxval=100.0)
    v = v_kick_vec(m1, m2, chi1, chi2)
    assert jnp.all(v >= 0.0)


def test_v_kick_vec_misalignment_activates_kick_mode():
    """Misalignment activates a kick mode invisible to aligned-same-sign
    spins.  This is the qualitative statement that motivates the whole
    misalignment-mode work.

    At ``q = 1`` with equal *aligned same-sign* spins every aligned
    kick contribution vanishes (no mass asymmetry, no aligned-spin
    difference, no perpendicular component) — ``v_kick ≈ 0``.  The
    same masses with *in-plane anti-aligned* spins activate the
    superkick at the few × ``10³ km/s`` cluster-ejection scale.
    """
    m1, m2 = jnp.asarray(30.0), jnp.asarray(30.0)
    chi_mag = 0.9
    v_aligned_same = v_kick_vec(
        m1,
        m2,
        jnp.asarray([0.0, 0.0, chi_mag]),
        jnp.asarray([0.0, 0.0, chi_mag]),
    )
    v_inplane_anti = v_kick_vec(
        m1,
        m2,
        jnp.asarray([chi_mag, 0.0, 0.0]),
        jnp.asarray([-chi_mag, 0.0, 0.0]),
    )
    # Aligned-same-sign at q=1 has zero of every contribution.
    assert float(v_aligned_same) < 1.0  # numerical zero in km/s
    # Anti-aligned in-plane activates the superkick comfortably above
    # cluster escape (retention-activates-correctly precondition).
    assert float(v_inplane_anti) > 1000.0
    # And the ratio is enormous — misalignment is the difference
    # between "no kick" and "definitely ejected".
    assert float(v_inplane_anti) / max(float(v_aligned_same), 1e-6) > 1e3


def test_v_kick_vec_differentiable_in_spin_components():
    def f(chi1_x: jnp.ndarray) -> jnp.ndarray:
        chi1 = jnp.asarray([chi1_x, 0.1, 0.5])
        chi2 = jnp.asarray([-0.2, 0.3, -0.1])
        return v_kick_vec(jnp.asarray(40.0), jnp.asarray(20.0), chi1, chi2)

    g = jax.grad(f)(jnp.asarray(0.2))
    assert jnp.isfinite(g)


def test_v_kick_vec_jit_stable():
    @jax.jit
    def vk(chi1: jnp.ndarray, chi2: jnp.ndarray) -> jnp.ndarray:
        return v_kick_vec(jnp.asarray(40.0), jnp.asarray(20.0), chi1, chi2)

    v1 = vk(jnp.asarray([0.0, 0.0, 0.3]), jnp.asarray([0.0, 0.0, 0.0]))
    v2 = vk(jnp.asarray([0.3, 0.0, 0.0]), jnp.asarray([0.0, 0.3, 0.0]))
    assert jnp.isfinite(v1) and jnp.isfinite(v2)
