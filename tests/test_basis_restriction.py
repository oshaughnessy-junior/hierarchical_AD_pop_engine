"""Acceptance tests for ``TensorProductBasis.restrict_to`` /
``prolong_to`` — the cross-basis modal operators that move
coefficients between the high-resolution density basis and the
lower-resolution solver basis (PLAN_engine_v3_basis.md §Step 3).

Scope
-----
These are pure-modal operators on Legendre coefficient arrays — no
quadrature, no interpolation, no physics.  Restrict drops the
high modes; prolong zero-pads.  The acceptance criteria are
correspondingly tight:

* ``restrict_to(prolong_to(c, fine), self) == c`` exactly (bit-level
  identity, because zero-pad then chop is a no-op).
* ``prolong_to(restrict_to(c, coarse), self)`` zeros out the
  truncated tail.
* Functions that fit inside the coarse modal subspace are preserved
  through restrict→prolong with no loss.
* Domain or shape violations raise ``ValueError`` (these are the
  guard rails that keep silent bugs out of the solver).

The two operators must also be JIT-friendly (they're called from the
inner Picard loop in Step 4).

Float-64 mode is enabled — matches the rest of the basis test suite
and is required for the bit-level identity checks below.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.basis import (  # noqa: E402
    LegendreLobatto1D,
    TensorProductBasis,
)

TOL_EXACT = 1e-14   # bit-level identity (slicing + zero-pad)
TOL_MODAL = 1e-12   # low-mode preservation through roundtrip


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _make_basis_pair(
    n_m_density: int = 24,
    n_chi_density: int = 16,
    n_m_solver: int = 12,
    n_chi_solver: int = 8,
) -> tuple[TensorProductBasis, TensorProductBasis]:
    """Two bases on the same (log m, χ) interval; density is finer than
    solver on every axis.  Sized small for fast tests."""
    density = TensorProductBasis.for_binary_grid(
        n_basis_m=n_m_density, n_basis_chi=n_chi_density
    )
    solver = TensorProductBasis.for_binary_grid(
        n_basis_m=n_m_solver, n_basis_chi=n_chi_solver
    )
    return density, solver


def _random_coeffs(shape: tuple[int, int], seed: int = 0) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape))


# ---------------------------------------------------------------------------
# Basic correctness.
# ---------------------------------------------------------------------------
def test_restrict_truncates_high_modes():
    """Restrict keeps the top-left ``(n_t_m, n_t_χ)`` block and drops
    everything else."""
    density, solver = _make_basis_pair()
    coeffs = _random_coeffs(density.shape, seed=1)
    out = density.restrict_to(coeffs, solver)
    assert out.shape == solver.shape
    # The output must equal the top-left block of the input.
    expected = coeffs[: solver.shape[0], : solver.shape[1]]
    assert jnp.allclose(out, expected, atol=TOL_EXACT)


def test_prolong_zero_pads_high_modes():
    """Prolong places the source block in the top-left and zeros the rest."""
    density, solver = _make_basis_pair()
    coeffs = _random_coeffs(solver.shape, seed=2)
    out = solver.prolong_to(coeffs, density)
    assert out.shape == density.shape
    # Top-left block: identical to input.
    assert jnp.allclose(
        out[: solver.shape[0], : solver.shape[1]], coeffs, atol=TOL_EXACT
    )
    # Right-of-block (high χ modes): zero.
    assert jnp.all(out[:, solver.shape[1]:] == 0.0)
    # Below-block (high m modes): zero.
    assert jnp.all(out[solver.shape[0]:, :] == 0.0)


# ---------------------------------------------------------------------------
# Inverse / roundtrip identities — the headline acceptance.
# ---------------------------------------------------------------------------
def test_restrict_of_prolong_is_identity():
    """``restrict(prolong(c)) == c`` at the bit level.

    This is the modal one-sided inverse: zero-pad then truncate
    recovers the original coarse coefficients exactly.  No tolerance
    needed — it's literally a slice of a padded array.
    """
    density, solver = _make_basis_pair()
    coeffs_solver = _random_coeffs(solver.shape, seed=3)
    prolonged = solver.prolong_to(coeffs_solver, density)
    roundtrip = density.restrict_to(prolonged, solver)
    # Bit-level equal: no floating-point arithmetic in either operator.
    assert jnp.array_equal(roundtrip, coeffs_solver)


def test_prolong_of_restrict_zeros_truncated_tail():
    """``prolong(restrict(c))`` keeps the low-mode block and zeros the
    high-mode tail of ``c``.  Not an identity on the original ``c``
    (information is lost), but a clean projector onto the coarse
    modal subspace embedded in the fine basis.
    """
    density, solver = _make_basis_pair()
    coeffs_density = _random_coeffs(density.shape, seed=4)
    restricted = density.restrict_to(coeffs_density, solver)
    roundtrip = solver.prolong_to(restricted, density)
    assert roundtrip.shape == density.shape
    # Low-mode block: preserved.
    low_block = coeffs_density[: solver.shape[0], : solver.shape[1]]
    assert jnp.allclose(
        roundtrip[: solver.shape[0], : solver.shape[1]],
        low_block,
        atol=TOL_EXACT,
    )
    # High-mode tail: zero.
    assert jnp.all(roundtrip[solver.shape[0]:, :] == 0.0)
    assert jnp.all(roundtrip[:, solver.shape[1]:] == 0.0)


def test_low_mode_function_survives_restrict_prolong_roundtrip():
    """A coefficient array supported entirely in the solver-basis
    subspace (i.e., zero above ``solver.shape`` modes) survives a
    restrict→prolong roundtrip with no loss.  This is the property
    the v3 plan leans on: the solver acts on this subspace, and
    prolonging back to the density basis is exact for it.
    """
    density, solver = _make_basis_pair()
    coeffs_density = jnp.zeros(density.shape, dtype=jnp.float64)
    # Drop a non-trivial pattern entirely within the solver block.
    low_block = _random_coeffs(solver.shape, seed=5)
    coeffs_density = coeffs_density.at[
        : solver.shape[0], : solver.shape[1]
    ].set(low_block)

    restricted = density.restrict_to(coeffs_density, solver)
    roundtrip = solver.prolong_to(restricted, density)
    assert jnp.allclose(roundtrip, coeffs_density, atol=TOL_MODAL)


# ---------------------------------------------------------------------------
# Same-shape edge case — degenerate identity.
# ---------------------------------------------------------------------------
def test_restrict_to_same_basis_is_identity():
    """Restricting to a basis of identical shape (and domain) returns
    the original coefficients verbatim.  Useful for code that
    dispatches generically over (density-equal, density-coarser)
    configurations without special-casing the equal case."""
    density, _ = _make_basis_pair()
    twin = TensorProductBasis.for_binary_grid(
        n_basis_m=density.shape[0], n_basis_chi=density.shape[1]
    )
    coeffs = _random_coeffs(density.shape, seed=6)
    out = density.restrict_to(coeffs, twin)
    assert out.shape == coeffs.shape
    assert jnp.array_equal(out, coeffs)


def test_prolong_to_same_basis_is_identity():
    """Mirror of the previous test for prolong."""
    _, solver = _make_basis_pair()
    twin = TensorProductBasis.for_binary_grid(
        n_basis_m=solver.shape[0], n_basis_chi=solver.shape[1]
    )
    coeffs = _random_coeffs(solver.shape, seed=7)
    out = solver.prolong_to(coeffs, twin)
    assert out.shape == coeffs.shape
    assert jnp.array_equal(out, coeffs)


# ---------------------------------------------------------------------------
# Guard rails — operator validity checks.
# ---------------------------------------------------------------------------
def test_restrict_rejects_larger_target():
    """``restrict_to`` requires target.shape ≤ source.shape on every
    axis.  Calling with a finer target should raise."""
    density, solver = _make_basis_pair()
    coeffs = _random_coeffs(solver.shape, seed=8)
    with pytest.raises(ValueError, match="restrict_to"):
        solver.restrict_to(coeffs, density)


def test_prolong_rejects_smaller_target():
    """``prolong_to`` requires target.shape ≥ source.shape on every
    axis.  Calling with a coarser target should raise."""
    density, solver = _make_basis_pair()
    coeffs = _random_coeffs(density.shape, seed=9)
    with pytest.raises(ValueError, match="prolong_to"):
        density.prolong_to(coeffs, solver)


def test_mixed_shape_violation_rejected():
    """A target that is coarser on one axis but finer on the other is
    neither a restriction nor a prolongation; both operators reject it.
    """
    base = TensorProductBasis.for_binary_grid(
        n_basis_m=16, n_basis_chi=12
    )
    # Mixed: m smaller, χ larger.
    mixed = TensorProductBasis.for_binary_grid(
        n_basis_m=8, n_basis_chi=20
    )
    coeffs = _random_coeffs(base.shape, seed=10)
    with pytest.raises(ValueError):
        # restrict: χ goes up (16→20 modes), violation.
        base.restrict_to(coeffs, mixed)
    coeffs_mixed = _random_coeffs(mixed.shape, seed=11)
    with pytest.raises(ValueError):
        # prolong from mixed → base: m goes up (8→16), χ goes down (20→12).
        # The χ-axis violates the ≥ requirement.
        mixed.prolong_to(coeffs_mixed, base)


def test_domain_mismatch_raises():
    """Different physical intervals on the m or χ axis are rejected by
    both operators.  This is the silent-bug guard: if a downstream
    refactor accidentally builds two bases on different log-m windows,
    modal restrict/prolong is no longer well-defined and the
    operator must refuse rather than silently produce garbage."""
    # Same nominal shape, different m-domain.
    basis_a = TensorProductBasis.from_1d(
        LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=12),
        LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=8),
    )
    basis_b = TensorProductBasis.from_1d(
        LegendreLobatto1D.on(domain=(0.0, 2.0), n_basis=12),  # diff!
        LegendreLobatto1D.on(domain=(0.0, 1.0), n_basis=8),
    )
    coeffs = _random_coeffs(basis_a.shape, seed=12)
    with pytest.raises(ValueError, match="domain mismatch"):
        basis_a.restrict_to(coeffs, basis_b)
    with pytest.raises(ValueError, match="domain mismatch"):
        basis_a.prolong_to(coeffs, basis_b)


# ---------------------------------------------------------------------------
# JIT compatibility — required for Step 4 Picard inner loop.
# ---------------------------------------------------------------------------
def test_restrict_to_is_jit_traceable():
    density, solver = _make_basis_pair()

    @jax.jit
    def restrict(coeffs):
        return density.restrict_to(coeffs, solver)

    coeffs = _random_coeffs(density.shape, seed=13)
    out = restrict(coeffs)
    assert out.shape == solver.shape
    expected = coeffs[: solver.shape[0], : solver.shape[1]]
    assert jnp.allclose(out, expected, atol=TOL_EXACT)


def test_prolong_to_is_jit_traceable():
    density, solver = _make_basis_pair()

    @jax.jit
    def prolong(coeffs):
        return solver.prolong_to(coeffs, density)

    coeffs = _random_coeffs(solver.shape, seed=14)
    out = prolong(coeffs)
    assert out.shape == density.shape
    assert jnp.allclose(
        out[: solver.shape[0], : solver.shape[1]],
        coeffs,
        atol=TOL_EXACT,
    )
    assert jnp.all(out[solver.shape[0]:, :] == 0.0)
    assert jnp.all(out[:, solver.shape[1]:] == 0.0)


# ---------------------------------------------------------------------------
# Function-space sanity: round-trip preserves a function that lives in
# the coarse subspace, evaluated on a dense grid.
# ---------------------------------------------------------------------------
def test_function_space_roundtrip_preserves_low_resolution_function():
    """Build a function that is exactly representable in the solver
    basis (a few low-order Legendre modes), evaluate at dense nodes
    in the density basis, project, restrict-then-prolong, evaluate
    again — the reconstruction must match the original function.

    This is the function-space restatement of
    ``test_low_mode_function_survives_restrict_prolong_roundtrip``
    and is the closest analog to the v2 solver's actual usage: the
    density-basis evaluation is what observables see.
    """
    density, solver = _make_basis_pair(
        n_m_density=32, n_chi_density=16, n_m_solver=12, n_chi_solver=6
    )

    # Build a function in the solver-basis subspace by writing down
    # coefficients in the solver basis and prolonging to density.
    rng = np.random.default_rng(20)
    solver_coeffs = jnp.asarray(
        rng.standard_normal(solver.shape) * np.exp(
            -0.5 * np.arange(solver.shape[0])[:, None]
        ) * np.exp(
            -0.5 * np.arange(solver.shape[1])[None, :]
        )
    )  # rapidly decaying so it really sits in the solver subspace
    density_coeffs_truth = solver.prolong_to(solver_coeffs, density)

    # The full roundtrip: density → restrict → solver → prolong → density.
    restricted = density.restrict_to(density_coeffs_truth, solver)
    density_coeffs_back = solver.prolong_to(restricted, density)

    # Coefficient-level equality.
    assert jnp.allclose(
        density_coeffs_back, density_coeffs_truth, atol=TOL_EXACT
    )

    # Evaluation on a dense (m, χ) grid agrees too.
    m_eval = jnp.linspace(
        density.basis_m.domain[0] + 0.05,
        density.basis_m.domain[1] - 0.05,
        20,
    )
    chi_eval = jnp.linspace(
        density.basis_chi.domain[0] + 0.05,
        density.basis_chi.domain[1] - 0.05,
        15,
    )
    M, C = jnp.meshgrid(m_eval, chi_eval, indexing="ij")
    val_truth = density.evaluate(density_coeffs_truth, M, C)
    val_back = density.evaluate(density_coeffs_back, M, C)
    assert jnp.allclose(val_truth, val_back, atol=TOL_MODAL)
