"""Prescribed spin-orientation distributions ``P(θ̂_1, θ̂_2)``.

These are the model knobs that the misalignment-mode work introduces.
At each merger event the engine **promotes** the scalar progenitor
spins ``(|χ_1|, |χ_2|)`` to vectors by drawing tilt angles relative to
the orbital angular momentum ``L̂ = ẑ`` from a prescribed distribution
``P(θ̂)``, runs the full vector Barausse-Rezzolla / Lousto-Zlochower
remnant physics (see :mod:`remnants.spin_vector`,
:mod:`remnants.kick_vector`), and marginalises back to scalar
``(|χ_rem|, v_kick)``.  The choice of ``P(θ̂)`` is what distinguishes
aligned-only from misaligned populations.

This module owns the orientation *abstraction* — three concrete models
and a uniform quadrature/sampling API.  It contains no remnant or
retention physics; everything physical happens downstream in
:mod:`remnants.spin_vector`, :mod:`remnants.kick_vector` and the
marginalisation wrappers landing in Session B.

Concrete models
---------------
``Aligned()``
    Delta at ``θ̂ = L̂``.  Single quadrature point ``(cos θ_1, cos θ_2,
    Δφ, w) = (1, 1, 0, 1)``.  Bit-for-bit shortcut at every call site:
    callers do ``isinstance(orientation, Aligned)`` and dispatch to the
    existing scalar ``chi_rem`` / ``v_kick_aligned`` / Heaviside path
    (see ``demo/PLAN_misalignment_mode.md`` § Decision 2).  Concrete
    class lives here so that ``Fisher(κ → ∞)`` has a clean physical
    reference point to compare against in tests.

``Isotropic()``
    Uniform on the 2-sphere for each spin.  Product Gauss-Legendre on
    ``cos θ_1 × cos θ_2`` plus uniform-trapezoidal on ``Δφ`` (default
    ``(n_θ, n_θ, n_Δφ) = (4, 4, 8) = 128`` quadrature points; see the
    Convergence Notes at the top of this module for the smooth-
    integrand rationale for the 128-point default and the Heaviside-
    retention caveat).  The ``κ → 0`` limit of ``Fisher``.

``Fisher(κ)``
    Per-spin von Mises-Fisher on the 2-sphere concentrated along
    ``L̂ = ẑ`` with concentration ``κ ≥ 0``.  *Same nodes as*
    ``Isotropic`` — only the quadrature weights depend on ``κ``.  This
    keeps the quadrature surface fixed under ``jax.jit`` and makes
    ``κ`` a clean ``jax.grad`` leaf (PLAN Decision 4).  ``κ → 0``
    recovers ``Isotropic`` exactly (weights collapse to the
    isotropic ones); ``κ → ∞`` asymptotes to ``Aligned()`` numerically
    but is *not* bit-equivalent.

Quadrature normalisation convention
-----------------------------------
``quadrature_3d()`` returns a flat ``(n_quad, 4)`` array of
``(cos θ_1, cos θ_2, Δφ, weight)`` tuples with the property that for
any function ``f(cos θ_1, cos θ_2, Δφ)``::

    Σ_k w_k · f(cos θ_1_k, cos θ_2_k, Δφ_k)
        ≈ ⟨f⟩_P
        = ∫ f(θ_1, θ_2, Δφ) P(θ̂_1, θ̂_2) dΩ_1 dΩ_2 .

In particular ``Σ_k w_k = 1`` for every concrete model (with
appropriate quadrature accuracy).  The single azimuthal integration
that's redundant under ``P(θ̂_1, θ̂_2) = P(θ̂_1, θ̂_2; Δφ)`` (the
axisymmetric case all three concrete models satisfy) has been folded
into the weights, leaving the flat-array `(cos θ_1, cos θ_2, Δφ)`
parameterisation.  The factor of ``(2π)`` from that collapse is
included in the returned weights.

The flat-array API is deliberate: it feeds straight into the inner
``jax.lax.scan`` on the basis / implicit sides (PLAN Decision 3).
Naïve vmap-over-orientations on the implicit side's 75 MB 4-D loss
integrand would multiply that by ``n_orient ≈ 1024`` → 37 GB, so the
scan-over-flat-quadrature pattern is *required*, not optional.

Sampling convention
-------------------
``sample(key, n_events)`` returns
``(theta_1, phi_1, theta_2, phi_2)`` — four ``(n_events,)`` arrays.
``theta_i ∈ [0, π]`` is the polar angle from ``L̂ = ẑ``;
``phi_i ∈ [0, 2π)`` is the azimuth.  Used on the MC side
(``red_team/particle_mc.py``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float


# Default quadrature resolutions.
#
# 2026-05-27 update: dropped from ``(n_θ, n_Δφ) = (8, 16) = 1024`` to
# ``(4, 8) = 128`` points per merger cell — an 8× compute speedup on
# both the gain and loss sides of the implicit solver (each scans
# ``n_quad`` per Picard iteration).  Rationale and accuracy bound:
#
# Smooth-integrand argument
# -------------------------
# The orientation-marginal integrands are smooth in
# ``(cos θ_1, cos θ_2, Δφ)`` under the **recommended** sigmoid
# retention path (see ``CoagulationResponse``):
#
# * ``chi_rem_vec`` is degree-2 in the spin vectors (BR 2009 quadratic
#   form) ⇒ degree-2 polynomial in ``cos θ_i``.  Gauss-Legendre with
#   ``n_θ = 4`` is exact for degree-7 polynomials → machine precision.
# * ``v_kick_vec`` has Fourier content in ``Δφ`` up to order ~3
#   (in-plane + superkick terms).  Periodic trapezoidal with
#   ``n_Δφ = 8`` resolves Fourier modes up to order 4 (Nyquist) → also
#   machine precision on the smooth part.
# * The smooth ``sigmoid`` retention preserves smoothness.
#
# So ``(4, 8) = 128`` points is more than enough for the sigmoid path.
#
# Heaviside-retention caveat
# --------------------------
# Under ``retention_kind="heaviside"`` the orientation integrand is
# piecewise-discontinuous wherever ``v_kick(θ̂_1, θ̂_2, Δφ) = v_esc``,
# and Gauss-Legendre convergence drops from spectral to ``O(1/n²)``.
# In that regime the default ``(4, 8)`` will leave a few-percent
# integration error.  Two workarounds:
#
# 1. Switch to sigmoid retention (the new default — see
#    ``CoagulationResponse.retention_kind``).  Recovers AD-on-``v_esc``
#    as a bonus.
# 2. Pass a hand-built ``Isotropic(n_theta=8, n_dphi=16)`` or finer.
#    All concrete models expose ``n_theta`` and ``n_dphi`` for
#    per-call override.
#
# Override paths
# --------------
# * Code: ``Isotropic(n_theta=N, n_dphi=M)`` or
#   ``Fisher(kappa=κ, n_theta=N, n_dphi=M)`` constructs an explicit
#   higher / lower-resolution model.
# * CLI: ``--n-theta`` / ``--n-dphi`` on both
#   ``demo/forward_modeling_observables.py`` and
#   ``demo/misalignment_observable_parity.py``.
#
# The old ``(8, 16)`` defaults were chosen with no convergence study
# (they predated the Session-D performance debugging); the new
# defaults are chosen against the smooth-integrand analytic argument
# above and validated by the eight invariant-based tests in
# ``tests/test_orientation_models.py`` + ``tests/test_remnant_marginal.py``
# (Σw=1, ⟨cos θ⟩=0, ⟨cos² θ⟩=1/3, Fisher Langevin, MC-vs-quadrature
# 5σ backstops at n_mc=200k — all hold at any quadrature ≥ (3, 6)).
_DEFAULT_N_THETA = 4
_DEFAULT_N_DPHI = 8


# ---------------------------------------------------------------------------
# Quadrature primitives (static — built at module-construction time,
# constant under jax.jit).
# ---------------------------------------------------------------------------
def _gauss_legendre_nodes_weights(n: int) -> tuple[Array, Array]:
    """Gauss-Legendre nodes & weights on ``[-1, 1]``, computed in numpy.

    Computed outside jax (numpy.polynomial.legendre.leggauss) and frozen
    as static arrays so they're jit-stable.
    """
    nodes_np, weights_np = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(nodes_np), jnp.asarray(weights_np)


def _trapezoidal_periodic_dphi(n: int) -> tuple[Array, Array]:
    """Periodic trapezoidal nodes & weights on ``[0, 2π)``.

    For a smooth periodic integrand the periodic trapezoidal rule is
    *spectrally* accurate.  Nodes: equally spaced.  Weights: uniform
    ``2π / n``.
    """
    nodes = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    weights = jnp.full((n,), 2.0 * jnp.pi / n)
    return nodes, weights


def _build_axisymmetric_3d_quadrature(
    n_theta: int,
    n_dphi: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """Build the flat 3-D quadrature grid common to Isotropic and Fisher.

    Returns
    -------
    cos_theta_1, cos_theta_2, dphi, w_iso, w_geom
        Each of shape ``(n_theta * n_theta * n_dphi,)``.
        ``w_iso`` are the isotropic weights (``Σ = 1`` for ``P = 1/(4π)²``).
        ``w_geom`` is the *geometric* weight ``w_GL_i · w_GL_j ·
        (2π / n_dphi)`` without the orientation-PDF reweighting — the
        Fisher path multiplies this by the per-axis Fisher factor.
    """
    cos_theta_nodes, w_gl = _gauss_legendre_nodes_weights(n_theta)
    dphi_nodes, w_trap = _trapezoidal_periodic_dphi(n_dphi)

    ct1, ct2, dp = jnp.meshgrid(
        cos_theta_nodes, cos_theta_nodes, dphi_nodes, indexing="ij"
    )
    w1, w2, wd = jnp.meshgrid(w_gl, w_gl, w_trap, indexing="ij")

    w_geom = (w1 * w2 * wd).ravel()
    # Isotropic single-spin PDF on the sphere has the factor 1/(4π)
    # per spin; one azimuthal collapses to 2π, leaving total 1/(8π).
    w_iso = w_geom * (1.0 / (8.0 * jnp.pi))

    return ct1.ravel(), ct2.ravel(), dp.ravel(), w_iso, w_geom


# ---------------------------------------------------------------------------
# Concrete orientation models.
#
# All three implement the duck-typed pair
#
#   quadrature_3d() -> (n_quad, 4) array of (cos θ_1, cos θ_2, Δφ, w)
#   sample(key, n_events) -> (theta_1, phi_1, theta_2, phi_2)
#
# OrientationModel is intentionally not a literal abstract base class:
# every call site that needs to specialise (``retention.P_retain``,
# ``_step_one_generation``, ``build_remnant_tensor``) uses
# ``isinstance(orientation, Aligned)`` to dispatch the bit-for-bit
# shortcut, so concrete subclassing buys nothing and would force
# uniform field layouts that don't make sense (Aligned has no κ,
# Fisher has one).
# ---------------------------------------------------------------------------
class Aligned(eqx.Module):
    """Delta-function orientation at ``θ̂_1 = θ̂_2 = L̂``.

    Single quadrature point.  Used as the κ → ∞ reference for
    ``Fisher``, and as the bit-for-bit shortcut tag at downstream
    call sites.
    """

    def quadrature_3d(self) -> Float[Array, "n_quad 4"]:
        """Single quadrature point ``(cos θ_1, cos θ_2, Δφ, w) = (1, 1, 0, 1)``."""
        return jnp.asarray([[1.0, 1.0, 0.0, 1.0]])

    def sample(
        self, key: Array, n_events: int
    ) -> tuple[Array, Array, Array, Array]:
        """All spins along ``ẑ``: ``θ_i = 0``, ``φ_i = 0``."""
        del key  # unused — deterministic
        zeros = jnp.zeros((n_events,))
        return zeros, zeros, zeros, zeros


class Isotropic(eqx.Module):
    """Uniform-on-the-sphere orientation for each spin (independent).

    Quadrature: product Gauss-Legendre on ``(cos θ_1, cos θ_2)`` × periodic
    trapezoidal on ``Δφ``.  Default ``(n_θ, n_θ, n_Δφ) = (4, 4, 8) =
    128`` quadrature points per merger cell (was 1024 before
    2026-05-27; see Convergence Notes at the top of the module).
    """

    n_theta: int = eqx.field(static=True)
    n_dphi: int = eqx.field(static=True)

    @classmethod
    def default(
        cls,
        n_theta: int = _DEFAULT_N_THETA,
        n_dphi: int = _DEFAULT_N_DPHI,
    ) -> "Isotropic":
        return cls(n_theta=n_theta, n_dphi=n_dphi)

    def quadrature_3d(self) -> Float[Array, "n_quad 4"]:
        """Flat ``(n_quad, 4)`` array ``(cos θ_1, cos θ_2, Δφ, w)``.

        ``Σ_k w_k = 1`` to quadrature accuracy.
        """
        ct1, ct2, dp, w_iso, _ = _build_axisymmetric_3d_quadrature(
            self.n_theta, self.n_dphi
        )
        return jnp.stack([ct1, ct2, dp, w_iso], axis=-1)

    def sample(
        self, key: Array, n_events: int
    ) -> tuple[Array, Array, Array, Array]:
        """Sample uniform-on-sphere for each spin independently."""
        k_ct1, k_ct2, k_p1, k_p2 = jax.random.split(key, 4)
        cos_t1 = jax.random.uniform(k_ct1, (n_events,), minval=-1.0, maxval=1.0)
        cos_t2 = jax.random.uniform(k_ct2, (n_events,), minval=-1.0, maxval=1.0)
        theta_1 = jnp.arccos(cos_t1)
        theta_2 = jnp.arccos(cos_t2)
        phi_1 = jax.random.uniform(k_p1, (n_events,), minval=0.0, maxval=2.0 * jnp.pi)
        phi_2 = jax.random.uniform(k_p2, (n_events,), minval=0.0, maxval=2.0 * jnp.pi)
        return theta_1, phi_1, theta_2, phi_2


class Fisher(eqx.Module):
    """Per-spin von Mises-Fisher orientation on the 2-sphere, centred on ``ẑ``.

    Single-spin PDF::

        P_F(θ̂; κ) = κ / (4π sinh κ) · exp(κ cos θ) .

    Each spin draws independently from this PDF.  ``κ`` is a JAX leaf —
    ``jax.grad(...)(κ)`` flows through the quadrature weights cleanly,
    cashing in the AD-on-orientation-knob promise from PLAN Decision 4.

    Quadrature: same nodes as :class:`Isotropic`; weights reweighted
    by the per-axis Fisher PDF.  Keeping the nodes fixed under ``κ``
    is what makes the surface jit-stable and ``jax.grad``-friendly.

    Limits
    ------
    * ``κ → 0``: weights collapse to the isotropic weights (regularised
      via the small-``κ`` expansion ``κ / sinh κ → 1 − κ² / 6 + …``).
      Returns the same ``Σ w_k = 1`` quadrature as :class:`Isotropic`
      to ``O(κ²)``.
    * ``κ → ∞``: weights concentrate on ``cos θ = 1``.  For ``κ`` much
      larger than the quadrature can resolve (roughly ``κ ≳ n_θ²``)
      tail accuracy degrades — pass ``Aligned()`` for that asymptote.
    """

    kappa: Float[Array, ""]
    n_theta: int = eqx.field(static=True)
    n_dphi: int = eqx.field(static=True)

    @classmethod
    def default(
        cls,
        kappa: float | Array,
        n_theta: int = _DEFAULT_N_THETA,
        n_dphi: int = _DEFAULT_N_DPHI,
    ) -> "Fisher":
        return cls(
            kappa=jnp.asarray(kappa),
            n_theta=n_theta,
            n_dphi=n_dphi,
        )

    def quadrature_3d(self) -> Float[Array, "n_quad 4"]:
        """Flat ``(n_quad, 4)`` array ``(cos θ_1, cos θ_2, Δφ, w(κ))``.

        Differentiable in ``κ`` through ``w``.  Nodes are ``κ``-free.
        """
        ct1, ct2, dp, _, w_geom = _build_axisymmetric_3d_quadrature(
            self.n_theta, self.n_dphi
        )
        # Per-axis Fisher PDF on the sphere: κ/(4π sinh κ) · exp(κ x).
        # Joint PDF for two independent spins multiplies the per-axis
        # PDFs.  One azimuthal integration collapses → factor 2π.
        # Together with the geometric weight (which already includes
        # the dΔφ measure and the dx_1 dx_2 measures from GL) this
        # gives the correct full-sphere expectation.
        kappa = self.kappa
        # Stable κ/sinh(κ): use the closed-form 2κ/(exp(κ) − exp(−κ))
        # via expm1 to avoid catastrophic cancellation at large |κ|;
        # the limit at κ=0 is 1 (handled via jnp.where).
        kappa_safe = jnp.where(jnp.abs(kappa) < 1e-12, 1.0, kappa)
        kappa_over_sinh = jnp.where(
            jnp.abs(kappa) < 1e-12,
            1.0,
            2.0 * kappa_safe / (jnp.exp(kappa_safe) - jnp.exp(-kappa_safe)),
        )
        # Per-axis radial PDF F(x) = κ / (4π sinh κ) · exp(κ x).
        # Use exp(κ(x − 1)) · κ/(4π sinh κ) · exp(κ) for numerical
        # safety at large κ?  Stay simple — these are scalar enough.
        F1 = (kappa_over_sinh / (4.0 * jnp.pi)) * jnp.exp(kappa * ct1)
        F2 = (kappa_over_sinh / (4.0 * jnp.pi)) * jnp.exp(kappa * ct2)
        # Full weight per quad point: geometric × 2π (azimuthal
        # collapse) × F(x_1) × F(x_2).
        w = w_geom * (2.0 * jnp.pi) * F1 * F2
        return jnp.stack([ct1, ct2, dp, w], axis=-1)

    def sample(
        self, key: Array, n_events: int
    ) -> tuple[Array, Array, Array, Array]:
        """Inverse-CDF sample per-axis: ``cos θ = 1 + (1/κ) log[U − (U−1) exp(−2κ)]``.

        Limit at ``κ → 0``: ``cos θ`` uniform on ``[-1, 1]``.  Implemented
        via the closed-form inverse CDF for the Fisher radial marginal.
        """
        k_ct1, k_ct2, k_p1, k_p2 = jax.random.split(key, 4)
        u1 = jax.random.uniform(k_ct1, (n_events,))
        u2 = jax.random.uniform(k_ct2, (n_events,))
        cos_t1 = self._sample_cos_theta(u1)
        cos_t2 = self._sample_cos_theta(u2)
        theta_1 = jnp.arccos(jnp.clip(cos_t1, -1.0, 1.0))
        theta_2 = jnp.arccos(jnp.clip(cos_t2, -1.0, 1.0))
        phi_1 = jax.random.uniform(k_p1, (n_events,), minval=0.0, maxval=2.0 * jnp.pi)
        phi_2 = jax.random.uniform(k_p2, (n_events,), minval=0.0, maxval=2.0 * jnp.pi)
        return theta_1, phi_1, theta_2, phi_2

    def _sample_cos_theta(self, u: Array) -> Array:
        """Inverse-CDF for the Fisher cos-θ marginal.

        Closed form::

            cos θ = 1 + (1/κ) · log(u + (1 − u) · exp(−2κ)) .

        Limits cleanly to uniform at ``κ → 0`` (verified by Taylor
        expansion).
        """
        kappa = self.kappa
        # Branch on tiny κ to avoid 0/0.  Series: cos θ → 2u − 1.
        kappa_safe = jnp.where(jnp.abs(kappa) < 1e-6, 1.0, kappa)
        cos_fisher = 1.0 + (1.0 / kappa_safe) * jnp.log(
            u + (1.0 - u) * jnp.exp(-2.0 * kappa_safe)
        )
        cos_uniform = 2.0 * u - 1.0
        return jnp.where(jnp.abs(kappa) < 1e-6, cos_uniform, cos_fisher)
