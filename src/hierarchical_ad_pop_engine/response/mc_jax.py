r""":class:`MCResponseJax` — JAX-pure MC-backed response for ``solver="mc_jax"``.

Session **J4** of ``demo/PLAN_jax_mc.md`` — the JAX-traceable companion
to :class:`MCResponse` in ``mc.py``.  Unblocks M4 (SVI under the MC
path) because the entire chain

    θ_pop → sample_natals_jax → particle_mc_jax → MCEvaluatorJax →
    log_rate_density

is end-to-end AD-traceable, in contrast to the numpy ``solver="mc"``
path whose stash carries no gradient flow back to θ_pop.

Dispatch flow under ``solver="mc_jax"``::

    CoagulationPopulation.log_prob
        ↓
    universe.log_rate_density
        ↓
    response.log_rate_density  ← dispatches on `solver`
        ↓ solver == "mc_jax"
    self.mc_jax_response.log_rate_density
        ↓
    self.mc_jax_response.evaluator.log_rate_density
        ↓
    rate_density_4d → log

The mc_jax response owns the J2 cascade output + the J3 evaluator
both as **dynamic** eqx fields so AD flows uniformly.  Per-likelihood-
call ``particle_mc_jax`` runs **inside the traced computation** —
the key is fixed at factory-construction time (deterministic SVI
objective) and the same key is reused on every step.  Variability
across MC realisations is handled by the standard "average over
multiple keys" pattern at the outer test layer, not here.

Hard constraints honoured
-------------------------
* ``[[feedback-no-fake-physics-stand-ins]]`` — particle_mc_jax under
  Aligned mode runs the same scalar BR + LZ + Heaviside-retention
  triple as numpy ``particle_mc``.  Non-Aligned raises
  ``NotImplementedError`` (J2b follow-up).
* ``[[feedback-no-fudge-factors]]`` — bandwidth from
  :class:`MCEvaluatorJax` is Silverman-on-n_eff; nothing tuned.
* ``[[feedback-perturbative-unreliable-implicit-production]]`` — ships
  additively.  Default solver in ``build_gaussian_coagulation`` is
  unchanged.
* ``[[feedback-red-team-validation]]`` — numpy ``MCResponse`` /
  ``MCEvaluator`` are untouched; remain parallel validation backstops.

Related
-------
* ``[[demo/PLAN_jax_mc]]`` § J4 — the PLAN this implements.
* ``[[project-jax-mc-j2-done]]`` — particle_mc_jax (cascade).
* ``[[project-jax-mc-j3-done]]`` — MCEvaluatorJax (KDE).
* ``[[feedback-gaussian-kde-over-smooths-cascade-peaks]]`` — user
  objection logged at J3 ship; M4 SVI failure-mode predictor.
* ``[[project-integration-test-in-flight]]`` — implicit SVI test that
  the J4 M4 test mirrors.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike

from ..evaluation.mc_jax import MCEvaluatorJax
from ..kernels import log_pairing_kernel
from ..spin_orientation import Aligned

if TYPE_CHECKING:
    from .coagulation import CoagulationResponse


__all__ = ["MCResponseJax", "build_mc_jax_response"]


# ---------------------------------------------------------------------------
# Stash carrier — JAX-array dynamic fields.
# ---------------------------------------------------------------------------
class MCResponseJax(eqx.Module):
    r"""JAX-pure particle-MC stash + ``log_rate_density`` route.

    Built **inside the traced computation** by
    :func:`build_mc_jax_response` from a :class:`CoagulationResponse`'s
    natal / kernel / orientation parameters.  All carriers are JAX
    arrays so AD flows from ``log_rate_density(m1, m2, χ1, χ2)`` back
    to the natal / kernel / log_R_env parameters that built the
    cascade.

    Attributes
    ----------
    evaluator : MCEvaluatorJax
        The J3 evaluator; itself an eqx.Module holding all the
        cascade-derived state (samples, weights, bandwidths,
        log_R_env, kernel_a/b).
    n_init : int
        Initial particle count (= cfg.n_seed).  Static field — used
        for density-convention rescaling: MCEvaluatorJax.density returns
        a count density with ``∫N dm dχ = W_total``; the implicit /
        basis solvers' density integrates to ``(1 - f_dep)``.
        Dividing by n_init brings the MC path into the same
        probability-density convention as the grid solvers, so the
        rate-density formula ``½ R_env K · N · N`` is cross-method
        comparable (matches numpy MCResponse pattern).

    Methods
    -------
    log_rate_density(m1, m2, chi1, chi2, env) -> Array
        Mirror of :meth:`CoagulationResponse.log_rate_density` — returns
        ``log[½·R_env·K(m1,m2)·N(m1,χ1)·N(m2,χ2)]``.  AD-traceable
        end-to-end.  ``env`` is ignored (v_esc was baked into the
        cascade at factory time, matching numpy MCResponse).
    """

    evaluator: MCEvaluatorJax
    # n_init is static so the rescaling factor is a Python int, not a
    # dynamic JAX scalar (would force a recompile if it changed).  This
    # field doesn't need to be AD-traceable.
    n_init: int = eqx.field(static=True)

    def log_rate_density(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
        env: dict,
    ) -> Array:
        r"""``log[½ R_env K(m1,m2) N(m1,χ1) N(m2,χ2)]`` via the J3 KDE.

        AD-traceable end-to-end:

        * sample positions in MCEvaluatorJax carry tangents from
          natal hyperparameters (J1 reparametrisation).
        * sample weights in MCEvaluatorJax carry tangents from natal
          + kernel through the J2 cascade.
        * MCEvaluatorJax bandwidths depend on weights → also AD-traced.
        * log_R_env and kernel_a/b enter directly.

        Density-convention rescaling (matches numpy MCResponse): divide
        N by n_init so the MC pdf integrates to (1 - f_dep), apples-
        to-apples with the implicit / basis grid solvers.
        """
        del env  # MC driver baked v_esc at construction time

        m1_arr = jnp.asarray(m1)
        m2_arr = jnp.asarray(m2)
        chi1_arr = jnp.asarray(chi1)
        chi2_arr = jnp.asarray(chi2)

        N1 = self.evaluator.density(m1_arr, chi1_arr)
        N2 = self.evaluator.density(m2_arr, chi2_arr)
        # Probability-density convention: divide by n_init so
        # ∫N dm dχ ≈ (1 - f_dep) bit-stably (W_total / n_init).
        N1_pdf = N1 / float(self.n_init)
        N2_pdf = N2 / float(self.n_init)

        log_K = log_pairing_kernel(
            m1_arr, m2_arr,
            a=self.evaluator.kernel_a, b=self.evaluator.kernel_b,
        )
        # log[½ R_env K N1 N2] = log_R_env − log 2 + log K + log N1 + log N2.
        log_N1 = jnp.log(jnp.maximum(N1_pdf, 1e-300))
        log_N2 = jnp.log(jnp.maximum(N2_pdf, 1e-300))
        return (
            self.evaluator.log_R_env
            - jnp.log(2.0)
            + log_K
            + log_N1
            + log_N2
        )

    def total_rate(self, *, n_quad: int = 64) -> Array:
        r"""Scalar ``∫∫∫∫ ½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂) dω``.

        Uses the J3 evaluator's ``total_rate`` (grid quadrature on the
        anisotropic Gaussian KDE) divided by ``n_init²`` so the density
        convention matches :meth:`log_rate_density` bit-stably.  Without
        this rescaling :meth:`log_rate_density` and :meth:`total_rate`
        would be inconsistent by a factor of ``n_init²`` ≈ 10⁶ at
        production resolution — exactly the M4 SVI failure mode caught
        2026-05-28 workstation round 1.

        Mirrors the implicit / basis solvers' total_rate convention:
        ``E[N_obs] = total_rate`` at the calibrated ``log_R_env``.

        AD-traceable end-to-end (the evaluator's total_rate is pure
        jnp).
        """
        return self.evaluator.total_rate(n_quad=n_quad) / float(self.n_init) ** 2


# ---------------------------------------------------------------------------
# Factory — JAX-traceable MC driver + J3 evaluator build.
# ---------------------------------------------------------------------------
def build_mc_jax_response(
    response: "CoagulationResponse",
    v_esc: ArrayLike,
    *,
    key: jax.Array,
    n_seed: int = 5000,
    depletion_fraction: float = 0.5,
    n_steps: Optional[int] = None,
    k_ratio: float = 1.5,
) -> MCResponseJax:
    r"""Build an :class:`MCResponseJax` by running ``particle_mc_jax`` traced.

    This factory is **JAX-traceable** — unlike :func:`build_mc_response`,
    which calls the numpy ``particle_mc`` driver eagerly, this one
    routes through :func:`particle_mc_jax` and stays inside the
    surrounding ``jax.jit`` / ``jax.grad`` trace.  Consequence: it can
    be called inside numpyro ``model()`` and SVI gradients flow back
    to ``θ_pop``.

    Parameters
    ----------
    response : CoagulationResponse
        Source of natal / kernel / orientation parameters.  Currently
        ``natal_mass_kind="gaussian"`` only, matching numpy
        MCResponse's scope.  Non-Aligned orientation raises
        NotImplementedError (J2b follow-up).
    v_esc : ArrayLike
        Host escape velocity (km/s).  Baked into the cascade at
        construction time; ``log_rate_density`` ignores ``env["v_esc"]``
        downstream (same convention as numpy MCResponse).
    key : jax.Array
        PRNG key driving the cascade.  Fixed at factory call time so
        the SVI objective is deterministic in θ.  Each likelihood
        evaluation under the surrounding numpyro context retraces this
        factory with the SAME key (since it's an eqx-leaf-equal
        closure), yielding the same cascade dynamics modulo the
        evolving natal samples.
    n_seed : int, default 5000
        Initial particle count (= MCConfig.n_seed).
    depletion_fraction : float, default 0.5
        Cascade evolution knob.  Per
        ``[[feedback-mc-depletion-tracks-engine]]`` set this from the
        engine's f_natal_depleted in the M4 integration test.
    n_steps : int, optional
        Override for the J2 inner-loop step budget.  If ``None``,
        defaults to ``int(depletion_fraction * n_seed * k_ratio)``
        — see ``k_ratio`` below.
    k_ratio : float, default 1.5
        Estimate of ``K_max / ⟨K⟩`` over the natal pool.  Round-3
        Bernoulli K-gate (see
        ``demo/PLAN_jax_mc_time_decoupling_diagnosis.md``) sizes
        ``n_steps`` so the expected number of K-accepts matches
        numpy's ``n_max_mergers = depletion × n_init``.  The default
        ``1.5`` is a mild overshoot of typical Gaussian-natal ×
        power-law-kernel regimes (workstation-round-3 demo measured
        ``1.24`` at depletion=1.0); for production SVI estimate this
        at init time and pass explicitly to avoid retracing every
        gradient step (a JAX-traced ``mu_natal`` can't be cast to a
        Python int for ``jax.lax.scan``).

    Returns
    -------
    MCResponseJax
        Eqx-Module-wrapped J3 evaluator + n_init.  All dynamic
        fields carry AD tangents from the input ``response``.
    """
    from red_team.backup_particle_mc import MCConfig, particle_mc_jax

    if response.natal_mass_kind != "gaussian":
        raise NotImplementedError(
            "MCResponseJax currently supports natal_mass_kind='gaussian' "
            f"only; got {response.natal_mass_kind!r}.  Adding pl_peak / "
            "power-law natal sampling is a follow-up to Session J4 — "
            "see demo/PLAN_jax_mc.md."
        )

    # Reject non-Aligned at factory time — particle_mc_jax already does
    # this inside, but failing here gives a cleaner error stack.
    orientation = response.orientation
    if orientation is not None and not isinstance(orientation, Aligned):
        raise NotImplementedError(
            "MCResponseJax currently supports only Aligned orientation; "
            "non-Aligned per-event vector physics is the J2b follow-up "
            "(see demo/PLAN_jax_mc.md § J2)."
        )

    natal_mass = response.natal_mass_params
    natal_spin = response.natal_spin_params
    kernel = response.kernel_params

    # MCConfig fields are typed as float but Python's dataclass doesn't
    # validate at instantiation — passing JAX scalars works and is what
    # lets AD flow through cfg.mu_natal etc.  Note m_min / m_max enter
    # the truncated-normal bounds via standardisation; JAX handles their
    # derivative through truncnorm correctly.
    cfg = MCConfig(
        mu_natal=jnp.asarray(natal_mass["mu"]),
        sigma_natal=jnp.asarray(natal_mass["sigma"]),
        m_min=jnp.asarray(natal_mass["m_min"]),
        m_max=jnp.asarray(natal_mass["m_max"]),
        natal_spin_loc=jnp.asarray(natal_spin["loc"]),
        natal_spin_scale=jnp.asarray(natal_spin["scale"]),
        kernel_a=jnp.asarray(kernel["a"]),
        kernel_b=jnp.asarray(kernel["b"]),
        v_esc=jnp.asarray(v_esc),
        n_seed=int(n_seed),
        depletion_fraction=float(depletion_fraction),
        orientation=orientation,
    )

    out = particle_mc_jax(cfg, key, n_steps=n_steps, k_ratio=float(k_ratio))

    # Build the J3 evaluator over the surviving pool + cascade weights.
    # Surviving-pool selection here must stay JAX-traceable; under AD
    # we can't use boolean indexing on a tracer, so we fall back to
    # multiplying by the active_mask and keeping full-length arrays.
    # MCEvaluatorJax.from_run handles full-length arrays via the
    # weights infrastructure — zero-weighted (inactive) samples
    # contribute nothing to the KDE.
    #
    # Trick: instead of boolean-indexing on active_mask, we pass
    # weights = w_active * active_mask.  Inactive slots get weight 0,
    # which makes the Gaussian KDE ignore them (their kernel
    # contributions multiply by 0).  The bandwidth still includes them
    # in the σ̂_w variance computation but their zero weight nulls
    # their contribution to σ̂² and to the mean.  This is the
    # AD-traceable surviving-pool filter.
    m_active_full = out["m_active"]
    chi_active_full = out["chi_active"]
    w_active_full = out["w_active"]
    active_mask_f = out["active_mask"].astype(m_active_full.dtype)
    # Mask: zero-out inactive samples in weights.  Keeps full-length
    # arrays so jit shape is static.
    masked_weights = w_active_full * active_mask_f

    evaluator = MCEvaluatorJax.from_run(
        m_active_full,
        chi_active_full,
        out["mergers"],
        log_R_env=jnp.asarray(response.log_R_env),
        kernel_params={
            "a": jnp.asarray(kernel["a"]),
            "b": jnp.asarray(kernel["b"]),
        },
        weights=masked_weights,
    )
    return MCResponseJax(
        evaluator=evaluator,
        n_init=int(n_seed),
    )
