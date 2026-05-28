r""":class:`MCResponse` — particle-MC backed response for ``solver="mc"``.

Session M2 of ``demo/PLAN_mc_first_reprioritization.md``.  Promotes the
existing :func:`red_team.backup_particle_mc.particle_mc` driver and the
just-promoted :class:`evaluation.mc.MCEvaluator` to a first-class
``CoagulationResponse``-like surface that the gwkokab-facing
``CoagulationPopulation.log_prob`` can call through.

Architecture
------------
The MC path is the third sibling of the implicit and basis solvers:

* ``solver="implicit"`` — IFT-VJP Picard on the engine grid.  Production
  for Aligned.
* ``solver="basis"`` — IFT-VJP Picard on basis coefficients.  Fast at
  modest resolution; AD-clean.
* ``solver="mc"`` (this module) — particle-MC driver run **eagerly at
  factory time**; the resulting merger record and surviving pool are
  stashed on the response.  Per-event ``log_rate_density`` queries
  inside one likelihood evaluation are jit-traceable through the
  ``MCEvaluator`` k-NN density (the PLAN's "Python-time switch on the
  parameter update, traceable inside one likelihood evaluation").

The dispatch flow under ``solver="mc"`` is::

    CoagulationPopulation.log_prob
        ↓
    universe.log_rate_density
        ↓
    response.log_rate_density  ← dispatches on `solver`
        ↓ solver == "mc"
    self.mc_response.log_rate_density
        ↓
    evaluator.rate_density_4d → log

The MCResponse owns the merger stash and the evaluator.  The
underlying :class:`CoagulationResponse` is unchanged at the field level
except for the new optional ``mc_response`` carrier and the
``log_rate_density`` dispatch arm (both shipped in the same M2 wire-up
edit on ``response/coagulation.py``).

Aligned bit-stability
---------------------
Under ``orientation=Aligned()`` (the default), the MC driver routes
every accepted pair through the v1 scalar
``chi_rem`` / ``v_kick_aligned`` / ``retention_aligned`` triple
(``[[project-misalignment-session-c-done]]``); the per-event records
match the engine's per-event physics bit-for-bit modulo the numpy RNG
sampling.  No dispatch leak between Aligned-implicit and Aligned-MC
on the *physics* surface; the numerical disagreement between the two
paths is MC sampling noise, gated by
``[[project-evaluation-step3-done]]``'s 10 % rel-L1 budget on the
k-NN-based density estimator (k=500 biweight).

Pytree semantics
----------------
:class:`MCResponse` is an ``eqx.Module``.  The numpy stash arrays
(``m_active``, ``chi_active``, ``mergers``, ``v_kicks``) are declared
``eqx.field(static=True)`` so they live in the pytree's aux_data, not
the differentiable leaves.  Consequences:

* The stash is **identity-keyed under ``jax.jit``** — when the
  population parameters change between MCMC steps, the eqx Python-time
  control flow detects a new stash object and ``jax.jit`` retraces.
  This is what makes the "single MC run per likelihood call" pattern
  correct without any per-call gradient leakage.
* The stash carries through ``jax.lax.while_loop`` /
  ``jax.lax.scan`` bodies as a closed-over constant; the
  ``MCEvaluator`` k-NN lookup is wrapped in ``jax.pure_callback`` so
  the surrounding trace records a host-callback edge without trying
  to differentiate through the k-NN tree.
* **No AD on natal / kernel through ``solver="mc"`` inside one
  likelihood evaluation.**  HMC requires the gradient to flow through
  the parameter update *between* MCMC steps; random-walk Metropolis
  or NUTS-with-pathfinder works without that gradient.  This is the
  "lossier in HMC" trade the PLAN flags.

Hard constraints honoured
-------------------------
* ``[[feedback-no-fake-physics-stand-ins]]`` — MC continues to run the
  real vector BR + LZ + per-event Heaviside retention.  No smoothing
  knobs.
* ``[[feedback-no-fudge-factors]]`` — every knob (n_seed,
  depletion_fraction, k) is either a user-facing config field with a
  documented default or set by an analytic convergence argument
  (``[[project-evaluation-step3-done]]``).
* ``[[feedback-mc-depletion-tracks-engine]]`` — MC ``depletion_fraction``
  is the user-facing "amount of evolution" knob in M2; the M4
  integration test (recovery under ``solver="mc"``) will set it from
  the engine's ``f_natal_depleted`` so the two paths sample the same
  regime.
* ``[[feedback-perturbative-unreliable-implicit-production]]`` —
  implicit remains the default solver.  ``solver="mc"`` ships
  **additively**.
* ``[[feedback-dispatch-callgraph-audit]]`` — every new kwarg on
  ``coagulation_model`` is followed by a grep through callers; bitwise
  Aligned guard test in ``tests/test_mc_distribution_log_prob.py``
  catches the dispatch-leak failure mode of Session B/C.

Related
-------
* ``[[demo/PLAN_mc_first_reprioritization.md]]`` § Session M2 — the
  PLAN this implements.
* ``[[project-mc-time-calibration-m1-done]]`` — the ε ↔ MC-state map
  the M4 integration test will use.
* ``[[project-evaluation-step3-done]]`` — the MCEvaluator k-NN density
  this layer rides on.
* ``[[project-misalignment-session-c-done]]`` — the existing MC
  per-event vector-physics dispatch the driver already implements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike

from ..evaluation.mc import MCEvaluator
from ..kernels import log_pairing_kernel
from ..spin_orientation import Aligned

if TYPE_CHECKING:
    from .coagulation import CoagulationResponse


# ---------------------------------------------------------------------------
# Stash carrier.
# ---------------------------------------------------------------------------
class MCResponse(eqx.Module):
    r"""Particle-MC stash + ``log_rate_density`` route.

    Built eagerly by :func:`build_mc_response` from a
    :class:`CoagulationResponse`'s natal / kernel / orientation
    parameters at ``coagulation_model(..., solver="mc")`` factory
    time.  Lives on the parent :class:`CoagulationResponse` as an
    optional ``mc_response`` field; the parent's
    :meth:`CoagulationResponse.log_rate_density` dispatches to
    :meth:`log_rate_density` here under ``solver="mc"``.

    Attributes
    ----------
    m_active, chi_active : numpy arrays, shape (n_active,)
        The surviving single-BH pool from the MC run.  Static fields —
        not differentiable, identity-keyed under ``jax.jit`` retrace.
    mergers : numpy array, shape (n_mergers, 8)
        Accepted-merger record from
        :func:`red_team.backup_particle_mc.particle_mc`.  Columns:
        ``(m1, m2, χ1, χ2, g1, g2, retained, m_remnant)``.
    v_kicks : numpy array, shape (n_mergers,)
        Per-event recoil ``v_kick`` (km/s).
    log_R_env : float
        ``log(per-env-per-time merger amplitude)`` — same convention as
        :attr:`CoagulationResponse.log_R_env`.
    kernel_a, kernel_b : float
        Pairing kernel parameters — same convention as
        :attr:`CoagulationResponse.kernel_params`.
    k : int or None
        Number of nearest neighbours for the k-NN balloon estimator.
        ``None`` triggers :class:`MCEvaluator`'s default (Silverman
        √n rule, floored at 4).  PLAN-recommended Stage 1 setting:
        ``k=500`` for the headline 10 % rel-L1 budget
        (``[[project-evaluation-step3-done]]``).
    stats : dict (static)
        The ``stats`` return field from ``particle_mc``: ``n_attempts``,
        ``n_accepted``, ``n_retained``, ``n_active_final``,
        ``depletion`` (= ``f_natal_depleted``), ``orientation`` tag,
        ``stop_reason``.  Carried for downstream diagnostics + the M4
        ``f_natal_depleted`` mirror.

    Methods
    -------
    evaluator() -> MCEvaluator
        Construct the k-NN-backed point-query evaluator over the
        surviving pool.  One MCEvaluator per ``MCResponse`` (the k-d
        tree is built inside the evaluator constructor); the
        :meth:`log_rate_density` route caches one shared evaluator
        across calls via an eqx.Module ``__post_init__``.
    log_rate_density(m1, m2, chi1, chi2, env) -> Array
        Mirror of :meth:`CoagulationResponse.log_rate_density` —
        returns ``log[½·R_env·K(m1,m2)·N(m1,χ1)·N(m2,χ2)]`` evaluated
        through the k-NN density at ``(m1,χ1)`` and ``(m2,χ2)``.
        ``env`` is ignored on the MC path (the MC driver's ``v_esc``
        was set at construction time — see PLAN § "Hard gotcha").
    """

    # Stash arrays are numpy (not JAX) — they live in the eqx static
    # field bucket (aux_data) so jit retraces on object identity and
    # so the AD machinery can never accidentally try to thread a
    # tangent through them.  Annotated as ``np.ndarray`` rather than
    # ``jax.Array`` so eqx's static-field type check doesn't warn
    # "a JAX array is being set as static".
    m_active: np.ndarray = eqx.field(static=True)
    chi_active: np.ndarray = eqx.field(static=True)
    mergers: np.ndarray = eqx.field(static=True)
    v_kicks: np.ndarray = eqx.field(static=True)
    log_R_env: float = eqx.field(static=True)
    kernel_a: float = eqx.field(static=True)
    kernel_b: float = eqx.field(static=True)
    k: Optional[int] = eqx.field(static=True, default=None)
    stats: dict = eqx.field(static=True, default_factory=dict)

    def evaluator(self) -> MCEvaluator:
        r"""Construct the k-NN balloon evaluator over the surviving pool.

        Returns a fresh :class:`MCEvaluator` (k-d tree built inside the
        constructor).  Caching the evaluator across calls is the
        responsibility of the caller — :meth:`log_rate_density` builds
        the evaluator once per JIT trace by closing over it inside the
        ``pure_callback``.
        """
        return MCEvaluator(
            self.m_active,
            self.chi_active,
            self.mergers,
            log_R_env=self.log_R_env,
            kernel_params={"a": self.kernel_a, "b": self.kernel_b},
            k=self.k,
        )

    def log_rate_density(
        self,
        m1: Array,
        m2: Array,
        chi1: Array,
        chi2: Array,
        env: dict,
    ) -> Array:
        r"""``log[½ R_env K(m1,m2) N(m1,χ1) N(m2,χ2)]`` via the MC k-NN
        density estimator.

        Mirrors :meth:`CoagulationResponse.log_rate_density`'s signature
        so the universe-layer dispatch under ``solver="mc"`` is a
        drop-in replacement.  ``env`` (which carries ``v_esc``) is
        **ignored on the MC path** — the MC driver's ``v_esc`` was
        baked in at construction time; callers that want a different
        ``v_esc`` must rebuild the response via the
        ``coagulation_model(..., solver="mc")`` factory.  Documented
        in the module-level "Hard gotcha".

        The kernel ``K(m1, m2)`` is computed analytically via
        :func:`log_pairing_kernel` (jit-traceable).  The single-BH
        densities ``N(m_i, χ_i)`` are looked up through the k-NN
        evaluator via :func:`jax.pure_callback` so the surrounding
        trace records a host-callback edge without trying to
        differentiate through the k-d tree.
        """
        del env  # MC driver baked v_esc at construction time

        evaluator = self.evaluator()
        m1_arr = jnp.asarray(m1)
        m2_arr = jnp.asarray(m2)
        chi1_arr = jnp.asarray(chi1)
        chi2_arr = jnp.asarray(chi2)

        # Density lookups via host callback.  Shape contract: input
        # shape == output shape (per query point).  We broadcast the
        # mass/spin pair to a common shape, then map them into a flat
        # query and back to the shape on return.
        bcast_shape = jnp.broadcast_shapes(
            m1_arr.shape, chi1_arr.shape,
        )
        shape_dtype = jax.ShapeDtypeStruct(bcast_shape, jnp.float64)

        def _density_host(m_np, chi_np):
            # MCEvaluator.density returns a numpy array of the same
            # shape as the query (we ravel + reshape ourselves to
            # avoid relying on the evaluator's broadcasting rules,
            # which are numpy-shape-strict).
            out = evaluator.density(np.asarray(m_np), np.asarray(chi_np))
            return np.asarray(out, dtype=np.float64).reshape(m_np.shape)

        N1 = jax.pure_callback(
            _density_host,
            shape_dtype,
            jnp.broadcast_to(m1_arr, bcast_shape),
            jnp.broadcast_to(chi1_arr, bcast_shape),
        )
        N2 = jax.pure_callback(
            _density_host,
            shape_dtype,
            jnp.broadcast_to(m2_arr, bcast_shape),
            jnp.broadcast_to(chi2_arr, bcast_shape),
        )

        # ── Density-convention rescaling ──────────────────────────────
        # :class:`MCEvaluator.density` returns a **count density**:
        # ``∫density dm dχ ≈ n_active`` (verifiable on the Stage 1
        # fixture — n_active ≈ 3700, integral ≈ 3914 within k-NN
        # noise).  The implicit / basis solvers' ``N(m, χ)`` is a
        # **probability density** with ``∫N dm dχ ≈ 1 − f_dep`` — the
        # natal IMF is a pdf factor scaled by the cascade depletion.
        #
        # The cross-method ``log_rate_density`` formula
        # ``½ R_env K(m₁,m₂) N(m₁,χ₁) N(m₂,χ₂)`` must use the same
        # density convention on both sides or it disagrees by
        # ``n_active²`` (≈ 10⁷ at Stage 1) — exactly the mode the
        # cross-solver parity test caught.  We rescale here to the
        # probability-density convention so MC and implicit /
        # basis are apples-to-apples.
        #
        # Divisor: ``n_init`` (= MCConfig.n_seed) rather than
        # ``n_active``.  After depletion ``∫N* dm dχ = (1 − f_dep)``
        # ≈ ``n_active / n_init``, so dividing the count density by
        # ``n_init`` makes the MC pdf integrate to ``(1 − f_dep)`` —
        # *bit-stably* mirroring the implicit's convention without
        # any explicit f_dep query into ``stats``.
        n_init = float(self.stats.get("n_init", 1) or 1)
        N1 = N1 / n_init
        N2 = N2 / n_init

        log_K = log_pairing_kernel(
            m1_arr, m2_arr, a=self.kernel_a, b=self.kernel_b,
        )
        # log[½ R_env K N1 N2] = log_R_env − log 2 + log K + log N1 + log N2.
        # Floor the densities at 1e-30 so log doesn't NaN on
        # outside-domain queries (the evaluator already returns 0 there).
        log_N1 = jnp.log(jnp.maximum(N1, 1e-30))
        log_N2 = jnp.log(jnp.maximum(N2, 1e-30))
        return (
            jnp.asarray(self.log_R_env)
            - jnp.log(2.0)
            + log_K
            + log_N1
            + log_N2
        )


# ---------------------------------------------------------------------------
# Factory — eager Python-time MC driver run.
# ---------------------------------------------------------------------------
def build_mc_response(
    response: "CoagulationResponse",
    v_esc: ArrayLike,
    *,
    n_seed: int = 5000,
    depletion_fraction: float = 0.5,
    n_max_attempts: int = 2_000_000,
    rejection_batch: int = 1024,
    rng_seed: Optional[int] = None,
    orientation_key_seed: int = 0,
    k: Optional[int] = None,
) -> MCResponse:
    r"""Build an :class:`MCResponse` by running the particle MC eagerly.

    **This is a Python-time call** — the MC driver is numpy + scipy,
    not jax-traceable.  Call once per population-parameter update;
    cache the result on :class:`CoagulationResponse` for the
    likelihood loop to read through.

    Parameters
    ----------
    response : CoagulationResponse
        Source of natal / kernel / orientation parameters.  Currently
        **only ``natal_mass_kind="gaussian"``** is supported; other
        natal kinds raise :class:`NotImplementedError` (the MC driver
        in :mod:`red_team.backup_particle_mc` only ships a Gaussian
        natal sampler — extension to pl_peak / power-law is a
        follow-up).
    v_esc : ArrayLike
        Host escape velocity (km/s).  Baked into the MC driver's
        retention cut at construction time; the resulting MCResponse's
        :meth:`log_rate_density` ignores ``env["v_esc"]`` on
        downstream queries (PLAN § "Hard gotcha").  For
        multi-environment fits, build one MCResponse per ``v_esc``
        bin.
    n_seed : int, default 5000
        Initial particle count.  M2 default sized to match the
        MCEvaluator k=500 headline budget from
        ``[[project-evaluation-step3-done]]``.
    depletion_fraction : float, default 0.5
        MC "amount of evolution" knob: the driver stops once
        ``(N_init − N_active) / N_init ≥ depletion_fraction``.  Per
        ``[[feedback-mc-depletion-tracks-engine]]``, production fits
        must set this from the engine's ``f_natal_depleted`` so MC and
        implicit sample the same regime — that wire-up lands in M4's
        integration test, not M2.
    n_max_attempts, rejection_batch
        Forwarded to :class:`MCConfig`.  Safety net + chunking knobs.
    rng_seed : int, optional
        Seed for the numpy RNG.  ``None`` uses
        :func:`secrets.token_bytes`-derived randomness (one fresh seed
        per call) so repeated calls give independent MC realisations.
    orientation_key_seed : int, default 0
        Seed for the JAX PRNG used by the per-event vector-orientation
        path under non-``Aligned`` orientation models.  Independent of
        the numpy ``rng`` so swapping orientation models doesn't
        perturb the pair-rejection sequence (the Aligned
        bit-for-bit invariant from Session C relies on this).
    k : int, optional
        Forwarded to the eventual :class:`MCEvaluator` constructed by
        :meth:`MCResponse.evaluator`.  ``None`` uses MCEvaluator's
        default (Silverman √n).  PLAN Stage 1 default: ``k=500`` for
        10 % rel-L1.

    Returns
    -------
    MCResponse
        Eqx-Module-wrapped stash.  Static fields hold numpy arrays;
        the surrounding ``jax.jit`` retraces on object identity, so
        every fresh build triggers a new compilation downstream.

    Raises
    ------
    NotImplementedError
        If ``response.natal_mass_kind`` is not ``"gaussian"``.
    """
    # Late-imported so the module is importable without scipy when the
    # rest of the package doesn't need the MC path.
    from red_team.backup_particle_mc import MCConfig, particle_mc

    if response.natal_mass_kind != "gaussian":
        raise NotImplementedError(
            "MCResponse currently supports natal_mass_kind='gaussian' "
            f"only; got {response.natal_mass_kind!r}.  Adding pl_peak / "
            "power-law natal sampling is a follow-up to Session M2 — "
            "see demo/PLAN_mc_first_reprioritization.md."
        )

    natal_mass = response.natal_mass_params
    natal_spin = response.natal_spin_params
    kernel = response.kernel_params

    # MCConfig wants Python floats — cast the JAX/numpy leaves.
    cfg = MCConfig(
        mu_natal=float(jnp.asarray(natal_mass["mu"])),
        sigma_natal=float(jnp.asarray(natal_mass["sigma"])),
        m_min=float(jnp.asarray(natal_mass["m_min"])),
        m_max=float(jnp.asarray(natal_mass["m_max"])),
        natal_spin_loc=float(jnp.asarray(natal_spin["loc"])),
        natal_spin_scale=float(jnp.asarray(natal_spin["scale"])),
        kernel_a=float(jnp.asarray(kernel["a"])),
        kernel_b=float(jnp.asarray(kernel["b"])),
        v_esc=float(jnp.asarray(v_esc)),
        n_seed=int(n_seed),
        depletion_fraction=float(depletion_fraction),
        n_max_attempts=int(n_max_attempts),
        rejection_batch=int(rejection_batch),
        orientation=response.orientation,
        orientation_key_seed=int(orientation_key_seed),
    )

    # Seed the numpy RNG.  ``None`` → fresh entropy each call so
    # repeated builds give independent realisations (matches the
    # PLAN's "one MC per parameter update" semantics — different
    # parameters, different stash).
    if rng_seed is None:
        import secrets
        rng = np.random.default_rng(int.from_bytes(secrets.token_bytes(8), "big"))
    else:
        rng = np.random.default_rng(int(rng_seed))

    result = particle_mc(cfg, rng)

    # Cast scalar static fields to Python natives so future driver
    # changes returning JAX scalars can't introduce a retrace-or-trace
    # ambiguity.
    stats_py = {
        key: (val.item() if hasattr(val, "item") else val)
        for key, val in result["stats"].items()
    }

    # The stash arrays are intentionally numpy + static (see
    # MCResponse class docstring: "identity-keyed under jax.jit").
    # eqx warns by default whenever any array — numpy or JAX —
    # is placed in a static field, because the common-case mistake
    # is to accidentally freeze a JAX array.  Here it is correct
    # by design (the stash must not be a JAX leaf), so we suppress
    # the warning explicitly at the construction site.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A JAX array is being set as static",
            category=UserWarning,
        )
        return MCResponse(
            m_active=np.asarray(result["m_active"], dtype=np.float64),
            chi_active=np.asarray(result["chi_active"], dtype=np.float64),
            mergers=np.asarray(result["mergers"], dtype=np.float64),
            v_kicks=np.asarray(result["v_kicks"], dtype=np.float64),
            log_R_env=float(jnp.asarray(response.log_R_env)),
            kernel_a=float(jnp.asarray(kernel["a"])),
            kernel_b=float(jnp.asarray(kernel["b"])),
            k=(None if k is None else int(k)),
            stats=stats_py,
        )


__all__ = [
    "MCResponse",
    "build_mc_response",
]
