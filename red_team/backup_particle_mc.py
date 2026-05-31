"""Backup implementation 1 — pure particle Monte Carlo.

The simplest possible coagulation simulator. No grids, no per-generation
densities, no truncated Volterra series. Just a particle list whose
members merge with each other, depleting the pool as the cascade runs.

Why this implementation cannot be wrong by construction:

1. State is a list of continuous ``(m, χ)`` floats. No discretisation.
2. There is no "generation" abstraction in the dynamics — the cascade is
   a sequence of pair-mergers; generation labels are optional diagnostics.
3. The pool depletes when particles merge: two singles become one
   remnant (if retained) or zero (if not). No "non-depleting source"
   approximation.
4. Each accepted pair corresponds to *one physical merger event*. There
   is no abstraction between density and rate.

If this disagrees with the engine, the engine has a bug. If it agrees
with the Doctor-perturbative-with-loss backup (Backup 2), we have two
independent witnesses to the "correct" total ``N(m, χ)``.

Misalignment mode — Session C wiring
------------------------------------
``MCConfig.orientation`` accepts any concrete :mod:`spin_orientation`
model (``Aligned`` / ``Isotropic`` / ``Fisher(κ)``).  Defaults to
``None`` which is treated as ``Aligned()``: the per-merger code path is
the **bit-for-bit existing** scalar ``chi_rem`` / ``v_kick_aligned`` /
``retention_aligned`` triple.  Under any non-``Aligned`` model the MC
draws one orientation realisation per accepted pair from
``orientation.sample(key, 1)``, promotes the natal spin magnitudes to
vectors, evaluates the full vector Barausse-Rezzolla 2009 remnant spin
and Lousto-Zlochower 2008/2011 kick, and applies a per-event Heaviside
retention against the per-event ``v_kick`` (no MC-side marginalisation
needed; each accepted pair is a single physical event).  The
``v_kicks`` 1-D array on the return dict carries the per-event kick
magnitude alongside the existing ``mergers`` 8-column schema, which is
left untouched for downstream-reader compatibility.

The natal ``chi`` distribution is sampled in ``[0, 1]`` and interpreted
as **signed χ** under ``Aligned`` (v1 semantics) and as ``|χ|`` under
non-``Aligned`` (the orientation distribution carries the sign / tilt).
Same numerical values, different interpretation — no schema change.

Run::

    python -m red_team.backup_particle_mc       # smoke at Stage 1 θ*
"""

from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import truncnorm

# Allowed engine imports — physics primitives only. Red-team rule is the
# same as for sim_coagulation.py: may not import response.* (the thing
# under test) but may import the lowest-level physics fits.
from hierarchical_ad_pop_engine.remnants.mass import M_rem as _M_rem_jax
from hierarchical_ad_pop_engine.remnants.spin import chi_rem as _chi_rem_jax
from hierarchical_ad_pop_engine.remnants.kick import (
    retention_aligned as _retention_aligned_jax,
    v_kick_aligned as _v_kick_aligned_jax,
)
from hierarchical_ad_pop_engine.remnants.marginal import (
    chi_rem_at_orientation as _chi_rem_at_orientation_jax,
    v_kick_at_orientation as _v_kick_at_orientation_jax,
)
from hierarchical_ad_pop_engine.spin_orientation import Aligned as _Aligned

from red_team.sim_coagulation import make_marginal


# ---------------------------------------------------------------------------
# Numpy wrappers — same convention as sim_coagulation. Vectorised because
# we call them on small batches in the hot loop.
# ---------------------------------------------------------------------------
def M_rem(m1, m2, chi1, chi2):
    return np.asarray(_M_rem_jax(m1, m2, chi1, chi2))


def chi_rem_(m1, m2, chi1, chi2):
    return np.asarray(_chi_rem_jax(m1, m2, chi1, chi2))


def retention_aligned(m1, m2, chi1, chi2, v_esc):
    return np.asarray(_retention_aligned_jax(m1, m2, chi1, chi2, v_esc))


def kernel_K(m1, m2, a, b):
    """Same kernel as ``sim_coagulation.kernel_K`` — duplicated to keep this
    backup importable on its own."""
    m_big = np.maximum(m1, m2)
    m_sml = np.minimum(m1, m2)
    q = m_sml / m_big
    return (m1 + m2) ** a * q ** b


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
@dataclass
class MCConfig:
    mu_natal: float
    sigma_natal: float
    m_min: float
    m_max: float
    natal_spin_loc: float
    natal_spin_scale: float
    kernel_a: float
    kernel_b: float
    v_esc: float
    n_seed: int = 20_000
    # Maximum number of merger *attempts* (rejected + accepted). The cascade
    # naturally terminates when the active pool can't form more accepted
    # pairs, but this caps runtime in pathological cases.
    n_max_attempts: int = 2_000_000
    # Stopping criterion 1 — hard cap on accepted mergers. If ``None`` the
    # depletion-fraction criterion takes over.
    n_max_mergers: Optional[int] = None
    # Stopping criterion 2 — stop once the active pool has shrunk by this
    # fraction relative to ``n_seed``. Defaults to 0.05 (5 %) which places
    # the cascade in the engine's "small effective ε" regime — see
    # `compare_all` output, where engine ≈ Backup 2 implicit at ε ≈ 0.025.
    # Running to full depletion (the original default) leaves only a
    # handful of high-generation remnants and the KDE looks featureless.
    #
    # Per `feedback-mc-depletion-tracks-engine` this is **not** a free
    # knob in production fits: the demo / acceptance harnesses set it
    # from the engine's ``f_natal_depleted`` so the two paths sample the
    # same regime.  Left as a config field here because the standalone
    # MC smoke (this module's ``__main__``) doesn't have an engine in
    # the loop.
    depletion_fraction: float = 0.05
    # Sample budget on each call to the K-rejection batch — bigger means
    # less Python overhead but more wasted work if acceptance is high.
    rejection_batch: int = 1024
    # Misalignment mode — Session C.  ``None`` is treated as
    # :class:`~hierarchical_ad_pop_engine.spin_orientation.Aligned` and
    # routes every accepted pair through the existing scalar v1
    # ``chi_rem`` / ``v_kick_aligned`` / ``retention_aligned`` code path
    # bit-for-bit.  Any other concrete ``OrientationModel`` activates the
    # per-event vector-physics path (see
    # ``demo/PLAN_misalignment_mode.md`` § Session C, step 8).
    orientation: object = None
    # Master jax PRNG seed used to derive per-event sampling keys under
    # non-``Aligned`` orientations.  Independent of the numpy ``rng`` so
    # that switching orientation models does not perturb the pair-
    # rejection sequence (Aligned bit-for-bit invariant relies on the
    # numpy ``rng`` driving everything that touches the active pool).
    orientation_key_seed: int = 0

    # ------------------------------------------------------------------
    # Session M3b — importance-sampled / stratified natal draws.
    # All three fields default to ``None``; in that case
    # :func:`sample_natals` falls through to the v1 truncnorm path and
    # returns ``weights = np.ones(n_seed)``, so every existing caller
    # is bit-for-bit unaffected.
    # ------------------------------------------------------------------
    # ``proposal``: caller-supplied importance proposal.  Signature
    # ``q(rng) -> (m, chi, log_q_per_sample)``.  When set, samples are
    # drawn from this and per-particle weights become
    # ``w = exp(log p_natal(m, chi) - log q(m, chi))``.  Overrides
    # ``stratify`` if both are set.
    proposal: Optional[Callable[[np.random.Generator],
                                Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
    # ``stratify``: if set to ``(n_m_cells, n_chi_cells)``, divides the
    # ``(log m, χ)`` rectangle into that many cells and draws roughly
    # ``n_seed / (n_m_cells × n_chi_cells)`` particles uniformly per
    # cell; weights follow ``w = p_natal / q_stratified`` automatically.
    # Useful for resolving narrow natal features (the
    # :file:`demo/PLAN_m3_stratified_natal.md` § M3b headline use case).
    stratify: Optional[Tuple[int, int]] = None
    # ``weight_array``: explicit caller-supplied per-particle weights
    # at sample time.  When set, multiplies *into* whatever weights the
    # ``proposal`` / ``stratify`` paths produced (or replaces the
    # uniform-ones for the default v1 path).  Shape ``(n_seed,)``.  Use
    # case: caller has its own importance proposal handled externally
    # and just wants to inject the resulting weights.
    weight_array: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Two-pulse natal IMF (``natal_family="two_gaussian"``) — the
    # methods-paper two-component experiment (INSTRUCTIONS_gwkokab_stage1
    # §5.1).  All three default ``None`` ⇒ single-pulse, and every code
    # path is bit-for-bit unchanged (``_two_pulse_active`` is ``False``).
    #
    # When set, the natal mass marginal becomes the normalised mixture
    #   (1 − λ)·TruncN(μ₁, σ₁) + λ·TruncN(μ₂, σ₂)
    # with ``(μ₁, σ₁) = (mu_natal, sigma_natal)`` the FIRST pulse and
    # ``(μ₂, σ₂) = (mu_natal_2, sigma_natal_2)`` the second; ``λ =
    # lambda_natal`` the weight of pulse 2.  The natal SPIN marginal is
    # the single ``natal_spin_{loc,scale}`` truncnorm shared by both
    # pulses (the two-pulse experiments hold spins narrow + common).
    #
    # AD: the JAX sampler routes λ through SMOOTH importance weights
    # (fixed 50/50 mixture proposal; λ enters only the weight numerator)
    # so SVI/NUTS can recover λ, μ₁, μ₂, σ₁, σ₂ — see
    # :func:`sample_natals_jax`.  Caller owns the μ₁ < μ₂ label-switch
    # convention.
    mu_natal_2: Optional[float] = None
    sigma_natal_2: Optional[float] = None
    lambda_natal: Optional[float] = None


def _orientation_is_aligned(orientation: object) -> bool:
    """``True`` if ``orientation`` should route to the v1 scalar path.

    ``None`` is treated as ``Aligned()`` so callers that pre-date Session
    C continue to land on the bit-for-bit existing code path.
    """
    return orientation is None or isinstance(orientation, _Aligned)


def _per_event_remnant_misaligned(
    key: jax.Array,
    m1: float,
    m2: float,
    chi1_mag: float,
    chi2_mag: float,
    v_esc: float,
    orientation: object,
) -> tuple[float, float, float, bool]:
    """One MC-event remnant evaluation under a non-``Aligned`` orientation.

    Samples ``(θ_1, φ_1, θ_2, φ_2)`` from ``orientation.sample(key, 1)``,
    promotes the scalar progenitor magnitudes to vector spins via the
    convention ``φ_1 = 0``, ``φ_2 = Δφ`` (azimuthal-symmetry collapse:
    ``|χ_rem|`` and ``v_kick`` are invariant under rigid rotation about
    ``L̂``), and returns the scalar ``(m_rem, |χ_rem|, v_kick, retain)``
    for this event.

    ``m_rem`` uses the parallel projection ``χ_z = χ_mag · cos θ`` —
    matches the convention the engine's ``_step_one_generation``
    marginalised path uses for ``M_rem``'s ``chi_eff`` input (see
    ``project-misalignment-session-b-done``).

    Retention is **per-event Heaviside** ``v_kick < v_esc``.  No
    orientation-side marginalisation: each accepted pair is one physical
    event with a definite outcome.
    """
    theta_1, phi_1, theta_2, phi_2 = orientation.sample(key, n_events=1)
    cos_t1 = float(jnp.cos(theta_1[0]))
    cos_t2 = float(jnp.cos(theta_2[0]))
    dphi = float(phi_2[0] - phi_1[0])

    # Parallel projection feeds M_rem's chi_eff input.
    chi_z_1 = chi1_mag * cos_t1
    chi_z_2 = chi2_mag * cos_t2
    m_rem = float(_M_rem_jax(m1, m2, chi_z_1, chi_z_2))

    chi_rem_mag = float(
        _chi_rem_at_orientation_jax(
            m1, m2, chi1_mag, chi2_mag, cos_t1, cos_t2, dphi
        )
    )
    v_kick = float(
        _v_kick_at_orientation_jax(
            m1, m2, chi1_mag, chi2_mag, cos_t1, cos_t2, dphi
        )
    )
    retain = v_kick < float(v_esc)
    return m_rem, chi_rem_mag, v_kick, retain


# ---------------------------------------------------------------------------
# Particle MC.
# ---------------------------------------------------------------------------
def _natal_log_pdf(m: np.ndarray, chi: np.ndarray, cfg: MCConfig) -> np.ndarray:
    r"""Joint log p_natal(m, χ) for the IS-weight numerator.

    Product of independent truncnorm distributions in ``m`` (on
    ``[m_min, m_max]``) and ``χ`` (on ``[0, 1]``) — mirrors the v1
    sampling distribution used by :func:`sample_natals` when no
    proposal is supplied.  Returned in log space so the
    ``log p − log q`` subtraction in :func:`sample_natals` is
    numerically clean across ~30 orders of magnitude in the tails.
    """
    a_m = (cfg.m_min - cfg.mu_natal) / cfg.sigma_natal
    b_m = (cfg.m_max - cfg.mu_natal) / cfg.sigma_natal
    log_p_m = truncnorm.logpdf(
        m, a_m, b_m, loc=cfg.mu_natal, scale=cfg.sigma_natal,
    )
    a_c = (0.0 - cfg.natal_spin_loc) / cfg.natal_spin_scale
    b_c = (1.0 - cfg.natal_spin_loc) / cfg.natal_spin_scale
    log_p_chi = truncnorm.logpdf(
        chi, a_c, b_c, loc=cfg.natal_spin_loc, scale=cfg.natal_spin_scale,
    )
    return log_p_m + log_p_chi


def _stratified_natal(
    cfg: MCConfig, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Stratified-uniform sampler on ``(log m, χ)`` cells, with IS weights.

    Per ``demo/PLAN_m3_stratified_natal.md`` § "M3b — driver-side
    weighting + workstation gate" / Stratified proposal at the
    engine grid edges:

    * Divide ``[log m_min, log m_max] × [0, 1]`` into
      ``n_m_cells × n_chi_cells`` equal-area cells.
    * Draw ``round(n_seed / (n_m_cells · n_chi_cells))`` samples
      uniformly per cell in ``(log m, χ)``.
    * Per-particle weight ``w_i = p_natal(m_i, χ_i) / q_strat(m_i, χ_i)``
      with ``q_strat`` the joint density of the stratified-uniform
      proposal in the ``(m, χ)`` measure (Jacobian from
      ``d(log m) = dm / m`` already folded in).

    The proposal density in ``(m, χ)``-coordinates is
    ``q_strat(m, χ) = 1 / (m · Δlog m · Δχ)`` with ``Δlog m =
    log(m_max) - log(m_min)`` and ``Δχ = 1``; uniform across cells
    because we draw equally per cell.

    Returns ``(m, chi, weights)`` with ``weights > 0`` everywhere
    inside the support (truncnorm density is positive on
    ``[m_min, m_max] × [0, 1]``).
    """
    n_m_cells, n_chi_cells = cfg.stratify  # type: ignore[misc]
    n_total_cells = int(n_m_cells * n_chi_cells)
    if n_total_cells <= 0:
        raise ValueError(
            f"stratify={cfg.stratify!r} must have positive entries"
        )
    n_per_cell = max(1, cfg.n_seed // n_total_cells)
    log_m_edges = np.linspace(
        np.log(cfg.m_min), np.log(cfg.m_max), n_m_cells + 1,
    )
    chi_edges = np.linspace(0.0, 1.0, n_chi_cells + 1)
    m_list = []
    chi_list = []
    for i in range(n_m_cells):
        for j in range(n_chi_cells):
            log_m_cell = rng.uniform(
                log_m_edges[i], log_m_edges[i + 1], n_per_cell,
            )
            chi_cell = rng.uniform(
                chi_edges[j], chi_edges[j + 1], n_per_cell,
            )
            m_list.append(np.exp(log_m_cell))
            chi_list.append(chi_cell)
    m = np.concatenate(m_list)
    chi = np.concatenate(chi_list)
    log_m_range = float(np.log(cfg.m_max) - np.log(cfg.m_min))
    # log q_strat(m, χ) = -log m - log(Δlog m) - log(Δχ) with Δχ=1.
    log_q = -np.log(m) - np.log(log_m_range)
    log_w = _natal_log_pdf(m, chi, cfg) - log_q
    weights = np.exp(log_w)
    return m, chi, weights


def sample_natals(
    cfg: MCConfig, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Draw natal ``(m, χ)`` samples + per-particle importance weights.

    Returns ``(m, chi, weights)``.

    Three sampling modes, resolved in priority order:

    1. ``cfg.proposal`` (caller-supplied importance proposal):
       ``q(rng) -> (m, chi, log_q_per_sample)``.  Weights
       ``w = exp(log p_natal - log q)``.
    2. ``cfg.stratify = (n_m_cells, n_chi_cells)``: stratified-
       uniform draw on ``(log m, χ)`` cells via
       :func:`_stratified_natal`.
    3. Default: original v1 truncnorm draw in both axes; uniform
       weights ``np.ones(n_seed)``.  PRNG draw sequence is
       bit-for-bit identical to the pre-M3b implementation so the
       downstream K-rejection loop stays in lockstep with the
       Aligned-baseline cascade dynamics.

    ``cfg.weight_array``, if set, is multiplied *into* whichever
    base weights mode (1)/(2)/(3) produced — useful for an external
    caller who computed weights upstream.
    """
    if cfg.proposal is not None:
        m, chi, log_q = cfg.proposal(rng)
        m = np.asarray(m, dtype=np.float64).ravel()
        chi = np.asarray(chi, dtype=np.float64).ravel()
        log_q = np.asarray(log_q, dtype=np.float64).ravel()
        if not (m.shape == chi.shape == log_q.shape):
            raise ValueError(
                f"proposal returned mismatched shapes m={m.shape} "
                f"chi={chi.shape} log_q={log_q.shape}"
            )
        log_w = _natal_log_pdf(m, chi, cfg) - log_q
        weights = np.exp(log_w)
    elif cfg.stratify is not None:
        m, chi, weights = _stratified_natal(cfg, rng)
    else:
        # ---------- v1 bit-for-bit path. ----------
        a_m = (cfg.m_min - cfg.mu_natal) / cfg.sigma_natal
        b_m = (cfg.m_max - cfg.mu_natal) / cfg.sigma_natal
        m = truncnorm.rvs(
            a_m, b_m, loc=cfg.mu_natal, scale=cfg.sigma_natal,
            size=cfg.n_seed, random_state=rng,
        ).astype(np.float64)
        a_c = (0.0 - cfg.natal_spin_loc) / cfg.natal_spin_scale
        b_c = (1.0 - cfg.natal_spin_loc) / cfg.natal_spin_scale
        chi = truncnorm.rvs(
            a_c, b_c, loc=cfg.natal_spin_loc, scale=cfg.natal_spin_scale,
            size=cfg.n_seed, random_state=rng,
        ).astype(np.float64)
        weights = np.ones(cfg.n_seed, dtype=np.float64)

    if cfg.weight_array is not None:
        wa = np.asarray(cfg.weight_array, dtype=np.float64).ravel()
        if wa.shape != m.shape:
            raise ValueError(
                f"cfg.weight_array shape {wa.shape} != natal-sample "
                f"shape {m.shape}"
            )
        weights = weights * wa

    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "non-finite weight from sample_natals — check proposal "
            "support and natal-pdf domain"
        )
    if np.any(weights < 0.0):
        raise ValueError(
            "negative weight from sample_natals — proposal must yield "
            "log_q ≤ log_p_natal (importance weights are non-negative "
            "by construction)"
        )
    # Filter zero-weight samples.  These are particles drawn by the
    # stratified / proposal sampler in regions where ``p_natal``
    # underflows to zero (e.g. ``m`` very far from ``mu_natal`` —
    # truncnorm logpdf can hit −2500 at the support's tail and exp
    # rounds to 0.0 in float64).  Such samples contribute nothing to
    # any weighted estimator AND would inflate the M3a MCEvaluator's
    # k-NN bandwidth without contributing to its kernel sum, so we
    # drop them at the sampler boundary to keep the downstream API
    # contract (``weights > 0``) intact.  Under the v1 path
    # (uniform weights = 1) no particles are filtered, so bit-for-bit
    # behaviour is preserved.
    keep = weights > 0.0
    if not np.all(keep):
        m = m[keep]
        chi = chi[keep]
        weights = weights[keep]
    if m.size == 0:
        raise ValueError(
            "sample_natals filtered to zero particles — proposal "
            "has no overlap with the natal support"
        )
    return m, chi, weights


# ===========================================================================
# Session J1 — JAX-pure natal sampler.
#
# Lives alongside the numpy ``sample_natals`` / ``_natal_log_pdf`` /
# ``_stratified_natal`` (kept untouched as the validation backstop per
# ``demo/PLAN_jax_mc.md`` § "Hard rule").  All J1 surfaces have a ``_jax``
# suffix and accept a ``jax.Array`` key instead of a ``np.random.Generator``;
# returns are ``jax.Array`` triples ``(m, chi, weights)``.
#
# Reparametrised in ``mu_natal`` / ``sigma_natal`` / ``natal_spin_loc`` /
# ``natal_spin_scale`` so gradients flow through the natal hyperparameters
# (this is what unblocks the eventual M4 SVI integration test under
# ``solver="mc_jax"``).
# ===========================================================================
def _two_pulse_active(cfg: MCConfig) -> bool:
    r"""``True`` iff the two-pulse natal mass marginal is configured.

    Activated by ``cfg.lambda_natal is not None``.  In that case
    ``mu_natal_2`` and ``sigma_natal_2`` must also be set.  When
    ``False`` every natal code path is the bit-for-bit single-pulse v1.
    """
    if cfg.lambda_natal is None:
        return False
    if cfg.mu_natal_2 is None or cfg.sigma_natal_2 is None:
        raise ValueError(
            "two-pulse natal requires mu_natal_2 and sigma_natal_2 to be "
            "set alongside lambda_natal"
        )
    return True


def _two_pulse_mass_component_logpdfs_jax(m, cfg: MCConfig):
    r"""``(log c₁(m), log c₂(m))`` — the two truncnorm mass components.

    Each component is a truncated normal on ``[m_min, m_max]``; returned
    in log space.  AD-clean in ``mu_natal{,_2}`` / ``sigma_natal{,_2}``.
    """
    from jax.scipy.stats import truncnorm as _jt
    m_min = jnp.asarray(cfg.m_min)
    m_max = jnp.asarray(cfg.m_max)
    mu1 = jnp.asarray(cfg.mu_natal)
    sig1 = jnp.asarray(cfg.sigma_natal)
    mu2 = jnp.asarray(cfg.mu_natal_2)
    sig2 = jnp.asarray(cfg.sigma_natal_2)
    a1, b1 = (m_min - mu1) / sig1, (m_max - mu1) / sig1
    a2, b2 = (m_min - mu2) / sig2, (m_max - mu2) / sig2
    log_c1 = _jt.logpdf(m, a1, b1, loc=mu1, scale=sig1)
    log_c2 = _jt.logpdf(m, a2, b2, loc=mu2, scale=sig2)
    return log_c1, log_c2


def _two_pulse_mass_logpdf_jax(m, cfg: MCConfig):
    r"""log f_mass(m) for the two-pulse mixture (normalised over support).

    ``log[(1−λ)·c₁(m) + λ·c₂(m)]`` with the λ-gradient flowing smoothly
    via ``log1p(-λ)`` / ``log(λ)``.
    """
    lam = jnp.asarray(cfg.lambda_natal)
    log_c1, log_c2 = _two_pulse_mass_component_logpdfs_jax(m, cfg)
    return jnp.logaddexp(jnp.log1p(-lam) + log_c1, jnp.log(lam) + log_c2)


def _natal_log_pdf_jax(m, chi, cfg: MCConfig):
    r"""JAX mirror of :func:`_natal_log_pdf` — joint log p_natal(m, χ).

    Uses ``jax.scipy.stats.truncnorm.logpdf``; output dtype follows JAX
    default precision (float32 unless ``jax_enable_x64``).  Differentiable
    in ``cfg.mu_natal`` / ``sigma_natal`` / ``natal_spin_loc`` /
    ``natal_spin_scale`` via standard truncnorm autodiff.  When the
    two-pulse marginal is active (:func:`_two_pulse_active`) the mass
    factor is the two-Gaussian mixture (differentiable additionally in
    ``mu_natal_2`` / ``sigma_natal_2`` / ``lambda_natal``); the spin
    factor is unchanged.
    """
    from jax.scipy.stats import truncnorm as _jt
    loc_c = jnp.asarray(cfg.natal_spin_loc)
    sc_c = jnp.asarray(cfg.natal_spin_scale)
    if _two_pulse_active(cfg):
        log_p_m = _two_pulse_mass_logpdf_jax(m, cfg)
    else:
        mu = jnp.asarray(cfg.mu_natal)
        sig = jnp.asarray(cfg.sigma_natal)
        a_m = (jnp.asarray(cfg.m_min) - mu) / sig
        b_m = (jnp.asarray(cfg.m_max) - mu) / sig
        log_p_m = _jt.logpdf(m, a_m, b_m, loc=mu, scale=sig)
    a_c = (0.0 - loc_c) / sc_c
    b_c = (1.0 - loc_c) / sc_c
    log_p_chi = _jt.logpdf(chi, a_c, b_c, loc=loc_c, scale=sc_c)
    return log_p_m + log_p_chi


def _stratified_natal_jax(cfg: MCConfig, key: jax.Array):
    r"""JAX mirror of :func:`_stratified_natal`.

    Stratified-uniform sampler on the ``(log m, χ)`` rectangle with
    auto-computed Jacobian-correct weights ``w = p_natal / q_strat``.
    Returns ``(m, chi, weights)`` as JAX arrays of length
    ``n_m_cells × n_chi_cells × n_per_cell`` (≤ ``cfg.n_seed``).

    Vectorised by construction — no Python loop over cells, so this
    is jit-compatible (under static ``stratify`` shape).
    """
    n_m_cells, n_chi_cells = cfg.stratify  # type: ignore[misc]
    n_total_cells = int(n_m_cells * n_chi_cells)
    if n_total_cells <= 0:
        raise ValueError(
            f"stratify={cfg.stratify!r} must have positive entries"
        )
    n_per_cell = max(1, cfg.n_seed // n_total_cells)
    log_m_edges = jnp.linspace(
        jnp.log(jnp.asarray(cfg.m_min)),
        jnp.log(jnp.asarray(cfg.m_max)),
        n_m_cells + 1,
    )
    chi_edges = jnp.linspace(0.0, 1.0, n_chi_cells + 1)
    key_m, key_chi = jax.random.split(key)
    u_m = jax.random.uniform(
        key_m, shape=(n_m_cells, n_chi_cells, n_per_cell),
    )
    u_chi = jax.random.uniform(
        key_chi, shape=(n_m_cells, n_chi_cells, n_per_cell),
    )
    log_m_lo = log_m_edges[:n_m_cells, None, None]
    log_m_hi = log_m_edges[1:, None, None]
    chi_lo = chi_edges[None, :n_chi_cells, None]
    chi_hi = chi_edges[None, 1:, None]
    log_m = log_m_lo + (log_m_hi - log_m_lo) * u_m
    chi_cells = chi_lo + (chi_hi - chi_lo) * u_chi
    m = jnp.exp(log_m).ravel()
    chi = chi_cells.ravel()
    log_m_range = jnp.log(jnp.asarray(cfg.m_max)) - jnp.log(jnp.asarray(cfg.m_min))
    # log q_strat(m, χ) = -log m - log(Δlog m) - log(Δχ) with Δχ=1.
    log_q = -jnp.log(m) - jnp.log(log_m_range)
    log_w = _natal_log_pdf_jax(m, chi, cfg) - log_q
    weights = jnp.exp(log_w)
    return m, chi, weights


def sample_natals_jax(
    cfg: MCConfig, key: jax.Array, *, filter_zero_weights: bool = True,
):
    r"""JAX-pure version of :func:`sample_natals`.

    Returns ``(m, chi, weights)`` as JAX arrays.  Same three sampling
    modes as the numpy version:

    1. ``cfg.proposal`` if set: caller-supplied ``q(key)`` →
       ``(m, chi, log_q)``.  Returns JAX arrays of any length.
    2. ``cfg.stratify`` if set: stratified-uniform on ``(log m, χ)``
       cells via :func:`_stratified_natal_jax`.
    3. Default: ``jax.random.truncated_normal`` in both axes;
       reparametrised so gradients in ``mu_natal`` / ``sigma_natal``
       / ``natal_spin_loc`` / ``natal_spin_scale`` flow through the
       affine ``loc + scale·z`` transform.

    ``cfg.weight_array`` multiplies into whichever base-mode weights.

    Parameters
    ----------
    cfg : MCConfig
        Same dataclass as the numpy path.  Float fields are
        ``jnp.asarray``-converted inside so AD on natal hyperparameters
        works without the caller having to JAX-ify the dataclass.
    key : jax.Array
        PRNG key (replaces the numpy ``rng``).
    filter_zero_weights : bool, default True
        If True, samples with zero weight (proposal landing in regions
        where ``p_natal`` underflows to 0) are filtered out before
        return — same convention as the numpy ``sample_natals``.
        Note: this turns the output shape into a dynamic Python-time
        quantity, so under ``jax.jit`` the caller must set
        ``filter_zero_weights=False`` and handle the masking
        explicitly (J2's particle_mc loop will follow that pattern).

    Notes
    -----
    Reparametrisation: ``jax.random.truncated_normal(key, a, b,
    shape)`` returns standard-truncated samples; we shift+scale by
    ``mu`` / ``sigma`` (and ``loc_chi`` / ``scale_chi``).  The affine
    transform is the reparametrisation trick — gradients in mu/sigma
    flow through it cleanly.  Note that ``a`` and ``b`` themselves
    depend on ``mu`` / ``sigma`` via the truncation bounds; JAX
    handles that derivative through the underlying truncnorm sampler.
    """
    if cfg.proposal is not None:
        m, chi, log_q = cfg.proposal(key)
        m = jnp.asarray(m).ravel()
        chi = jnp.asarray(chi).ravel()
        log_q = jnp.asarray(log_q).ravel()
        if not (m.shape == chi.shape == log_q.shape):
            raise ValueError(
                f"proposal returned mismatched shapes m={m.shape} "
                f"chi={chi.shape} log_q={log_q.shape}"
            )
        log_w = _natal_log_pdf_jax(m, chi, cfg) - log_q
        weights = jnp.exp(log_w)
    elif cfg.stratify is not None:
        m, chi, weights = _stratified_natal_jax(cfg, key)
    elif _two_pulse_active(cfg):
        # ---------- Two-pulse IS-weight path. ----------
        # Sample masses from a FIXED 50/50 mixture proposal of the two
        # components (so the proposal is λ-independent); carry the true
        # λ-mixture through smooth per-particle importance weights.  λ
        # enters ONLY the weight numerator ⇒ AD in λ flows cleanly; μ₁,
        # σ₁, μ₂, σ₂ enter both the reparametrised sample positions and
        # the weights.  Spin is the single shared natal truncnorm,
        # sampled from its exact pdf ⇒ contributes weight 1.
        mu1 = jnp.asarray(cfg.mu_natal)
        sig1 = jnp.asarray(cfg.sigma_natal)
        mu2 = jnp.asarray(cfg.mu_natal_2)
        sig2 = jnp.asarray(cfg.sigma_natal_2)
        loc_c = jnp.asarray(cfg.natal_spin_loc)
        sc_c = jnp.asarray(cfg.natal_spin_scale)
        m_min = jnp.asarray(cfg.m_min)
        m_max = jnp.asarray(cfg.m_max)
        a1, b1 = (m_min - mu1) / sig1, (m_max - mu1) / sig1
        a2, b2 = (m_min - mu2) / sig2, (m_max - mu2) / sig2
        a_c, b_c = (0.0 - loc_c) / sc_c, (1.0 - loc_c) / sc_c
        key_m1, key_m2, key_lab, key_chi = jax.random.split(key, 4)
        # Reparametrised draw from each component; select per-particle by
        # a fixed 50/50 label (label is λ-independent, so the discrete
        # assignment never blocks λ's gradient; each chosen component is
        # reparametrised so ∂position/∂μ_i,σ_i flows).
        m1 = mu1 + sig1 * jax.random.truncated_normal(
            key_m1, a1, b1, shape=(cfg.n_seed,))
        m2 = mu2 + sig2 * jax.random.truncated_normal(
            key_m2, a2, b2, shape=(cfg.n_seed,))
        pick2 = jax.random.uniform(key_lab, shape=(cfg.n_seed,)) < 0.5
        m = jnp.where(pick2, m2, m1)
        z_chi = jax.random.truncated_normal(
            key_chi, a_c, b_c, shape=(cfg.n_seed,))
        chi = loc_c + sc_c * z_chi
        # IS weight = f_mix(m;λ) / q_prop(m),  q_prop = ½c₁ + ½c₂.
        log_f_mix = _two_pulse_mass_logpdf_jax(m, cfg)
        log_c1, log_c2 = _two_pulse_mass_component_logpdfs_jax(m, cfg)
        log_q = jnp.logaddexp(jnp.log(0.5) + log_c1, jnp.log(0.5) + log_c2)
        weights = jnp.exp(log_f_mix - log_q)
    else:
        # ---------- Reparametrised truncnorm path. ----------
        mu = jnp.asarray(cfg.mu_natal)
        sig = jnp.asarray(cfg.sigma_natal)
        loc_c = jnp.asarray(cfg.natal_spin_loc)
        sc_c = jnp.asarray(cfg.natal_spin_scale)
        a_m = (jnp.asarray(cfg.m_min) - mu) / sig
        b_m = (jnp.asarray(cfg.m_max) - mu) / sig
        a_c = (0.0 - loc_c) / sc_c
        b_c = (1.0 - loc_c) / sc_c
        key_m, key_chi = jax.random.split(key)
        # ``jax.random.truncated_normal`` takes (lower, upper) in
        # standard-normal coordinates; shift+scale gives the
        # reparametrisation.
        z_m = jax.random.truncated_normal(
            key_m, a_m, b_m, shape=(cfg.n_seed,),
        )
        m = mu + sig * z_m
        z_chi = jax.random.truncated_normal(
            key_chi, a_c, b_c, shape=(cfg.n_seed,),
        )
        chi = loc_c + sc_c * z_chi
        weights = jnp.ones(cfg.n_seed, dtype=m.dtype)

    if cfg.weight_array is not None:
        wa = jnp.asarray(cfg.weight_array).ravel()
        if wa.shape != m.shape:
            raise ValueError(
                f"cfg.weight_array shape {wa.shape} != natal-sample "
                f"shape {m.shape}"
            )
        weights = weights * wa

    if filter_zero_weights:
        # Eager-only — dynamic shape.  Validation + zero-weight filter
        # both live behind this flag because ``bool(jnp.all(...))`` is
        # a TracerBoolConversionError under jax.jit / jax.grad / SVI
        # tracing; callers that need jit-compatibility (J2's
        # ``particle_mc_jax``, J4's ``build_mc_jax_response``, M4 SVI)
        # set ``filter_zero_weights=False`` and the validation is
        # skipped along with the dynamic-shape filter.
        #
        # The default truncnorm + uniform-ones path produces trivially
        # valid weights (finite + positive by construction); the
        # proposal / stratify paths produce positive weights via
        # ``exp(log_w)`` which is always non-negative.  The only mode
        # where validation actually catches anything is a caller-
        # supplied buggy proposal — and that caller is eager by API
        # contract anyway (proposals return concrete arrays, not
        # tracers).
        if not bool(jnp.all(jnp.isfinite(weights))):
            raise ValueError(
                "non-finite weight from sample_natals_jax — check proposal "
                "support and natal-pdf domain"
            )
        if bool(jnp.any(weights < 0.0)):
            raise ValueError(
                "negative weight from sample_natals_jax — proposal must "
                "yield log_q ≤ log_p_natal (importance weights are "
                "non-negative by construction)"
            )
        keep = np.asarray(weights > 0.0)
        if not bool(keep.all()):
            keep_idx = jnp.asarray(np.flatnonzero(keep))
            m = m[keep_idx]
            chi = chi[keep_idx]
            weights = weights[keep_idx]
        if m.size == 0:
            raise ValueError(
                "sample_natals_jax filtered to zero particles — "
                "proposal has no overlap with the natal support"
            )

    return m, chi, weights


# ===========================================================================
# Session J2 — JAX-pure particle MC hot loop.
#
# K-rejection-as-importance-weighting design (PLAN_jax_mc.md § J2): every
# candidate pair becomes a merger event with weight increment
# ``w_event = K_ij / K_max`` instead of being accepted with that probability.
# This makes the inner loop deterministic-once-keys-are-drawn (every step
# always merges), which is what ``jax.lax.scan`` requires.
#
# Validation: sandbox-derived ([[demo/PLAN_jax_mc]] § J2 + this session's
# j2_sandbox_derivation.py) that:
#  - per-attempt observable shape matches discrete in expectation
#    (INV-A / INV-B exact in expectation, INV-C bias → 0 as 1/√n_init).
#  - At production n_init = 20000 the cascade-fidelity bias is < 0.2 %,
#    well below MC SE.
#
# Aligned-only first pass: non-Aligned vector physics adds per-event
# orientation sampling + ``jax.random.split`` inside the scan and is a
# follow-up (J2b).  Calls with ``cfg.orientation`` set to a non-``None`` /
# non-``Aligned`` model raise ``NotImplementedError``.
#
# Numpy ``particle_mc`` stays as the validation backstop per
# [[feedback-red-team-validation]] and [[demo/PLAN_jax_mc]] § "Hard rule".
# ===========================================================================
def _kernel_K_jax(m1, m2, a, b):
    """JAX mirror of :func:`kernel_K`.  Inputs are scalars or arrays."""
    m_big = jnp.maximum(m1, m2)
    m_sml = jnp.minimum(m1, m2)
    q = m_sml / m_big
    return (m1 + m2) ** a * q ** b


def _kernel_upper_bound_jax(m_min_active, m_max_active, a, b):
    """JAX mirror of :func:`_kernel_upper_bound`.

    Differs slightly from the numpy version: takes scalar
    ``m_min_active`` / ``m_max_active`` rather than the active array, so
    it can be invoked inside ``lax.scan`` without dynamic indexing.
    Same 1.05 safety margin.
    """
    cand_hi = _kernel_K_jax(2 * m_max_active, 2 * m_max_active, a, b)
    cand_lo = _kernel_K_jax(2 * m_min_active, 2 * m_min_active, a, b)
    cand_mi = _kernel_K_jax(
        m_max_active + m_min_active, m_max_active + m_min_active, a, b,
    )
    return jnp.maximum(jnp.maximum(cand_hi, cand_lo), cand_mi) * 1.05


def particle_mc_jax(
    cfg: "MCConfig",
    key: jax.Array,
    *,
    n_steps: Optional[int] = None,
    k_ratio: float = 1.5,
):
    """JAX-pure particle MC under Bernoulli K-rejection.

    **Round-3 update (2026-05-29):** switched from rejection-as-weighting
    (J2 original) to **Bernoulli K-rejection with a straight-through
    estimator** for AD.  Numpy and JAX now share the same cascade
    dynamics in expectation — pool depletion, retention, and remnant
    injection are all K-gated by a per-step Bernoulli draw, exactly
    mirroring numpy ``particle_mc``.  The previous design conflated
    pool depletion (which numpy guards by K-rejection) with rate-
    equivalent event recording (which numpy treats as a discrete
    accept/reject), producing a systematic deficit on both
    ``Σw_mergers`` and the K-weighted rate integral ``∫∫K·N·N``.  See
    ``demo/PLAN_jax_mc_time_decoupling_diagnosis.md`` for the root
    cause and ``[[project-jax-mc-workstation-round3]]`` for the
    headline numbers.

    Returns a dict mirroring the numpy :func:`particle_mc` schema fields
    that downstream consumers actually use:

    ``m_active`` / ``chi_active`` / ``w_active`` / ``gen_active``
        Per-surviving-particle state; weights propagate the M3a/M3b
        ``w_desc = 0.5(w_i + w_j)`` natal-density rule.
    ``mergers``
        Per-step record ``(m1, m2, chi1, chi2, g1, g2, retained, m_rem)``,
        heavier first.  Shape ``(n_steps, 8)``.  Inactive or
        K-rejected steps return NaN rows.
    ``v_kicks``
        Per-step aligned kick magnitude.  Shape ``(n_steps,)``.  NaN on
        inactive or K-rejected steps.
    ``w_mergers``
        Per-step descendant weight ``w_desc = 0.5(w_i + w_j)`` on
        K-accepted active steps; ``0`` on K-rejected or inactive steps.
        Under uniform initial weights every accepted step contributes
        unit weight (matches numpy ``n_accepted`` count exactly per
        run, not just in expectation).  Pass this directly to
        ``MCEvaluator(weights=...)`` for weighted density estimation.
        **AD note:** the K-gate uses a straight-through estimator so
        ``∂w_mergers / ∂kernel_{a,b}`` flows through ``w_event =
        K_ij / K_max`` even though the forward pass is binary.
    ``step_active``
        Bool array, shape ``(n_steps,)``.  ``True`` if the step was
        both pool-active (``n_active >= 2``) and K-accepted —
        i.e. it produced a real merger event.  ``False`` for rejected
        or post-exhaustion steps.
    ``stats``
        Dict with ``n_init``, ``n_steps``, ``n_active_final``,
        ``depletion``, ``n_eff_active``, ``n_eff_mergers``,
        ``sum_w_mergers``, ``n_accepted`` (Σ step_active).

    Parameters
    ----------
    cfg : MCConfig
        Same dataclass as the numpy path.  ``cfg.orientation`` must be
        ``None`` or :class:`~hierarchical_ad_pop_engine.spin_orientation.Aligned`
        for J2 (non-Aligned is a follow-up).
    key : jax.Array
        PRNG key; split internally for natal sampling + per-step pair
        selection + K-gate Bernoulli draw.
    n_steps : int, optional
        Fixed inner-loop step budget.  If ``None``, computed from
        ``cfg.depletion_fraction`` and ``k_ratio`` as
        ``int(depletion × n_init × k_ratio)``, targeting
        ``depletion × n_init`` accepted events in expectation (matches
        numpy's ``n_max_mergers`` stopping criterion).
    k_ratio : float, default 1.5
        Estimate of ``K_max / ⟨K⟩`` over the natal pool.  Used to size
        ``n_steps`` so the expected number of K-accepts matches
        numpy's ``n_max_mergers = depletion × n_init``.  The default
        1.5 is a mild overshoot of typical Gaussian-natal × power-law-
        kernel regimes (workstation-round-3 demo measured 1.24 at
        depletion=1.0; production fits should set this from a brief
        natal-time MC estimate, e.g. via
        :func:`estimate_k_ratio`).

        **Why static, not derived from natal moments inside the
        trace:** under SVI, the natal mu/sigma are JAX tracers, so we
        can't cast them to a Python int for ``jax.lax.scan``'s static
        ``length`` argument without forcing a retrace per gradient
        step.  Pass ``k_ratio`` explicitly (from a one-time estimate
        at SVI init) to avoid the retrace.

    Notes
    -----
    AD support:

    * ``∂Σ m_active / ∂cfg.mu_natal``: cleanly via J1's truncated_normal
      reparametrisation.  This is the load-bearing M4 gradient.
    * ``∂Σ w_mergers / ∂cfg.kernel_b``: finite via the **straight-
      through estimator** on the K-gate.  Forward uses the hard
      Bernoulli accept (0/1); backward propagates the gradient as
      ``∂(K_ij/K_max)/∂kernel_b``.  Has zero bias on the K-weighted
      shape and a small (controllable) bias at the
      ``K_ij ≈ K_max`` boundary.
    * ``∂retained / ∂cfg.v_esc``: zero (Heaviside).  Recovers under
      misalignment + soft-sigmoid retention (follow-up J2b).

    Examples
    --------
    >>> import jax
    >>> cfg = MCConfig(mu_natal=8.0, sigma_natal=1.0, m_min=4.0, m_max=80.0,
    ...                natal_spin_loc=0.0, natal_spin_scale=0.10,
    ...                kernel_a=0.0, kernel_b=1.0, v_esc=1000.0,
    ...                n_seed=200, depletion_fraction=0.10)
    >>> out = particle_mc_jax(cfg, jax.random.PRNGKey(2026))
    >>> out["m_active"].shape  # doctest: +SKIP
    (..., )
    """
    if not _orientation_is_aligned(cfg.orientation):
        raise NotImplementedError(
            "particle_mc_jax currently supports only Aligned orientation; "
            "non-Aligned per-event vector physics is the J2b follow-up "
            "(see demo/PLAN_jax_mc.md § J2)."
        )

    key_natal, key_loop = jax.random.split(key)
    # Disable filtering — keeps shape static for the scan carry.  Default
    # truncnorm path produces no zero-weight samples anyway.
    m_arr0, chi_arr0, w_arr0 = sample_natals_jax(
        cfg, key_natal, filter_zero_weights=False,
    )
    n_init = int(m_arr0.shape[0])

    if n_steps is None:
        # Round-3 default: under Bernoulli K-rejection, each scan step
        # accepts with probability ⟨K⟩/K_max ≈ 1/k_ratio.  To match
        # numpy's ``n_max_mergers = depletion × n_init`` accepted-event
        # cap in expectation, run k_ratio × that many steps.
        #
        # The 1.5 "drops-per-accept" factor in the pre-round-3 formula
        # was wrong twice over: (1) it assumed every step was an
        # accepted event (it isn't under Bernoulli gating), (2) it
        # baked numpy's depletion-driven stop into JAX's count-driven
        # stop.  See
        # ``demo/PLAN_jax_mc_time_decoupling_diagnosis.md`` § 5.
        n_steps = max(1, int(cfg.depletion_fraction * n_init * float(k_ratio)))

    active0 = jnp.ones(n_init, dtype=bool)
    gen0 = jnp.ones(n_init, dtype=jnp.int32)

    # Carry pytree for jax.lax.scan.
    carry0 = (m_arr0, chi_arr0, w_arr0, active0, gen0, key_loop)

    a_k = jnp.asarray(cfg.kernel_a)
    b_k = jnp.asarray(cfg.kernel_b)
    v_esc = jnp.asarray(cfg.v_esc)

    def step_fn(carry, _):
        m_arr, chi_arr, w_arr, active, gen, key = carry

        n_active = jnp.sum(active)
        step_active_now = n_active >= 2

        # K_max from current active pool (NaN-safe via inf padding).
        m_active_for_min = jnp.where(active, m_arr, jnp.inf)
        m_active_for_max = jnp.where(active, m_arr, -jnp.inf)
        m_min_active = jnp.min(m_active_for_min)
        m_max_active = jnp.max(m_active_for_max)
        K_max = _kernel_upper_bound_jax(m_min_active, m_max_active, a_k, b_k)
        # Sanity guard — at pool exhaustion m_min becomes inf and K_max
        # is undefined.  Mask via step_active_now.
        K_max = jnp.where(step_active_now, K_max, 1.0)

        # Sample i, j from active pool with i != j.
        # Round-3: also split a key for the Bernoulli K-gate.
        key, k_i, k_j, k_u = jax.random.split(key, 4)
        p_active = jnp.where(active, 1.0, 0.0)
        denom = jnp.maximum(p_active.sum(), 1.0)
        p_i = p_active / denom
        i = jax.random.choice(k_i, n_init, p=p_i)
        # Mask out i so j != i.
        p_j_unnorm = jnp.where(
            jnp.arange(n_init) == i, 0.0, p_active,
        )
        denom_j = jnp.maximum(p_j_unnorm.sum(), 1.0)
        p_j = p_j_unnorm / denom_j
        j = jax.random.choice(k_j, n_init, p=p_j)

        m_i = m_arr[i]
        m_j = m_arr[j]
        chi_i = chi_arr[i]
        chi_j = chi_arr[j]
        w_i = w_arr[i]
        w_j = w_arr[j]
        g_i = gen[i]
        g_j = gen[j]

        # Round-3: Bernoulli K-rejection (mirrors numpy ``particle_mc``).
        # w_event ∈ [0, 1] is the per-attempt acceptance probability.
        # Forward: ``hard_accept`` is a 0/1 Bernoulli draw, so the cascade
        # dynamics (pool depletion, retention, remnant injection) only
        # advance on K-accepted attempts — bit-identical semantics to
        # numpy's discrete-rejection loop.
        # Backward (AD): the straight-through estimator ``st_accept``
        # carries the gradient of ``w_event`` to whatever the discrete
        # accept gate multiplies in the forward pass.  This is the
        # zero-bias-on-shape, small-bias-at-boundary path documented in
        # the diagnosis memo § 5.B.1.
        K_ij = _kernel_K_jax(m_i, m_j, a_k, b_k)
        w_event = K_ij / K_max
        u = jax.random.uniform(k_u)
        hard_accept = u < w_event
        # Straight-through: forward = hard (0.0 or 1.0), backward = w_event.
        st_accept = w_event + jax.lax.stop_gradient(
            hard_accept.astype(w_event.dtype) - w_event
        )

        # Aligned remnant physics (no per-event orientation work — those
        # paths require extra keys inside the scan).
        m_new = _M_rem_jax(m_i, m_j, chi_i, chi_j)
        chi_new = _chi_rem_jax(m_i, m_j, chi_i, chi_j)
        v_kick_evt = _v_kick_aligned_jax(m_i, m_j, chi_i, chi_j)
        retain = _retention_aligned_jax(m_i, m_j, chi_i, chi_j, v_esc) > 0.5

        # Identify heavier / lighter parent (so we keep the heavier slot
        # and write the remnant there — matches numpy's cosmetic m1 ≥ m2
        # convention).
        heavier_is_i = m_i >= m_j
        idx_heavy = jnp.where(heavier_is_i, i, j)
        idx_light = jnp.where(heavier_is_i, j, i)

        # Descendant cumulative weight: M3a/M3b natal-density average.
        # **Round-3:** the ``w_event`` multiplicative factor is REMOVED
        # here — accepted events under Bernoulli K-rejection have unit
        # effective weight, matching numpy's v1 ``w_desc = 0.5(w_i + w_j)``
        # bit-for-bit.  The K-gate carries gradient flow via the
        # straight-through ``st_accept`` factor applied below to
        # ``w_out`` (not to ``w_desc`` itself, so the heavy-slot weight
        # evolves on the same trajectory as numpy).
        w_desc = 0.5 * (w_i + w_j)

        # Round-3: gate state mutations on Bernoulli K-accept AND
        # pool-active.  Both light-slot deactivation and heavy-slot
        # mass/spin updates only happen on accepted attempts (numpy
        # parity).
        accept_mask = step_active_now & hard_accept
        do_retain = accept_mask & retain
        do_drop_both = accept_mask & (~retain)

        new_active = active
        new_active = new_active.at[idx_light].set(
            jnp.where(accept_mask, False, new_active[idx_light]),
        )
        new_active = new_active.at[idx_heavy].set(
            jnp.where(do_drop_both, False, new_active[idx_heavy]),
        )
        # idx_heavy stays True on retention (no-op since it was True).

        new_m_arr = m_arr.at[idx_heavy].set(
            jnp.where(do_retain, m_new, m_arr[idx_heavy]),
        )
        new_chi_arr = chi_arr.at[idx_heavy].set(
            jnp.where(do_retain, chi_new, chi_arr[idx_heavy]),
        )
        new_w_arr = w_arr.at[idx_heavy].set(
            jnp.where(do_retain, w_desc, w_arr[idx_heavy]),
        )
        new_gen = gen.at[idx_heavy].set(
            jnp.where(do_retain, jnp.maximum(g_i, g_j) + 1, gen[idx_heavy]).astype(jnp.int32),
        )

        # Per-step records (heavier-first, masked NaN on rejected /
        # inactive steps — only K-accepted active steps produce real
        # merger events).
        m1_rec = jnp.where(heavier_is_i, m_i, m_j)
        m2_rec = jnp.where(heavier_is_i, m_j, m_i)
        chi1_rec = jnp.where(heavier_is_i, chi_i, chi_j)
        chi2_rec = jnp.where(heavier_is_i, chi_j, chi_i)
        g1_rec = jnp.where(heavier_is_i, g_i, g_j).astype(m_arr.dtype)
        g2_rec = jnp.where(heavier_is_i, g_j, g_i).astype(m_arr.dtype)
        retain_rec = retain.astype(m_arr.dtype)
        m_rem_rec = jnp.where(retain, m_new, jnp.nan)
        rec_active = jnp.stack(
            [m1_rec, m2_rec, chi1_rec, chi2_rec, g1_rec, g2_rec, retain_rec, m_rem_rec],
        )
        nan_row = jnp.full_like(rec_active, jnp.nan)
        rec = jnp.where(accept_mask, rec_active, nan_row)

        v_kick_out = jnp.where(accept_mask, v_kick_evt, jnp.nan)
        # Round-3: w_out carries the straight-through K-gate so that
        # ``∂Σw_mergers/∂kernel_b`` (the J2 AD invariant) keeps flowing
        # smoothly even though the cascade dynamics are now hard-gated.
        # Forward: w_desc on K-accepted active steps, 0 otherwise.
        # Backward: gradient w.r.t. K params propagates via st_accept.
        w_out = st_accept * jnp.where(step_active_now, w_desc, 0.0)
        step_flag = accept_mask.astype(jnp.int32)

        new_carry = (new_m_arr, new_chi_arr, new_w_arr, new_active, new_gen, key)
        out = (rec, v_kick_out, w_out, step_flag)
        return new_carry, out

    final_carry, outputs = jax.lax.scan(step_fn, carry0, jnp.arange(n_steps))
    (m_arr_f, chi_arr_f, w_arr_f, active_f, gen_f, _key_f) = final_carry
    (mergers_arr, v_kicks_arr, w_mergers_arr, step_flags) = outputs

    # ``active_f`` is dynamic — eager-mode boolean indexing produces
    # ragged outputs (used by KS-test / weight-histogram callers).
    # Under ``jax.grad`` / ``jax.jit`` boolean indexing on a traced mask
    # is not supported; in that case the caller should index with the
    # raw full-length arrays + multiply by the active mask.
    #
    # Convention: return both forms — ``m_active`` / ``chi_active`` /
    # ``w_active`` / ``gen_active`` are full-length n_init arrays
    # alongside the ``active_mask`` boolean.  The caller picks the
    # form they need (eager indexing via ``out['m_active'][out['active_mask']]``,
    # or weighted sums via ``jnp.sum(out['m_active'] * out['active_mask'])``).
    # Plain-eager users keep the original API by indexing in their own
    # code; the J1 sandbox-test pattern (KS-test driver) handles this
    # cleanly outside the AD scope.
    out_dict = dict(
        m_active=m_arr_f,
        chi_active=chi_arr_f,
        w_active=w_arr_f,
        gen_active=gen_f,
        active_mask=active_f,
        mergers=mergers_arr,
        v_kicks=v_kicks_arr,
        w_mergers=w_mergers_arr,
        step_active=step_flags.astype(bool),
        stats=dict(
            n_init=n_init,
            n_steps=int(n_steps),
            # Differentiable counters — stay as jnp arrays.  Caller
            # converts to Python scalars eagerly if desired.
            n_active_final=jnp.sum(active_f),
            depletion=1.0 - jnp.sum(active_f) / max(n_init, 1),
            sum_w_mergers=jnp.sum(w_mergers_arr),
            # Round-3: under Bernoulli K-gating, n_accepted is the
            # discrete count of K-accepted active steps — mirrors
            # numpy ``particle_mc``'s ``stats['n_accepted']`` exactly.
            n_accepted=jnp.sum(step_flags),
            orientation="Aligned",
        ),
    )
    return out_dict


# ===========================================================================
# Session B.2 — deterministic soft K-gate (slot pre-allocation).
#
# The principled successor to round-3's B.1 Bernoulli K-gate
# (``particle_mc_jax`` above).  Motivation + full derivation:
#   * demo/PLAN_jax_mc_time_decoupling_diagnosis.md § 5.B.2 + § 8
#   * project-jax-mc-b2-soft-gate-followup
#   * sandbox derivation outputs/b2_soft_gate_derivation.py (numpy,
#     verified Q1/Q2/Q3 below)
#
# B.1's two costs and how B.2 removes them
# ----------------------------------------
#   (1) ``k_ratio`` magic factor.  B.1 sizes n_steps = depletion·n_init·
#       k_ratio so the EXPECTED number of Bernoulli K-accepts hits
#       numpy's depletion·n_init cap; k_ratio ≈ K_max/⟨K⟩ has to be
#       estimated and drifts with the regime.  B.2 carries a cumulative
#       net-removal BUDGET and clips the last contributing step, pinning
#       depletion to EXACTLY depletion·n_init.  ``n_steps`` then only has
#       to be a generous UPPER BOUND — too-large just adds cheap no-op
#       steps, it can't bias depletion.  (Sandbox Q1: depletion = 0.1000
#       with std 0.0000 for every n_steps ≥ ~1.5·D·n_init.)
#   (2) Straight-through bias.  B.1's forward pass is a binary Bernoulli
#       accept, so AD needs a straight-through estimator (zero-bias on
#       the K-weighted shape, small bias at the K_ij≈K_max boundary).
#       B.2's forward pass is the smooth occupancy flow
#       δ = (K_ij/K_max)·min(o_i,o_j) — differentiable everywhere, no
#       estimator.  (Sandbox Q3: with the pair sequence held fixed — the
#       AD-faithful condition — the soft observable is C0-smooth in
#       kernel_b; the discrete mirror is piecewise-constant with zero
#       gradient between accept-flips.)
#
# Mechanism (occupancy flow + slot pre-allocation)
# ------------------------------------------------
# Pre-allocate S = n_init + n_steps slots: [0:n_init] natal parents
# (occupancy o=1), [n_init:S] remnant tail (o=0, "born" as the cascade
# runs).  Each scan step k:
#   * draw a pair (i, j) with probability ∝ occupancy o (the soft analog
#     of numpy's uniform-over-active draw),
#   * δ = (K_ij/K_max)·min(o_i, o_j)  (clipped by the depletion budget),
#   * o_i -= δ; o_j -= δ,
#   * remnant born into tail slot ``n_init + k`` with occupancy δ (if
#     retained) — the scan counter IS the write index, so no dynamic
#     write-pointer is needed.
# Net occupancy removed per step: δ if retained (2 consumed − 1 born),
# 2δ if not.  W_total = Σ o·w = (1 − depletion)·n_init bit-stably
# (sandbox Q2 confirmed exact; KS(m), KS(χ) vs numpy well under the
# 0.05 critical value).
#
# Aligned-only first pass, same as B.1: physics fits are called in
# pair-selection order (matching numpy ``particle_mc`` / B.1 bit-for-
# bit); non-Aligned raises.  Retention stays Heaviside under Aligned, so
# ∂/∂v_esc is still zero here (recovers under misalignment + soft-sigmoid
# retention, the J2b follow-up) — B.2 fixes the KERNEL-gradient quality
# and the depletion-control knob, not the v_esc gradient.
#
# Additive: numpy ``particle_mc`` and B.1 ``particle_mc_jax`` are both
# untouched.  ``build_mc_jax_response(gate="soft")`` opts in; the default
# stays B.1 until B.2 is workstation-confirmed.
# ===========================================================================
def particle_mc_jax_soft(
    cfg: "MCConfig",
    key: jax.Array,
    *,
    n_steps: Optional[int] = None,
    n_steps_mult: float = 3.0,
    occ_tol: float = 1e-9,
):
    """JAX-pure particle MC under the B.2 deterministic soft K-gate.

    Drop-in companion to :func:`particle_mc_jax` (B.1).  Returns a dict
    with the same load-bearing keys plus a continuous ``occupancy``
    field (B.1's ``active_mask`` is the boolean ``occupancy > occ_tol``).

    Returns
    -------
    dict with keys

    ``m_active`` / ``chi_active`` / ``w_active`` / ``gen_active``
        Full-length ``S = n_init + n_steps`` slot arrays.  Parent slots
        ``[0:n_init]`` plus remnant-tail slots ``[n_init:S]``.
    ``occupancy``
        Per-slot occupancy ``o ∈ [0, 1]`` — the *continuous* surviving
        fraction of each slot.  This is the B.2 analog of B.1's binary
        ``active_mask``; pass ``w_active · occupancy`` as the density
        estimator's weights (it carries kernel/natal AD tangents, unlike
        the hard mask).  ``Σ occupancy·w_active = (1−depletion)·n_init``.
    ``active_mask``
        ``occupancy > occ_tol`` — boolean, for eager callers / schema
        parity with B.1.
    ``mergers`` / ``v_kicks`` / ``w_mergers`` / ``step_active``
        Per-step records, shapes ``(n_steps, 8)`` / ``(n_steps,)``.
        ``w_mergers`` is the occupancy-scaled descendant weight
        ``0.5(w_i+w_j)·δ`` (so Σ over steps tracks the cascade's
        cumulative merged mass).  Non-event steps carry NaN / 0.
    ``stats``
        ``n_init``, ``n_steps``, ``n_slots`` (= S), ``W_total``,
        ``depletion`` (= net_removed/n_init, pinned to the target),
        ``net_removed``, ``orientation``.

    Parameters
    ----------
    n_steps : int, optional
        Inner-loop step budget.  If ``None``, ``int(depletion_fraction ·
        n_init · n_steps_mult)``.  Unlike B.1's k_ratio, this is an
        UPPER BOUND, not a precision knob — the depletion budget pins
        the achieved depletion regardless (provided n_steps is large
        enough to reach the budget; ``n_steps_mult=3`` is a safe margin
        for Gaussian-natal × power-law-kernel regimes where
        K_max/⟨K⟩ ≈ 1.2–1.5).
    n_steps_mult : float, default 3.0
        Multiplier for the auto ``n_steps``.  Generous by design.
    occ_tol : float, default 1e-9
        Occupancy below this counts as "merged away" for pair-eligibility
        and the ``active_mask`` boolean.
    """
    if not _orientation_is_aligned(cfg.orientation):
        raise NotImplementedError(
            "particle_mc_jax_soft currently supports only Aligned "
            "orientation; non-Aligned per-event vector physics is the "
            "J2b follow-up (see demo/PLAN_jax_mc.md § J2)."
        )

    key_natal, key_loop = jax.random.split(key)
    m0, chi0, w0 = sample_natals_jax(
        cfg, key_natal, filter_zero_weights=False,
    )
    n_init = int(m0.shape[0])

    if n_steps is None:
        n_steps = max(
            1, int(cfg.depletion_fraction * n_init * float(n_steps_mult))
        )
    S = n_init + n_steps

    # Pre-allocated slot arrays: parents [0:n_init] + remnant tail.
    m_arr0 = jnp.zeros(S, dtype=m0.dtype).at[:n_init].set(m0)
    chi_arr0 = jnp.zeros(S, dtype=chi0.dtype).at[:n_init].set(chi0)
    w_arr0 = jnp.zeros(S, dtype=w0.dtype).at[:n_init].set(w0)
    occ0 = jnp.zeros(S, dtype=m0.dtype).at[:n_init].set(1.0)
    gen0 = jnp.zeros(S, dtype=jnp.int32).at[:n_init].set(1)

    a_k = jnp.asarray(cfg.kernel_a)
    b_k = jnp.asarray(cfg.kernel_b)
    v_esc = jnp.asarray(cfg.v_esc)
    R_target = jnp.asarray(cfg.depletion_fraction) * n_init

    carry0 = (m_arr0, chi_arr0, w_arr0, occ0, gen0,
              jnp.asarray(0.0, dtype=m0.dtype), key_loop)

    def step_fn(carry, k):
        m_arr, chi_arr, w_arr, occ, gen, net_removed, key = carry

        occ_active = occ > occ_tol
        n_active = jnp.sum(occ_active)
        budget_left = R_target - net_removed
        step_active_now = (n_active >= 2) & (budget_left > occ_tol)

        # K_max from current occupied pool (NaN-safe via inf padding).
        m_for_min = jnp.where(occ_active, m_arr, jnp.inf)
        m_for_max = jnp.where(occ_active, m_arr, -jnp.inf)
        m_min_active = jnp.min(m_for_min)
        m_max_active = jnp.max(m_for_max)
        K_max = _kernel_upper_bound_jax(m_min_active, m_max_active, a_k, b_k)
        K_max = jnp.where(step_active_now, K_max, 1.0)

        # Occupancy-weighted pair draw (soft analog of numpy's uniform-
        # over-active draw): P(slot) ∝ o.  Two splits, j != i.
        key, k_i, k_j = jax.random.split(key, 3)
        p_occ = jnp.where(occ_active, occ, 0.0)
        denom_i = jnp.maximum(p_occ.sum(), occ_tol)
        p_i = p_occ / denom_i
        i = jax.random.choice(k_i, S, p=p_i)
        p_j_unnorm = jnp.where(jnp.arange(S) == i, 0.0, p_occ)
        denom_j = jnp.maximum(p_j_unnorm.sum(), occ_tol)
        p_j = p_j_unnorm / denom_j
        j = jax.random.choice(k_j, S, p=p_j)

        m_i = m_arr[i]; m_j = m_arr[j]
        chi_i = chi_arr[i]; chi_j = chi_arr[j]
        w_i = w_arr[i]; w_j = w_arr[j]
        g_i = gen[i]; g_j = gen[j]
        o_i = occ[i]; o_j = occ[j]

        # Soft K-gate: deterministic occupancy flow.  δ = (K_ij/K_max) ·
        # min(o_i, o_j) — smooth in the kernel params, bounded by the
        # smaller partner's occupancy so neither slot can go negative.
        K_ij = _kernel_K_jax(m_i, m_j, a_k, b_k)
        w_event = K_ij / K_max
        delta_raw = w_event * jnp.minimum(o_i, o_j)

        # Aligned remnant physics — pair-selection order (matches numpy /
        # B.1 bit-for-bit; heavier-first is applied only to the record).
        m_new = _M_rem_jax(m_i, m_j, chi_i, chi_j)
        chi_new = _chi_rem_jax(m_i, m_j, chi_i, chi_j)
        v_kick_evt = _v_kick_aligned_jax(m_i, m_j, chi_i, chi_j)
        retain = _retention_aligned_jax(m_i, m_j, chi_i, chi_j, v_esc) > 0.5

        # Net occupancy removed by a FULL-δ step is ``mult · δ`` with
        # ``mult = 1`` if retained (2 consumed − 1 born) else ``2``.
        mult = jnp.where(retain, 1.0, 2.0)

        # Depletion-budget cap — division-free, gradient-SAFE.  Cap δ so
        # this step's net removal cannot overshoot the remaining budget:
        #   δ_cap = max(budget_left, 0) / mult     (mult ∈ {1, 2})
        #   δ     = min(δ_raw, δ_cap)
        # This pins cumulative net removal to R_target exactly (when
        # δ_raw > δ_cap the step contributes mult·δ_cap = budget_left) and
        # — crucially — has a BOUNDED gradient.  The earlier form
        # ``clip_scale = budget_left / max(net_full, occ_tol)`` divided by
        # the step's own removal, whose derivative −budget_left/net_full²
        # blows up to inf/NaN as net_full → 0 (e.g. SVI pushing kernel_b
        # into a regime where most pairs have tiny K).  That NaN poisoned
        # the whole SVI run under gate="soft" (M4 ELBO → nan) while the
        # forward pass stayed finite — a backward-only hazard.  ``mult`` is
        # 1 or 2, so ``budget_left / mult`` has no small-denominator
        # explosion, and ``minimum`` is gradient-safe.  Once the budget is
        # spent, δ_cap = 0 ⇒ δ = 0 (cheap no-op steps).
        delta_cap = jnp.maximum(budget_left, 0.0) / mult
        delta = jnp.minimum(delta_raw, delta_cap)
        delta = jnp.where(step_active_now, delta, 0.0)
        net_this = mult * delta

        # Occupancy updates on the two parents.
        new_occ = occ.at[i].add(-delta)
        new_occ = new_occ.at[j].add(-delta)

        # Remnant born into tail slot (n_init + k) — the scan counter is
        # the static write index, so no write-pointer.  Occupancy δ if
        # retained, else 0 (the slot stays empty and contributes nothing).
        tail = n_init + k
        tail_occ = jnp.where(retain, delta, 0.0)
        new_m_arr = m_arr.at[tail].set(m_new)
        new_chi_arr = chi_arr.at[tail].set(chi_new)
        new_w_arr = w_arr.at[tail].set(0.5 * (w_i + w_j))
        new_occ = new_occ.at[tail].set(tail_occ)
        new_gen = gen.at[tail].set(
            (jnp.maximum(g_i, g_j) + 1).astype(jnp.int32)
        )

        new_net_removed = net_removed + net_this

        # Per-step records (heavier-first; NaN/0 on non-events).
        is_event = (delta > occ_tol)
        heavier_is_i = m_i >= m_j
        m1_rec = jnp.where(heavier_is_i, m_i, m_j)
        m2_rec = jnp.where(heavier_is_i, m_j, m_i)
        chi1_rec = jnp.where(heavier_is_i, chi_i, chi_j)
        chi2_rec = jnp.where(heavier_is_i, chi_j, chi_i)
        g1_rec = jnp.where(heavier_is_i, g_i, g_j).astype(m_arr.dtype)
        g2_rec = jnp.where(heavier_is_i, g_j, g_i).astype(m_arr.dtype)
        retain_rec = retain.astype(m_arr.dtype)
        m_rem_rec = jnp.where(retain, m_new, jnp.nan)
        rec_event = jnp.stack(
            [m1_rec, m2_rec, chi1_rec, chi2_rec, g1_rec, g2_rec,
             retain_rec, m_rem_rec]
        )
        nan_row = jnp.full_like(rec_event, jnp.nan)
        rec = jnp.where(is_event, rec_event, nan_row)
        v_kick_out = jnp.where(is_event, v_kick_evt, jnp.nan)
        # Occupancy-scaled descendant weight (smooth in kernel params via
        # δ — no straight-through needed, the B.2 AD headline).
        w_out = 0.5 * (w_i + w_j) * delta
        step_flag = is_event.astype(jnp.int32)

        new_carry = (new_m_arr, new_chi_arr, new_w_arr, new_occ, new_gen,
                     new_net_removed, key)
        return new_carry, (rec, v_kick_out, w_out, step_flag)

    final_carry, outputs = jax.lax.scan(
        step_fn, carry0, jnp.arange(n_steps)
    )
    (m_f, chi_f, w_f, occ_f, gen_f, net_removed_f, _key_f) = final_carry
    (mergers_arr, v_kicks_arr, w_mergers_arr, step_flags) = outputs

    W_total = jnp.sum(occ_f * w_f)
    out_dict = dict(
        m_active=m_f,
        chi_active=chi_f,
        w_active=w_f,
        gen_active=gen_f,
        occupancy=occ_f,
        active_mask=occ_f > occ_tol,
        mergers=mergers_arr,
        v_kicks=v_kicks_arr,
        w_mergers=w_mergers_arr,
        step_active=step_flags.astype(bool),
        stats=dict(
            n_init=n_init,
            n_steps=int(n_steps),
            n_slots=int(S),
            W_total=W_total,
            net_removed=net_removed_f,
            depletion=net_removed_f / max(n_init, 1),
            sum_w_mergers=jnp.sum(w_mergers_arr),
            orientation="Aligned",
        ),
    )
    return out_dict


def _kish_n_eff(weights: np.ndarray) -> float:
    """Kish (1965) effective sample size ``(Σ w)² / Σ w²``.

    Coincides with ``len(weights)`` for uniform weights.  Pinned at
    `0.0` for an empty array.  Used in :func:`particle_mc`'s stats
    return so callers can monitor whether stratification or IS has
    actually bought sampling efficiency — same convention as the
    M3a ``MCEvaluator.n_eff`` field.
    """
    if weights.size == 0:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    s = float(w.sum())
    ss = float(np.sum(w * w))
    return s * s / ss if ss > 0 else 0.0


def _kernel_upper_bound(m_active: np.ndarray, a: float, b: float) -> float:
    """Conservative K_max over the active pool."""
    mlo = float(m_active.min())
    mhi = float(m_active.max())
    candidates = [
        kernel_K(np.array(2 * mhi), np.array(2 * mhi), a, b),  # equal-mass heavy
        kernel_K(np.array(2 * mlo), np.array(2 * mlo), a, b),  # equal-mass light
        kernel_K(np.array(mhi + mlo), np.array(mhi + mlo), a, b),  # asymmetric (q=mlo/mhi)
    ]
    return float(max(candidates)) * 1.05


def particle_mc(
    cfg: MCConfig,
    rng: np.random.Generator,
    *,
    track_generations: bool = True,
) -> dict:
    """Run the cascade to natural termination (or budget exhaustion).

    Returns a dict with keys:
        ``m_active``, ``chi_active``, ``gen_active`` — surviving particles.
        ``mergers``  : structured array shape ``(N_accept, 8)`` of accepted
            merger events: (m1, m2, χ1, χ2, g1, g2, retained, m_remnant_or_nan).
            Heavier first.  Under non-``Aligned`` orientation the χ columns
            are the **progenitor magnitudes** ``|χ_1|, |χ_2|`` (the
            orientation distribution carries the sign / tilt).
        ``v_kicks`` : 1-D array shape ``(N_accept,)`` of the per-event
            recoil ``v_kick`` (km/s).  Under ``Aligned`` this is the
            scalar :func:`v_kick_aligned`; under any other orientation
            the per-event sample.  Added in Session C — older readers
            that only consume ``mergers`` are unaffected.
        ``w_active`` : 1-D array shape ``(N_active_final,)`` — Session
            M3b per-surviving-particle importance weight.  Uniform
            ``1.0`` under the default v1 path (no proposal / stratify /
            weight_array); non-trivial when an IS proposal or
            stratified natal sampler is in use.  Pair with
            ``MCEvaluator(weights=...)`` for weighted density
            estimation.  Older readers that consume only
            ``m_active`` / ``chi_active`` are unaffected.
        ``w_mergers`` : 1-D array shape ``(N_accept,)`` — Session M3b
            per-accepted-merger descendant weight (``w_desc = 0.5 ·
            (w_i + w_j)`` from the merger of parents ``i, j``).
            Same uniform-1.0 invariant under v1 path.  Pair with
            rows of ``mergers`` when computing weighted observable-
            marginals.
        ``stats`` : dict with n_attempts, n_accepted, n_retained,
            n_active_final, plus Session M3b additions
            ``n_eff_active`` / ``n_eff_mergers`` (Kish 1965 effective
            sample size; equals the count for uniform weights).

    Notes
    -----
    The hot loop is intentionally Python-level because the bookkeeping
    (active-mask updates, pair removals) doesn't vectorise cleanly. We
    batch the *K-rejection step* (drawing many candidate pairs at once)
    because that's where ~all the wallclock goes — typically O(1/⟨K⟩)
    rejections per accepted pair.

    Aligned bit-for-bit invariant: when ``cfg.orientation is None`` or
    ``Aligned()`` the only numerical change versus pre-Session-C code is
    the new ``v_kicks`` array on the return dict — every other quantity
    (active particles, mergers list, stats counters, rng draws) is
    literally unchanged.  The Heaviside retention is what was already
    routed; the new ``v_kick_aligned`` evaluation on each accepted pair
    is read-only and does not perturb the numpy ``rng`` sequence.
    """
    aligned_path = _orientation_is_aligned(cfg.orientation)
    orient_key_state = jax.random.PRNGKey(int(cfg.orientation_key_seed))
    m_arr, chi_arr, w_arr = sample_natals(cfg, rng)
    n_init = m_arr.size
    gen_arr = np.ones(n_init, dtype=np.int32) if track_generations else None
    active = np.ones(n_init, dtype=bool)
    # Session M3b — per-particle importance weight, propagated through
    # the cascade by the arithmetic-mean rule on accepted mergers
    # (``w_desc = 0.5·(w_i + w_j)``).  Uniform initial weights (the v1
    # path) give ``w_arr == 1`` for ever, so the rejection-loop
    # dynamics are bit-for-bit unchanged — only the new ``w_active`` /
    # ``w_mergers`` return-dict entries appear post-M3b.  See
    # ``demo/PLAN_m3_stratified_natal.md`` § "Weight propagation rule".
    # Per-merger weight record (one float per accepted merger,
    # indexed in lockstep with ``mergers_records``).
    w_mergers_records: list[float] = []

    # Resolve stopping criterion. If the user passed an explicit
    # n_max_mergers we use that; otherwise compute one from
    # depletion_fraction. We also stop on n_max_attempts as a safety net.
    if cfg.n_max_mergers is not None:
        n_max_mergers = cfg.n_max_mergers
    else:
        # Each merger removes ≈1 (retained) or 2 (non-retained) from the
        # pool; for high-retention regimes ≈ 1 per merger. Stopping at
        # depletion_fraction means roughly that many mergers.
        n_max_mergers = max(1, int(cfg.depletion_fraction * n_init))
    n_active_floor = max(2, int(n_init * (1.0 - cfg.depletion_fraction))) if cfg.n_max_mergers is None else 2

    mergers_records: list[np.ndarray] = []
    # Per-event v_kick (km/s).  Aligned: v_kick_aligned scalar.
    # Non-Aligned: per-event vector-LZ sample.  Indexed in lockstep with
    # ``mergers_records`` so the i-th entry corresponds to the i-th
    # accepted merger.
    v_kicks_records: list[float] = []

    n_attempts = 0
    n_accepted = 0
    n_retained = 0
    no_progress_streak = 0
    max_no_progress = 5  # how many full rejection batches we tolerate
    stop_reason = "n_max_mergers"  # updated as we go

    while n_accepted < n_max_mergers and n_attempts < cfg.n_max_attempts:
        if int(active.sum()) < n_active_floor:
            stop_reason = "depletion_fraction"
            break
        active_idx = np.flatnonzero(active)
        n_active = active_idx.size
        if n_active < 2:
            stop_reason = "pool_exhausted"
            break

        # Update K_max from current pool — pool shrinks and broadens, so the
        # static bound from natals would become loose over time.
        K_max = _kernel_upper_bound(m_arr[active_idx], cfg.kernel_a, cfg.kernel_b)
        if not np.isfinite(K_max) or K_max <= 0:
            stop_reason = "K_max_invalid"
            break

        # Batch K-rejection. Draw `rejection_batch` candidate pairs from the
        # active list; accept by K_ij / K_max.
        batch = min(cfg.rejection_batch, n_active * (n_active - 1) // 2 or 1)
        ia_in_active = rng.integers(0, n_active, size=batch)
        ib_in_active = rng.integers(0, n_active, size=batch)
        # Disallow self-pairs by resampling i_b for collisions.
        same = ia_in_active == ib_in_active
        while same.any():
            ib_in_active[same] = rng.integers(0, n_active, size=int(same.sum()))
            same = ia_in_active == ib_in_active

        ia = active_idx[ia_in_active]
        ib = active_idx[ib_in_active]
        m_a = m_arr[ia]; m_b = m_arr[ib]
        chi_a = chi_arr[ia]; chi_b = chi_arr[ib]
        K_vals = kernel_K(m_a, m_b, cfg.kernel_a, cfg.kernel_b)
        u = rng.random(size=batch)
        K_accept = u < (K_vals / K_max)

        n_attempts += batch
        if not K_accept.any():
            no_progress_streak += 1
            if no_progress_streak > max_no_progress:
                stop_reason = "no_K_acceptance"
                break
            continue
        no_progress_streak = 0

        # We process accepted pairs *one at a time* because each merger
        # mutates the active set, which would invalidate later pair indices
        # in the batch. (We accept the Python-loop cost here; the dominant
        # work is the K-rejection above which IS vectorised.)
        accept_idx = np.flatnonzero(K_accept)
        for k in accept_idx:
            i_a = ia[k]; i_b = ib[k]
            # Reject silently if either particle has already merged earlier
            # in this same batch.
            if not (active[i_a] and active[i_b]):
                continue
            n_accepted += 1
            ma = m_arr[i_a]; mb = m_arr[i_b]
            ca = chi_arr[i_a]; cb = chi_arr[i_b]
            ga = (gen_arr[i_a] if gen_arr is not None else 1)
            gb = (gen_arr[i_b] if gen_arr is not None else 1)

            if aligned_path:
                # ---------- Aligned bit-for-bit existing v1 path. ----------
                retain = bool(retention_aligned(ma, mb, ca, cb, cfg.v_esc) > 0.5)
                if retain:
                    m_new = float(M_rem(ma, mb, ca, cb))
                    chi_new = float(chi_rem_(ma, mb, ca, cb))
                # v_kick for the v_kicks return array — read-only, does
                # not affect the active-pool sequence.  Evaluating both
                # branches keeps the per-event work uniform regardless of
                # whether the pair was retained.
                v_kick_evt = float(_v_kick_aligned_jax(ma, mb, ca, cb))
            else:
                # ---------- Non-Aligned: per-event vector physics. ---------
                # Advance the orientation key stream.  Independent of the
                # numpy ``rng`` so the active-pool sequence stays in lock-
                # step with the Aligned baseline up to the retention
                # decision.
                orient_key_state, evt_key = jax.random.split(orient_key_state)
                m_new, chi_new, v_kick_evt, retain = (
                    _per_event_remnant_misaligned(
                        evt_key, float(ma), float(mb),
                        float(ca), float(cb),
                        cfg.v_esc, cfg.orientation,
                    )
                )

            # Session M3b — descendant weight = arithmetic mean of
            # parents.  Under uniform initial weights (v1 path), wa =
            # wb = 1 ⇒ w_new = 1 ⇒ this is a no-op that keeps the
            # active-pool weights uniform forever.  Under stratified /
            # importance-proposal initial weights, the average is what
            # carries the natal IS information through the cascade
            # while preserving per-particle scale (each particle
            # continues to represent ~one bin's worth of probability
            # mass).  See ``demo/PLAN_m3_stratified_natal.md`` §
            # "Weight propagation rule (LOCKED)".
            wa = float(w_arr[i_a]); wb = float(w_arr[i_b])
            w_new = 0.5 * (wa + wb)
            if retain:
                # Replace heavier-slot with the remnant; deactivate the
                # other. Heavier slot first so the bookkeeping conserves
                # m1 ≥ m2 cosmetically.
                if ma >= mb:
                    m_arr[i_a] = m_new; chi_arr[i_a] = chi_new
                    w_arr[i_a] = w_new
                    if gen_arr is not None: gen_arr[i_a] = max(ga, gb) + 1
                    active[i_b] = False
                else:
                    m_arr[i_b] = m_new; chi_arr[i_b] = chi_new
                    w_arr[i_b] = w_new
                    if gen_arr is not None: gen_arr[i_b] = max(ga, gb) + 1
                    active[i_a] = False
                n_retained += 1
                event_rem = m_new
            else:
                active[i_a] = False; active[i_b] = False
                event_rem = np.nan

            # Heavier-first event record.
            if ma >= mb:
                rec = (ma, mb, ca, cb, ga, gb, float(retain), event_rem)
            else:
                rec = (mb, ma, cb, ca, gb, ga, float(retain), event_rem)
            mergers_records.append(rec)
            v_kicks_records.append(v_kick_evt)
            # Per-merger weight is the descendant weight (which is also
            # what the engine-side comparison should use when binning
            # mergers — the merger event "weighs" w_new in the
            # rate-density sense after the arithmetic-mean propagation).
            w_mergers_records.append(w_new)

            if n_accepted >= n_max_mergers:
                break

    out_idx = np.flatnonzero(active)
    return dict(
        m_active=m_arr[out_idx].copy(),
        chi_active=chi_arr[out_idx].copy(),
        gen_active=(gen_arr[out_idx].copy() if gen_arr is not None else None),
        mergers=(
            np.asarray(mergers_records, dtype=np.float64)
            if mergers_records else np.empty((0, 8), dtype=np.float64)
        ),
        v_kicks=(
            np.asarray(v_kicks_records, dtype=np.float64)
            if v_kicks_records else np.empty((0,), dtype=np.float64)
        ),
        # Session M3b additions — per-particle / per-event importance
        # weights.  Under the v1 (uniform-weight) path these are
        # uniformly 1.0 and existing readers can safely ignore them.
        # ``w_active`` is the per-surviving-particle weight (pair with
        # ``m_active`` / ``chi_active``; pass straight into
        # ``MCEvaluator(weights=...)`` for density estimation).
        # ``w_mergers`` is the per-accepted-merger descendant weight
        # (pair with rows of ``mergers``; multiply per-event histogram
        # bins by these when computing weighted observable-marginals).
        w_active=w_arr[out_idx].copy(),
        w_mergers=(
            np.asarray(w_mergers_records, dtype=np.float64)
            if w_mergers_records else np.empty((0,), dtype=np.float64)
        ),
        stats=dict(
            n_attempts=n_attempts,
            n_accepted=n_accepted,
            n_retained=n_retained,
            n_active_final=int(active.sum()),
            n_init=n_init,
            stop_reason=stop_reason,
            depletion=1.0 - (int(active.sum()) / max(n_init, 1)),
            orientation=type(cfg.orientation).__name__ if cfg.orientation is not None else "Aligned",
            # M3b — Kish effective sample sizes (1965 design-effect).
            # For uniform weights, both equal ``n_active_final`` /
            # ``n_accepted`` exactly.  Useful for the workstation
            # gate's stratification-vs-uniform comparison.
            n_eff_active=_kish_n_eff(w_arr[out_idx]),
            n_eff_mergers=(
                _kish_n_eff(np.asarray(w_mergers_records, dtype=np.float64))
                if w_mergers_records else 0.0
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Smoke driver — Stage 1 θ*, KDE plots.
# ---------------------------------------------------------------------------
_STAGE1 = dict(
    mu_natal=8.0,
    sigma_natal=1.0,
    m_min=4.0,
    m_max=80.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.10,
    kernel_a=0.0,
    kernel_b=1.0,
    v_esc=1000.0,
)


def _plot_active_marginals(result: dict, cfg: MCConfig, out_dir: Path) -> None:
    r"""Raw HISTOGRAMS of the surviving population — no KDE.

    KDE smoothing is murder on the deep-cascade peaks (3G+ remnants pile
    at discrete ``M_rem`` cascade locations with only ~100 counts/bin, so
    a kernel bandwidth wide enough to look smooth washes the peaks out).
    Per the standing preference + ``feedback-gaussian-kde-over-smooths-
    cascade-peaks``, this smoke driver plots step histograms instead.
    Occupancy-weighted if the result carries an ``occupancy`` field (the
    B.2 soft-gate output); uniform otherwise (numpy / B.1).
    """
    import matplotlib.pyplot as plt

    m_active = np.asarray(result["m_active"])
    chi_active = np.asarray(result["chi_active"])
    gen_active = result["gen_active"]
    gen_active = None if gen_active is None else np.asarray(gen_active)
    # Occupancy weights for the soft-gate output; uniform otherwise.
    w = result.get("occupancy", None)
    w = np.ones_like(m_active) if w is None else np.asarray(w)

    n_bins_m = 40
    m_edges = np.geomspace(cfg.m_min, cfg.m_max * 2, n_bins_m + 1)
    m_centers = np.sqrt(m_edges[:-1] * m_edges[1:])
    m_binw = np.diff(m_edges)
    n_bins_chi = 50
    chi_edges = np.linspace(0.0, 1.0, n_bins_chi + 1)
    chi_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    chi_binw = np.diff(chi_edges)[0]

    def _hist(x, weights, edges, binw):
        h, _ = np.histogram(x, bins=edges, weights=weights)
        return h / binw

    # ---- dN/dm (raw step histogram) ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    peak = 0.0
    if m_active.size >= 1:
        dN_dm = _hist(m_active, w, m_edges, m_binw)
        peak = max(peak, dN_dm.max())
        ax.loglog(m_centers, np.maximum(dN_dm, 1e-30), drawstyle="steps-mid",
                  lw=2.0, color="C3",
                  label=f"all surviving (Σw={w.sum():.0f})")
    if gen_active is not None and gen_active.size:
        G = int(gen_active.max())
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(G, 1)))
        for g in range(1, G + 1):
            mask = gen_active == g
            if int(mask.sum()) < 1:
                continue
            dN = _hist(m_active[mask], w[mask], m_edges, m_binw)
            ax.loglog(m_centers, np.maximum(dN, 1e-30), drawstyle="steps-mid",
                      lw=1.3, ls="--", color=colors[g - 1],
                      label=f"{g}G (Σw={w[mask].sum():.0f})")
    ax.set_xlabel(r"single-BH mass $m \;[M_\odot]$")
    ax.set_ylabel(r"$dN/dm$ (raw histogram, per $M_\odot$)")
    ax.set_title("Backup 1 — pure particle MC, surviving population $dN/dm$")
    if peak > 0:
        ax.set_ylim(bottom=peak * 1e-6)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    fig.tight_layout()
    out = out_dir / "backup1_dN_dm.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")

    # ---- dN/dχ (raw step histogram) ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    peak = 0.0
    if chi_active.size >= 1:
        dN_dchi = _hist(chi_active, w, chi_edges, chi_binw)
        peak = max(peak, dN_dchi.max())
        ax.semilogy(chi_centers, np.maximum(dN_dchi, 1e-30),
                    drawstyle="steps-mid", lw=2.0, color="C3",
                    label=f"all surviving (Σw={w.sum():.0f})")
    if gen_active is not None and gen_active.size:
        G = int(gen_active.max())
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(G, 1)))
        for g in range(1, G + 1):
            mask = gen_active == g
            if int(mask.sum()) < 1:
                continue
            dN = _hist(chi_active[mask], w[mask], chi_edges, chi_binw)
            ax.semilogy(chi_centers, np.maximum(dN, 1e-30),
                        drawstyle="steps-mid", lw=1.3, ls="--",
                        color=colors[g - 1],
                        label=f"{g}G (Σw={w[mask].sum():.0f})")
    ax.axvline(0.6865, color="0.4", lw=0.7, ls=":", label=r"$\chi_{\rm rem}(1,0)$")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(r"$dN/d\chi$ (raw histogram)")
    ax.set_title("Backup 1 — pure particle MC, surviving population $dN/d\\chi$")
    ax.set_xlim(0, 1)
    if peak > 0:
        ax.set_ylim(bottom=peak * 1e-4)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    fig.tight_layout()
    out = out_dir / "backup1_dN_dchi.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _smoke_main(
    run_hash: Optional[str] = None,
    n_seed: int = 20_000,
    n_max_mergers: Optional[int] = None,
    depletion_fraction: float = 0.05,
) -> None:
    here = Path(__file__).parent
    rh = run_hash or secrets.token_hex(3)
    fig_dir = here / f"figures_{rh}"
    art_dir = here / f"artifacts_{rh}"
    fig_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    print(f"Run hash: {rh}")
    print(f"  figures → {fig_dir.relative_to(here.parent)}/")
    print(f"  artifacts → {art_dir.relative_to(here.parent)}/")

    cfg = MCConfig(
        n_seed=n_seed,
        n_max_mergers=n_max_mergers,
        depletion_fraction=depletion_fraction,
        **_STAGE1,
    )
    rng = np.random.default_rng(20260518)
    print(f"Config: {cfg}")

    print("Running pure particle MC ...")
    result = particle_mc(cfg, rng, track_generations=True)
    s = result["stats"]
    print()
    print("Stats:")
    print(f"  n_init             = {s['n_init']}")
    print(f"  n_attempts         = {s['n_attempts']}")
    print(f"  n_accepted         = {s['n_accepted']}")
    print(f"  n_retained         = {s['n_retained']}")
    print(f"  n_active_final     = {s['n_active_final']}")
    print(f"  depletion fraction = {s['depletion']:.4f}")
    print(f"  retain fraction    = {s['n_retained']/max(s['n_accepted'],1):.3f}")
    print(f"  stop reason        = {s['stop_reason']}")

    gen_active = result["gen_active"]
    if gen_active is not None and gen_active.size:
        print()
        print("Surviving population by generation:")
        for g in range(1, int(gen_active.max()) + 1):
            n_g = int((gen_active == g).sum())
            print(f"  {g}G surviving: N = {n_g}")

    np.savez(
        art_dir / "backup1_mc_result.npz",
        m_active=result["m_active"],
        chi_active=result["chi_active"],
        gen_active=(
            result["gen_active"] if result["gen_active"] is not None
            else np.zeros(0)
        ),
        mergers=result["mergers"],
    )
    print(f"  wrote {art_dir / 'backup1_mc_result.npz'}")

    _plot_active_marginals(result, cfg, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-hash", default=None)
    parser.add_argument("--n-seed", type=int, default=20_000)
    parser.add_argument(
        "--n-max-mergers", type=int, default=None,
        help="Hard cap on accepted mergers. If given, overrides "
        "--depletion-fraction.",
    )
    parser.add_argument(
        "--depletion-fraction", type=float, default=0.05,
        help="Stop when the active pool has shrunk by this fraction "
        "relative to n_seed. Defaults to 0.05 (early-cascade regime "
        "matching engine ≈ Backup 2 implicit at ε ≈ 0.025). Set to 0.5 "
        "for deeper cascade; 1.0 for full exhaustion.",
    )
    args = parser.parse_args()
    _smoke_main(
        run_hash=args.run_hash,
        n_seed=args.n_seed,
        n_max_mergers=args.n_max_mergers,
        depletion_fraction=args.depletion_fraction,
    )
