r"""Stage-2 selection — VT(θ)-weighted Poisson-mean estimator for gwkokab.

Stage 1 fits the *detected* rate with **flat** selection (``VT ≡ 1``,
efficiency folded into ``log_rate``); the Poisson mean is just
``T_obs·∫ intensity dx`` (``gwkokab_bridge.coagulation_poisson_mean_estimator``).
Stage 2 makes selection explicit: the expected detected count becomes

.. math::

    \mu \;=\; T_{\rm obs}\int VT(x)\,\frac{dN}{dx}(x)\,d^4x ,
    \qquad x=(m_1,m_2,\chi_1,\chi_2),

where ``VT(x)`` is the sensitive space-time volume (a *selection* weight,
0 ≤ relative response) and ``dN/dx = exp(log_rate_density)`` is the engine's
rate-density (intensity).  This module supplies ``VT`` toy models and a
``poisson_mean_estimator`` with gwkokab's ``(model, **kw) -> (μ, var)``
contract that drop-in replaces the flat estimator in
:func:`gwkokab_bridge.make_coagulation_dist_fn` / the Stage-1 wiring.

**Primary toy VT — Euclidean sensitive volume.**  For an inspiral-dominated
SNR the optimal SNR scales as ``ρ ∝ M_chirp^{5/6}`` (at fixed distance), and
the detectable comoving volume in the local/Euclidean limit goes as
``V ∝ ρ^3``.  Hence

.. math::

    VT(m_1,m_2)\;\propto\;\rho^3\;\propto\;M_{\rm chirp}^{5/6\,\cdot\,3}
    \;=\;M_{\rm chirp}^{15/6}\;=\;M_{\rm chirp}^{5/2},

a smooth power law in the chirp mass — **no threshold, no smoothing scale**,
so it is AD-clean by construction.  This is :func:`vt_powerlaw` (default
``exponent = 15/6``).  A thresholded-SNR surrogate
(``expit(SNR−8)``; ``PLAN_gwkokab_fit.md`` Stage-2 option *a*) is provided as
:func:`snr_threshold_pdet` for when a sharp horizon matters; it carries an
explicit ``steepness`` (the Heaviside→sigmoid relaxation, same spirit as the
engine's soft-sigmoid retention around ``v_esc``), NOT a data-derived scale.

**Estimator — deterministic separable grid.**  :func:`make_vt_poisson_mean_estimator`
integrates ``VT·intensity`` on a tensor grid ``m₁×m₂×χ₁×χ₂`` by evaluating the
**black-box** ``exp(model.log_prob)`` at the nodes (so it is *solver-agnostic*
— mc_jax / basis / implicit alike) and contracting with trapezoid weights
(log-trapezoid in mass, matching ``MCEvaluatorJax.total_rate``'s measure).
Deterministic (no Monte-Carlo noise in ``μ`` → clean SVI/NUTS gradients) and
**exactly linear in VT**: with ``VT ≡ 1`` it reduces to this module's own flat
grid integral (:func:`make_flat_grid_poisson_mean_estimator`), with ``VT ≡ c``
to ``c×`` that.  ``log_rate_density`` already carries the ``½ R_env`` and the
unordered-pair ``½`` (``log[½ R K N N]``), so integrating ``exp(log_prob)``
over the **full** ``m₁×m₂`` square yields ``∫ intensity`` directly — no extra
factor.

Honesty / scope notes:
* This grid integral spans the model's box support ``[m_min, m_max_factor·m_max]``
  and is **independent** of ``MCEvaluatorJax.total_rate`` (different domain +
  the engine's separable-χ trick); the open ``total_rate`` metric bias
  (``[[project-mc-jax-total-rate-bias]]``, §4.3 — KDE oversmoothing +
  quadrature-span clipping) is orthogonal.  For an apples-to-apples flat
  baseline under selection, use :func:`make_flat_grid_poisson_mean_estimator`
  (same method, ``VT≡1``), not the grid ``coagulation_poisson_mean_estimator``.
* ``var`` is returned ``0`` (deterministic quadrature).  Keep
  ``variance_cut_threshold=None`` downstream.

Hard rules honoured: additive (new module; nothing in ``response/`` or the
bridge changes); no fudge factors (VT is an explicit physical power law, ``μ``
linear in it); the SNR surrogate is a *declared* toy (``PLAN`` option *a*), not
a fake stand-in for IMRPhenomD (option *b*).
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "chirp_mass",
    "vt_powerlaw",
    "snr_threshold_pdet",
    "make_vt_poisson_mean_estimator",
    "make_flat_grid_poisson_mean_estimator",
]

# Selection weight: VT(m1, m2, chi1, chi2) -> relative sensitive volume.
VTFn = Callable[[Array, Array, Array, Array], Array]


def chirp_mass(m1: Array, m2: Array) -> Array:
    r"""Chirp mass ``M_c = (m₁ m₂)^{3/5} / (m₁+m₂)^{1/5}`` (symmetric in m₁,m₂)."""
    m1 = jnp.asarray(m1)
    m2 = jnp.asarray(m2)
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def vt_powerlaw(
    *,
    exponent: float = 15.0 / 6.0,
    m_chirp_ref: float = 10.0,
    vt_ref: float = 1.0,
) -> VTFn:
    r"""Euclidean sensitive-volume toy ``VT ∝ M_chirp^{exponent}``.

    ``VT(m₁,m₂) = vt_ref · (M_chirp(m₁,m₂) / m_chirp_ref)^{exponent}``, χ-independent.

    Parameters
    ----------
    exponent
        Default ``15/6`` — the ``ρ ∝ M_chirp^{5/6}`` SNR scaling cubed into a
        Euclidean volume ``V ∝ ρ³``.  ``exponent = 0`` recovers flat selection.
    m_chirp_ref
        Chirp mass (M⊙) at which ``VT = vt_ref`` — sets the normalisation point
        (an explicit physical scale, not a tuned knob).
    vt_ref
        Sensitive volume at ``m_chirp_ref`` (absorbs ``T_obs``/efficiency units;
        the absolute scale is degenerate with ``log_rate`` exactly as in the
        flat case — it is the SHAPE in ``M_chirp`` that selection adds).
    """
    exponent = float(exponent)
    m_chirp_ref = float(m_chirp_ref)
    vt_ref = float(vt_ref)

    def vt_fn(m1: Array, m2: Array, chi1: Array, chi2: Array) -> Array:
        mc = chirp_mass(m1, m2)
        return vt_ref * (mc / m_chirp_ref) ** exponent

    return vt_fn


def snr_threshold_pdet(
    *,
    snr_ref: float = 8.0,
    m_chirp_ref: float = 10.0,
    snr_threshold: float = 8.0,
    steepness: float = 1.0,
    vt_ref: float = 1.0,
) -> VTFn:
    r"""Thresholded-SNR detection-probability surrogate (PLAN Stage-2 option *a*).

    ``SNR(m₁,m₂) = snr_ref · (M_chirp / m_chirp_ref)^{5/6}``;
    ``pdet = vt_ref · σ(steepness·(SNR − snr_threshold))`` with ``σ = expit``.

    A *declared toy* — a smooth (Heaviside→sigmoid) horizon, NOT a real
    IMRPhenomD optimal-SNR (that is option *b*, a follow-up).  ``steepness`` is
    the relaxation scale (same role as the engine's soft-sigmoid retention),
    exposed explicitly; ``snr_threshold = 8`` is the physical detection cut.
    Returns values in ``(0, vt_ref)`` so, as a pdet, ``μ_pdet ≤ μ_flat``.
    """
    snr_ref = float(snr_ref)
    m_chirp_ref = float(m_chirp_ref)
    snr_threshold = float(snr_threshold)
    steepness = float(steepness)
    vt_ref = float(vt_ref)

    def pdet_fn(m1: Array, m2: Array, chi1: Array, chi2: Array) -> Array:
        mc = chirp_mass(m1, m2)
        snr = snr_ref * (mc / m_chirp_ref) ** (5.0 / 6.0)
        return vt_ref * jax.nn.sigmoid(steepness * (snr - snr_threshold))

    return pdet_fn


def _grids(
    m_min: float, m_hi: float, n_m: int, n_chi: int
) -> Tuple[Array, Array, Array, Array]:
    r"""Geomspaced mass grid (+ log-trapezoid weights) and linear χ grid (+
    trapezoid weights).  Mass measure matches ``MCEvaluatorJax.total_rate``."""
    log_m = jnp.linspace(jnp.log(m_min), jnp.log(m_hi), n_m)
    m_grid = jnp.exp(log_m)
    dlog = jnp.diff(log_m)
    w_log_m = jnp.concatenate([dlog[:1] * 0.5, 0.5 * (dlog[:-1] + dlog[1:]), dlog[-1:] * 0.5])
    w_m = w_log_m * m_grid  # ∫ f dm on a geomspaced grid (∫ f m dlnm)

    chi_grid = jnp.linspace(0.0, 1.0, n_chi)
    dchi = jnp.diff(chi_grid)
    w_chi = jnp.concatenate([dchi[:1] * 0.5, 0.5 * (dchi[:-1] + dchi[1:]), dchi[-1:] * 0.5])
    return m_grid, w_m, chi_grid, w_chi


def make_vt_poisson_mean_estimator(
    vt_fn: VTFn,
    *,
    m_min: float,
    m_max: float,
    m_max_factor: float = 2.5,
    n_m: int = 40,
    n_chi: int = 16,
    env: Any = None,
) -> Callable[..., Tuple[Array, Array]]:
    r"""gwkokab ``poisson_mean_estimator`` for ``μ = T_obs·∫ VT·intensity d⁴x``.

    Returns ``estimator(model, *, T_obs=1.0, **_ignored) -> (μ, var=0)``.  The
    grid (nodes + trapezoid weights) is built once here and reused every call;
    only ``exp(model.log_prob)`` at the nodes (and thus ``μ``) depends on the
    sampled population parameters, so the estimator is deterministic and
    AD-clean.

    ``model`` is duck-typed: anything exposing ``log_prob(value=(...,4))`` (the
    bridge ``CoagulationPopulation``); ``env`` is unused here (the bridge bakes
    ``v_esc`` into the response) and kept only for signature symmetry.

    Integration domain: ``m ∈ [m_min, m_max_factor·m_max]`` (captures retained
    hierarchical remnants above the natal cap), ``χ ∈ [0, 1]``.  Full m₁×m₂
    square — ``log_prob`` already carries the unordered ``½``.
    """
    m_hi = float(m_max) * float(m_max_factor)
    m_grid, w_m, chi_grid, w_chi = _grids(float(m_min), m_hi, int(n_m), int(n_chi))

    # 4-D tensor nodes (m1, m2, chi1, chi2) and the separable weight tensor.
    M1, M2, C1, C2 = jnp.meshgrid(m_grid, m_grid, chi_grid, chi_grid, indexing="ij")
    W = (w_m[:, None, None, None] * w_m[None, :, None, None]
         * w_chi[None, None, :, None] * w_chi[None, None, None, :])
    VT = vt_fn(M1, M2, C1, C2)               # constant in θ
    nodes = jnp.stack([M1, M2, C1, C2], axis=-1)  # (n_m, n_m, n_chi, n_chi, 4)

    def estimator(model: Any, *, T_obs: float = 1.0, **_ignored: Any) -> Tuple[Array, Array]:
        log_intensity = model.log_prob(nodes)     # (n_m, n_m, n_chi, n_chi)
        intensity = jnp.exp(log_intensity)
        mu = jnp.asarray(T_obs) * jnp.sum(VT * intensity * W)
        return mu, jnp.zeros_like(mu)

    return estimator


def make_flat_grid_poisson_mean_estimator(
    *,
    m_min: float,
    m_max: float,
    m_max_factor: float = 2.5,
    n_m: int = 40,
    n_chi: int = 16,
    env: Any = None,
) -> Callable[..., Tuple[Array, Array]]:
    r"""``VT ≡ 1`` special case of :func:`make_vt_poisson_mean_estimator`.

    The apples-to-apples flat baseline for selection comparisons (same grid /
    method as the VT estimator).  NB: this is *not* identical to the bridge's
    grid ``coagulation_poisson_mean_estimator`` (which uses the evaluator's own
    ``[min(m_active), max(m_active)]`` domain + separable-χ quadrature); see the
    module docstring.
    """
    return make_vt_poisson_mean_estimator(
        lambda m1, m2, c1, c2: jnp.ones_like(chirp_mass(m1, m2)),
        m_min=m_min, m_max=m_max, m_max_factor=m_max_factor,
        n_m=n_m, n_chi=n_chi, env=env,
    )
