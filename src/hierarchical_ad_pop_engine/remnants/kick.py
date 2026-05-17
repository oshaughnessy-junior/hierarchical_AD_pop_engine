"""Remnant kick magnitude — aligned-spin model only.

For aligned-spin progenitors the kick has two contributions:

.. math::

    v_m       &= A \\, \\eta^2 \\, \\frac{1-q}{1+q} \\, \\bigl[1 + B \\eta \\bigr]
                \\quad\\text{(mass asymmetry, Fitchett-Detweiler)} \\\\
    v_\\parallel &= K \\, \\eta^2 \\, \\frac{|\\chi_2 - q \\chi_1|}{1+q}
                \\quad\\text{(aligned-spin component)}

with ``A = 1.2 × 10^4`` km/s, ``B = -0.93``, ``K = 6 × 10^4`` km/s
(Lousto & Zlochower 2008, 2011; González et al. 2007). Total:

.. math::

    v_{\\rm kick} = \\sqrt{v_m^2 + v_\\parallel^2} .

This is **pure aligned-spin physics** — no in-plane "superkick" component
and no tilt marginalisation. Tilt-marginalised kicks (which would introduce
a real CDF over kick magnitudes from isotropic / partially aligned tilt
distributions) are a separate v2 module.

Retention
---------

Because the aligned kick is deterministic, retention is a Heaviside step:

.. math::

    P_{\\rm retain}(m_1, m_2, \\chi_1, \\chi_2 \\mid v_{\\rm esc})
        = \\mathbf{1}\\bigl[\\, v_{\\rm kick}(m_1, m_2, \\chi_1, \\chi_2) < v_{\\rm esc} \\,\\bigr] .

**Autodiff consequence**: gradients of the retained-population rate with
respect to ``v_esc`` are zero almost everywhere (Heaviside has no gradient).
This is an honest limitation of the aligned-only physics — to recover AD in
``v_esc``, add tilt marginalisation (a smooth kick CDF over tilts) in v2.
Gradients with respect to natal IMF, spin, kernel, and other parameters
flow normally because those change the integrand of the retained-volume
integral smoothly.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .mass import eta as _eta


# Lousto-Zlochower coefficients (km/s).
_A = 1.2e4
_B = -0.93
_K = 6.0e4


def v_kick_aligned(m1: Array, m2: Array, chi1: Array, chi2: Array) -> Array:
    """Aligned-spin remnant kick magnitude in km/s.

    Convention: internally swaps so the heavier mass plays the role of ``m_1``
    in the Lousto-Zlochower formula. Smooth and differentiable in
    ``(m_1, m_2, \\chi_1, \\chi_2)``.
    """
    m_big = jnp.maximum(m1, m2)
    m_sml = jnp.minimum(m1, m2)
    chi_big = jnp.where(m1 >= m2, chi1, chi2)
    chi_sml = jnp.where(m1 >= m2, chi2, chi1)
    q = m_sml / m_big
    e = _eta(m_big, m_sml)
    v_m = _A * jnp.square(e) * (1.0 - q) / (1.0 + q) * (1.0 + _B * e)
    v_par = _K * jnp.square(e) * jnp.abs(chi_sml - q * chi_big) / (1.0 + q)
    return jnp.sqrt(jnp.square(v_m) + jnp.square(v_par))


def retention_aligned(
    m1: Array,
    m2: Array,
    chi1: Array,
    chi2: Array,
    v_esc: Array,
) -> Array:
    """Heaviside retention for aligned-spin progenitors.

    Returns ``1.0`` if ``v_kick < v_esc``, else ``0.0``. **Not differentiable
    in v_esc by construction** — see module docstring.
    """
    v = v_kick_aligned(m1, m2, chi1, chi2)
    return jnp.where(v < v_esc, 1.0, 0.0)
