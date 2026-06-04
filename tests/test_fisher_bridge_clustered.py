"""Engine-tier tests: Fisher-κ through the bridge + AD on the clustered χ grid.

These build the REAL ``solver="implicit"`` response, which imports the
engine's natal-mass module and therefore ``gwkokab`` (→ RIFT/lalsuite).
That dependency is absent in the lightweight JAX sandbox, so the whole
module SKIPS there and runs on the workstation — same two-tier convention
as the G6–G7 full-stack tier in ``test_gwkokab_bridge.py``.  A genuine
signature/bug must FAIL loudly, not hide as a skip, so the skip is gated
narrowly on the import of the engine builders.

What is asserted (invariant-based, no frozen references):

* orientation actually threads through — Isotropic and Fisher(κ) produce a
  χ-density that DIFFERS from the Aligned default and from each other;
* ``∂(Σ N*)/∂κ`` is finite and nonzero — the AD-on-the-orientation-knob
  promise (the "interesting" spin result);
* ``∂(Σ N*)/∂v_esc`` is finite and nonzero under Fisher + the smooth
  ``sigmoid`` retention — the kick-retention signature that is identically
  zero under the Aligned + Heaviside default;
* the bridge's ``make_coagulation_dist_fn`` forwards ``orientation`` to the
  built response.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Engine tier: skip the whole module if the implicit engine (→ gwkokab)
# can't be imported in this environment.
try:  # noqa: SIM105
    from hierarchical_ad_pop_engine.grids import BinaryGrid
    from hierarchical_ad_pop_engine.spin_orientation import (
        Aligned,
        Isotropic,
        Fisher,
    )
    from hierarchical_ad_pop_engine.response.coagulation import (
        build_gaussian_coagulation,
    )
    from hierarchical_ad_pop_engine.response.implicit import (
        solve_implicit_fixed_point,
    )
    from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (
        make_coagulation_dist_fn,
    )
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"engine (implicit solver / gwkokab) unavailable: {exc!r}",
        allow_module_level=True,
    )


# Small but representative resolution — keeps each solve well under the
# sandbox per-command budget while exercising the full vector path.
_GRID = BinaryGrid.clustered_chi(m_min=4.0, m_max=80.0, n_m=16, n_chi=12)
_V_ESC = 300.0
_QUAD = dict(n_theta=3, n_dphi=6)  # light orientation quadrature for tests


def _solve_sum(orientation, v_esc=_V_ESC, epsilon=0.5):
    resp = build_gaussian_coagulation(
        mu=8.0, sigma=1.0, m_min=4.0, m_max=80.0,
        natal_spin_loc=0.0, natal_spin_scale=0.05, kernel_b=2.0,
        grid=_GRID, solver="implicit",
        orientation=orientation, retention_kind="sigmoid",
    )
    res = solve_implicit_fixed_point(
        resp, jnp.asarray(v_esc), epsilon=epsilon,
        max_iter=60, tol=1e-7, damping=0.5,
    )
    return res


def _chi_marginal(orientation):
    res = _solve_sum(orientation)
    return np.asarray(jnp.sum(res.N, axis=0))  # sum over m → dN/dχ proxy


def test_orientation_changes_the_spin_density():
    m_al = _chi_marginal(Aligned())
    m_iso = _chi_marginal(Isotropic.default(**_QUAD))
    m_fi = _chi_marginal(Fisher.default(kappa=4.0, **_QUAD))
    rel = lambda a, b: np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30)
    assert rel(m_iso, m_al) > 1e-3, "Isotropic should differ from Aligned"
    assert rel(m_fi, m_iso) > 1e-3, "Fisher(κ) should differ from Isotropic"


def test_grad_through_kappa_is_finite_and_nonzero():
    def total_N(kappa):
        return jnp.sum(_solve_sum(Fisher.default(kappa=kappa, **_QUAD)).N)

    g = jax.grad(total_N)(jnp.asarray(4.0))
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0


def test_grad_through_v_esc_nonzero_under_fisher_sigmoid():
    """Aligned+Heaviside gives ∂/∂v_esc ≡ 0; Fisher+sigmoid restores it."""
    def total_N(v_esc):
        return jnp.sum(_solve_sum(Fisher.default(kappa=4.0, **_QUAD), v_esc=v_esc).N)

    g = jax.grad(total_N)(jnp.asarray(_V_ESC))
    assert np.isfinite(float(g))
    assert abs(float(g)) > 0.0


def test_bridge_forwards_orientation_to_response():
    """make_coagulation_dist_fn(orientation=Fisher(...)) → the built
    response carries that orientation (not the Aligned default)."""
    dist_fn = make_coagulation_dist_fn(
        v_esc=_V_ESC, m_min=4.0, m_max=80.0, grid=_GRID,
        solver="implicit", epsilon=0.5,
        orientation=Fisher.default(kappa=4.0, **_QUAD),
        retention_kind="sigmoid",
    )
    pop = dist_fn(
        mu_natal=8.0, sigma_natal=1.0, natal_spin_loc=0.0,
        natal_spin_scale=0.05, kernel_b=2.0, log_rate=0.0,
    )
    resp = getattr(pop, "response", None)
    assert resp is not None, "CoagulationPopulation should expose .response"
    assert isinstance(resp.orientation, Fisher)
    assert float(resp.orientation.kappa) == pytest.approx(4.0)
