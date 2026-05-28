r"""Session J4 sandbox-tier tests — JAX MCResponseJax surface.

Per ``demo/PLAN_jax_mc.md`` § J4.  Pins five sandbox invariants on the
``MCResponseJax`` class itself (the factory + ``solver="mc_jax"``
dispatch through ``build_gaussian_coagulation`` is workstation-tier
because the full ``CoagulationResponse`` import path requires gwkokab):

1. **Smoke** — MCResponseJax constructs from a J2 cascade output +
   J3 evaluator; ``log_rate_density`` returns a finite scalar.
2. **Log-rate algebra parity** — for known θ*, the MCResponseJax
   ``log_rate_density`` agrees with the MCEvaluatorJax
   ``log_rate_density`` minus ``2 · log(n_init)`` (the density-
   convention rescaling explicit in the code).
3. **AD-through-mu_natal end-to-end** — gradient of
   ``log_rate_density`` w.r.t. ``mu_natal`` is finite and matches FD.
   This is the load-bearing M4 SVI gradient: confirms the chain
   J1 (sample_natals_jax) → J2 (particle_mc_jax) → J3 (KDE) → J4
   (log_rate_density) carries gradients through θ_pop.
4. **AD-through-log_R_env** — gradient of ``log_rate_density`` w.r.t.
   ``log_R_env`` is exactly 1.0 (additive shift).
5. **env is ignored** — passing different ``env["v_esc"]`` values
   returns the same log_rate_density (v_esc was baked into the
   cascade at factory time, matching numpy MCResponse).

Workstation-tier integration (factory + solver dispatch + SVI) is
in ``tests/test_integration_stage1_recovery_mc_jax.py`` (M4).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _stub_gwkokab_if_absent() -> None:
    """The full ``response/__init__.py`` chain transitively imports
    gwkokab via the natal / basis modules.  This sandbox file only
    exercises ``MCResponseJax``, which doesn't need gwkokab at all —
    stub the missing modules so the load completes.  No-op when
    gwkokab is available (workstation)."""
    try:
        import gwkokab  # noqa: F401
        return
    except ImportError:
        pass
    for name in (
        "gwkokab", "gwkokab.models", "gwkokab.models.utils",
        "gwkokab.models.redshift", "gwkokab.utils", "gwkokab.utils.kernel",
    ):
        mod = types.ModuleType(name)
        mod.__path__ = []  # make it a package
        sys.modules[name] = mod

    def _stub(*a, **k):
        return 0.0

    sys.modules["gwkokab.models.utils"].doubly_truncated_power_law_log_norm_constant = _stub
    sys.modules["gwkokab.utils.kernel"].log_planck_taper_window = _stub
    sys.modules["gwkokab.models.redshift"].MadauDickinsonRedshift = type("MadauDickinsonRedshift", (), {})
    sys.modules["gwkokab.models.redshift"].PowerlawRedshift = type("PowerlawRedshift", (), {})


_stub_gwkokab_if_absent()

# Pre-populate the package + subpackage namespace with empty modules so
# the response/__init__.py side-effects (which transitively import
# gwkokab) don't run when we load mc_jax.py as a submodule.  The
# relative imports inside mc_jax.py (``from ..evaluation.mc_jax``,
# ``from ..kernels``, ``from ..spin_orientation``) resolve against the
# pre-populated namespace + actual sibling modules.
_PKG = "hierarchical_ad_pop_engine"
if f"{_PKG}.response" not in sys.modules:
    _resp_pkg = types.ModuleType(f"{_PKG}.response")
    _resp_pkg.__path__ = [str(_REPO / "src" / _PKG / "response")]
    sys.modules[f"{_PKG}.response"] = _resp_pkg
    # The parent package needs to exist too — but it has no gwkokab dep
    # at __init__ time, so a normal import is fine.
    import hierarchical_ad_pop_engine  # noqa: F401
    setattr(sys.modules[_PKG], "response", _resp_pkg)

from hierarchical_ad_pop_engine.evaluation.mc_jax import MCEvaluatorJax  # noqa: E402
_MC_JAX_PATH = (
    _REPO / "src" / _PKG / "response" / "mc_jax.py"
)
_spec = importlib.util.spec_from_file_location(
    f"{_PKG}.response.mc_jax", str(_MC_JAX_PATH),
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[f"{_PKG}.response.mc_jax"] = _mod
_spec.loader.exec_module(_mod)
MCResponseJax = _mod.MCResponseJax

from red_team.backup_particle_mc import MCConfig, particle_mc_jax  # noqa: E402


THETA = dict(
    mu_natal=8.0, sigma_natal=1.0, m_min=4.0, m_max=80.0,
    natal_spin_loc=0.05, natal_spin_scale=0.10,
    kernel_a=0.0, kernel_b=1.0, v_esc=1000.0,
)
N_SEED = 2000


def _build_response(cfg, key, log_R_env):
    """Mirror of build_mc_jax_response's body — no CoagulationResponse
    dependency so this runs in the sandbox without gwkokab.
    """
    out = particle_mc_jax(cfg, key)
    m_active = out["m_active"]
    chi_active = out["chi_active"]
    w_active = out["w_active"]
    active_mask = out["active_mask"].astype(m_active.dtype)
    masked_weights = w_active * active_mask
    evaluator = MCEvaluatorJax.from_run(
        m_active,
        chi_active,
        out["mergers"],
        log_R_env=jnp.asarray(log_R_env),
        kernel_params={"a": jnp.asarray(cfg.kernel_a),
                       "b": jnp.asarray(cfg.kernel_b)},
        weights=masked_weights,
    )
    return MCResponseJax(evaluator=evaluator, n_init=int(cfg.n_seed))


# ---------------------------------------------------------------------------
# INV-1: smoke
# ---------------------------------------------------------------------------
def test_smoke_constructs_returns_finite_log_rate():
    cfg = MCConfig(n_seed=N_SEED, depletion_fraction=0.10, **THETA)
    r = _build_response(cfg, jax.random.PRNGKey(2026), log_R_env=0.0)
    assert r.n_init == N_SEED
    lrd = r.log_rate_density(
        jnp.asarray(8.0), jnp.asarray(12.0),
        jnp.asarray(0.1), jnp.asarray(0.2),
        {"v_esc": jnp.asarray(1000.0)},
    )
    print(f"\n[INV-1] log_rate_density at sample point = {float(lrd):.4f}")
    assert jnp.isfinite(lrd)


# ---------------------------------------------------------------------------
# INV-2: log_rate algebra parity — diff from evaluator.log_rate_density
# is exactly -2 · log(n_init)
# ---------------------------------------------------------------------------
def test_log_rate_density_matches_evaluator_minus_log_n_init_squared():
    cfg = MCConfig(n_seed=N_SEED, depletion_fraction=0.10, **THETA)
    r = _build_response(cfg, jax.random.PRNGKey(2026), log_R_env=0.0)
    args = (jnp.asarray(8.0), jnp.asarray(12.0),
            jnp.asarray(0.1), jnp.asarray(0.2))
    response_lrd = r.log_rate_density(*args, {"v_esc": jnp.asarray(1000.0)})
    evaluator_lrd = r.evaluator.log_rate_density(*args)
    diff = float(response_lrd - evaluator_lrd)
    expected = -2.0 * float(jnp.log(jnp.asarray(N_SEED)))
    print(f"\n[INV-2] response - evaluator = {diff:.4f}, "
          f"expected -2·log(n_init) = {expected:.4f}")
    np.testing.assert_allclose(diff, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# INV-3: AD through mu_natal — load-bearing M4 SVI gradient
# ---------------------------------------------------------------------------
def test_grad_through_mu_natal_matches_fd():
    key = jax.random.PRNGKey(2026)
    cfg_base = MCConfig(n_seed=N_SEED, depletion_fraction=0.10, **THETA)

    def f(mu):
        cfg = replace(cfg_base, mu_natal=mu)
        r = _build_response(cfg, key, log_R_env=0.0)
        return r.log_rate_density(
            jnp.asarray(8.0), jnp.asarray(12.0),
            jnp.asarray(0.1), jnp.asarray(0.2),
            {"v_esc": jnp.asarray(1000.0)},
        )

    mu_ref = jnp.asarray(THETA["mu_natal"])
    g_ad = float(jax.grad(f)(mu_ref))
    h = 1e-2
    g_fd = float((f(mu_ref + h) - f(mu_ref - h)) / (2.0 * h))
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-9)
    print(f"\n[INV-3] AD ∂log_rate / ∂mu_natal: AD = {g_ad:.4e}, FD = {g_fd:.4e}, rel = {rel:.3e}")
    assert jnp.isfinite(g_ad)
    assert abs(g_ad) > 0.0
    # Loose tolerance — log_rate_density is the composition of several
    # finite-difference-noisy operations (cascade + KDE + log); 10% is
    # comfortable above MC SE.
    assert rel < 0.10, f"AD-vs-FD rel = {rel:.3e}"


# ---------------------------------------------------------------------------
# INV-4: AD through log_R_env — trivially additive
# ---------------------------------------------------------------------------
def test_grad_through_log_R_env_is_one():
    key = jax.random.PRNGKey(2026)
    cfg_base = MCConfig(n_seed=N_SEED, depletion_fraction=0.10, **THETA)

    def f(log_R):
        r = _build_response(cfg_base, key, log_R_env=log_R)
        return r.log_rate_density(
            jnp.asarray(8.0), jnp.asarray(12.0),
            jnp.asarray(0.1), jnp.asarray(0.2),
            {"v_esc": jnp.asarray(1000.0)},
        )

    log_R_ref = jnp.asarray(0.0)
    g = float(jax.grad(f)(log_R_ref))
    print(f"\n[INV-4] ∂log_rate / ∂log_R_env = {g:.6f}  (expect 1.000000)")
    np.testing.assert_allclose(g, 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# INV-5: env is ignored — v_esc was baked at factory time
# ---------------------------------------------------------------------------
def test_env_ignored():
    cfg = MCConfig(n_seed=N_SEED, depletion_fraction=0.10, **THETA)
    r = _build_response(cfg, jax.random.PRNGKey(2026), log_R_env=0.0)
    args = (jnp.asarray(8.0), jnp.asarray(12.0),
            jnp.asarray(0.1), jnp.asarray(0.2))
    a = float(r.log_rate_density(*args, {"v_esc": jnp.asarray(500.0)}))
    b = float(r.log_rate_density(*args, {"v_esc": jnp.asarray(1500.0)}))
    np.testing.assert_allclose(a, b, rtol=1e-9)
    print(f"\n[INV-5] env['v_esc'] ignored: lrd(v_esc=500)={a:.4f}, "
          f"lrd(v_esc=1500)={b:.4f}")
