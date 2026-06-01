r"""H1 cross-method gate — PLAN_cross_method_injection_recovery Step 3.

Asserts that the IMPLICIT recovery density and the MC injection density agree on
the population SHAPE (``dN/dm``, ``dN/dχ``) at the injection θ and across a small
θ-ball, at the depletion-matched ε.  This is the guard that makes inject-MC /
recover-implicit trustworthy: if the two methods disagree on the shape at θ*, the
recovered posterior is a cross-method systematic, not a measurement — so this
must run (and hold) BEFORE any cross-method fit is believed.

The metric is Jensen–Shannon divergence (nats, bounded by ln 2 ≈ 0.693) on the
renormalised marginals, so density-convention constants cancel and only genuine
shape disagreement (the kernel_b / σ residuals) registers.

Full-stack tier: needs the engine (→ gwkokab → RIFT/lalsuite) and jax; SKIPs in
the lightweight sandbox, runs on the workstation.  The threshold is intentionally
LOOSE (a regression guard + diagnostic, printed every run); tightening it tracks
resolving the residual systematics, and that tightening must be backed by a
convergence argument, not vibes (`feedback_no_fudge_factors`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parent.parent
for p in (str(_REPO), str(_REPO / "src"), str(_REPO / "demo")):
    if p not in sys.path:
        sys.path.insert(0, p)

fit = pytest.importorskip("fit_stage1_gwkokab")

_ENV_IMPORT_ERRORS = (ImportError, ModuleNotFoundError, OSError)

# Generous diagnostic bound: ~7× the same-physics compare_all scale (≈0.007).
# A cross-method shape disagreement above this is a real systematic to chase.
_JS_GATE = 0.10

_M_MIN, _M_MAX, _V_ESC, _KA, _KMAX, _NSEED = 4.0, 80.0, 1000.0, 0.0, 3, 400
_INJECT_DEPLETION = 0.10

_THETA_STAR = dict(
    mu_natal=8.0, sigma_natal=1.0, natal_spin_loc=0.05, natal_spin_scale=0.10,
    kernel_b=1.0, log_rate=0.0,
)
# small θ-ball around the injection point (H1 requires agreement on a
# neighbourhood, not just one fiducial).
_THETA_BALL = [
    {},                                   # θ*
    {"mu_natal": 9.0},
    {"sigma_natal": 1.3},
    {"kernel_b": 1.4},
]


def _build_responses(theta):
    """Return (impl_response, mc_evaluator, grid, ε*) at θ, or skip on env error."""
    try:
        from hierarchical_ad_pop_engine.grids import BinaryGrid
        from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (
            make_coagulation_dist_fn,
        )
        from hierarchical_ad_pop_engine.response.implicit import (
            solve_implicit_fixed_point,
        )

        grid = BinaryGrid.default(m_min=_M_MIN, m_max=_M_MAX, n_m=64, n_chi=20)
        th = {**_THETA_STAR, **theta}

        # MC injection at θ → evaluator + achieved depletion.
        mc_dist = make_coagulation_dist_fn(
            v_esc=_V_ESC, m_min=_M_MIN, m_max=_M_MAX, kernel_a=_KA, k_max=_KMAX,
            n_seed=_NSEED, depletion_fraction=_INJECT_DEPLETION, gate="soft",
            grid=grid, mc_jax_key=jax.random.PRNGKey(0), solver="mc_jax")
        mc_resp = mc_dist(**th)._response.mc_jax_response
        ev = mc_resp.evaluator
        n_init = int(getattr(mc_resp, "n_init", _NSEED))
        W = float(jnp.sum(jnp.asarray(ev.weights)))
        f_target = max(0.0, min(0.999, 1.0 - W / max(n_init, 1)))

        # implicit response at θ + depletion-matched ε.
        impl_resp = make_coagulation_dist_fn(
            v_esc=_V_ESC, m_min=_M_MIN, m_max=_M_MAX, kernel_a=_KA, k_max=_KMAX,
            n_seed=_NSEED, grid=grid, solver="implicit", epsilon=1.0)(**th)._response
        v = jnp.asarray(_V_ESC)
        eps_star, _f, status = fit._match_epsilon_to_fdep(
            lambda e: float(
                solve_implicit_fixed_point(impl_resp, v, epsilon=float(e)).f_natal_depleted),
            f_target, lo=1e-3, hi=50.0)
        return impl_resp, ev, grid, eps_star, status
    except _ENV_IMPORT_ERRORS as exc:
        pytest.skip(f"engine stack not importable here (environment): "
                    f"{type(exc).__name__}: {exc}")


@pytest.mark.parametrize("theta", _THETA_BALL,
                         ids=["star", "mu+1", "sig+0.3", "kb+0.4"])
def test_cross_method_js_below_gate(theta):
    impl_resp, ev, grid, eps_star, status = _build_responses(theta)
    assert status == "ok", f"ε-match failed at θ={theta} (status={status})"
    js_m, js_c, js_2d = fit.cross_method_js(
        impl_resp, ev, grid, _V_ESC, epsilon=eps_star)
    print(f"\n[H1] θ+{theta or 'star'}: ε*={eps_star:.4g}  "
          f"JS dN/dm={js_m:.4f}  dN/dχ={js_c:.4f}  N(m,χ)2D={js_2d:.4f} nats")
    assert np.isfinite(js_m) and np.isfinite(js_c) and np.isfinite(js_2d)
    assert js_m < _JS_GATE, f"dN/dm JS {js_m:.4f} ≥ gate {_JS_GATE} at θ={theta}"
    assert js_c < _JS_GATE, f"dN/dχ JS {js_c:.4f} ≥ gate {_JS_GATE} at θ={theta}"
    # 2D JS is the structural metric; gate it a bit looser (it sees the
    # correlation the 1D marginals miss).  Printed regardless for diagnosis.
    assert js_2d < 1.5 * _JS_GATE, \
        f"N(m,χ) 2D JS {js_2d:.4f} ≥ gate {1.5 * _JS_GATE} at θ={theta}"
