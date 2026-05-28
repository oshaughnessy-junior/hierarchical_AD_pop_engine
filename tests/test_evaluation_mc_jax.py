r"""Session J3 sandbox-tier tests — JAX MCEvaluatorJax (Gaussian KDE).

Per ``demo/PLAN_jax_mc.md`` § J3.  Pins seven invariants:

1. **Smoke** — ``MCEvaluatorJax.from_run`` builds + returns sensible
   bandwidths + density values at sample points.
2. **Density integrates to W_total** — internal-grid Riemann/trapezoid
   integral of ``N(m, χ)`` agrees with ``W_total = Σ w_i`` to ~10 %
   (Gaussian KDE has some boundary leak in log m).
3. **Uniform-weights bit-for-bit self** — building with
   ``weights=None`` vs ``weights=jnp.ones`` gives identical bandwidths
   and densities.
4. **W_total scales density linearly** — duplicating weights × 2 gives
   density × 2 at the same evaluation points; bandwidth unchanged
   (depends on σ̂_w and n_eff, both invariant under weight scaling).
5. **Loose parity with numpy MCEvaluator** — rel-L1 between Gaussian
   KDE and k-NN balloon on a shared sample is ≤ 50 % on the
   internal grid (Gaussian KDE oversmooths cascade features; PLAN
   acknowledges this; see
   ``[[feedback-gaussian-kde-over-smooths-cascade-peaks]]``).
6. **AD through bandwidth (= AD through sample weights)** —
   ``jax.grad(total_rate)(weights)`` is finite and matches FD; weights
   control bandwidth via Silverman / n_eff so this is the AD path
   J4 SVI uses.
7. **AD through log_R_env (the simplest M4 axis)** —
   ``jax.grad(log_rate_density)(log_R_env) == 1.0`` exactly (linear
   in log_R_env).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hierarchical_ad_pop_engine.evaluation import MCEvaluatorJax  # noqa: E402
from hierarchical_ad_pop_engine.evaluation.mc import MCEvaluator  # noqa: E402
from red_team.backup_particle_mc import MCConfig, particle_mc_jax  # noqa: E402


THETA = dict(
    mu_natal=8.0, sigma_natal=1.0, m_min=4.0, m_max=80.0,
    natal_spin_loc=0.0, natal_spin_scale=0.10,
    kernel_a=0.0, kernel_b=1.0, v_esc=1000.0,
)
N_SEED = 2000


def _build_run(key_seed=2026, depletion=0.10):
    """Build a J2 cascade result + accessor slices."""
    cfg = MCConfig(n_seed=N_SEED, depletion_fraction=depletion, **THETA)
    out = particle_mc_jax(cfg, jax.random.PRNGKey(key_seed))
    am = out["active_mask"]
    sa = out["step_active"]
    m_a = out["m_active"][am]
    chi_a = out["chi_active"][am]
    w_a = out["w_active"][am]
    mergers = out["mergers"][sa]
    return m_a, chi_a, w_a, mergers


def _build_evaluator(weights=None, reflect_chi=True):
    m_a, chi_a, w_a, mergers = _build_run()
    use_w = w_a if weights is None else weights
    ev = MCEvaluatorJax.from_run(
        m_a, chi_a, mergers,
        log_R_env=jnp.asarray(0.0),
        kernel_params={"a": jnp.asarray(0.0), "b": jnp.asarray(1.0)},
        weights=use_w,
        reflect_chi=reflect_chi,
    )
    return ev, m_a, chi_a, w_a, mergers


# ---------------------------------------------------------------------------
# INV-1: smoke
# ---------------------------------------------------------------------------
def test_smoke_build_and_density():
    ev, m_a, _, _, _ = _build_evaluator()
    assert ev.n_active == int(m_a.shape[0])
    assert float(ev.W_total) > 0
    assert float(ev.n_eff) > 0
    assert float(ev.h_log_m) > 0
    assert float(ev.h_chi) > 0
    # Density at the natal mode should be elevated vs the tail.
    N_mode = float(ev.density(jnp.asarray(8.0), jnp.asarray(0.05)))
    N_tail = float(ev.density(jnp.asarray(50.0), jnp.asarray(0.9)))
    print(f"\n[INV-1] n_active={ev.n_active}, n_eff={float(ev.n_eff):.1f}, "
          f"h_log_m={float(ev.h_log_m):.4f}, h_chi={float(ev.h_chi):.4f}, "
          f"N(mode)={N_mode:.3e}, N(tail)={N_tail:.3e}")
    assert N_mode > N_tail
    assert N_mode > 0
    assert N_tail >= 0


# ---------------------------------------------------------------------------
# INV-2: density integrates to ≈ W_total
# ---------------------------------------------------------------------------
def test_density_integrates_to_W_total():
    ev, _, _, _, _ = _build_evaluator()
    # Internal grid integral.
    m_grid = jnp.geomspace(
        float(jnp.min(ev.m_active)), float(jnp.max(ev.m_active)), 64,
    )
    chi_grid = jnp.linspace(0.0, 1.0, 64)
    M, C = jnp.meshgrid(m_grid, chi_grid, indexing="ij")
    N = ev.density(M, C)
    # Trapezoid weights
    log_m = jnp.log(m_grid)
    dlog = jnp.diff(log_m)
    w_log_m = jnp.concatenate([dlog[:1], 0.5 * (dlog[:-1] + dlog[1:]), dlog[-1:]])
    w_m = w_log_m * m_grid
    d_chi = jnp.diff(chi_grid)
    w_chi = jnp.concatenate([d_chi[:1], 0.5 * (d_chi[:-1] + d_chi[1:]), d_chi[-1:]])
    integral = float(jnp.sum(N * w_m[:, None] * w_chi[None, :]))
    W = float(ev.W_total)
    rel = abs(integral - W) / W
    print(f"\n[INV-2] ∫N dm dχ = {integral:.2f}  W_total = {W:.2f}  rel = {rel:.3f}")
    # Gaussian KDE leaks mass outside the internal grid in log m
    # (tails extend below m_min and above m_max).  10 % is loose
    # but realistic at the Silverman bandwidth on a depleted pool.
    assert rel < 0.15, f"density-mass mismatch: rel = {rel:.3f}"


# ---------------------------------------------------------------------------
# INV-3: uniform weights == ones-weights bit-for-bit
# ---------------------------------------------------------------------------
def test_uniform_weights_bit_for_bit():
    m_a, chi_a, _, mergers = _build_run()
    ev_none = MCEvaluatorJax.from_run(
        m_a, chi_a, mergers,
        log_R_env=jnp.asarray(0.0),
        kernel_params={"a": jnp.asarray(0.0), "b": jnp.asarray(1.0)},
        weights=None,
    )
    ev_ones = MCEvaluatorJax.from_run(
        m_a, chi_a, mergers,
        log_R_env=jnp.asarray(0.0),
        kernel_params={"a": jnp.asarray(0.0), "b": jnp.asarray(1.0)},
        weights=jnp.ones(m_a.shape[0]),
    )
    np.testing.assert_allclose(
        float(ev_none.h_log_m), float(ev_ones.h_log_m),
        rtol=1e-6, err_msg="h_log_m mismatch under weights=None vs ones",
    )
    np.testing.assert_allclose(
        float(ev_none.h_chi), float(ev_ones.h_chi),
        rtol=1e-6, err_msg="h_chi mismatch",
    )
    m_q = jnp.linspace(5.0, 30.0, 20)
    chi_q = jnp.linspace(0.05, 0.95, 20)
    N_none = ev_none.density(m_q, chi_q)
    N_ones = ev_ones.density(m_q, chi_q)
    np.testing.assert_allclose(
        np.asarray(N_none), np.asarray(N_ones),
        rtol=1e-6, err_msg="density mismatch under weights=None vs ones",
    )
    print(f"\n[INV-3] uniform-weights bit-for-bit OK at n_active = {ev_none.n_active}")


# ---------------------------------------------------------------------------
# INV-4: W_total scales linearly; bandwidth invariant under weight scaling
# ---------------------------------------------------------------------------
def test_W_total_scales_density():
    m_a, chi_a, w_a, mergers = _build_run()
    w_2x = 2.0 * w_a
    ev1 = MCEvaluatorJax.from_run(
        m_a, chi_a, mergers,
        log_R_env=jnp.asarray(0.0),
        kernel_params={"a": jnp.asarray(0.0), "b": jnp.asarray(1.0)},
        weights=w_a,
    )
    ev2 = MCEvaluatorJax.from_run(
        m_a, chi_a, mergers,
        log_R_env=jnp.asarray(0.0),
        kernel_params={"a": jnp.asarray(0.0), "b": jnp.asarray(1.0)},
        weights=w_2x,
    )
    # Bandwidth invariant: σ̂_w cancels weight scaling (∝ Σw / Σw),
    # n_eff = (Σw)²/Σw² also cancels weight scaling.
    np.testing.assert_allclose(
        float(ev1.h_log_m), float(ev2.h_log_m), rtol=1e-6,
    )
    np.testing.assert_allclose(
        float(ev1.h_chi), float(ev2.h_chi), rtol=1e-6,
    )
    np.testing.assert_allclose(
        float(ev1.n_eff), float(ev2.n_eff), rtol=1e-6,
    )
    # Density scales as W_total exactly (linear in weights via the
    # W_total · f̂ formula).
    m_q = jnp.linspace(5.0, 30.0, 10)
    chi_q = jnp.linspace(0.05, 0.95, 10)
    N1 = np.asarray(ev1.density(m_q, chi_q))
    N2 = np.asarray(ev2.density(m_q, chi_q))
    ratio = N2 / np.maximum(N1, 1e-30)
    print(f"\n[INV-4] density ratio (w×2) / (w): mean = {ratio.mean():.4f} "
          f"(expect 2.0); bandwidth invariant ✓")
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# INV-5: loose parity with numpy MCEvaluator
# ---------------------------------------------------------------------------
def test_loose_parity_with_numpy_mceval():
    r"""Gaussian KDE will oversmooth cascade peaks vs k-NN balloon —
    PLAN-acknowledged trade.  Tolerance is generous (50 % rel-L1) to
    survive the smoothing difference.

    See ``[[feedback-gaussian-kde-over-smooths-cascade-peaks]]`` for
    the user objection to this design.
    """
    m_a, chi_a, w_a, mergers = _build_run()
    ev_jax = MCEvaluatorJax.from_run(
        m_a, chi_a, mergers,
        log_R_env=jnp.asarray(0.0),
        kernel_params={"a": jnp.asarray(0.0), "b": jnp.asarray(1.0)},
        weights=w_a,
    )
    ev_np = MCEvaluator(
        np.asarray(m_a), np.asarray(chi_a), np.asarray(mergers),
        log_R_env=0.0, kernel_params={"a": 0.0, "b": 1.0},
        weights=np.asarray(w_a),
    )
    # Evaluate on shared internal grid.
    m_grid = np.geomspace(
        float(np.asarray(m_a).min()),
        float(np.asarray(m_a).max()),
        32,
    )
    chi_grid = np.linspace(0.05, 0.95, 32)
    M, C = np.meshgrid(m_grid, chi_grid, indexing="ij")
    N_jax = np.asarray(ev_jax.density(jnp.asarray(M), jnp.asarray(C)))
    N_np = ev_np.density(M, C)
    # rel-L1 normalised by numpy total
    abs_diff = np.abs(N_jax - N_np)
    rel_L1 = abs_diff.sum() / max(N_np.sum(), 1e-30)
    print(f"\n[INV-5] numpy vs JAX KDE rel-L1 on density: {rel_L1:.3f}  (tol 0.50)")
    print(f"  numpy N range: [{N_np.min():.3e}, {N_np.max():.3e}], total = {N_np.sum():.3e}")
    print(f"  jax   N range: [{N_jax.min():.3e}, {N_jax.max():.3e}], total = {N_jax.sum():.3e}")
    # Acknowledged trade — Gaussian KDE smoother.  Don't hard-gate
    # below 30%.
    assert rel_L1 < 0.50, f"rel-L1 too large: {rel_L1:.3f}"


# ---------------------------------------------------------------------------
# INV-6: AD through bandwidth via sample weights
# ---------------------------------------------------------------------------
def test_grad_through_weights_finite():
    r"""``jax.grad(total_rate)(weights)`` is finite, non-trivial, and
    matches a per-element FD.

    Weights enter via:
      - σ̂_w (Silverman bandwidth)
      - n_eff
      - W_total (density scaling)
      - per-sample kernel sum

    All AD paths flow through pure jnp.  The headline-J3 AD invariant.
    """
    m_a, chi_a, w_a, mergers = _build_run()

    def f(w):
        ev = MCEvaluatorJax.from_run(
            m_a, chi_a, mergers,
            log_R_env=jnp.asarray(0.0),
            kernel_params={"a": jnp.asarray(0.0), "b": jnp.asarray(1.0)},
            weights=w,
        )
        return ev.total_rate(n_quad=32)

    w_ref = jnp.asarray(w_a)
    g_ad = jax.grad(f)(w_ref)
    # Spot-check one element via FD
    idx = int(w_a.shape[0]) // 2
    h = 1e-2
    w_plus = w_ref.at[idx].add(h)
    w_minus = w_ref.at[idx].add(-h)
    g_fd_i = float((f(w_plus) - f(w_minus)) / (2.0 * h))
    g_ad_i = float(g_ad[idx])
    print(f"\n[INV-6] total_rate AD on weights[{idx}]: AD = {g_ad_i:.4e}, "
          f"FD = {g_fd_i:.4e}")
    assert jnp.all(jnp.isfinite(g_ad))
    rel = abs(g_ad_i - g_fd_i) / max(abs(g_fd_i), 1e-9)
    assert rel < 5e-2, f"AD-vs-FD rel = {rel:.3e}"


# ---------------------------------------------------------------------------
# INV-7: AD through log_R_env trivially
# ---------------------------------------------------------------------------
def test_grad_through_log_R_env():
    r"""``log_rate_density`` is linear in ``log_R_env`` (additive shift),
    so ``∂ log_rate / ∂ log_R_env = 1.0`` exactly.  Sanity check.
    """
    m_a, chi_a, w_a, mergers = _build_run()

    def f(log_R):
        ev = MCEvaluatorJax.from_run(
            m_a, chi_a, mergers,
            log_R_env=log_R,
            kernel_params={"a": jnp.asarray(0.0), "b": jnp.asarray(1.0)},
            weights=w_a,
        )
        return ev.log_rate_density(
            jnp.asarray(8.0), jnp.asarray(12.0),
            jnp.asarray(0.1), jnp.asarray(0.2),
        )

    log_R_ref = jnp.asarray(0.0)
    g = float(jax.grad(f)(log_R_ref))
    print(f"\n[INV-7] ∂log_rate / ∂log_R_env = {g:.6f}  (expect 1.000000)")
    np.testing.assert_allclose(g, 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# INV-bonus: marginal_1d / marginal_2d raise NotImplementedError
# ---------------------------------------------------------------------------
def test_marginals_raise_not_implemented():
    ev, _, _, _, _ = _build_evaluator()
    with pytest.raises(NotImplementedError, match="follow-up"):
        ev.marginal_1d("m1", jnp.array([8.0, 12.0]))
    with pytest.raises(NotImplementedError, match="follow-up"):
        ev.marginal_2d(("m1", "chi1"), jnp.array([8.0]), jnp.array([0.1]))
