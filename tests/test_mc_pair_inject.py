r"""Exact-particle MC injection sampler — PLAN_cross_method_injection_recovery
Step 1.  Validates ``demo/fit_stage1_gwkokab.pair_sample_from_mc``: drawing
merger pairs from the surviving MC particle pool ∝ w_i·w_j·K(m_i,m_j), with no
grid and no jitter (the injection-side fix for cross-method recovery).

Workstation tier (needs jax). Uses a DUCK-TYPED evaluator — only the attributes
the sampler reads (``m_active, chi_active, weights, kernel_a, kernel_b,
mergers``) — so it runs without the engine/gwkokab stack.
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

# pair_sample_from_mc imports _log_pairing_kernel_jax from the (jax-only)
# evaluation.mc_jax module; skip cleanly if that stack isn't importable here.
fit = pytest.importorskip("fit_stage1_gwkokab")


class _FakeEvaluator:
    """Minimal stand-in exposing only what pair_sample_from_mc reads."""

    def __init__(self, m, chi, w, a, b):
        self.m_active = jnp.asarray(m)
        self.chi_active = jnp.asarray(chi)
        self.weights = jnp.asarray(w)
        self.kernel_a = jnp.asarray(a)
        self.kernel_b = jnp.asarray(b)
        # (n_mergers, 8) — only .shape[0] is read by the harness print.
        self.mergers = jnp.zeros((3, 8))


def _kernel(m1, m2, a, b):
    big = max(m1, m2)
    sml = min(m1, m2)
    return (m1 + m2) ** a * (sml / big) ** b


def test_draws_are_exact_particles_ordered_no_self_pairs():
    # distinct masses ⇒ a self-pair would show as m1==m2, so it's checkable.
    m = [8.0, 12.0, 20.0, 33.0]
    chi = [0.02, 0.05, 0.70, 0.66]
    w = [1.0, 1.0, 1.0, 1.0]
    ev = _FakeEvaluator(m, chi, w, a=0.0, b=2.0)
    ev_chi_of_m = dict(zip(m, chi))

    pts = np.asarray(fit.pair_sample_from_mc(ev, 4000, jax.random.PRNGKey(0)))
    m1, m2, c1, c2 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    mset = set(m)
    assert set(np.round(m1, 6)).issubset({round(x, 6) for x in mset})
    assert set(np.round(m2, 6)).issubset({round(x, 6) for x in mset})
    assert np.all(m1 >= m2)                       # ordering
    assert np.all(m1 != m2)                       # distinct masses ⇒ no self-pair
    # spins travel with their particle (the sort carries χ correctly)
    for mm, cc in zip(m1, c1):
        assert abs(cc - ev_chi_of_m[round(float(mm), 6)]) < 1e-9
    print(f"\n[P1] {len(pts)} exact-particle pairs: in-pool, m1≥m2, no self-pairs")


def test_pair_frequency_matches_wwK():
    r"""Empirical unordered-pair frequency ∝ w_i w_j K(m_i,m_j)."""
    m = [8.0, 12.0, 20.0, 33.0]
    chi = [0.02, 0.05, 0.70, 0.66]
    w = [0.5, 1.0, 2.0, 1.5]
    a, b = 1.0, 1.5
    ev = _FakeEvaluator(m, chi, w, a=a, b=b)

    n = len(m)
    # analytic unordered-pair weights p_{ij} ∝ w_i w_j K, i<j
    pairs, pw = [], []
    for ii in range(n):
        for jj in range(ii + 1, n):
            pairs.append((ii, jj))
            pw.append(w[ii] * w[jj] * _kernel(m[ii], m[jj], a, b))
    pw = np.asarray(pw) / np.sum(pw)

    N = 60000
    pts = np.asarray(fit.pair_sample_from_mc(ev, N, jax.random.PRNGKey(1)))
    # map each event to its unordered mass-pair index
    m_to_i = {round(x, 6): k for k, x in enumerate(m)}
    emp = np.zeros(len(pairs))
    pidx = {tuple(sorted(p)): k for k, p in enumerate(pairs)}
    for m1, m2 in zip(pts[:, 0], pts[:, 1]):
        i_, j_ = m_to_i[round(float(m1), 6)], m_to_i[round(float(m2), 6)]
        emp[pidx[tuple(sorted((i_, j_)))]] += 1
    emp /= emp.sum()

    # 1/sqrt(N) sampling noise ⇒ generous abs tol
    np.testing.assert_allclose(emp, pw, atol=0.02)
    print(f"\n[P2] empirical pair freq matches ∝ w·w·K: {np.round(emp,3)} "
          f"vs {np.round(pw,3)}")
