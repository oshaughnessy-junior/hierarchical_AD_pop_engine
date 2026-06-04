"""Soft convergence gate for the χ-resolution sweep (engine tier).

Asserts the *ordering* invariant the paper's resolution summary relies on:
the continuous rel-L1 truncation measure (vs the highest-resolution
benchmark) strictly decreases as ``n_chi`` is refined, for both marginals.
No frozen reference values — resolutions and physics are tiny, the gate is
comparative only.  The quantitative table for the paper comes from
``demo/resolution_sweep_chi.py`` at paper resolution (see the paper repo's
Makefile target).

Engine tier: needs the implicit solver (→ gwkokab); skips when that import
is unavailable, same convention as ``test_fisher_bridge_clustered.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

try:
    from hierarchical_ad_pop_engine.response.implicit import (  # noqa: F401
        solve_implicit_fixed_point,
    )
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"engine (implicit solver / gwkokab) unavailable: {exc!r}",
        allow_module_level=True,
    )

_DEMO_DIR = (Path(__file__).resolve().parent.parent / "demo").resolve()
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))
import resolution_sweep_chi as _rsc  # noqa: E402


def _args(tmp_path, **over):
    base = dict(
        m_min=4.0, m_max=80.0, mu=8.0, sigma=1.0, natal_spin_scale=0.05,
        kernel_b=2.0, epsilon=0.6, v_esc=300.0, kappa=0.0,
        n_theta=3, n_dphi=6, max_iter=80, max_gap_factor=2.0,
        n_chi_list="8,12,20", bench_n_m=16,
        n_m_list="10,14,20", bench_n_chi=12,
        n_fine=801, render="pchip-log",
        cache_dir=str(tmp_path / "cache"), outdir=str(tmp_path / "out"),
        solve_only=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


@pytest.mark.slow
def test_rel_l1_decreases_with_n_chi(tmp_path):
    results = _rsc.run_sweep(_args(tmp_path))
    sweep = results["chi_sweep"]
    # benchmark row (last) is 0 by construction; compare the others
    rel_chi = [r["rel_chi"] for r in sweep[:-1]]
    rel_m = [r["rel_m"] for r in sweep[:-1]]
    assert all(r > 0 for r in rel_chi + rel_m)
    assert rel_chi[0] > rel_chi[-1], "dN/dchi truncation must shrink with n_chi"
    assert rel_m[0] > rel_m[-1], "dN/dm truncation must shrink with n_chi"
    # strict monotone ordering across the (well-separated) sweep points
    assert rel_chi == sorted(rel_chi, reverse=True)


@pytest.mark.slow
def test_rel_l1_decreases_with_n_m(tmp_path):
    results = _rsc.run_sweep(_args(tmp_path))
    sweep = results["m_sweep"]
    rel_m = [r["rel_m"] for r in sweep[:-1]]
    assert all(r > 0 for r in rel_m)
    assert rel_m == sorted(rel_m, reverse=True)


@pytest.mark.slow
def test_render_kinds_agree_at_benchmark(tmp_path):
    """hat and pchip-log reconstructions of the *same* solve agree in
    rel-L1 to a few percent — the rendering choice is not doing physics."""
    import numpy as np

    args = _args(tmp_path)
    sol = _rsc.solve_marginals(16, 20, args)
    chi_fine = np.linspace(1e-3, 1.0 - 1e-3, 801)
    a = _rsc.render_marginal(sol["chi"], sol["dN_dchi"], chi_fine, "hat")
    b = _rsc.render_marginal(sol["chi"], sol["dN_dchi"], chi_fine, "pchip-log")
    assert _rsc.rel_l1(a, b, chi_fine) < 0.10
