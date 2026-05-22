r"""Standalone smoke test for the observables module.

Runs without ``gwkokab`` (uses a synthetic Gaussian-in-log-m × beta-like
single-BH density ``N(m, χ)`` rather than the production natal call),
so it works in environments where the engine's gwkokab-backed natal
imports are unavailable.  The pytest suite in
``tests/test_observables.py`` uses the production
``build_default_coagulation`` — run that on a workstation with the
full dependency stack for the canonical regression check.

This script is the one to run **inside the sandbox** (or any minimal
JAX install) to confirm that an observables refactor hasn't broken
the marginal arithmetic.  All printed relative diffs should be at
machine precision (``≤ 1e-12``) and the AD vs finite-difference check
should agree to ``≤ 1e-10``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from hierarchical_ad_pop_engine import observables as obs
from hierarchical_ad_pop_engine.grids import BinaryGrid


def synthetic_density(grid: BinaryGrid) -> jnp.ndarray:
    r"""Gaussian-in-``log m`` × beta-like-in-``χ`` density, no gwkokab.

    Smooth, non-negative, well inside the grid's interior — exactly
    what the marginal arithmetic needs for a clean consistency check.
    """
    m = grid.m
    chi = grid.chi
    mu_log = jnp.log(20.0)
    sig_log = 0.3
    N_m = jnp.exp(-0.5 * ((jnp.log(m) - mu_log) / sig_log) ** 2) / m
    N_chi = chi**2 * (1.0 - chi) ** 2
    return N_m[:, None] * N_chi[None, :]


def main() -> int:
    grid = BinaryGrid.default(m_min=5.0, m_max=80.0, n_m=20, n_chi=8)
    N = synthetic_density(grid)
    kp = dict(a=jnp.asarray(0.3), b=jnp.asarray(1.0))
    log_R = jnp.asarray(0.5)

    C_total = obs.total_rate(N, grid, kp, log_R)
    w_m, w_chi = obs.trapezoid_weights(grid)
    print(f"Total per-environment rate: {float(C_total):.6e}\n")

    # 1-D marginals reduce to total rate.
    print("=== 1-D marginals → total rate ===")
    for axis in obs.VALID_AXES:
        x, dC = obs.marginal_1d(N, grid, kp, log_R, axis=axis)
        w = w_m if axis in (obs.AXIS_M1, obs.AXIS_M2) else w_chi
        rel = float(abs(jnp.sum(dC * w) / C_total - 1.0))
        status = "OK" if rel < 1e-12 else "FAIL"
        print(f"  {axis:>5s}: rel diff = {rel:.2e}  [{status}]")

    # 2-D marginals reduce to total rate.
    print("\n=== 2-D marginals → total rate ===")
    pairs = [
        (obs.AXIS_M1, obs.AXIS_M2),
        (obs.AXIS_M1, obs.AXIS_CHI1),
        (obs.AXIS_M1, obs.AXIS_CHI2),
        (obs.AXIS_M2, obs.AXIS_CHI1),
        (obs.AXIS_M2, obs.AXIS_CHI2),
        (obs.AXIS_CHI1, obs.AXIS_CHI2),
    ]
    for a, b in pairs:
        _, _, dens = obs.marginal_2d(N, grid, kp, log_R, axes=(a, b))
        wa = w_m if a in (obs.AXIS_M1, obs.AXIS_M2) else w_chi
        wb = w_m if b in (obs.AXIS_M1, obs.AXIS_M2) else w_chi
        integ = float(jnp.einsum("ij,i,j->", dens, wa, wb))
        rel = abs(integ / float(C_total) - 1.0)
        status = "OK" if rel < 1e-12 else "FAIL"
        print(f"  ({a:>4s}, {b:>4s}): rel diff = {rel:.2e}  [{status}]")

    # Symmetries.
    print("\n=== Symmetries ===")
    _, dC_m1 = obs.marginal_1d(N, grid, kp, log_R, axis=obs.AXIS_M1)
    _, dC_m2 = obs.marginal_1d(N, grid, kp, log_R, axis=obs.AXIS_M2)
    sym_m = float(jnp.max(jnp.abs(dC_m1 - dC_m2)))
    print(f"  ‖dC/dm1 − dC/dm2‖∞ = {sym_m:.2e}")
    _, dC_c1 = obs.marginal_1d(N, grid, kp, log_R, axis=obs.AXIS_CHI1)
    _, dC_c2 = obs.marginal_1d(N, grid, kp, log_R, axis=obs.AXIS_CHI2)
    sym_c = float(jnp.max(jnp.abs(dC_c1 - dC_c2)))
    print(f"  ‖dC/dχ1 − dC/dχ2‖∞ = {sym_c:.2e}")

    # Independent numpy reimplementation.
    print("\n=== Numpy 4-D quadrature backstop ===")
    m_np = np.asarray(grid.m)
    log_m_np = np.asarray(grid.log_m)
    chi_np = np.asarray(grid.chi)
    N_np = np.asarray(N)
    a_np, b_np = 0.3, 1.0
    R_np = float(np.exp(0.5))
    m1, m2 = m_np[:, None], m_np[None, :]
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    K = (m1 + m2) ** a_np * q**b_np
    R4 = 0.5 * R_np * (
        K[:, None, :, None]
        * N_np[:, :, None, None]
        * N_np[None, None, :, :]
    )
    dlog = np.diff(log_m_np)
    dlog_full = np.concatenate(
        [dlog[:1] * 0.5, 0.5 * (dlog[:-1] + dlog[1:]), dlog[-1:] * 0.5]
    )
    w_m_np = m_np * dlog_full
    d = np.diff(chi_np)
    w_chi_np = np.concatenate([d[:1] * 0.5, 0.5 * (d[:-1] + d[1:]), d[-1:] * 0.5])
    C_np = np.einsum("ijkl,i,j,k,l->", R4, w_m_np, w_chi_np, w_m_np, w_chi_np)
    rel = abs(C_np / float(C_total) - 1.0)
    print(f"  Total rate (numpy):   {C_np:.10e}")
    print(f"  Total rate (JAX):     {float(C_total):.10e}")
    print(f"  rel diff:             {rel:.2e}")

    # AD vs finite difference.
    print("\n=== AD vs finite difference ===")

    def f(lR):
        return obs.total_rate(N, grid, kp, lR)

    g_ad = float(jax.grad(f)(jnp.asarray(0.5)))
    h = 1e-5
    g_fd = (float(f(jnp.asarray(0.5 + h))) - float(f(jnp.asarray(0.5 - h)))) / (2 * h)
    rel = abs(g_ad / g_fd - 1.0)
    print(f"  jax.grad: {g_ad:.6e}")
    print(f"  FD:       {g_fd:.6e}")
    print(f"  rel diff: {rel:.2e}")

    # Ordered-mass fold.
    print("\n=== Ordered-mass fold ===")
    _, _, dens_mm = obs.marginal_2d(
        N, grid, kp, log_R, axes=(obs.AXIS_M1, obs.AXIS_M2)
    )
    folded = obs.fold_ordered_m(grid.m, dens_mm)
    C_fold = float(jnp.einsum("ij,i,j->", folded, w_m, w_m))
    rel = abs(C_fold / float(C_total) - 1.0)
    print(f"  ∫∫_{{m1≥m2}} fold = {C_fold:.6e}")
    print(f"  total rate       = {float(C_total):.6e}")
    print(f"  rel diff:          {rel:.2e}")
    print("\nAll checks complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
