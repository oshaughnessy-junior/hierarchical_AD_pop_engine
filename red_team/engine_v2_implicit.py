"""Engine v2 prototype — implicit Picard solver in 2D ``(m, χ)``.

The current engine ``CoagulationResponse._solve`` is a perturbative Volterra
truncation that empirically matches Backup 2 implicit only at unphysically
small ε ≈ 0.025. Real hierarchical-merger cascades operate at ε ~ O(1):
2G/3G peaks are observable in GW catalogues. To reach that regime we need
either (i) loss-term inclusion + 1/n! Volterra coefficients in a longer
perturbative chain — fragile — or (ii) the implicit fixed-point solver
sketched here, which captures all orders by construction.

The implicit equation::

    N(m, χ) = N_natal(m, χ) + C[N, N](m, χ)

where the bilinear collision operator ``C[N, N] = gain − loss`` is::

    gain(m, χ) = (1/2) ∫∫ K(m₁,m₂) · N(m₁,χ₁) · N(m₂,χ₂) · P_retain
                       · δ(m − M_rem) · δ(χ − χ_rem)
                       dm₁ dm₂ dχ₁ dχ₂

    loss(m, χ) = N(m, χ) · ∫∫ K(m, m') · N(m', χ') · P_retain(m, m', χ, χ')
                                dm' dχ'

Solved by damped Picard iteration. The same engine primitives (grid,
``_step_one_generation`` for gain, ``log_pairing_kernel``,
``retention_aligned``, ``M_rem``, ``chi_rem``) are reused — only the
*accumulator structure* is replaced.

If this prototype agrees with Backup 1 (pure particle MC) and Backup 2
(Doctor port) at moderate ε, the path is clear: refactor
``CoagulationResponse`` to use this solver as the production
``_log_total_density``. AD support via implicit-function-theorem custom
VJPs is a follow-up.

Smoke driver::

    python -m red_team.engine_v2_implicit
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np

# Engine primitives. This is the comparison driver (red-team Phase 1+),
# so importing from response.* is allowed here (the rule applies only to
# sim_coagulation.py).
from hierarchical_ad_pop_engine.response.coagulation import (
    _step_one_generation,
    build_gaussian_coagulation,
    CoagulationResponse,
)
from hierarchical_ad_pop_engine.kernels import (
    log_pairing_kernel,
    log_pairing_kernel_grid,
)
from hierarchical_ad_pop_engine.remnants.kick import retention_aligned


# ---------------------------------------------------------------------------
# Trapezoid weights on the engine grids.
# ---------------------------------------------------------------------------
def _dm_log(grid_m: jnp.ndarray, grid_log_m: jnp.ndarray) -> jnp.ndarray:
    """``dm`` weights on a geomspace grid (dm = m · dlog m, trapezoid)."""
    dlog_m = jnp.diff(grid_log_m)
    dlog_m_full = jnp.concatenate(
        [dlog_m[:1] * 0.5, 0.5 * (dlog_m[:-1] + dlog_m[1:]), dlog_m[-1:] * 0.5]
    )
    return grid_m * dlog_m_full


def _dchi(grid_chi: jnp.ndarray) -> jnp.ndarray:
    """``dχ`` weights on a linear grid (trapezoid)."""
    d = jnp.diff(grid_chi)
    return jnp.concatenate([d[:1] * 0.5, 0.5 * (d[:-1] + d[1:]), d[-1:] * 0.5])


# ---------------------------------------------------------------------------
# Loss density — the part the engine's current accumulator omits.
# ---------------------------------------------------------------------------
def loss_density(
    N: jnp.ndarray,
    response: CoagulationResponse,
    v_esc: jnp.ndarray,
) -> jnp.ndarray:
    """``N(m, χ) · ⟨K · N · P_retain⟩(m, χ)`` on the engine's grid.

    Memory: builds a (N_m, N_chi, N_m, N_chi) 4D array. For (96, 32, 96, 32)
    that's ~75 MB at float64 — fits comfortably.
    """
    grid = response.grid
    m = grid.m
    chi = grid.chi
    # 4D broadcast over (m, χ, m', χ').
    m1 = m[:, None, None, None]
    chi1 = chi[None, :, None, None]
    m2 = m[None, None, :, None]
    chi2 = chi[None, None, None, :]

    log_K = log_pairing_kernel(m1, m2, **response.kernel_params)  # (N_m, 1, N_m, 1)
    P = retention_aligned(m1, m2, chi1, chi2, v_esc)  # (N_m, N_chi, N_m, N_chi)
    K_P = jnp.exp(log_K) * P  # (N_m, N_chi, N_m, N_chi) — broadcast on j

    integrand = K_P * N[None, None, :, :]  # multiply by N(m', χ')

    dm_w = _dm_log(grid.m, grid.log_m)  # (N_m,)
    dchi_w = _dchi(grid.chi)  # (N_chi,)

    # Σ over (m', χ'): integrand[i, j, k, l] × dm[k] × dχ[l]
    avg = jnp.einsum("ijkl,k,l->ij", integrand, dm_w, dchi_w)
    return N * avg


# ---------------------------------------------------------------------------
# Gain density — engine's existing _step_one_generation with sym = 1/2 for
# the unordered-pair correction, **divided by output cell area** so the
# returned quantity is a true density at (m_out, χ_out) on the same
# measure scale as loss_density and N_natal.
#
# Why the division: `_step_one_generation` returns the bilinear-scattered
# *weight* per output cell — it already absorbed the four progenitor
# measure factors ``dm₁ dχ₁ dm₂ dχ₂`` via its ``cell`` term, and the
# bilinear scatter assigns *count* per output cell. To get density at
# (m_out, χ_out) we divide by ``dm_out · dχ_out``, the Jacobian of the
# discrete-density → discrete-count map. On a geomspace m grid, this
# Jacobian varies by ~20× across the grid; a constant rescaling
# cannot fix it.
# ---------------------------------------------------------------------------
def gain_density(
    N: jnp.ndarray,
    log_K_mm: jnp.ndarray,
    response: CoagulationResponse,
    v_esc: jnp.ndarray,
) -> jnp.ndarray:
    log_N_safe = jnp.log(jnp.maximum(N, 1e-30))
    raw_weight_per_cell = _step_one_generation(
        log_N_safe,
        log_N_safe,
        log_K_mm,
        response.grid,
        v_esc,
        0.5,  # (1/2) ∫∫ K N N P_retain — unordered-pair correction
    )
    grid = response.grid
    dm_out = _dm_log(grid.m, grid.log_m)  # (N_m,)
    dchi_out = _dchi(grid.chi)  # (N_chi,)
    return raw_weight_per_cell / (dm_out[:, None] * dchi_out[None, :])


# ---------------------------------------------------------------------------
# Solver.
# ---------------------------------------------------------------------------
def solve_implicit(
    response: CoagulationResponse,
    v_esc: jnp.ndarray,
    *,
    epsilon: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    damping: float = 0.5,
    verbose: bool = False,
) -> dict:
    """Damped Picard fixed-point for ``N = N_natal + ε · (gain(N) − loss(N))``.

    ``epsilon`` has units of [time] and absolute physical meaning: it is
    the cluster evolution time × the kernel's rate constant. ε ≈ 1 sets
    the cascade to a fully-developed steady state; ε ≪ 1 leaves the
    natal population mostly intact.

    Both ``gain_density`` and ``loss_density`` are returned in proper
    density units (number per unit ``m · χ``) by dividing the
    bilinear-scattered count-per-cell by the output cell area
    ``dm_out · dχ_out``. This is the Jacobian of the discrete
    density ↔ count map; on a geomspace m grid it varies ~20× across
    the grid, so it cannot be absorbed into a constant rescaling. With
    this correction, ε is on the same absolute scale as Backup 2's ε
    (both have units of evolution time).

    Returns dict with keys ``log_N``, ``N``, ``n_iter``, ``final_diff``,
    ``history`` (per-iteration ``∫N dm dχ`` for monitoring), ``epsilon``,
    ``f_natal_depleted`` (the fraction of the natal pool consumed by the
    loss term at the converged fixed point — see below), and the raw
    ``loss_integral`` / ``natal_integral`` numerators / denominators.

    The depletion fraction is

    .. math::

        f_{\\rm dep} = 1 - \\frac{\\int N^*(m,\\chi)\\,dm\\,d\\chi}
                                {\\int N_{\\rm natal}(m,\\chi)\\,dm\\,d\\chi}

    **Why not** ``ε·∫L/∫N_natal``: integrating the steady-state equation
    and using ``∫G = (1/2)·∫L`` (gain counts each pair once; the double
    integral in L counts it twice) gives
    ``∫N* = ∫N_natal − (ε/2)·∫L``, so ``ε·∫L/∫N_natal`` overestimates
    the true depletion by a factor of 2.  The formula above is exact at
    the Picard fixed point.
    """
    log_K_mm = log_pairing_kernel_grid(response.grid.m, **response.kernel_params)
    log_N_natal = response._natal_log_density()
    N_natal = jnp.exp(log_N_natal)

    # Iteration state.
    N = N_natal
    grid = response.grid
    dm_w = _dm_log(grid.m, grid.log_m)
    dchi_w = _dchi(grid.chi)
    history = []

    final_diff = float("inf")
    n_iter = 0
    loss = loss_density(N, response, v_esc)  # initial; overwritten in loop
    for it in range(max_iter):
        gain = gain_density(N, log_K_mm, response, v_esc)
        loss = loss_density(N, response, v_esc)
        N_rhs = N_natal + epsilon * (gain - loss)
        N_rhs = jnp.maximum(N_rhs, 0.0)  # numerical floor
        N_new = (1.0 - damping) * N + damping * N_rhs
        final_diff = float(jnp.max(jnp.abs(N_new - N)))
        # Track ∫N dm dχ for monitoring convergence.
        integral = float(jnp.einsum("ij,i,j->", N_new, dm_w, dchi_w))
        history.append(integral)
        N = N_new
        n_iter = it + 1
        if verbose:
            print(f"    iter {it+1:3d}: ‖ΔN‖∞ = {final_diff:.3e},  ∫N = {integral:.4g}")
        if final_diff < tol:
            break

    # Final loss evaluation at the converged N* (needed for depletion fraction).
    loss = loss_density(N, response, v_esc)
    loss_integral = float(jnp.einsum("ij,i,j->", loss, dm_w, dchi_w))
    natal_integral = float(jnp.einsum("ij,i,j->", N_natal, dm_w, dchi_w))
    n_star_integral = float(jnp.einsum("ij,i,j->", N, dm_w, dchi_w))
    # f_dep = 1 - ∫N*/∫N_natal.  Integrating the steady-state equation gives
    # ∫N* = ∫N_natal - (ε/2)·loss_integral, so using ε·loss_integral/∫N_natal
    # directly overestimates the true depletion by 2×.  The formula below is
    # exact at the Picard fixed point with no factor-of-2 ambiguity.
    f_natal_depleted = max(0.0, min(1.0, 1.0 - n_star_integral / max(natal_integral, 1e-30)))

    return dict(
        log_N=jnp.log(jnp.maximum(N, 1e-30)),
        N=N,
        n_iter=n_iter,
        final_diff=final_diff,
        history=np.asarray(history),
        epsilon=epsilon,
        loss_integral=loss_integral,
        natal_integral=natal_integral,
        f_natal_depleted=f_natal_depleted,
    )


# ---------------------------------------------------------------------------
# Marginal helpers — for plotting and JS comparison.
# ---------------------------------------------------------------------------
def marginals(
    N: jnp.ndarray, response: CoagulationResponse
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (m, dN/dm, χ, dN/dχ) by trapezoidal marginalisation."""
    grid = response.grid
    m = np.asarray(grid.m)
    chi = np.asarray(grid.chi)
    dm_w = np.asarray(_dm_log(grid.m, grid.log_m))
    dchi_w = np.asarray(_dchi(grid.chi))
    N_np = np.asarray(N)
    dN_dm = N_np @ dchi_w
    dN_dchi = N_np.T @ dm_w
    return m, dN_dm, chi, dN_dchi


# ---------------------------------------------------------------------------
# Smoke driver — Stage 1 θ*, plot N(m), N(χ), convergence trace.
# ---------------------------------------------------------------------------
_STAGE1 = dict(
    mu=8.0,
    sigma=1.0,
    m_min=4.0,
    m_max=80.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.10,
    kernel_a=0.0,
    kernel_b=1.0,
    log_R_env=0.0,
    k_max=4,  # unused by implicit solver but required by build_gaussian_coagulation
)
V_ESC = 1000.0


def _smoke_main(run_hash: Optional[str] = None, epsilon: float = 1.0) -> None:
    import matplotlib.pyplot as plt

    here = Path(__file__).parent
    rh = run_hash or secrets.token_hex(3)
    fig_dir = here / f"figures_{rh}"
    art_dir = here / f"artifacts_{rh}"
    fig_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    print(f"Run hash: {rh}")
    print(f"  figures → {fig_dir.relative_to(here.parent)}/")

    response = build_gaussian_coagulation(**_STAGE1)

    print(f"Solving implicit Picard (ε={epsilon}) ...")
    result = solve_implicit(
        response, jnp.asarray(V_ESC),
        epsilon=epsilon,
        max_iter=200, tol=1e-7, damping=0.4, verbose=True,
    )
    print(f"  converged in {result['n_iter']} iterations, "
          f"|ΔN|∞ = {result['final_diff']:.2e}")
    print(f"  ∫N dm dχ converged to {result['history'][-1]:.4g}")

    np.savez(
        art_dir / "engine_v2_implicit.npz",
        N=np.asarray(result["N"]),
        log_N=np.asarray(result["log_N"]),
        history=result["history"],
    )

    # Plots.
    m, dN_dm, chi, dN_dchi = marginals(result["N"], response)
    log_N_natal = np.asarray(response._natal_log_density())
    N_natal_np = np.exp(log_N_natal)
    dN_dm_natal = N_natal_np @ np.asarray(_dchi(response.grid.chi))
    dN_dchi_natal = N_natal_np.T @ np.asarray(_dm_log(response.grid.m, response.grid.log_m))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.loglog(m, np.maximum(dN_dm_natal, 1e-30), color="k", lw=1.5, label=r"$N_{\rm natal}$")
    ax.loglog(m, np.maximum(dN_dm, 1e-30), color="C0", lw=2.0,
              label=f"engine v2 implicit ({result['n_iter']} iters)")
    ax.set_xlabel(r"$m \;[M_\odot]$")
    ax.set_ylabel(r"$dN/dm$")
    ax.set_title("Engine v2 — mass marginal")
    ax.set_ylim(bottom=dN_dm.max() * 1e-7)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small")

    ax = axes[1]
    ax.semilogy(chi, np.maximum(dN_dchi_natal, 1e-30), color="k", lw=1.5, label=r"$N_{\rm natal}$")
    ax.semilogy(chi, np.maximum(dN_dchi, 1e-30), color="C0", lw=2.0,
                label="engine v2 implicit")
    ax.axvline(0.6865, color="0.4", lw=0.7, ls=":")
    ax.set_xlabel(r"$\chi$")
    ax.set_ylabel(r"$dN/d\chi$")
    ax.set_title("Engine v2 — spin marginal")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=dN_dchi.max() * 1e-5)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small")

    ax = axes[2]
    ax.plot(result["history"], marker="o", ms=3)
    ax.set_xlabel("Picard iteration")
    ax.set_ylabel(r"$\int N \,dm\,d\chi$")
    ax.set_title("Convergence trace")
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    out = fig_dir / "engine_v2_implicit.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-hash", default=None)
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="Coupling × time. Larger ε → deeper cascade. Note units "
        "differ from Backup 2 — see solve_implicit docstring.",
    )
    args = parser.parse_args()
    _smoke_main(run_hash=args.run_hash, epsilon=args.epsilon)
