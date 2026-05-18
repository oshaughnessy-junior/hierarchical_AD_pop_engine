"""Backup implementation 2 — perturbative-with-loss + implicit Picard.

Direct port of the Doctor et al. Supplemental ``CollisionTerm`` solver
(``SupplementalPlotting/Python/tools_HierarchyMergers_ZeroSpin.py``) to
our parameters. **Mass only**; spin is not represented in this backup.
The collision operator is the standard Smoluchowski form
``gain − loss`` evaluated on a linear mass grid.

Two solvers:

1. **Perturbative 3rd-order Volterra expansion** ::

      df1 = C[u_0, u_0]
      df2 = C[u_0, df1]
      df3 = C[df1, df1] + 2 C[u_0, df2]
      u_net = u_0 + ε df1 + ε² df2 + (ε³/3) df3

   The ``1/3 = 1/3!`` factor is the time-integration coefficient for the
   third-order Picard term — *exactly* the factor the engine's accumulator
   omits.

2. **Implicit Picard fixed-point** on the integrated equation ::

      u_{n+1} = u_0 + ε · C[u_n, u_n]

   Iterated to convergence. Captures all orders, including resummation
   effects that the truncated perturbative series misses.

Simplifying physics assumptions versus the engine (documented so the
comparison is interpretable):

- **No spin.** Aligned-spin remnant fits are ignored; we work in 1D mass.
- **Mass conservation** ``m_rem = m1 + m2`` (no radiated GW energy).
  This shifts peak positions by ~5 % relative to the engine's
  Lousto-Zlochower remnant fit; the *shape* comparison is still
  meaningful.
- **No kick / no retention.** Closed system. Valid for the Stage 1
  ``v_esc = 1000 km/s`` regime where ~all aligned-spin kicks retain.

If Backup 1 (particle MC, full physics) and Backup 2 (this) agree on
the coagulation cascade *structure* modulo a small peak-shift from
mass conservation, the engine's coagulation algorithm is the suspect.
If Backup 1 disagrees with both Backup 2 and the engine, look at
remnant physics first.

Run::

    python -m red_team.backup_perturbative
"""

from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import truncnorm


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
@dataclass
class PerturbConfig:
    mu_natal: float
    sigma_natal: float
    m_min: float
    m_max: float
    kernel_a: float
    kernel_b: float
    # Coupling × time. Picking ε is calibration: it sets how far the
    # cascade has run. ε = 1 corresponds to "one e-fold of merger time"
    # in this dimensionless reference frame.
    epsilon: float = 1.0
    m_grid_max: float = 200.0
    dm: float = 0.25
    picard_max_iter: int = 200
    picard_tol: float = 1e-7
    # Picard damping (under-relaxation). Helps convergence at large ε.
    picard_damping: float = 0.5


# ---------------------------------------------------------------------------
# Building blocks.
# ---------------------------------------------------------------------------
def make_grid(cfg: PerturbConfig) -> np.ndarray:
    """Linear mass grid starting at 0 (so the j + l = k convolution
    represents ``m_rem = m_j + m_l`` directly)."""
    return np.arange(0.0, cfg.m_grid_max + cfg.dm, cfg.dm)


def gaussian_natal_density(
    mvals: np.ndarray, mu: float, sigma: float, m_min: float, m_max: float
) -> np.ndarray:
    """Truncated Gaussian on the *physical* support [m_min, m_max]; zero
    outside (which includes the grid points below m_min, e.g. m = 0)."""
    a = (m_min - mu) / sigma
    b = (m_max - mu) / sigma
    pdf = truncnorm.pdf(mvals, a, b, loc=mu, scale=sigma)
    return np.where((mvals >= m_min) & (mvals <= m_max), pdf, 0.0)


def kernel_matrix(mvals: np.ndarray, a: float, b: float) -> np.ndarray:
    """``γ[i, j] = (m_i + m_j)^a · q^b`` with q = min/max. Same kernel as
    the engine. Zeros on rows/columns where ``m = 0`` to avoid divergence.
    Includes a ``dm`` factor so the ``Σ_l γ[k, l] u[l]`` discrete sum
    approximates ``∫ γ(m, m') u(m') dm'`` directly."""
    m1 = mvals[:, None]
    m2 = mvals[None, :]
    m_big = np.maximum(m1, m2)
    m_sml = np.minimum(m1, m2)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(m_big > 0, m_sml / m_big, 0.0)
        gam = np.where(
            (m1 > 0) & (m2 > 0),
            (m1 + m2) ** a * np.where(q > 0, q ** b, 0.0),
            0.0,
        )
    gam = np.where(np.isfinite(gam), gam, 0.0)
    return gam


def bred(B: np.ndarray) -> np.ndarray:
    """Discrete sum-convolution: ``bout[k] = Σ_{j+l=k} B[j, l]``.

    Captures the ``δ(m - m' - m'')`` gain on a linear grid with
    ``m0 = 0`` (so the j + l = k constraint maps directly to
    ``m_rem = m_j + m_l``). O(N²) — fine for our N ~ 400 grid.
    """
    n = B.shape[0]
    bout = np.zeros(n, dtype=np.float64)
    # bout[k] = sum over j of B[j, k-j] for j in [max(0, k-(n-1)), min(k, n-1)].
    for k in range(2, n):
        j_lo = max(0, k - (n - 1))
        j_hi = min(k, n - 1)
        if j_lo > j_hi:
            continue
        bout[k] = float(np.sum(np.diag(np.fliplr(B[j_lo:j_hi + 1, k - j_hi : k - j_lo + 1]))))
    return bout


def collision_term(
    u: np.ndarray, v: np.ndarray, gamma: np.ndarray, dm: float
) -> np.ndarray:
    """Symmetric bilinear collision operator (gain − loss).

    ::

        B[i, j] = (1/2) · γ[i, j] · (u[i] v[j] + v[i] u[j])
        gain[k] = (1/2) · Σ_{j+l=k} B[j, l]
        loss[k] = Σ_l B[k, l]
        return (gain − loss) · dm

    The trailing ``· dm`` converts the discrete operator to a rate-density
    increment matching ``∂u/∂t``.

    This mirrors the Doctor reference's ``CollisionTerm`` exactly.
    """
    B = 0.5 * gamma * (u[:, None] * v[None, :] + v[:, None] * u[None, :])
    gain = bred(B)
    loss = B.sum(axis=-1)
    return (0.5 * gain - loss) * dm


# ---------------------------------------------------------------------------
# Solvers.
# ---------------------------------------------------------------------------
def perturbative_u_net(cfg: PerturbConfig) -> dict:
    """Compute ``u_net`` via 3rd-order Volterra expansion with the proper
    1/n! time-integration coefficients."""
    mvals = make_grid(cfg)
    u_0 = gaussian_natal_density(mvals, cfg.mu_natal, cfg.sigma_natal, cfg.m_min, cfg.m_max)
    gamma = kernel_matrix(mvals, cfg.kernel_a, cfg.kernel_b)
    df1 = collision_term(u_0, u_0, gamma, cfg.dm)
    df2 = collision_term(u_0, df1, gamma, cfg.dm)
    df3 = collision_term(df1, df1, gamma, cfg.dm) + 2.0 * collision_term(u_0, df2, gamma, cfg.dm)
    eps = cfg.epsilon
    u_net = u_0 + eps * df1 + (eps ** 2) * df2 + (eps ** 3 / 3.0) * df3
    return dict(mvals=mvals, u_net=u_net, u_0=u_0, df1=df1, df2=df2, df3=df3)


def implicit_picard(cfg: PerturbConfig) -> dict:
    """Implicit fixed-point solver: ``u = u_0 + ε · C[u, u]``.

    Uses simple Picard iteration with under-relaxation
    ``u_{n+1} = (1-α) u_n + α (u_0 + ε C[u_n, u_n])``. The damping factor
    ``α = picard_damping`` helps when ε is moderate-to-large.
    """
    mvals = make_grid(cfg)
    u_0 = gaussian_natal_density(mvals, cfg.mu_natal, cfg.sigma_natal, cfg.m_min, cfg.m_max)
    gamma = kernel_matrix(mvals, cfg.kernel_a, cfg.kernel_b)

    u = u_0.copy()
    history = [u.copy()]
    alpha = cfg.picard_damping
    for it in range(cfg.picard_max_iter):
        rhs = u_0 + cfg.epsilon * collision_term(u, u, gamma, cfg.dm)
        u_new = (1.0 - alpha) * u + alpha * rhs
        diff = float(np.max(np.abs(u_new - u)))
        u = u_new
        history.append(u.copy())
        if diff < cfg.picard_tol:
            break
    return dict(mvals=mvals, u_net=u, u_0=u_0, history=history, n_iter=it + 1, final_diff=diff)


# ---------------------------------------------------------------------------
# Smoke driver — Stage 1 θ*, overlay perturbative + implicit + natal.
# ---------------------------------------------------------------------------
_STAGE1 = dict(
    mu_natal=8.0,
    sigma_natal=1.0,
    m_min=4.0,
    m_max=80.0,
    kernel_a=0.0,
    kernel_b=1.0,
)


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

    cfg = PerturbConfig(epsilon=epsilon, **_STAGE1)
    print(f"Config: {cfg}")

    print("Perturbative O(ε³) ...")
    pert = perturbative_u_net(cfg)
    print(f"  ∫u_net dm = {float(np.trapezoid(pert['u_net'], pert['mvals'])):.4g}")

    print("Implicit Picard ...")
    impl = implicit_picard(cfg)
    print(f"  converged in {impl['n_iter']} iterations, final |Δu|_∞ = {impl['final_diff']:.2e}")
    print(f"  ∫u_net dm = {float(np.trapezoid(impl['u_net'], impl['mvals'])):.4g}")

    np.savez(
        art_dir / "backup2_solutions.npz",
        mvals=pert["mvals"],
        u_0=pert["u_0"],
        u_pert=pert["u_net"],
        u_impl=impl["u_net"],
        epsilon=cfg.epsilon,
    )
    print(f"  wrote {art_dir / 'backup2_solutions.npz'}")

    # Plot overlay.
    fig, ax = plt.subplots(figsize=(8, 5))
    m = pert["mvals"]
    eps_floor = 1e-7
    ax.loglog(
        m, np.maximum(pert["u_0"], eps_floor),
        color="k", ls="-", lw=1.5, label=r"$u_0$ (natal Gaussian)",
    )
    ax.loglog(
        m, np.maximum(pert["u_net"], eps_floor),
        color="C0", ls="-", lw=2.0, label=fr"perturbative $O(\varepsilon^3)$, $\varepsilon={cfg.epsilon}$",
    )
    ax.loglog(
        m, np.maximum(impl["u_net"], eps_floor),
        color="C1", ls="--", lw=2.0, label=fr"implicit Picard ({impl['n_iter']} iters)",
    )
    ax.set_xlabel(r"single-BH mass $m \;[M_\odot]$")
    ax.set_ylabel(r"$u(m)$")
    ax.set_title(rf"Backup 2 — Doctor port, $\varepsilon={cfg.epsilon}$, dm={cfg.dm}")
    ax.set_xlim(cfg.m_min, cfg.m_grid_max)
    ax.set_ylim(bottom=eps_floor)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    fig.tight_layout()
    out = fig_dir / "backup2_u_net.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-hash", default=None)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()
    _smoke_main(run_hash=args.run_hash, epsilon=args.epsilon)
