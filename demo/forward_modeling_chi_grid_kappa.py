"""Spin-results demo: clustered χ grid (believable) × Fisher-κ (interesting).

Two stories in one figure, both on the production ``solver="implicit"``
path:

* **Left — "believable" (reorganized χ grid).**  The cascade ``dN/dχ``
  marginal on the clustered grid vs a uniform grid of the *same* node
  count.  The clustered nodes (dense at the natal peak χ≈0.05 and the 2G
  cascade peak χ≈0.69) resolve the narrow cascade structure that the
  uniform grid blurs into a blocky step.

* **Right — "interesting" (Fisher-κ).**  The same ``dN/dχ`` marginal on the
  clustered grid as the spin-orientation concentration κ is swept from
  Isotropic (κ→0, kicks isotropic → broad retained-spin spread) toward
  aligned (large κ → the BR/LZ aligned cascade peak re-sharpens at χ≈0.69).
  This is the knob that makes the hierarchical spin signature physical and
  tunable rather than perfectly aligned by fiat.

Both panels marginalise the converged single-BH density ``N*(m, χ)`` over
``m`` (trapezoid on the engine measure) to a ``dN/dχ``.

Run (workstation / JAX env)::

    python demo/forward_modeling_chi_grid_kappa.py
    python demo/forward_modeling_chi_grid_kappa.py --n-m 64 --n-chi 24 \
        --kappas 0,2,8,30 --epsilon 0.6 --v-esc 300

Reduced defaults fit the Cowork sandbox per-command budget; pass larger
``--n-m / --n-chi`` for a production figure.  Output PNG:
``demo/figures_chi_grid_kappa/chi_grid_kappa.png``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.spin_orientation import Aligned, Isotropic, Fisher
from hierarchical_ad_pop_engine.response.coagulation import build_gaussian_coagulation
from hierarchical_ad_pop_engine.response.implicit import (
    solve_implicit_fixed_point,
    _dchi_weights,
    _dm_log_weights,
)


def _solve_chi_marginal(grid, orientation, *, mu, sigma, v_esc, epsilon,
                        natal_spin_scale, n_theta, n_dphi, max_iter):
    """Solve and return (chi_nodes, dN/dχ) on ``grid``."""
    o = orientation
    if isinstance(o, (Isotropic, Fisher)):
        # respect the requested light quadrature
        o = (Isotropic.default(n_theta=n_theta, n_dphi=n_dphi)
             if isinstance(o, Isotropic)
             else Fisher.default(kappa=o.kappa, n_theta=n_theta, n_dphi=n_dphi))
    resp = build_gaussian_coagulation(
        mu=mu, sigma=sigma, m_min=float(grid.m[0]), m_max=float(grid.m[-1]),
        natal_spin_loc=0.0, natal_spin_scale=natal_spin_scale, kernel_b=2.0,
        grid=grid, solver="implicit", orientation=o, retention_kind="sigmoid",
    )
    res = solve_implicit_fixed_point(
        resp, jnp.asarray(v_esc), epsilon=epsilon,
        max_iter=max_iter, tol=1e-7, damping=0.5,
    )
    dm_w = _dm_log_weights(grid.m, grid.log_m)
    dN_dchi = jnp.einsum("ij,i->j", res.N, dm_w)  # marginalise m
    return np.asarray(grid.chi), np.asarray(dN_dchi)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-m", type=int, default=24)
    p.add_argument("--n-chi", type=int, default=16)
    p.add_argument("--m-min", type=float, default=4.0)
    p.add_argument("--m-max", type=float, default=80.0)
    p.add_argument("--mu", type=float, default=8.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--natal-spin-scale", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.6)
    p.add_argument("--v-esc", type=float, default=300.0)
    p.add_argument("--kappas", type=str, default="0,2,8,30",
                   help="comma list; 0 => Isotropic")
    p.add_argument("--n-theta", type=int, default=4)
    p.add_argument("--n-dphi", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--outdir", type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        "figures_chi_grid_kappa"))
    args = p.parse_args()

    common = dict(mu=args.mu, sigma=args.sigma, v_esc=args.v_esc,
                  epsilon=args.epsilon, natal_spin_scale=args.natal_spin_scale,
                  n_theta=args.n_theta, n_dphi=args.n_dphi, max_iter=args.max_iter)

    grid_clu = BinaryGrid.clustered_chi(m_min=args.m_min, m_max=args.m_max,
                                        n_m=args.n_m, n_chi=args.n_chi)
    grid_uni = BinaryGrid.default(m_min=args.m_min, m_max=args.m_max,
                                  n_m=args.n_m, n_chi=args.n_chi)

    # Left panel: clustered vs uniform at matched n_chi (Isotropic).
    iso = Isotropic.default()
    chi_c, dN_c = _solve_chi_marginal(grid_clu, iso, **common)
    chi_u, dN_u = _solve_chi_marginal(grid_uni, iso, **common)

    # Right panel: κ sweep on the clustered grid.
    kappas = [float(k) for k in args.kappas.split(",")]
    sweep = []
    for k in kappas:
        o = Isotropic.default() if k == 0.0 else Fisher.default(kappa=k)
        chi_k, dN_k = _solve_chi_marginal(grid_clu, o, **common)
        sweep.append((k, chi_k, dN_k))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    axL.plot(chi_u, dN_u, "-o", ms=3, color="0.6",
             label=f"uniform χ (n_χ={args.n_chi})")
    axL.plot(chi_c, dN_c, "-o", ms=3, color="C3",
             label=f"clustered χ (n_χ={args.n_chi})")
    axL.set_title("Believable: clustered χ resolves the cascade peak")
    axL.set_xlabel(r"$\chi$"); axL.set_ylabel(r"$dN/d\chi$")
    axL.set_yscale("log"); axL.legend(fontsize=8)
    axL.axvline(0.6865, ls=":", color="k", lw=0.8)

    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(sweep)))
    for (k, chi_k, dN_k), c in zip(sweep, cmap):
        lab = "Isotropic (κ→0)" if k == 0.0 else f"Fisher κ={k:g}"
        axR.plot(chi_k, dN_k, "-o", ms=3, color=c, label=lab)
    axR.set_title("Interesting: Fisher-κ tunes the cascade spin")
    axR.set_xlabel(r"$\chi$"); axR.set_ylabel(r"$dN/d\chi$")
    axR.set_yscale("log"); axR.legend(fontsize=8)
    axR.axvline(0.6865, ls=":", color="k", lw=0.8)

    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "chi_grid_kappa.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
