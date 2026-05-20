"""Narrow 2-path comparison: package implicit Picard ↔ particle MC.

This is a **measurement instrument**, not a pass/fail oracle.  The goal is
to quantify agreement between the two fundamentally independent solvers on
the physics that each resolves well, and to surface the limits of each.

Two paths
---------
* **Implicit** — the production package solver
  ``hierarchical_ad_pop_engine.response.implicit.solve_implicit_fixed_point``.
  Full 2-D ``(m, χ)`` implicit Picard fixed-point, JAX ``while_loop`` + IFT
  custom VJP.  Handles all merger generations analytically at cost O(N_m²
  N_χ²) per iteration.
* **Particle MC** — ``red_team.backup_particle_mc.particle_mc``.
  Pure particle cloud in continuous ``(m, χ)``.  No grid, no approximation.
  Stopping fraction driven self-consistently from the implicit solver's
  ``f_natal_depleted = 1 − ∫N*/∫N_natal``.

JS at the grid's own resolution (primary metric)
-------------------------------------------------
A 32-point linear χ grid upsampled to 400 eval points and compared to a
smooth KDE is an apples-to-oranges comparison: the piecewise-linear
interpolation of the grid introduces "jaggyness" that the KDE doesn't
have, inflating JS even when the physics agrees.  The fix: evaluate the
MC KDE at the solver's grid points directly (96 m points, 32 χ points)
and compute JS there.  This makes both representations share the same
resolution.

The 400-point eval grids are still used for plotting; only the JS metrics
use the grid axis.

Natal baseline (representation noise floor)
-------------------------------------------
Running at ``ε_natal`` (default 1e-4) gives N* ≈ N_natal for both
solvers — any JS at that ε quantifies how much the grid analytic density
and the KDE-smoothed particle sample disagree on the same underlying
distribution, purely due to representation.  If JS(ε=1) ≈ JS(ε_natal),
the disagreement at ε=1 is representation noise, not physics.

Mass-range split
----------------
The 1G and 2G peaks (bulk, m < m_bulk_max, default 25 M☉) are well-
resolved by both methods at the default n_seed=20k.  Agreement there is
the primary signal.  The 3G+ tail (m > m_bulk_max) is exponentially
suppressed — at 20k particles the MC resolves it poorly; the implicit
solver handles it analytically.  JS on the full range is dominated by
this tail and should be interpreted alongside the MC-vs-MC shot-noise
estimate, not as a standalone pass/fail metric.

Run::

    python -m red_team.compare_implicit_vs_mc              # Stage 1 θ*, ε=1
    python -m red_team.compare_implicit_vs_mc --epsilon 0.5
    python -m red_team.compare_implicit_vs_mc --n-seed 100000
    python -m red_team.compare_implicit_vs_mc --no-natal-baseline  # faster
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from hierarchical_ad_pop_engine.response.coagulation import build_gaussian_coagulation
from hierarchical_ad_pop_engine.response.implicit import solve_implicit_fixed_point

from red_team import backup_particle_mc as B1
from red_team import engine_v2_implicit as V2  # for marginals() helper only
from red_team.phase1_overlay import THETA_STAR, _normalise, js_divergence
from red_team.sim_coagulation import make_marginal


# ---------------------------------------------------------------------------
# Implicit solver — package production path.
# ---------------------------------------------------------------------------
def _build_response(theta: dict):
    """Build a CoagulationResponse for the given theta dict."""
    # build_gaussian_coagulation with defaults (solver="perturbative") is fine
    # here — we bypass _log_total_density and call solve_implicit_fixed_point
    # directly, so the solver field on the response object is irrelevant.
    return build_gaussian_coagulation(
        mu=theta["mu_natal"],
        sigma=theta["sigma_natal"],
        m_min=theta["m_min"],
        m_max=theta["m_max"],
        natal_spin_loc=theta["natal_spin_loc"],
        natal_spin_scale=theta["natal_spin_scale"],
        kernel_a=theta["kernel_a"],
        kernel_b=theta["kernel_b"],
        log_R_env=0.0,
        k_max=theta["k_max"],
    )


def implicit_solve(
    theta: dict,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run the package implicit Picard solver.

    Returns
    -------
    m_grid, dN_dm_grid, chi_grid, dN_dchi_grid, diag
        Marginals on the solver's own (N_m,) and (N_chi,) grid axes.
        These are the canonical outputs; interpolation to finer eval grids
        is the caller's responsibility.
    """
    response = _build_response(theta)
    result = solve_implicit_fixed_point(
        response,
        jnp.asarray(theta["v_esc"]),
        epsilon=epsilon,
        max_iter=200,
        tol=1e-7,
        damping=0.4,
    )
    m_grid, dN_dm_grid, chi_grid, dN_dchi_grid = V2.marginals(result.N, response)
    diag = dict(
        n_iter=int(result.n_iter),
        final_diff=float(result.final_diff),
        f_natal_depleted=float(result.f_natal_depleted),
        natal_integral=float(result.natal_integral),
        loss_integral=float(result.loss_integral),
        m_grid=np.asarray(m_grid),
        chi_grid=np.asarray(chi_grid),
    )
    return (
        np.asarray(m_grid),
        np.asarray(dN_dm_grid),
        np.asarray(chi_grid),
        np.asarray(dN_dchi_grid),
        diag,
    )


# ---------------------------------------------------------------------------
# Particle MC — pure particle cloud.
# ---------------------------------------------------------------------------
def mc_run(
    theta: dict,
    n_seed: int,
    rng: np.random.Generator,
    depletion_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the pure particle MC.  Returns raw (m_active, chi_active) arrays."""
    cfg = B1.MCConfig(
        n_seed=n_seed,
        depletion_fraction=depletion_fraction,
        **{k: theta[k] for k in (
            "mu_natal", "sigma_natal", "m_min", "m_max",
            "natal_spin_loc", "natal_spin_scale",
            "kernel_a", "kernel_b", "v_esc",
        )},
    )
    result = B1.particle_mc(cfg, rng, track_generations=False)
    return result["m_active"], result["chi_active"], result["stats"]


def mc_marginals_on_grid(
    m_active: np.ndarray,
    chi_active: np.ndarray,
    m_grid: np.ndarray,
    chi_grid: np.ndarray,
    theta: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the MC KDE exactly at the solver's grid points.

    This is the apples-to-apples comparison: both representations live on
    the same (N_m,) and (N_chi,) axes, so JS measures physics agreement at
    the solver's resolution rather than interpolation-vs-KDE artefacts.
    """
    if m_active.size < 2:
        return np.zeros_like(m_grid), np.zeros_like(chi_grid)
    dN_dm = make_marginal(
        m_active, m_grid, kind="kde",
        bounds=(theta["m_min"], theta["m_max"] * 2.5),
        log_space=True,
    )
    dN_dchi = make_marginal(
        chi_active, chi_grid, kind="kde",
        bounds=(0.0, 1.0),
        log_space=False,
    )
    return dN_dm, dN_dchi


def mc_marginals_on_eval(
    m_active: np.ndarray,
    chi_active: np.ndarray,
    m_eval: np.ndarray,
    chi_eval: np.ndarray,
    theta: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the MC KDE on the fine eval grid (for plotting only)."""
    if m_active.size < 2:
        return np.zeros_like(m_eval), np.zeros_like(chi_eval)
    dN_dm = make_marginal(
        m_active, m_eval, kind="kde",
        bounds=(theta["m_min"], theta["m_max"] * 2.5),
        log_space=True,
    )
    dN_dchi = make_marginal(
        chi_active, chi_eval, kind="kde",
        bounds=(0.0, 1.0),
        log_space=False,
    )
    return dN_dm, dN_dchi


# ---------------------------------------------------------------------------
# JS helpers.
# ---------------------------------------------------------------------------
def js_subrange(
    p: np.ndarray,
    q: np.ndarray,
    x: np.ndarray,
    x_lo: float,
    x_hi: float,
) -> float:
    """JS divergence restricted to x ∈ [x_lo, x_hi]."""
    mask = (x >= x_lo) & (x <= x_hi)
    if mask.sum() < 4:
        return float("nan")
    return js_divergence(p[mask], q[mask], x[mask])


# ---------------------------------------------------------------------------
# Plotting helpers.
# ---------------------------------------------------------------------------
def _per_norm(arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    return _normalise(arr, x)


def plot_mass(
    m_eval: np.ndarray,
    imp: np.ndarray,
    mc1: np.ndarray,
    mc2: np.ndarray,
    m_bulk_max: float,
    js_bulk: float,
    js_full: float,
    js_noise_bulk: float,
    js_noise_full: float,
    js_natal_bulk: float,
    js_natal_full: float,
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    p_imp = _per_norm(imp, m_eval)
    p_mc1 = _per_norm(mc1, m_eval)
    p_mc2 = _per_norm(mc2, m_eval)
    eps_floor = max(p_imp.max(), p_mc1.max()) * 1e-6

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(m_eval, np.maximum(p_imp, eps_floor),
              color="C0", lw=2.2, ls="-",
              label="Package implicit Picard (production)")
    ax.loglog(m_eval, np.maximum(p_mc1, eps_floor),
              color="C2", lw=2.0, ls="--",
              label="Particle MC (seed 1)")
    ax.loglog(m_eval, np.maximum(p_mc2, eps_floor),
              color="C2", lw=0.9, ls=":",
              label="Particle MC (seed 2, shot-noise check)", alpha=0.7)
    lo = np.minimum(p_mc1, p_mc2)
    hi = np.maximum(p_mc1, p_mc2)
    ax.fill_between(m_eval, np.maximum(lo, eps_floor),
                    np.maximum(hi, eps_floor),
                    color="C2", alpha=0.15, label="MC shot-noise band")
    ax.axvline(m_bulk_max, color="0.5", lw=1.0, ls="-.",
               label=f"bulk/tail boundary ({m_bulk_max:.0f} $M_\\odot$)")

    ax.set_xlabel(r"single-BH mass $m \;[M_\odot]$")
    ax.set_ylabel(r"$p(m)$ (normalised)")
    ax.set_title(
        "Implicit Picard ↔ Particle MC — $p(m)$ at Stage 1 $\\theta^*$\n"
        "(JS computed at solver grid resolution)"
    )
    ax.set_ylim(bottom=eps_floor * 10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize="small")

    note = (
        f"JS bulk (m<{m_bulk_max:.0f}) [grid axis]:  {js_bulk:.4g}  "
        f"noise={js_noise_bulk:.4g}  natal={js_natal_bulk:.4g}\n"
        f"JS full [grid axis]:               {js_full:.4g}  "
        f"noise={js_noise_full:.4g}  natal={js_natal_full:.4g}"
    )
    ax.text(
        0.02, 0.03, note,
        transform=ax.transAxes, fontsize="x-small", family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.88),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_spin(
    chi_eval: np.ndarray,
    imp: np.ndarray,
    mc1: np.ndarray,
    mc2: np.ndarray,
    js_spin: float,
    js_noise_spin: float,
    js_natal_spin: float,
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    p_imp = _per_norm(imp, chi_eval)
    p_mc1 = _per_norm(mc1, chi_eval)
    p_mc2 = _per_norm(mc2, chi_eval)
    eps_floor = max(p_imp.max(), p_mc1.max()) * 1e-4

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(chi_eval, np.maximum(p_imp, eps_floor),
                color="C0", lw=2.2, ls="-",
                label="Package implicit Picard (production)")
    ax.semilogy(chi_eval, np.maximum(p_mc1, eps_floor),
                color="C2", lw=2.0, ls="--",
                label="Particle MC (seed 1)")
    ax.semilogy(chi_eval, np.maximum(p_mc2, eps_floor),
                color="C2", lw=0.9, ls=":",
                label="Particle MC (seed 2, shot-noise check)", alpha=0.7)
    lo = np.minimum(p_mc1, p_mc2)
    hi = np.maximum(p_mc1, p_mc2)
    ax.fill_between(chi_eval, np.maximum(lo, eps_floor),
                    np.maximum(hi, eps_floor),
                    color="C2", alpha=0.15, label="MC shot-noise band")
    ax.axvline(0.6865, color="0.4", lw=0.7, ls=":", label=r"$\chi_{\rm rem}(1,0)$")

    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(r"$p(\chi)$ (normalised)")
    ax.set_title(
        r"Implicit Picard ↔ Particle MC — $p(\chi)$ at Stage 1 $\theta^*$"
        "\n(JS computed at solver grid resolution)"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=eps_floor * 10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize="small")

    note = (
        f"JS(dN/dχ) [grid axis]:  {js_spin:.4g}  "
        f"noise={js_noise_spin:.4g}  natal={js_natal_spin:.4g}"
    )
    ax.text(
        0.02, 0.03, note,
        transform=ax.transAxes, fontsize="x-small", family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.88),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-hash", default=None)
    parser.add_argument(
        "--n-seed", type=int, default=20_000,
        help="Natal particles for the MC (per run). Default 20k gives good "
             "1G/2G statistics; 200k+ needed to resolve 3G+ peaks.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="Coupling ε for the implicit solver and MC stopping fraction.",
    )
    parser.add_argument(
        "--epsilon-natal", type=float, default=1e-4,
        help="Coupling for the natal-baseline run (representation-noise "
             "floor).  At this ε, N* ≈ N_natal; any JS measured here is "
             "grid-vs-KDE representation noise, not physics.",
    )
    parser.add_argument(
        "--no-natal-baseline", action="store_true",
        help="Skip the natal-baseline run (faster, but no representation "
             "noise floor estimate).",
    )
    parser.add_argument(
        "--m-bulk-max", type=float, default=25.0,
        help="Upper mass cutoff (M☉) for the 1G+2G 'bulk' JS. "
             "At Stage 1 θ*, 1G ≈ 8 M☉, 2G ≈ 15 M☉, 3G+ ≥ 25 M☉.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260518,
        help="Base RNG seed.  Two child seeds derived for the two MC runs.",
    )
    parser.add_argument(
        "--mc-depletion", type=float, default=None,
        help="Override the MC stopping fraction (normally from f_natal_depleted).",
    )
    args = parser.parse_args()

    here = Path(__file__).parent
    rh = args.run_hash or secrets.token_hex(3)
    fig_dir = here / f"figures_{rh}"
    art_dir = here / f"artifacts_{rh}"
    fig_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    print(f"Run hash: {rh}")
    print(f"  figures  → {fig_dir.relative_to(here.parent)}/")
    print(f"  artifacts → {art_dir.relative_to(here.parent)}/")

    # Fine eval grids — for plotting only.
    m_eval = np.geomspace(THETA_STAR["m_min"], THETA_STAR["m_max"] * 2.5, 400)
    chi_eval = np.linspace(0.001, 0.999, 400)

    rng = np.random.default_rng(args.seed)
    rng1, rng2, rng_nat = rng.spawn(3) if hasattr(rng, "spawn") else (
        np.random.default_rng(args.seed + 1),
        np.random.default_rng(args.seed + 2),
        np.random.default_rng(args.seed + 3),
    )

    # ------------------------------------------------------------------
    # 1. Package implicit solver at ε=args.epsilon.
    # ------------------------------------------------------------------
    print(f"\n[1/4] Package implicit Picard solver (ε={args.epsilon}) ...")
    m_grid, dN_dm_grid, chi_grid, dN_dchi_grid, imp_diag = implicit_solve(
        THETA_STAR, args.epsilon,
    )
    print(f"      converged in {imp_diag['n_iter']} iters, "
          f"|ΔN|∞ = {imp_diag['final_diff']:.2e}")
    print(f"      grid: {m_grid.size} m-points × {chi_grid.size} χ-points")
    print(f"      f_natal_depleted = {imp_diag['f_natal_depleted']:.4g}")

    # Interpolate to eval grids for plotting.
    dN_dm_eval = np.interp(np.log(m_eval), np.log(m_grid), dN_dm_grid)
    dN_dchi_eval = np.interp(chi_eval, chi_grid, dN_dchi_grid)

    # MC stopping fraction.
    if args.mc_depletion is not None:
        f_dep = args.mc_depletion
        dep_source = "CLI override"
    else:
        f_dep = imp_diag["f_natal_depleted"]
        dep_source = "implicit solver (self-consistent)"
    print(f"      MC stopping fraction: {f_dep:.4g}  [{dep_source}]")

    # ------------------------------------------------------------------
    # 2. Particle MC — seed 1.
    # ------------------------------------------------------------------
    print(f"\n[2/4] Particle MC — seed 1 (n_seed={args.n_seed}, "
          f"depletion={f_dep:.4g}) ...")
    m1_active, c1_active, mc1_stats = mc_run(
        THETA_STAR, args.n_seed, rng1, f_dep,
    )
    s = mc1_stats
    print(f"      n_accepted={s['n_accepted']}, n_retained={s['n_retained']}, "
          f"actual_depletion={s['depletion']:.4g}, stop={s['stop_reason']}")

    # ------------------------------------------------------------------
    # 3. Particle MC — seed 2 (shot-noise floor).
    # ------------------------------------------------------------------
    print(f"\n[3/4] Particle MC — seed 2 (shot-noise estimate) ...")
    m2_active, c2_active, mc2_stats = mc_run(
        THETA_STAR, args.n_seed, rng2, f_dep,
    )
    s2 = mc2_stats
    print(f"      n_accepted={s2['n_accepted']}, n_retained={s2['n_retained']}, "
          f"actual_depletion={s2['depletion']:.4g}, stop={s2['stop_reason']}")

    # ------------------------------------------------------------------
    # Evaluate MC KDE at the grid's own resolution (primary JS metric).
    # ------------------------------------------------------------------
    mc1_dm_grid, mc1_dchi_grid = mc_marginals_on_grid(
        m1_active, c1_active, m_grid, chi_grid, THETA_STAR,
    )
    mc2_dm_grid, mc2_dchi_grid = mc_marginals_on_grid(
        m2_active, c2_active, m_grid, chi_grid, THETA_STAR,
    )
    # Eval-axis marginals for plotting.
    mc1_dm_eval, mc1_dchi_eval = mc_marginals_on_eval(
        m1_active, c1_active, m_eval, chi_eval, THETA_STAR,
    )
    mc2_dm_eval, mc2_dchi_eval = mc_marginals_on_eval(
        m2_active, c2_active, m_eval, chi_eval, THETA_STAR,
    )

    # ------------------------------------------------------------------
    # 4. Natal baseline (representation noise floor).
    # ------------------------------------------------------------------
    js_natal_bulk = float("nan")
    js_natal_full = float("nan")
    js_natal_spin = float("nan")

    if not args.no_natal_baseline:
        print(f"\n[4/4] Natal baseline — ε={args.epsilon_natal} "
              "(representation noise floor) ...")
        m_grid_n, dN_dm_grid_n, chi_grid_n, dN_dchi_grid_n, nat_diag = \
            implicit_solve(THETA_STAR, args.epsilon_natal)
        print(f"      converged in {nat_diag['n_iter']} iters; "
              f"f_dep={nat_diag['f_natal_depleted']:.4g}")

        # MC at the same tiny ε — nearly all natal, ~no mergers.
        f_dep_nat = float(nat_diag["f_natal_depleted"])
        mn_active, cn_active, _ = mc_run(
            THETA_STAR, args.n_seed, rng_nat, max(f_dep_nat, 1e-6),
        )
        mc_nat_dm, mc_nat_dchi = mc_marginals_on_grid(
            mn_active, cn_active, m_grid_n, chi_grid_n, THETA_STAR,
        )
        js_natal_bulk = js_subrange(
            dN_dm_grid_n, mc_nat_dm, m_grid_n,
            float(THETA_STAR["m_min"]), args.m_bulk_max,
        )
        js_natal_full = js_divergence(dN_dm_grid_n, mc_nat_dm, m_grid_n)
        js_natal_spin = js_divergence(dN_dchi_grid_n, mc_nat_dchi, chi_grid_n)
        print(f"      JS(natal) bulk={js_natal_bulk:.4g}  "
              f"full={js_natal_full:.4g}  spin={js_natal_spin:.4g}")
    else:
        print("\n[4/4] Natal baseline skipped (--no-natal-baseline).")

    # ------------------------------------------------------------------
    # JS metrics at grid resolution.
    # ------------------------------------------------------------------
    m_lo = float(THETA_STAR["m_min"])

    js_bulk_imp_mc = js_subrange(dN_dm_grid, mc1_dm_grid, m_grid, m_lo, args.m_bulk_max)
    js_full_imp_mc = js_divergence(dN_dm_grid, mc1_dm_grid, m_grid)
    js_spin_imp_mc = js_divergence(dN_dchi_grid, mc1_dchi_grid, chi_grid)

    js_bulk_noise = js_subrange(mc1_dm_grid, mc2_dm_grid, m_grid, m_lo, args.m_bulk_max)
    js_full_noise = js_divergence(mc1_dm_grid, mc2_dm_grid, m_grid)
    js_spin_noise = js_divergence(mc1_dchi_grid, mc2_dchi_grid, chi_grid)

    # ------------------------------------------------------------------
    # Report table.
    # ------------------------------------------------------------------
    print(f"\nJS at solver grid resolution ({m_grid.size}m × {chi_grid.size}χ):")
    header = (
        f"\n{'metric':<34}  {'imp↔MC':>10}  {'MC↔MC':>10}  "
        f"{'natal':>10}  {'phys/noise':>11}  {'phys/natal':>11}"
    )
    sep = "-" * len(header)
    rows = [
        (f"dN/dm  bulk  (m ≤{args.m_bulk_max:.0f} M☉)",
         js_bulk_imp_mc, js_bulk_noise, js_natal_bulk),
        ("dN/dm  full  (all m)",
         js_full_imp_mc, js_full_noise, js_natal_full),
        ("dN/dχ  full  (all χ)",
         js_spin_imp_mc, js_spin_noise, js_natal_spin),
    ]
    print(header)
    print(sep)
    table_lines = [header.strip(), sep]
    for label, js_val, js_noi, js_nat in rows:
        r_noise = js_val / js_noi if (js_noi > 0 and np.isfinite(js_noi)) else float("nan")
        r_natal = js_val / js_nat if (js_nat > 0 and np.isfinite(js_nat)) else float("nan")
        line = (f"{label:<34}  {js_val:>10.4g}  {js_noi:>10.4g}  "
                f"{js_nat:>10.4g}  {r_noise:>11.1f}  {r_natal:>11.1f}")
        print(line)
        table_lines.append(line)

    print()
    print("  Column guide:")
    print("  imp↔MC   — implicit vs MC seed 1  (primary signal)")
    print("  MC↔MC    — MC seed 1 vs seed 2    (shot noise floor)")
    print("  natal    — JS at ε→0              (grid-vs-KDE representation noise)")
    print("  phys/noise — imp↔MC / MC↔MC ratio; ≈1 = shot noise, ≫1 = systematic")
    print("  phys/natal — imp↔MC / natal ratio;  ≈1 = representation noise, ≫1 = physics")
    print()

    # ------------------------------------------------------------------
    # Artifacts.
    # ------------------------------------------------------------------
    (art_dir / "compare_implicit_vs_mc_js.txt").write_text(
        "\n".join(table_lines) + "\n"
        + f"\nGrid: {m_grid.size}m × {chi_grid.size}χ\n"
        + f"Implicit diag: {imp_diag}\n"
        + f"MC seed1 stats: {mc1_stats}\n"
        + f"MC seed2 stats: {mc2_stats}\n"
        + f"JS natal: bulk={js_natal_bulk:.4g} full={js_natal_full:.4g} "
          f"spin={js_natal_spin:.4g}\n"
    )
    np.savez(
        art_dir / "compare_implicit_vs_mc.npz",
        m_eval=m_eval, chi_eval=chi_eval,
        m_grid=m_grid, chi_grid=chi_grid,
        # Grid-axis arrays (JS is computed here).
        imp_dm_grid=dN_dm_grid, imp_dchi_grid=dN_dchi_grid,
        mc1_dm_grid=mc1_dm_grid, mc1_dchi_grid=mc1_dchi_grid,
        mc2_dm_grid=mc2_dm_grid, mc2_dchi_grid=mc2_dchi_grid,
        # Eval-axis arrays (for plotting).
        imp_dm_eval=dN_dm_eval, imp_dchi_eval=dN_dchi_eval,
        mc1_dm_eval=mc1_dm_eval, mc1_dchi_eval=mc1_dchi_eval,
        mc2_dm_eval=mc2_dm_eval, mc2_dchi_eval=mc2_dchi_eval,
        # Scalar results.
        js_bulk_imp_mc=np.float64(js_bulk_imp_mc),
        js_full_imp_mc=np.float64(js_full_imp_mc),
        js_spin_imp_mc=np.float64(js_spin_imp_mc),
        js_bulk_noise=np.float64(js_bulk_noise),
        js_full_noise=np.float64(js_full_noise),
        js_spin_noise=np.float64(js_spin_noise),
        js_natal_bulk=np.float64(js_natal_bulk),
        js_natal_full=np.float64(js_natal_full),
        js_natal_spin=np.float64(js_natal_spin),
        f_natal_depleted=np.float64(f_dep),
        epsilon=np.float64(args.epsilon),
        m_bulk_max=np.float64(args.m_bulk_max),
        n_seed=np.int64(args.n_seed),
    )
    print(f"  artifacts written to {art_dir.relative_to(here.parent)}/")

    # ------------------------------------------------------------------
    # Plots (on eval grids for smooth visuals).
    # ------------------------------------------------------------------
    plot_mass(
        m_eval, dN_dm_eval, mc1_dm_eval, mc2_dm_eval,
        m_bulk_max=args.m_bulk_max,
        js_bulk=js_bulk_imp_mc, js_full=js_full_imp_mc,
        js_noise_bulk=js_bulk_noise, js_noise_full=js_full_noise,
        js_natal_bulk=js_natal_bulk, js_natal_full=js_natal_full,
        out=fig_dir / "compare_implicit_vs_mc_dN_dm.png",
    )
    plot_spin(
        chi_eval, dN_dchi_eval, mc1_dchi_eval, mc2_dchi_eval,
        js_spin=js_spin_imp_mc,
        js_noise_spin=js_spin_noise,
        js_natal_spin=js_natal_spin,
        out=fig_dir / "compare_implicit_vs_mc_dN_dchi.png",
    )


if __name__ == "__main__":
    main()
