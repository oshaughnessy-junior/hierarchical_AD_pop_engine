"""4-way overlay comparison — engine vs sim vs Backup 1 vs Backup 2.

Runs all four implementations at the Stage 1 θ* and produces a single
``dN/dm`` overlay with each curve normalised as a probability density.
Spin gets a 3-way overlay (Backup 2 is mass-only).

Goal: the human reads the figure and decides which implementations agree
on the "true" total ``N(m, χ)``. Pairwise Jensen-Shannon divergences in
the legend quantify pairwise disagreement.

Why this exists: red-team Phase 1 showed engine and sim disagree at 3G+
with JS ≈ 0.18-0.36; my first guesses (engine j!=l double-count,
simulator weighting) only partially closed the gap. We now have two
backups that *cannot be wrong by construction* (Backup 1 = pure
particle MC) or *match the Doctor reference exactly* (Backup 2). If the
backups agree with each other and not with the engine, the engine
needs further work — and the diagnostic will be either the loss term
(Backup 2 has it, engine doesn't) or higher-order Volterra
coefficients (Backup 2's 1/3! at 3rd order, engine has 1).

Run::

    python -m red_team.compare_all                  # default Stage 1 θ*
    python -m red_team.compare_all --epsilon 0.7    # tune Backup 2 cascade depth
"""

from __future__ import annotations

import argparse
import itertools
import secrets
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np

from hierarchical_ad_pop_engine.response.coagulation import (
    build_gaussian_coagulation,
)

from red_team import backup_particle_mc as B1
from red_team import backup_perturbative as B2
from red_team import engine_v2_implicit as V2
from red_team.phase1_overlay import (
    THETA_STAR,
    _normalise,
    _trapezoid_weights,
    engine_per_generation_marginals,
    js_divergence,
    simulator_per_generation_marginals,
)
from red_team.sim_coagulation import make_marginal


# ---------------------------------------------------------------------------
# Helpers — aggregate per-generation n_k into Σ_k n_k for the engine and
# the simulator, then KDE the particle-MC population on the common axis.
# ---------------------------------------------------------------------------
def engine_total_marginals(theta: dict, m_eval: np.ndarray, chi_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Engine's Σ_k n_k marginalised to ``dN/dm`` and ``dN/dχ`` on common axes."""
    dN_dm_per_gen, dN_dchi_per_gen = engine_per_generation_marginals(theta, m_eval, chi_eval)
    return dN_dm_per_gen.sum(axis=0), dN_dchi_per_gen.sum(axis=0)


def sim_total_marginals(theta: dict, m_eval: np.ndarray, chi_eval: np.ndarray,
                        n_seed: int, n_pairs_per_pass: int, rng: np.random.Generator,
                        kind: str = "kde") -> tuple[np.ndarray, np.ndarray]:
    """Simulator's Σ_k populations[k] → joint KDE on common axes."""
    sim_dN_dm, sim_dN_dchi, populations, _events = simulator_per_generation_marginals(
        theta, m_eval, chi_eval,
        n_seed=n_seed,
        n_pairs_per_pass=n_pairs_per_pass,
        rng=rng,
        kind=kind,
    )
    # Total = stack all generations' particles, KDE once.
    all_m_list = []
    all_chi_list = []
    for k in sorted(populations.keys()):
        m_k, chi_k = populations[k]
        if m_k.size:
            all_m_list.append(m_k)
            all_chi_list.append(chi_k)
    if not all_m_list:
        return np.zeros_like(m_eval), np.zeros_like(chi_eval)
    all_m = np.concatenate(all_m_list)
    all_chi = np.concatenate(all_chi_list)
    dN_dm = make_marginal(all_m, m_eval, kind=kind,
                          bounds=(m_eval[0], m_eval[-1]), log_space=True)
    dN_dchi = make_marginal(all_chi, chi_eval, kind=kind,
                            bounds=(0.0, 1.0), log_space=False)
    return dN_dm, dN_dchi


def backup1_marginals(theta: dict, m_eval: np.ndarray, chi_eval: np.ndarray,
                      n_seed: int, rng: np.random.Generator,
                      kind: str = "kde",
                      n_max_mergers: Optional[int] = None,
                      depletion_fraction: float = 0.05,
                      ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Backup 1 (pure particle MC) — surviving population KDE'd.

    Default stopping at depletion_fraction = 0.05 puts the cascade in the
    early-evolution regime where engine ≈ Backup 2 implicit at ε ≈ 0.025.
    Override via n_max_mergers if you want a deeper cascade.
    """
    cfg = B1.MCConfig(
        n_seed=n_seed,
        n_max_mergers=n_max_mergers,
        depletion_fraction=depletion_fraction,
        **{k: theta[k] for k in (
            "mu_natal", "sigma_natal", "m_min", "m_max",
            "natal_spin_loc", "natal_spin_scale",
            "kernel_a", "kernel_b", "v_esc",
        )},
    )
    result = B1.particle_mc(cfg, rng, track_generations=False)
    m_a = result["m_active"]
    chi_a = result["chi_active"]
    if m_a.size < 2:
        return np.zeros_like(m_eval), np.zeros_like(chi_eval), result["stats"]
    dN_dm = make_marginal(m_a, m_eval, kind=kind,
                          bounds=(theta["m_min"], theta["m_max"] * 2),
                          log_space=True)
    dN_dchi = make_marginal(chi_a, chi_eval, kind=kind,
                            bounds=(0.0, 1.0), log_space=False)
    return dN_dm, dN_dchi, result["stats"]


def engine_v2_marginals(theta: dict, m_eval: np.ndarray, chi_eval: np.ndarray,
                        epsilon: float = 1.0,
                        ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Engine v2 prototype — implicit Picard on (m, χ). Reuses engine
    primitives but with proper gain−loss accumulator and all-orders
    fixed-point iteration.

    ``epsilon`` here is engine-v2's coupling. Not 1:1 with Backup 2's ε
    (different grid measure / dimensionality); see
    ``engine_v2_implicit.solve_implicit`` docstring.
    """
    from hierarchical_ad_pop_engine.response.coagulation import (
        build_gaussian_coagulation,
    )
    response = build_gaussian_coagulation(
        mu=theta["mu_natal"], sigma=theta["sigma_natal"],
        m_min=theta["m_min"], m_max=theta["m_max"],
        natal_spin_loc=theta["natal_spin_loc"],
        natal_spin_scale=theta["natal_spin_scale"],
        kernel_a=theta["kernel_a"], kernel_b=theta["kernel_b"],
        log_R_env=0.0, k_max=theta["k_max"],
    )
    result = V2.solve_implicit(
        response, jnp.asarray(theta["v_esc"]),
        epsilon=epsilon,
        max_iter=200, tol=1e-7, damping=0.4,
    )
    m_grid, dN_dm_grid, chi_grid, dN_dchi_grid = V2.marginals(result["N"], response)
    log_m_grid = np.log(m_grid)
    log_m_eval = np.log(m_eval)
    dN_dm_eval = np.interp(log_m_eval, log_m_grid, dN_dm_grid)
    dN_dchi_eval = np.interp(chi_eval, chi_grid, dN_dchi_grid)
    diag = dict(
        n_iter=result["n_iter"],
        final_diff=result["final_diff"],
        f_natal_depleted=result["f_natal_depleted"],
        loss_integral=result["loss_integral"],
        natal_integral=result["natal_integral"],
    )
    return dN_dm_eval, dN_dchi_eval, diag


def backup2_mass_marginal(theta: dict, m_eval: np.ndarray, epsilon: float,
                          mode: str = "implicit") -> tuple[np.ndarray, dict]:
    """Backup 2 (Doctor port) — mass only, perturbative or implicit."""
    cfg = B2.PerturbConfig(epsilon=epsilon, **{k: theta[k] for k in (
        "mu_natal", "sigma_natal", "m_min", "m_max", "kernel_a", "kernel_b",
    )})
    if mode == "implicit":
        sol = B2.implicit_picard(cfg)
    else:
        sol = B2.perturbative_u_net(cfg)
    mvals = sol["mvals"]
    u_net = sol["u_net"]
    # Interpolate to common eval grid. u_net is on linear grid; eval is
    # geomspace. Use np.interp.
    u_on_eval = np.interp(m_eval, mvals, np.maximum(u_net, 0.0))
    diag = dict(
        n_iter=sol.get("n_iter"),
        final_diff=sol.get("final_diff"),
        integral=float(np.trapezoid(u_net, mvals)),
    )
    return u_on_eval, diag


# ---------------------------------------------------------------------------
# Plot.
# ---------------------------------------------------------------------------
def _per_norm(arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    Z = float(np.trapezoid(arr, x))
    return arr / Z if Z > 0 and np.isfinite(Z) else np.zeros_like(arr)


def plot_dN_dm_4way(
    m_eval: np.ndarray,
    curves: dict[str, np.ndarray],
    js_table: dict[tuple[str, str], float],
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    # Normalise each curve to probability density.
    norms = {k: _per_norm(v, m_eval) for k, v in curves.items()}
    eps = max(np.max(v) for v in norms.values() if np.max(v) > 0) * 1e-6

    style = {
        "engine":     dict(color="C0", ls="-",  lw=2.0, label="engine (Σ n_k, perturbative)"),
        "sim":        dict(color="C1", ls="--", lw=2.0, label="sim (importance-sampled MC)"),
        "backup1":    dict(color="C2", ls=":",  lw=2.5, label="Backup 1 (pure particle MC)"),
        "backup2":    dict(color="C3", ls="-.", lw=2.0, label="Backup 2 (Doctor implicit, 1D)"),
        "engine_v2":  dict(color="C4", ls="-",  lw=2.0, label="engine v2 (implicit Picard, 2D)"),
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, n in norms.items():
        if n.max() == 0:
            continue
        ax.loglog(m_eval, np.maximum(n, eps), **style[key])
    ax.set_xlabel(r"single-BH mass $m \;[M_\odot]$")
    ax.set_ylabel(r"$p(m)$ (each curve integrates to 1)")
    ax.set_title("4-way comparison — total $p(m)$ at Stage 1 $\\theta^*$")
    ax.set_ylim(bottom=eps * 10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False)
    # JS table inset as text.
    lines = []
    for (a, b), v in js_table.items():
        lines.append(f"JS({a}↔{b}) = {v:.3g}")
    ax.text(
        0.02, 0.02, "\n".join(lines),
        transform=ax.transAxes, fontsize="x-small", family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_dN_dchi_3way(
    chi_eval: np.ndarray,
    curves: dict[str, np.ndarray],
    js_table: dict[tuple[str, str], float],
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    norms = {k: _per_norm(v, chi_eval) for k, v in curves.items() if v.max() > 0}
    if not norms:
        return
    eps = max(v.max() for v in norms.values()) * 1e-4

    style = {
        "engine":     dict(color="C0", ls="-",  lw=2.0, label="engine (Σ n_k, perturbative)"),
        "sim":        dict(color="C1", ls="--", lw=2.0, label="sim (importance-sampled MC)"),
        "backup1":    dict(color="C2", ls=":",  lw=2.5, label="Backup 1 (pure particle MC)"),
        "engine_v2":  dict(color="C4", ls="-",  lw=2.0, label="engine v2 (implicit Picard, 2D)"),
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, n in norms.items():
        ax.semilogy(chi_eval, np.maximum(n, eps), **style[key])
    ax.axvline(0.6865, color="0.4", lw=0.7, ls=":", label=r"$\chi_{\rm rem}(1,0)$")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(r"$p(\chi)$ (each curve integrates to 1)")
    ax.set_title(r"3-way comparison — total $p(\chi)$ at Stage 1 $\theta^*$ (Backup 2 is mass-only)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=eps * 10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False)
    lines = []
    for (a, b), v in js_table.items():
        lines.append(f"JS({a}↔{b}) = {v:.3g}")
    ax.text(
        0.02, 0.02, "\n".join(lines),
        transform=ax.transAxes, fontsize="x-small", family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-hash", default=None)
    parser.add_argument("--n-seed", type=int, default=20_000)
    parser.add_argument("--n-pairs-per-pass", type=int, default=20_000)
    parser.add_argument("--epsilon", type=float, default=1.0,
                        help="Backup 2 coupling × time. Larger ε → deeper cascade. "
                        "User observation: ε ≈ 0.025 makes Backup 2 implicit "
                        "≈ engine in the Stage 1 setup.")
    parser.add_argument("--v2-epsilon", type=float, default=None,
                        help="Engine v2's coupling. After the cell-area "
                        "Jacobian fix, engine v2's ε is on the same absolute "
                        "scale as Backup 2's ε (both have [time] units). "
                        "Default: same as --epsilon.")
    parser.add_argument("--b1-depletion-fraction", type=float, default=None,
                        help="Backup 1 stopping criterion (fraction of natal "
                        "pool consumed). If unset, derived self-consistently "
                        "from engine v2's implicit fixed point: "
                        "f = ε · ∫L[N*]/∫N_natal. This is the principled "
                        "mapping between the implicit coupling ε and the "
                        "closed-system MC's stopping time. Override only for "
                        "sensitivity studies.")
    parser.add_argument("--b1-n-max-mergers", type=int, default=None,
                        help="Hard cap on Backup 1 mergers; overrides "
                        "--b1-depletion-fraction.")
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    here = Path(__file__).parent
    rh = args.run_hash or secrets.token_hex(3)
    fig_dir = here / f"figures_{rh}"
    art_dir = here / f"artifacts_{rh}"
    fig_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    print(f"Run hash: {rh}")
    print(f"  figures → {fig_dir.relative_to(here.parent)}/")
    print(f"  artifacts → {art_dir.relative_to(here.parent)}/")

    m_eval = np.geomspace(THETA_STAR["m_min"], THETA_STAR["m_max"] * 2.5, 400)
    chi_eval = np.linspace(0.001, 0.999, 400)

    rng = np.random.default_rng(args.seed)
    k_sim, k_b1 = rng.spawn(2) if hasattr(rng, "spawn") else (
        np.random.default_rng(args.seed + 1), np.random.default_rng(args.seed + 2),
    )

    # --- Engine ---
    print("Engine ...")
    eng_dm, eng_dchi = engine_total_marginals(THETA_STAR, m_eval, chi_eval)

    # --- Simulator ---
    print(f"Simulator (n_seed={args.n_seed}, n_pairs/pass={args.n_pairs_per_pass}) ...")
    sim_dm, sim_dchi = sim_total_marginals(
        THETA_STAR, m_eval, chi_eval, args.n_seed, args.n_pairs_per_pass, k_sim,
    )

    # --- Engine v2 prototype: 2D implicit Picard ---
    # Run *before* Backup 1 so we can use its self-consistent depletion
    # fraction for the closed-system particle MC. This is the principled
    # mapping between the implicit coupling ε and the MC stopping time:
    # f_dep = ε · ∫L[N*] / ∫N_natal at the converged fixed point.
    v2_eps = args.v2_epsilon if args.v2_epsilon is not None else args.epsilon
    print(f"Engine v2 — implicit Picard in 2D (m, χ), ε={v2_eps} ...")
    v2_dm, v2_dchi, v2_diag = engine_v2_marginals(THETA_STAR, m_eval, chi_eval,
                                                  epsilon=v2_eps)
    print(f"  Engine v2 diag: {v2_diag}")
    print(f"  Engine v2 natal-depletion fraction (= ε·∫L/∫N_natal): "
          f"f = {v2_diag['f_natal_depleted']:.4g}")

    # --- Backup 2: implicit Picard ---
    print(f"Backup 2 — Doctor implicit Picard (ε={args.epsilon}) ...")
    b2_dm, b2_diag = backup2_mass_marginal(
        THETA_STAR, m_eval, epsilon=args.epsilon, mode="implicit",
    )
    print(f"  Backup 2 diag: {b2_diag}")

    # --- Backup 1: pure particle MC ---
    if args.b1_depletion_fraction is None:
        # Self-consistent default: stop the MC when it has depleted the
        # same fraction of the natal pool that the implicit fixed point
        # predicts. This is the principled ε ↔ depletion mapping; no
        # empirical knob.
        b1_depletion = float(v2_diag["f_natal_depleted"])
        b1_dep_source = "engine_v2 (self-consistent)"
    else:
        b1_depletion = args.b1_depletion_fraction
        b1_dep_source = "CLI override"
    print(f"Backup 1 — pure particle MC (n_seed={args.n_seed}, "
          f"depletion_fraction={b1_depletion:.4g} [{b1_dep_source}]"
          f"{', n_max_mergers='+str(args.b1_n_max_mergers) if args.b1_n_max_mergers else ''}) ...")
    b1_dm, b1_dchi, b1_stats = backup1_marginals(
        THETA_STAR, m_eval, chi_eval, args.n_seed, k_b1,
        n_max_mergers=args.b1_n_max_mergers,
        depletion_fraction=b1_depletion,
    )
    print(f"  Backup 1 stats: {b1_stats}")

    # --- Pairwise JS divergence on dN/dm ---
    js_mass = {}
    curves_m = dict(engine=eng_dm, sim=sim_dm, backup1=b1_dm, backup2=b2_dm, engine_v2=v2_dm)
    for a, b in itertools.combinations(curves_m.keys(), 2):
        if curves_m[a].max() == 0 or curves_m[b].max() == 0:
            js_mass[(a, b)] = float("nan")
        else:
            js_mass[(a, b)] = js_divergence(curves_m[a], curves_m[b], m_eval)

    # --- Pairwise JS divergence on dN/dχ (no Backup 2) ---
    js_spin = {}
    curves_c = dict(engine=eng_dchi, sim=sim_dchi, backup1=b1_dchi, engine_v2=v2_dchi)
    for a, b in itertools.combinations(curves_c.keys(), 2):
        if curves_c[a].max() == 0 or curves_c[b].max() == 0:
            js_spin[(a, b)] = float("nan")
        else:
            js_spin[(a, b)] = js_divergence(curves_c[a], curves_c[b], chi_eval)

    # Report.
    print()
    print(f"{'pair':<24}  {'JS(dN/dm)':>12}  {'JS(dN/dχ)':>12}")
    table_lines = ["pair                       JS(dN/dm)     JS(dN/dchi)"]
    pairs = sorted(set(js_mass.keys()) | set(js_spin.keys()))
    for a, b in pairs:
        jm = js_mass.get((a, b), float("nan"))
        jc = js_spin.get((a, b), float("nan"))
        line = f"{a:<10}↔ {b:<12}  {jm:>12.4g}  {jc:>12.4g}"
        print(line)
        table_lines.append(line)
    (art_dir / "compare_all_js_table.txt").write_text("\n".join(table_lines) + "\n")

    # Save.
    np.savez(
        art_dir / "compare_all_artifacts.npz",
        m_eval=m_eval, chi_eval=chi_eval,
        eng_dm=eng_dm, eng_dchi=eng_dchi,
        sim_dm=sim_dm, sim_dchi=sim_dchi,
        b1_dm=b1_dm, b1_dchi=b1_dchi,
        b2_dm=b2_dm,
        v2_dm=v2_dm, v2_dchi=v2_dchi,
    )

    # Plot.
    plot_dN_dm_4way(m_eval, curves_m, js_mass, fig_dir / "compare_all_dN_dm.png")
    plot_dN_dchi_3way(chi_eval, curves_c, js_spin, fig_dir / "compare_all_dN_dchi.png")


if __name__ == "__main__":
    main()
