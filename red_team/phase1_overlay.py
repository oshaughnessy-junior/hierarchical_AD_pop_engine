"""Phase 1 — engine ↔ simulator per-generation overlay.

Runs the continuous-density engine and the discrete particle simulator at
the *same* θ* (Stage 1's truth), produces per-generation 1-D marginal
overlays of ``dN/dm`` and ``dN/dχ``, and computes Jensen-Shannon
divergence per generation per marginal on a common evaluation grid.

Line-style convention: viridis color per generation, solid for engine,
dashed for simulator. The human reads the overlay; the JS number in the
legend quantifies the discrepancy at a glance.

Why JS, not KS: KS on empirical CDFs is fragile when one or both
densities go to ~0 (the supremum can be dominated by a single tail
sample and bounces around with realisation noise). JS is bounded by
``log 2`` and stable to vanishing tail mass; small values mean
distributions agree on shape.

Run::

    python -m red_team.phase1_overlay              # default: KDE + θ*

Outputs (under ``red_team/figures_<hash>/`` and
``red_team/artifacts_<hash>/``)::

    phase1_per_gen_mass.png       # log-log dN/dm overlay
    phase1_per_gen_spin.png       # semilog-y dN/dχ overlay
    phase1_js_table.txt           # per-gen JS table (mass + spin)
    phase1_artifacts.npz          # eval grids + per-gen densities (both sides)
"""

from __future__ import annotations

import argparse
import secrets
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Engine — allowed here because this is the *comparison driver*, not the
# simulator. The red-team rule applies to red_team/sim_coagulation.py,
# which still doesn't import response.*.
from hierarchical_ad_pop_engine.response.coagulation import (
    build_gaussian_coagulation,
)

from red_team.sim_coagulation import (
    SimConfig,
    make_marginal,
    simulate_cascade,
)


# ---------------------------------------------------------------------------
# Stage 1 truth point — kept in sync with demo/demo_gwkokab_fit_stage1.py.
# ---------------------------------------------------------------------------
THETA_STAR = dict(
    mu_natal=8.0,
    sigma_natal=1.0,
    m_min=4.0,
    m_max=80.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.10,
    kernel_a=0.0,
    kernel_b=1.0,
    v_esc=1000.0,
    k_max=4,
)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    return np.concatenate(
        [dx[:1] * 0.5, 0.5 * (dx[:-1] + dx[1:]), dx[-1:] * 0.5]
    )


def _normalise(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Renormalise a density so ∫ p dx = 1 on the eval grid."""
    Z = float(np.trapezoid(p, x))
    if Z <= 0 or not np.isfinite(Z):
        return np.zeros_like(p)
    return p / Z


def js_divergence(
    p: np.ndarray, q: np.ndarray, x: np.ndarray, eps: float = 1e-30
) -> float:
    """Jensen-Shannon divergence in nats between densities ``p`` and ``q``
    evaluated on the common axis ``x``.

    Both inputs are normalised before the divergence is computed, so the
    metric is invariant under per-density rescaling. Bounded by
    ``log 2 ≈ 0.693``.
    """
    pn = _normalise(p, x)
    qn = _normalise(q, x)
    if pn.max() == 0 or qn.max() == 0:
        return float("nan")
    m = 0.5 * (pn + qn)
    dx = _trapezoid_weights(x)
    kl_pm = float(np.sum(np.where(pn > 0, pn * np.log((pn + eps) / (m + eps)), 0.0) * dx))
    kl_qm = float(np.sum(np.where(qn > 0, qn * np.log((qn + eps) / (m + eps)), 0.0) * dx))
    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------------------------------
# Engine side: solve at θ*, marginalise n_k(m,χ) to per-gen dN/dm, dN/dχ on
# the engine grid; then interpolate onto the common eval grid.
# ---------------------------------------------------------------------------
def engine_per_generation_marginals(
    theta: dict, m_eval: np.ndarray, chi_eval: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dN_dm[K, N_m_eval], dN_dchi[K, N_chi_eval])`` from the engine."""
    response = build_gaussian_coagulation(
        mu=theta["mu_natal"],
        sigma=theta["sigma_natal"],
        m_min=theta["m_min"],
        m_max=theta["m_max"],
        natal_spin_loc=theta["natal_spin_loc"],
        natal_spin_scale=theta["natal_spin_scale"],
        kernel_a=theta["kernel_a"],
        kernel_b=theta["kernel_b"],
        log_R_env=0.0,  # absolute rate is irrelevant; we compare shapes
        k_max=theta["k_max"],
    )
    log_n_k = response._solve(jnp.asarray(theta["v_esc"]))  # (K, N_m, N_chi)
    n_k = np.asarray(jnp.exp(log_n_k))
    grid = response.grid
    m_grid = np.asarray(grid.m)
    chi_grid = np.asarray(grid.chi)
    dchi = _trapezoid_weights(chi_grid)
    dm = _trapezoid_weights(m_grid)

    # Per-generation marginals on the engine grid.
    dN_dm_engine = np.einsum("kmc,c->km", n_k, dchi)  # (K, N_m)
    dN_dchi_engine = np.einsum("kmc,m->kc", n_k, dm)  # (K, N_chi)

    # Interpolate to common eval grids. For mass we interpolate in log m
    # (engine grid is geomspace) so curvature is captured cleanly.
    log_m_grid = np.log(m_grid)
    log_m_eval = np.log(m_eval)
    K = n_k.shape[0]
    dN_dm_eval = np.empty((K, m_eval.size))
    dN_dchi_eval = np.empty((K, chi_eval.size))
    for k in range(K):
        dN_dm_eval[k] = np.interp(log_m_eval, log_m_grid, dN_dm_engine[k])
        dN_dchi_eval[k] = np.interp(chi_eval, chi_grid, dN_dchi_engine[k])
    return dN_dm_eval, dN_dchi_eval


# ---------------------------------------------------------------------------
# Simulator side: per-generation populations → KDE → density on eval grid.
# ---------------------------------------------------------------------------
def simulator_per_generation_marginals(
    theta: dict,
    m_eval: np.ndarray,
    chi_eval: np.ndarray,
    n_seed: int,
    n_pairs_per_pass: int,
    rng: np.random.Generator,
    kind: str = "kde",
) -> tuple[
    np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray
]:
    """Return (dN/dm, dN/dχ, populations, merger_events)."""
    cfg = SimConfig(
        mu_natal=theta["mu_natal"],
        sigma_natal=theta["sigma_natal"],
        m_min=theta["m_min"],
        m_max=theta["m_max"],
        natal_spin_loc=theta["natal_spin_loc"],
        natal_spin_scale=theta["natal_spin_scale"],
        kernel_a=theta["kernel_a"],
        kernel_b=theta["kernel_b"],
        v_esc=theta["v_esc"],
        k_max=theta["k_max"],
        n_seed=n_seed,
        n_pairs_per_generation_pair=n_pairs_per_pass,
    )
    populations, events = simulate_cascade(cfg, rng)

    K = max(populations.keys())
    dN_dm = np.zeros((K, m_eval.size))
    dN_dchi = np.zeros((K, chi_eval.size))
    for k in range(1, K + 1):
        m_k, chi_k = populations[k]
        if m_k.size < 2:
            continue
        dN_dm[k - 1] = make_marginal(
            m_k, m_eval, kind=kind,
            bounds=(m_eval[0], m_eval[-1]), log_space=True,
        )
        dN_dchi[k - 1] = make_marginal(
            chi_k, chi_eval, kind=kind, bounds=(0.0, 1.0), log_space=False,
        )
    return dN_dm, dN_dchi, populations, events


# ---------------------------------------------------------------------------
# Overlay plots.
# ---------------------------------------------------------------------------
def _per_gen_normalise(arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-row trapezoidal-normalise so each generation integrates to 1
    on ``x``. Rows that integrate to ≤ 0 (empty generation) are zeroed
    out so they don't break log plotting."""
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        Z = float(np.trapezoid(arr[i], x))
        if Z > 0 and np.isfinite(Z):
            out[i] = arr[i] / Z
    return out


def plot_overlay_mass(
    m_eval: np.ndarray,
    eng: np.ndarray,
    sim: np.ndarray,
    js_per_gen: list[float],
    out: Path,
) -> None:
    """Per-generation overlay of ``dN/dm`` normalised so each curve is a
    probability density. JS in the legend judges shape agreement; the
    visual judges modal structure and tail behaviour."""
    import matplotlib.pyplot as plt

    K = eng.shape[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, K))
    eng_n = _per_gen_normalise(eng, m_eval)
    sim_n = _per_gen_normalise(sim, m_eval)
    eps = max(eng_n.max(), sim_n.max()) * 1e-6

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for k in range(K):
        if eng_n[k].max() == 0 and sim_n[k].max() == 0:
            continue
        e = np.maximum(eng_n[k], eps)
        s = np.maximum(sim_n[k], eps)
        ax.loglog(
            m_eval, e, color=colors[k], lw=2.0, ls="-",
            label=f"{k+1}G engine  (JS={js_per_gen[k]:.3g})",
        )
        ax.loglog(m_eval, s, color=colors[k], lw=2.0, ls="--", label=f"{k+1}G sim")
    ax.set_xlabel(r"primary mass $m \;[M_\odot]$")
    ax.set_ylabel(r"$p(m)$ per generation (each curve integrates to 1)")
    ax.set_title(r"Phase 1 — engine (solid) vs simulator (dashed) per-gen $p(m)$")
    ax.set_ylim(bottom=eps * 10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower left", frameon=False, fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_overlay_spin(
    chi_eval: np.ndarray,
    eng: np.ndarray,
    sim: np.ndarray,
    js_per_gen: list[float],
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    K = eng.shape[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, K))
    eng_n = _per_gen_normalise(eng, chi_eval)
    sim_n = _per_gen_normalise(sim, chi_eval)
    eps = max(eng_n.max(), sim_n.max()) * 1e-4

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for k in range(K):
        if eng_n[k].max() == 0 and sim_n[k].max() == 0:
            continue
        e = np.maximum(eng_n[k], eps)
        s = np.maximum(sim_n[k], eps)
        ax.semilogy(
            chi_eval, e, color=colors[k], lw=2.0, ls="-",
            label=f"{k+1}G engine  (JS={js_per_gen[k]:.3g})",
        )
        ax.semilogy(chi_eval, s, color=colors[k], lw=2.0, ls="--", label=f"{k+1}G sim")
    ax.axvline(0.6865, color="0.4", lw=0.7, ls=":", label=r"$\chi_{\rm rem}(1,0)$")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(r"$p(\chi)$ per generation (each curve integrates to 1)")
    ax.set_title(r"Phase 1 — engine (solid) vs simulator (dashed) per-gen $p(\chi)$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=eps * 10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower left", frameon=False, fontsize="small", ncol=2)
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
    parser.add_argument(
        "--hist",
        action="store_true",
        help="Use raw histograms instead of KDE on the simulator side (default is KDE).",
    )
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args()

    here = Path(__file__).parent
    rh = args.run_hash or secrets.token_hex(3)
    fig_dir = here / f"figures_{rh}"
    art_dir = here / f"artifacts_{rh}"
    fig_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    kind = "hist" if args.hist else "kde"
    print(f"Run hash: {rh}")
    print(f"  figures → {fig_dir.relative_to(here.parent)}/")
    print(f"  artifacts → {art_dir.relative_to(here.parent)}/")
    print(f"  smoothing: {kind}")

    # Common evaluation grids — fine so figures look smooth.
    m_eval = np.geomspace(THETA_STAR["m_min"], THETA_STAR["m_max"] * 2.0, 400)
    chi_eval = np.linspace(0.001, 0.999, 400)

    # Engine side.
    print("Solving engine at θ* ...")
    eng_dN_dm, eng_dN_dchi = engine_per_generation_marginals(
        THETA_STAR, m_eval, chi_eval
    )

    # Simulator side.
    print(f"Running simulator (N_seed={args.n_seed}, "
          f"n_pairs/pass={args.n_pairs_per_pass}) ...")
    rng = np.random.default_rng(args.seed)
    sim_dN_dm, sim_dN_dchi, populations, events = simulator_per_generation_marginals(
        THETA_STAR, m_eval, chi_eval,
        n_seed=args.n_seed,
        n_pairs_per_pass=args.n_pairs_per_pass,
        rng=rng,
        kind=kind,
    )

    K = eng_dN_dm.shape[0]

    # JS per generation per marginal.
    js_mass = [js_divergence(eng_dN_dm[k], sim_dN_dm[k], m_eval) for k in range(K)]
    js_spin = [js_divergence(eng_dN_dchi[k], sim_dN_dchi[k], chi_eval) for k in range(K)]

    # Report.
    print()
    print(f"{'Gen':>4}  {'N_sim':>8}  {'JS(dN/dm)':>10}  {'JS(dN/dχ)':>10}")
    table_lines = ["Gen   N_sim    JS(dN/dm)  JS(dN/dchi)"]
    for k in range(K):
        n_sim = populations.get(k + 1, (np.empty(0), np.empty(0)))[0].size
        line = f"{k+1:>4}  {n_sim:>8d}  {js_mass[k]:>10.4g}  {js_spin[k]:>10.4g}"
        print(line)
        table_lines.append(line)
    (art_dir / "phase1_js_table.txt").write_text("\n".join(table_lines) + "\n")
    print(f"  wrote {art_dir / 'phase1_js_table.txt'}")

    # Save artifacts.
    np.savez(
        art_dir / "phase1_artifacts.npz",
        m_eval=m_eval, chi_eval=chi_eval,
        eng_dN_dm=eng_dN_dm, eng_dN_dchi=eng_dN_dchi,
        sim_dN_dm=sim_dN_dm, sim_dN_dchi=sim_dN_dchi,
        js_mass=np.asarray(js_mass),
        js_spin=np.asarray(js_spin),
        theta_star=np.asarray(list(THETA_STAR.values()), dtype=object),
    )
    print(f"  wrote {art_dir / 'phase1_artifacts.npz'}")

    # Plots.
    print("Plotting overlays ...")
    plot_overlay_mass(m_eval, eng_dN_dm, sim_dN_dm, js_mass, fig_dir / "phase1_per_gen_mass.png")
    plot_overlay_spin(chi_eval, eng_dN_dchi, sim_dN_dchi, js_spin, fig_dir / "phase1_per_gen_spin.png")


if __name__ == "__main__":
    main()
