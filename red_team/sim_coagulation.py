"""Phase 0 — discrete-realization forward simulator for the coagulation cascade.

Particle Monte Carlo running against the same physics primitives the engine
uses (``M_rem``, ``chi_rem``, ``retention_aligned``). The algorithmic path
is completely independent of ``hierarchical_ad_pop_engine.response`` — we
track particles, sample pairs by rejection on the kernel, and apply
retention as a Bernoulli, rather than building 4D quadratures and bilinear
scatters on a grid.

Architectural rule (see ``demo/PLAN_red_team_validation.md``):

    May import:    engine physics primitives (M_rem, chi_rem,
                   retention_aligned, v_kick_aligned).
                   JAX is allowed transitively through them.
    May NOT import: hierarchical_ad_pop_engine.response.*

The simulator's outputs are designed to be hot-swappable into the engine's
Poisson likelihood: ``populations`` matches the structure of
``_solve``'s per-generation `n_k`; ``merger_events`` matches the per-event
support of ``log_rate_density``.

Smoke driver
------------

::

    python -m red_team.sim_coagulation         # run at Stage 1 θ*

Outputs (under ``red_team/figures_<hash>/`` and ``red_team/artifacts_<hash>/``)::

    sim_phase0_dN_dm1.png        # log-y per-generation mass
    sim_phase0_dN_dchi1.png      # log-y per-generation spin
    sim_phase0_catalog.npz       # raw merger-event catalog
    sim_phase0_populations.npz   # per-generation single-BH populations
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import truncnorm

# Allowed engine imports — physics primitives only. Wrapped with np.asarray
# at the call sites below so the outer driver is numpy-native. The
# architectural rule in PLAN_red_team_validation.md forbids importing
# `hierarchical_ad_pop_engine.response.*` from this module; that rule is
# enforced at code review (and by reading the import list right here).
from hierarchical_ad_pop_engine.remnants.mass import M_rem as _M_rem_jax
from hierarchical_ad_pop_engine.remnants.spin import chi_rem as _chi_rem_jax
from hierarchical_ad_pop_engine.remnants.kick import (
    retention_aligned as _retention_aligned_jax,
    v_kick_aligned as _v_kick_aligned_jax,
)


# ---------------------------------------------------------------------------
# Physics-primitive thin wrappers — keep numpy I/O at this boundary.
# ---------------------------------------------------------------------------
def M_rem(m1, m2, chi1, chi2):
    return np.asarray(_M_rem_jax(m1, m2, chi1, chi2))


def chi_rem(m1, m2, chi1, chi2):
    return np.asarray(_chi_rem_jax(m1, m2, chi1, chi2))


def retention_aligned(m1, m2, chi1, chi2, v_esc):
    return np.asarray(_retention_aligned_jax(m1, m2, chi1, chi2, v_esc))


def v_kick_aligned(m1, m2, chi1, chi2):
    return np.asarray(_v_kick_aligned_jax(m1, m2, chi1, chi2))


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """All physics + sample-size knobs for one simulator run."""

    # Natal IMF — single truncated normal on [m_min, m_max] (matches
    # Stage 1's build_gaussian_coagulation).
    mu_natal: float
    sigma_natal: float
    m_min: float
    m_max: float

    # Natal spin — truncated normal on [0, 1].
    natal_spin_loc: float
    natal_spin_scale: float

    # Pairing kernel K(m1, m2) ∝ (m1+m2)^a · q^b with q = min/max.
    kernel_a: float
    kernel_b: float

    # Environment.
    v_esc: float

    # Cascade depth.
    k_max: int

    # Sample budgets.
    n_seed: int = 10_000  # natal BHs sampled
    n_pairs_per_generation_pair: int = 10_000  # pairs drawn per (j, l) pass

    # Rejection-sampling settings.
    rejection_batch: int = 50_000
    max_rejection_attempts: int = 100_000_000


# ---------------------------------------------------------------------------
# Pairing kernel (numpy reimplementation; matches kernels.log_pairing_kernel
# up to the same formula, but algorithmically independent).
# ---------------------------------------------------------------------------
def kernel_K(m1: np.ndarray, m2: np.ndarray, a: float, b: float) -> np.ndarray:
    """Unnormalised pairing kernel K(m1, m2) = (m1+m2)^a · q^b, q = min/max."""
    m_big = np.maximum(m1, m2)
    m_sml = np.minimum(m1, m2)
    q = m_sml / m_big
    return (m1 + m2) ** a * q ** b


def _kernel_upper_bound(
    m_a: np.ndarray, m_b: np.ndarray, a: float, b: float, safety: float = 1.05
) -> float:
    """Safe upper bound on K over the cross product of two mass arrays.

    Picks the two-particle extremum analytically given the signs of (a, b)
    and inflates by ``safety`` to absorb edge effects.
    """
    m_big = float(max(m_a.max(), m_b.max()))
    m_sml = float(min(m_a.min(), m_b.min()))

    # Candidate extrema.
    candidates = []
    # Equal mass at the heavy end.
    candidates.append((2 * m_big) ** a * 1.0 ** b)
    # Equal mass at the light end.
    candidates.append((2 * m_sml) ** a * 1.0 ** b)
    # Asymmetric: m_big / m_sml mix.
    q_asym = m_sml / m_big
    candidates.append((m_big + m_sml) ** a * q_asym ** b)
    return float(max(candidates)) * safety


# ---------------------------------------------------------------------------
# Natal sampler.
# ---------------------------------------------------------------------------
def sample_natals(cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw N_seed natal BHs from f_IMF(m) × g_spin(χ) via scipy.stats."""
    a_m = (cfg.m_min - cfg.mu_natal) / cfg.sigma_natal
    b_m = (cfg.m_max - cfg.mu_natal) / cfg.sigma_natal
    m = truncnorm.rvs(
        a_m, b_m,
        loc=cfg.mu_natal, scale=cfg.sigma_natal,
        size=cfg.n_seed, random_state=rng,
    )
    a_c = (0.0 - cfg.natal_spin_loc) / cfg.natal_spin_scale
    b_c = (1.0 - cfg.natal_spin_loc) / cfg.natal_spin_scale
    chi = truncnorm.rvs(
        a_c, b_c,
        loc=cfg.natal_spin_loc, scale=cfg.natal_spin_scale,
        size=cfg.n_seed, random_state=rng,
    )
    return np.asarray(m, dtype=np.float64), np.asarray(chi, dtype=np.float64)


# ---------------------------------------------------------------------------
# Pair sampler — batched rejection on K, with replacement.
# ---------------------------------------------------------------------------
def sample_pairs_weighted_by_K(
    m_a: np.ndarray,
    m_b: np.ndarray,
    cfg: SimConfig,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_pairs`` (i_a, i_b) indices into populations a, b with
    acceptance probability ∝ K(m_a[i_a], m_b[i_b]).

    With replacement — matches the engine's non-depleting accumulator
    (see plan: "the engine treats n_1 as a non-depleting source").

    Returns ``(idx_a, idx_b)`` of shape ``(n_pairs,)``.
    """
    na, nb = m_a.size, m_b.size
    if na == 0 or nb == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    K_max = _kernel_upper_bound(m_a, m_b, cfg.kernel_a, cfg.kernel_b)

    idx_a_out = np.empty(n_pairs, dtype=np.int64)
    idx_b_out = np.empty(n_pairs, dtype=np.int64)
    filled = 0
    attempts = 0
    while filled < n_pairs:
        batch = min(cfg.rejection_batch, max(n_pairs - filled, 1))
        ia = rng.integers(0, na, size=batch)
        ib = rng.integers(0, nb, size=batch)
        K_ij = kernel_K(m_a[ia], m_b[ib], cfg.kernel_a, cfg.kernel_b)
        u = rng.random(size=batch)
        accept = u < (K_ij / K_max)
        n_new = int(accept.sum())
        n_take = min(n_new, n_pairs - filled)
        if n_take > 0:
            idx_a_out[filled : filled + n_take] = ia[accept][:n_take]
            idx_b_out[filled : filled + n_take] = ib[accept][:n_take]
            filled += n_take
        attempts += batch
        if attempts > cfg.max_rejection_attempts:
            raise RuntimeError(
                f"Rejection sampling failed to fill {n_pairs} pairs after "
                f"{attempts} attempts. Acceptance rate ≈ {filled / attempts:.3g}. "
                f"K_max bound = {K_max:.3g} may be loose, or K is near zero "
                f"over the support."
            )
    return idx_a_out, idx_b_out


# ---------------------------------------------------------------------------
# Cascade driver — discrete analog of CoagulationResponse._solve.
# ---------------------------------------------------------------------------
def simulate_cascade(
    cfg: SimConfig, rng: np.random.Generator
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Run the discrete coagulation cascade.

    Algorithm (mirrors ``_solve``'s accumulator, but as particle MC):

        populations[1] = natal sample
        for k = 2..k_max:
            for j ∈ {1, ..., k-1}:
                pair populations[j] × populations[k-1]
                apply retention; compute M_rem, chi_rem; advance.
            populations[k] = retained remnants from all (j, k-1) passes.

    Returns
    -------
    populations : dict[int, (m, χ)]
        Per-generation single-BH populations (only retained BHs).
        ``populations[k]`` is a tuple of two float64 arrays of equal length.
    merger_events : np.ndarray, shape (N_events, 7)
        Per-event record (m1, m2, χ1, χ2, g1, g2, retained), with m1 ≥ m2.
        ``g1`` is the generation of the heavier progenitor.
        ``retained`` is 1.0 if the merger was retained, else 0.0.
    """
    populations: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    populations[1] = sample_natals(cfg, rng)

    event_records: list[np.ndarray] = []  # collect (N_pairs, 7) blocks

    for k in range(2, cfg.k_max + 1):
        l = k - 1
        m_l, chi_l = populations[l]
        if m_l.size == 0:
            populations[k] = (np.empty(0), np.empty(0))
            continue

        # k-th gen accumulator: collect retained remnants from all (j, l) passes.
        m_k_parts: list[np.ndarray] = []
        chi_k_parts: list[np.ndarray] = []

        for j in range(1, l + 1):
            m_j, chi_j = populations[j]
            if m_j.size == 0:
                continue
            # NOTE: the engine's _solve applies a sym=0.5 factor when j == l
            # (unordered → ordered double-counting correction). The simulator
            # below treats all pairs as ordered samples, so absolute event
            # counts are 2× the engine's expectation for j == l passes.
            # Irrelevant for Phase 1 shape comparison; will need to be
            # corrected (e.g. halve n_pairs when j == l) for Phase 3 absolute-
            # rate parity with the engine.
            idx_j, idx_l = sample_pairs_weighted_by_K(
                m_j, m_l, cfg, cfg.n_pairs_per_generation_pair, rng
            )
            m_a, chi_a = m_j[idx_j], chi_j[idx_j]
            m_b, chi_b = m_l[idx_l], chi_l[idx_l]

            # Retention check — Heaviside on aligned-kick magnitude.
            retain_arr = retention_aligned(m_a, m_b, chi_a, chi_b, cfg.v_esc)
            retained = retain_arr > 0.5

            # Compute remnants (Lousto-Zlochower / Barausse-Rezzolla via engine).
            M_new = M_rem(m_a, m_b, chi_a, chi_b)
            chi_new = chi_rem(m_a, m_b, chi_a, chi_b)
            m_k_parts.append(M_new[retained])
            chi_k_parts.append(chi_new[retained])

            # Record events with m1 ≥ m2 convention; swap spins/gens to match.
            swap = m_a < m_b
            m1 = np.where(swap, m_b, m_a)
            m2 = np.where(swap, m_a, m_b)
            c1 = np.where(swap, chi_b, chi_a)
            c2 = np.where(swap, chi_a, chi_b)
            g1 = np.where(swap, l, j).astype(np.float64)
            g2 = np.where(swap, j, l).astype(np.float64)
            r = retained.astype(np.float64)
            block = np.stack([m1, m2, c1, c2, g1, g2, r], axis=1)
            event_records.append(block)

        if m_k_parts:
            populations[k] = (
                np.concatenate(m_k_parts).astype(np.float64),
                np.concatenate(chi_k_parts).astype(np.float64),
            )
        else:
            populations[k] = (np.empty(0), np.empty(0))

    if event_records:
        merger_events = np.concatenate(event_records, axis=0)
    else:
        merger_events = np.empty((0, 7), dtype=np.float64)
    return populations, merger_events


# ---------------------------------------------------------------------------
# Smoke driver — runs at Stage 1 θ*, prints per-generation budgets, writes
# log-y per-generation marginals.
# ---------------------------------------------------------------------------
# Mirrors demo.demo_gwkokab_fit_stage1.THETA_STAR / FIXED. Hard-coded rather
# than imported to keep the red-team module loosely coupled to the demo.
_STAGE1_THETA = dict(
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


def _smoke_plot_marginals(
    populations: dict[int, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt  # local import; not a top-level dep

    K = max(populations.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, K))

    # Per-generation mass spectrum, peak-normalised, log-y.
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bins_m = np.geomspace(2.0, 200.0, 80)
    for k in range(1, K + 1):
        m_k, _ = populations[k]
        if m_k.size == 0:
            continue
        h, edges = np.histogram(m_k, bins=bins_m)
        if h.max() == 0:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.loglog(
            centers, np.maximum(h, 1) / h.max(),
            lw=2.0, color=colors[k - 1], label=f"{k}G ({m_k.size} BHs)",
        )
    ax.set_xlabel(r"single-BH mass $m \;[M_\odot]$")
    ax.set_ylabel("per-generation count (peak-normalised, log-y)")
    ax.set_title("Phase 0 simulator — per-generation single-BH mass")
    ax.set_ylim(1e-4, 2.0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, title="Generation")
    fig.tight_layout()
    out = out_dir / "sim_phase0_dN_dm1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")

    # Per-generation spin spectrum, log-y.
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bins_c = np.linspace(0.0, 1.0, 40)
    for k in range(1, K + 1):
        _, chi_k = populations[k]
        if chi_k.size == 0:
            continue
        h, edges = np.histogram(chi_k, bins=bins_c)
        if h.max() == 0:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.semilogy(
            centers, np.maximum(h, 1) / h.max(),
            lw=2.0, color=colors[k - 1], label=f"{k}G",
        )
    ax.axvline(0.6865, color="0.4", lw=0.7, ls=":")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel("per-generation count (peak-normalised, log-y)")
    ax.set_title("Phase 0 simulator — per-generation single-BH spin")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-4, 2.0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, title="Generation")
    fig.tight_layout()
    out = out_dir / "sim_phase0_dN_dchi1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _smoke_driver(run_hash: Optional[str] = None) -> None:
    here = Path(__file__).parent
    rh = run_hash or secrets.token_hex(3)
    fig_dir = here / f"figures_{rh}"
    art_dir = here / f"artifacts_{rh}"
    fig_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    print(f"Run hash: {rh}")
    print(f"  figures → {fig_dir.relative_to(here.parent)}/")
    print(f"  artifacts → {art_dir.relative_to(here.parent)}/")

    cfg = SimConfig(**_STAGE1_THETA)
    rng = np.random.default_rng(20260518)
    print(f"Config: {cfg}")

    print("Running cascade ...")
    populations, events = simulate_cascade(cfg, rng)

    print()
    print(f"Per-generation populations (retained single BHs):")
    for k in sorted(populations.keys()):
        m_k, chi_k = populations[k]
        if m_k.size == 0:
            print(f"  {k}G: empty")
            continue
        print(
            f"  {k}G: N = {m_k.size:>6d}   "
            f"<m> = {m_k.mean():>6.2f} ± {m_k.std():>5.2f} M⊙   "
            f"<χ> = {chi_k.mean():>5.3f} ± {chi_k.std():>5.3f}"
        )

    print()
    print(f"Merger events: total {events.shape[0]}, "
          f"retained {int(events[:, 6].sum())} ({100 * events[:, 6].mean():.1f}%)")

    # Save artifacts.
    np.savez(art_dir / "sim_phase0_catalog.npz", events=events)
    np.savez(
        art_dir / "sim_phase0_populations.npz",
        **{
            f"m_{k}G": populations[k][0] for k in populations
        },
        **{
            f"chi_{k}G": populations[k][1] for k in populations
        },
    )
    print(f"  wrote {art_dir / 'sim_phase0_catalog.npz'}")
    print(f"  wrote {art_dir / 'sim_phase0_populations.npz'}")

    # Plots.
    print("Plotting per-generation marginals ...")
    _smoke_plot_marginals(populations, fig_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-hash",
        default=None,
        help="Output dir suffix; default: random 6-char hex.",
    )
    args = parser.parse_args()
    _smoke_driver(run_hash=args.run_hash)
