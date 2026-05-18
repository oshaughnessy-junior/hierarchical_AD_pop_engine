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
from scipy.stats import gaussian_kde, truncnorm

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
    """Run the discrete coagulation cascade with correct (j, l) weighting.

    Mirrors the engine's accumulator structure but as particle MC. For each
    ``k = 2..k_max``, the engine's ``n_k`` is

        n_k = Σ_{j=1..k-1} c(j, k-1) · ∫∫ K · n_j · n_{k-1} · P_retain dω
        c(j, l) = 1/2  if j == l    (unordered indistinguishable pair)
                = 1    if j != l    (distinguishable; no double-count)

    The simulator reproduces this mixture by:

        1. For each (j, l) combo, draw a uniform sample of candidate pairs
           from populations[j] × populations[l].
        2. Weight each candidate by ``c(j, l) · K · P_retain`` (i.e. the
           differential rate density for that pair).
        3. Pool weighted candidates across all (j, l) combos.
        4. Resample without replacement weights → unweighted size-``n_seed``
           draw representing ``n_k``, ready for use in subsequent generations.

    This is **importance sampling**: the proposal is the uniform pair
    distribution, the target is the K·P_retain-weighted distribution, and the
    resampling step converts importance weights to an unweighted draw.

    Parameters
    ----------
    cfg
        Physics + sample-size knobs.
    rng
        Numpy random Generator.

    Returns
    -------
    populations : dict[int, (m, χ)]
        Per-generation **unweighted** retained single-BH populations. Each
        entry holds two float64 arrays of equal length; the length is
        ``cfg.n_seed`` for k=1, and ``cfg.n_seed`` for k≥2 (resampled).
    merger_events : np.ndarray, shape (N_events, 8)
        Per-event record ``(m1, m2, χ1, χ2, g1, g2, retained, weight)`` with
        ``m1 ≥ m2``. ``weight`` is the importance weight before
        normalisation — useful for downstream weighted analyses.
    """
    populations: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    populations[1] = sample_natals(cfg, rng)

    event_records: list[np.ndarray] = []
    n_uniform_per_combo = cfg.n_pairs_per_generation_pair
    n_resample = cfg.n_seed

    for k in range(2, cfg.k_max + 1):
        l = k - 1
        m_l, chi_l = populations[l]
        if m_l.size == 0:
            populations[k] = (np.empty(0), np.empty(0))
            continue

        # Pool of weighted candidate remnants from all (j, l) contributions.
        M_pool_parts: list[np.ndarray] = []
        chi_pool_parts: list[np.ndarray] = []
        w_pool_parts: list[np.ndarray] = []

        for j in range(1, l + 1):
            m_j, chi_j = populations[j]
            if m_j.size == 0:
                continue

            # Step 1 — uniform proposal over populations[j] × populations[l].
            ia = rng.integers(0, m_j.size, size=n_uniform_per_combo)
            ib = rng.integers(0, m_l.size, size=n_uniform_per_combo)
            m_a, chi_a = m_j[ia], chi_j[ia]
            m_b, chi_b = m_l[ib], chi_l[ib]

            K_vals = kernel_K(m_a, m_b, cfg.kernel_a, cfg.kernel_b)
            retain_arr = retention_aligned(m_a, m_b, chi_a, chi_b, cfg.v_esc)
            retained = retain_arr > 0.5

            # Step 2 — importance weight = c(j, l) · K · P_retain.
            sym = 0.5 if j == l else 1.0
            w = sym * K_vals * retained.astype(np.float64)

            # Remnants for retained-only (zero-weight candidates would
            # contribute nothing on resample).
            keep = w > 0
            if keep.any():
                M_new = M_rem(m_a[keep], m_b[keep], chi_a[keep], chi_b[keep])
                chi_new = chi_rem(m_a[keep], m_b[keep], chi_a[keep], chi_b[keep])
                M_pool_parts.append(M_new)
                chi_pool_parts.append(chi_new)
                w_pool_parts.append(w[keep])

            # Event log — all candidates with their weights, retained or not.
            swap = m_a < m_b
            m1 = np.where(swap, m_b, m_a)
            m2 = np.where(swap, m_a, m_b)
            c1 = np.where(swap, chi_b, chi_a)
            c2 = np.where(swap, chi_a, chi_b)
            g1 = np.where(swap, l, j).astype(np.float64)
            g2 = np.where(swap, j, l).astype(np.float64)
            r = retained.astype(np.float64)
            block = np.stack(
                [m1, m2, c1, c2, g1, g2, r, sym * K_vals * r], axis=1
            )
            event_records.append(block)

        # Step 3-4 — pool, normalise weights, resample to fixed size.
        if M_pool_parts:
            M_pool = np.concatenate(M_pool_parts)
            chi_pool = np.concatenate(chi_pool_parts)
            w_pool = np.concatenate(w_pool_parts)
            W = float(w_pool.sum())
            if W <= 0:
                populations[k] = (np.empty(0), np.empty(0))
                continue
            p = w_pool / W
            idx = rng.choice(M_pool.size, size=n_resample, replace=True, p=p)
            populations[k] = (
                M_pool[idx].astype(np.float64),
                chi_pool[idx].astype(np.float64),
            )
        else:
            populations[k] = (np.empty(0), np.empty(0))

    if event_records:
        merger_events = np.concatenate(event_records, axis=0)
    else:
        merger_events = np.empty((0, 8), dtype=np.float64)
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


# ---------------------------------------------------------------------------
# 1-D density estimator with boundary reflection — used by the smoke driver
# AND by Phase 1's engine-vs-simulator overlay (red_team/phase1_overlay.py).
# ---------------------------------------------------------------------------
def make_marginal(
    data: np.ndarray,
    eval_points: np.ndarray,
    *,
    kind: str = "kde",
    bounds: Optional[tuple[float, float]] = None,
    log_space: bool = False,
    bw_method=None,
) -> np.ndarray:
    """Estimate a 1-D density of ``data`` and evaluate it at ``eval_points``.

    Parameters
    ----------
    data
        1-D array of samples.
    eval_points
        1-D array of points at which to return the density.
    kind
        - ``"kde"``: Gaussian KDE. If ``bounds`` is given, use the
          reflection trick (extend the sample with reflected copies at
          each boundary) so the density does not leak across — important
          for spin ∈ [0, 1] half-normal-near-0 distributions and for the
          natal-IMF tail at ``m_min``.
        - ``"hist"``: histogram with bin edges spanning ``eval_points``
          plus symmetric outer half-cells. Kept as a fallback so the user
          can see the raw MC realization when KDE smoothing looks
          suspicious.
    bounds
        ``(low, high)`` over which to enforce reflection. Ignored for
        kind="hist".
    log_space
        If True, the KDE is performed on ``log(data)`` and the density
        returned is ``dN/dm`` (Jacobian ``1/m``). Used for mass marginals
        on a geometric grid.
    bw_method
        Passed through to ``gaussian_kde``. ``None`` → Silverman's rule
        on the *original* sample (we compensate for the artificially
        larger std of the reflected sample explicitly).

    Returns
    -------
    density : 1-D array, ``len(eval_points)``.
        Density estimate in linear (not log) coordinates. Not
        absolute-rate-normalised — Phase 1 compares shapes after
        per-marginal renormalisation.
    """
    data = np.asarray(data, dtype=np.float64)
    eval_points = np.asarray(eval_points, dtype=np.float64)

    if kind == "hist":
        edges = np.empty(eval_points.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (eval_points[:-1] + eval_points[1:])
        edges[0] = eval_points[0] - 0.5 * (eval_points[1] - eval_points[0])
        edges[-1] = eval_points[-1] + 0.5 * (eval_points[-1] - eval_points[-2])
        h, _ = np.histogram(data, bins=edges, density=True)
        return h
    if kind != "kde":
        raise ValueError(f"unknown kind={kind!r}; want 'kde' or 'hist'")

    if data.size < 2:
        return np.zeros_like(eval_points)

    # Work in the transformed coordinate.
    if log_space:
        sample = np.log(data)
        eval_t = np.log(eval_points)
        bounds_t = (
            (np.log(bounds[0]), np.log(bounds[1])) if bounds is not None else None
        )
    else:
        sample = data
        eval_t = eval_points
        bounds_t = bounds

    # Silverman bandwidth from the *original* sample so reflection does
    # not artificially widen the kernel.
    sigma_orig = float(sample.std(ddof=1)) + 1e-30
    n_orig = sample.size
    if bw_method is None:
        h = 1.06 * sigma_orig * (n_orig ** (-1.0 / 5.0))
    else:
        # Caller-supplied; gaussian_kde will interpret.
        h = None  # delegate to scipy

    if bounds_t is not None:
        lo, hi = bounds_t
        extended = np.concatenate([sample, 2 * lo - sample, 2 * hi - sample])
        if h is not None:
            sigma_ext = float(extended.std(ddof=1)) + 1e-30
            kde = gaussian_kde(extended, bw_method=h / sigma_ext)
        else:
            kde = gaussian_kde(extended, bw_method=bw_method)
    else:
        if h is not None:
            kde = gaussian_kde(sample, bw_method=h / sigma_orig)
        else:
            kde = gaussian_kde(sample, bw_method=bw_method)

    density_t = kde(eval_t)

    # Apply Jacobian if we worked in log-space (dN/dlogm → dN/dm).
    if log_space:
        density_t = density_t / eval_points
    return density_t


def _smoke_plot_marginals(
    populations: dict[int, tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    *,
    kind: str = "kde",
    m_min: float = 4.0,
    m_max: float = 80.0,
) -> None:
    """Per-generation single-BH mass and spin marginals.

    Uses :func:`make_marginal` so the user can flip between KDE (default,
    smooth, reveals MC-realisation noise as wiggles in the tails) and
    histogram (raw bin counts; sharper but artefact-prone). The KDE is
    boundary-reflected at ``[m_min, m_max]`` for mass and ``[0, 1]`` for
    spin so half-normal-near-0 distributions don't leak across.
    """
    import matplotlib.pyplot as plt  # local import; not a top-level dep

    K = max(populations.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, K))

    # Evaluation grids — fine, so KDE detail is visible at the figure level.
    m_eval = np.geomspace(m_min, m_max * 2.0, 400)
    chi_eval = np.linspace(0.0, 1.0, 400)

    # ---- Mass ----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for k in range(1, K + 1):
        m_k, _ = populations[k]
        if m_k.size < 2:
            continue
        dN_dm = make_marginal(
            m_k, m_eval, kind=kind, bounds=(m_min, m_max * 2.0), log_space=True
        )
        if dN_dm.max() <= 0:
            continue
        ax.loglog(
            m_eval, np.maximum(dN_dm, 1e-30) / dN_dm.max(),
            lw=2.0, color=colors[k - 1], label=f"{k}G ({m_k.size} BHs)",
        )
    ax.set_xlabel(r"single-BH mass $m \;[M_\odot]$")
    ax.set_ylabel(f"per-generation $dN/dm$ (peak-normalised, {kind})")
    ax.set_title(f"Phase 0 simulator — per-generation $dN/dm$ ({kind})")
    ax.set_ylim(1e-4, 2.0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, title="Generation")
    fig.tight_layout()
    out = out_dir / "sim_phase0_dN_dm1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")

    # ---- Spin ----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for k in range(1, K + 1):
        _, chi_k = populations[k]
        if chi_k.size < 2:
            continue
        dN_dchi = make_marginal(
            chi_k, chi_eval, kind=kind, bounds=(0.0, 1.0), log_space=False
        )
        if dN_dchi.max() <= 0:
            continue
        ax.semilogy(
            chi_eval, np.maximum(dN_dchi, 1e-30) / dN_dchi.max(),
            lw=2.0, color=colors[k - 1], label=f"{k}G",
        )
    ax.axvline(0.6865, color="0.4", lw=0.7, ls=":", label=r"$\chi_{\rm rem}(1,0)$")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(f"per-generation $dN/d\\chi$ (peak-normalised, {kind})")
    ax.set_title(f"Phase 0 simulator — per-generation $dN/d\\chi$ ({kind})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-4, 2.0)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, title="Generation")
    fig.tight_layout()
    out = out_dir / "sim_phase0_dN_dchi1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _smoke_driver(run_hash: Optional[str] = None, kind: str = "kde") -> None:
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
    print(f"Plotting per-generation marginals ({kind}) ...")
    _smoke_plot_marginals(
        populations, fig_dir, kind=kind,
        m_min=cfg.m_min, m_max=cfg.m_max,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-hash",
        default=None,
        help="Output dir suffix; default: random 6-char hex.",
    )
    parser.add_argument(
        "--hist",
        action="store_true",
        help="Use raw histograms instead of boundary-reflected KDE for the "
        "marginal plots. Useful for seeing the raw MC realization.",
    )
    args = parser.parse_args()
    _smoke_driver(run_hash=args.run_hash, kind="hist" if args.hist else "kde")
