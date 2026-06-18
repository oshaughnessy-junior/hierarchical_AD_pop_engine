"""Rate-plane sanity check for the axion hierarchy toy model.

This script exists because fixed-size tracer populations are easy to misuse:
the particles carry *shape*, not abundance.  The corrected tracer path below
propagates scalar generation weights ``N_k`` separately and weights each
candidate two-body event by

    c_jl N_j N_l K(m1, m2) / n_proposal .

The output compares three mass-plane shapes:

1. no-axion weighted tracer recurrence with the high-mass locality kernel;
2. no-axion implicit Picard forward rate with the separable power-law kernel;
3. axion-filter weighted tracer recurrence with the high-mass locality kernel.

All panels are peak-normalized in ``d^2R / dlnm1 dlnm2`` so this is a shape
check only.  The channel table printed to stdout is the important bookkeeping
audit: if high-generation channels explode there, the figure is telling us the
model, not only the plot, is too aggressive.

Run:

    MPLCONFIGDIR=.mplconfig XDG_CACHE_HOME=.cache PYTHONPATH=src:. \\
      /Users/rossma/miniconda3/envs/junior_tools/bin/python \\
      demo/axion_rate_plane_comparison.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

from hierarchical_ad_pop_engine import observables as obs
from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.response.coagulation import build_gaussian_coagulation
from hierarchical_ad_pop_engine.response.implicit import solve_implicit_fixed_point

from red_team.sim_coagulation import M_rem, chi_rem, retention_aligned

from demo.axion_spin_filter_forward_demo import (
    AxionCascadeConfig,
    apply_axion_filter,
)


FIG_DIR = Path(__file__).parent / "figures"
ART_DIR = Path(__file__).parent / "artifacts"
FIG_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)


def sample_truncnorm(rng, mean, sigma, lo, hi, size):
    a = (lo - mean) / sigma
    b = (hi - mean) / sigma
    return np.asarray(
        truncnorm.rvs(a, b, loc=mean, scale=sigma, size=size, random_state=rng),
        dtype=np.float64,
    )


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = np.cumsum(ws) / max(float(np.sum(ws)), 1e-300)
    return float(xs[np.searchsorted(cdf, q, side="left")])


def dln_hist2d(m1, m2, weights, log_edges):
    H, _, _ = np.histogram2d(
        np.log(m1), np.log(m2), bins=(log_edges, log_edges), weights=weights
    )
    dln = np.diff(log_edges)
    return H / np.maximum(dln[:, None] * dln[None, :], 1e-300)


def normalize_peak(H):
    H = np.asarray(H, dtype=np.float64)
    mx = np.nanmax(H)
    if not np.isfinite(mx) or mx <= 0:
        return H
    return H / mx


def high_mass_locality_kernel(
    m1: np.ndarray,
    m2: np.ndarray,
    *,
    a: float,
    b_low: float,
    m_turn: float,
    turn_width: float,
    sigma_low: float,
    sigma_high: float,
    locality_strength: float,
) -> np.ndarray:
    """Pairing kernel with a high-mass comparable-mass preference.

    Low masses use the familiar ``(m1+m2)^a q^b_low``.  Above ``m_turn``,
    the kernel multiplies in a log-mass locality gate that narrows toward
    ``sigma_high``.  This keeps G1+G1 and G1+G2 growth possible but suppresses
    high-mass mixed-generation encounters unless the two masses are comparable.
    """
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)
    mbig = np.maximum(m1, m2)
    msml = np.minimum(m1, m2)
    q = np.clip(msml / mbig, 1e-8, 1.0)
    mtot = m1 + m2
    high = 1.0 / (1.0 + np.exp(-(mtot - m_turn) / max(turn_width, 1e-6)))
    sigma_eff = sigma_low + (sigma_high - sigma_low) * high
    log_ratio = np.log(mbig / msml)
    locality = np.exp(-0.5 * (log_ratio / sigma_eff) ** 2)
    gate = (1.0 - high) + high * locality_strength * locality
    return (mtot**a) * (q**b_low) * gate


def simulate_weighted_tracer(
    cfg: AxionCascadeConfig,
    *,
    use_axion: bool,
    cascade_epsilon: float,
    kernel_params: dict[str, float],
):
    """Finite-generation weighted tracer recurrence.

    ``populations[k]`` are equal-weight tracer samples from the normalized
    retained-remnant shape ``p_k``.  ``N[k]`` is the scalar abundance/rate
    carried by that generation.  Event rows are candidate mergers with physical
    channel weights; they are not raw tracer counts.
    """
    rng = np.random.default_rng(cfg.seed + (17 if use_axion else 0))

    m = sample_truncnorm(
        rng, cfg.mu_natal, cfg.sigma_natal, cfg.m_min, cfg.m_max, cfg.n_seed
    )
    chi_birth = sample_truncnorm(
        rng, cfg.natal_spin_loc, cfg.natal_spin_scale, 0.0, 0.98, cfg.n_seed
    )
    if use_axion:
        chi, _, _, _ = apply_axion_filter(m, chi_birth, cfg, rng)
    else:
        chi = chi_birth

    populations = {1: {"mass": m, "chi": chi}}
    N = {1: 1.0}
    events = []
    channel_rows = []

    for gen in range(2, cfg.k_max + 1):
        l = gen - 1
        pool_m = []
        pool_chi_birth = []
        pool_w = []

        for j in range(1, l + 1):
            pj = populations[j]
            pl = populations[l]
            ia = rng.integers(0, pj["mass"].size, size=cfg.n_pairs_per_channel)
            ib = rng.integers(0, pl["mass"].size, size=cfg.n_pairs_per_channel)

            ma = pj["mass"][ia]
            mb = pl["mass"][ib]
            ca = pj["chi"][ia]
            cb = pl["chi"][ib]
            kval = high_mass_locality_kernel(
                ma,
                mb,
                a=cfg.kernel_a,
                b_low=cfg.kernel_b,
                **kernel_params,
            )
            retain_prob = retention_aligned(ma, mb, ca, cb, 1000.0)
            sym = 0.5 if j == l else 1.0
            event_w = (
                cascade_epsilon
                * sym
                * N[j]
                * N[l]
                * kval
                / float(cfg.n_pairs_per_channel)
            )
            retain_w = event_w * retain_prob

            mnew = M_rem(ma, mb, ca, cb)
            chibirth_new = chi_rem(ma, mb, ca, cb)

            pool_m.append(mnew)
            pool_chi_birth.append(chibirth_new)
            pool_w.append(retain_w)

            swap = ma < mb
            events.append(
                np.column_stack(
                    [
                        np.where(swap, mb, ma),
                        np.where(swap, ma, mb),
                        np.full_like(ma, gen, dtype=np.float64),
                        np.full_like(ma, j, dtype=np.float64),
                        np.full_like(ma, l, dtype=np.float64),
                        event_w,
                    ]
                )
            )

            rate = float(np.sum(event_w))
            retained_rate = float(np.sum(retain_w))
            channel_rows.append((gen, j, l, rate, retained_rate))

        m_pool = np.concatenate(pool_m)
        chi_birth_pool = np.concatenate(pool_chi_birth)
        w_pool = np.concatenate(pool_w)
        N_gen = float(np.sum(w_pool))
        N[gen] = N_gen
        if (not np.isfinite(N_gen)) or N_gen <= 0:
            raise RuntimeError(f"Generation G{gen} has zero retained weight.")
        idx = rng.choice(m_pool.size, size=cfg.n_seed, replace=True, p=w_pool / N_gen)
        m_gen = m_pool[idx]
        chi_birth_gen = chi_birth_pool[idx]
        if use_axion:
            chi_gen, _, _, _ = apply_axion_filter(m_gen, chi_birth_gen, cfg, rng)
        else:
            chi_gen = chi_birth_gen
        populations[gen] = {"mass": m_gen, "chi": chi_gen}

    return {
        "populations": populations,
        "N": N,
        "events": np.concatenate(events, axis=0),
        "channels": np.asarray(channel_rows, dtype=np.float64),
    }


def implicit_rate_plane(cfg: AxionCascadeConfig, log_edges, *, implicit_epsilon: float):
    grid = BinaryGrid.default(m_min=cfg.m_min, m_max=cfg.m_max, n_m=72, n_chi=24)
    response = build_gaussian_coagulation(
        mu=cfg.mu_natal,
        sigma=cfg.sigma_natal,
        m_min=cfg.m_min,
        m_max=cfg.m_max,
        natal_spin_loc=cfg.natal_spin_loc,
        natal_spin_scale=cfg.natal_spin_scale,
        kernel_a=cfg.kernel_a,
        kernel_b=cfg.kernel_b,
        log_R_env=0.0,
        k_max=cfg.k_max,
        grid=grid,
    )
    result = solve_implicit_fixed_point(
        response,
        jnp.asarray(1000.0),
        epsilon=implicit_epsilon,
        max_iter=300,
        tol=1e-7,
        damping=0.35,
    )
    N_grid = obs.density_from_implicit(result)
    m_axis, _, dR_dm1dm2 = obs.marginal_2d(
        N_grid,
        response.grid,
        response.kernel_params,
        response.log_R_env,
        axes=(obs.AXIS_M1, obs.AXIS_M2),
    )
    ordered = np.asarray(obs.fold_ordered_m(m_axis, dR_dm1dm2))
    m = np.asarray(m_axis)
    # Convert from dR/dm1dm2 to dR/dlnm1dlnm2, then histogram/interpolate by
    # nearest cells onto the plotting grid for a direct visual comparison.
    H_dln = ordered * m[:, None] * m[None, :]

    centers = np.sqrt(np.exp(log_edges[:-1]) * np.exp(log_edges[1:]))
    out = np.zeros((centers.size, centers.size), dtype=np.float64)
    idx = np.searchsorted(m, centers)
    idx = np.clip(idx, 1, m.size - 1)
    left = m[idx - 1]
    right = m[idx]
    nearest = np.where(np.abs(centers - left) < np.abs(centers - right), idx - 1, idx)
    for a, ia in enumerate(nearest):
        for b, ib in enumerate(nearest):
            out[a, b] = H_dln[ia, ib]
    return out, {
        "n_iter": int(result.n_iter),
        "final_diff": float(result.final_diff),
        "f_natal_depleted": float(result.f_natal_depleted),
    }


def plot_comparison(noax_mc, axion_mc, imp_H, cfg, log_edges):
    edges = np.exp(log_edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    X, Y = np.meshgrid(edges, edges, indexing="ij")

    H_noax = dln_hist2d(
        noax_mc["events"][:, 0],
        noax_mc["events"][:, 1],
        noax_mc["events"][:, 5],
        log_edges,
    )
    H_ax = dln_hist2d(
        axion_mc["events"][:, 0],
        axion_mc["events"][:, 1],
        axion_mc["events"][:, 5],
        log_edges,
    )
    maps = [
        ("weighted tracer MC, no axion", H_noax),
        ("implicit Picard baseline, no axion", imp_H),
        ("weighted tracer MC, axion spins", H_ax),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    from matplotlib.colors import LogNorm

    for ax, (title, H) in zip(axes, maps):
        Z = normalize_peak(np.where(H > 0.0, H, np.nan))
        finite = np.isfinite(Z)
        vmin = 1e-4
        if finite.any():
            vmin = max(vmin, float(np.nanpercentile(Z[finite], 3)))
        pcm = ax.pcolormesh(
            X,
            Y,
            np.where(np.tril(np.ones_like(Z, dtype=bool)), Z, np.nan),
            cmap="Blues",
            norm=LogNorm(vmin=vmin, vmax=1.0),
            shading="auto",
        )
        ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color="0.25", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(edges[0], edges[-1])
        ax.set_title(title)
        ax.set_xlabel(r"$m_1\,[M_\odot]$")
        ax.grid(True, which="both", ls=":", alpha=0.25)
        for gen, color in [(2, "#3b6ea8"), (3, "#1f9e89"), (4, "#4cc26c"), (5, "#b8de29")]:
            mask = noax_mc["events"][:, 2].astype(int) == gen
            if title.startswith("weighted tracer MC, axion"):
                mask = axion_mc["events"][:, 2].astype(int) == gen
                ev = axion_mc["events"]
            else:
                ev = noax_mc["events"]
            if title.startswith("implicit") or not np.any(mask):
                continue
            w = ev[mask, 5]
            med1 = weighted_quantile(ev[mask, 0], w, 0.5)
            med2 = weighted_quantile(ev[mask, 1], w, 0.5)
            ax.plot(med1, med2, "o", ms=4, color=color, mec="white", mew=0.5)
            ax.text(med1 * 1.04, med2 * 1.03, f"G{gen}", color=color, weight="bold", fontsize=8)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02, label="peak-normalized")
    axes[0].set_ylabel(r"$m_2\,[M_\odot]$")

    out = FIG_DIR / "axion_rate_plane_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def print_channel_table(label, channels):
    print(label)
    print("  out  j  l      rate        retained      frac_of_total")
    total = float(np.sum(channels[:, 3]))
    for gen, j, l, rate, retained in channels:
        frac = rate / total if total > 0 else 0.0
        print(f"  G{int(gen):<2d}  {int(j):d}  {int(l):d}  {rate:10.3e}  {retained:10.3e}  {frac:10.3e}")
    print(f"  total event rate weight = {total:.3e}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--n-seed", type=int, default=25_000)
    parser.add_argument("--n-pairs-per-channel", type=int, default=35_000)
    parser.add_argument("--kernel-a", type=float, default=2.2)
    parser.add_argument("--kernel-b", type=float, default=2.8)
    parser.add_argument("--m-max", type=float, default=140.0)
    parser.add_argument("--m-local-turn", type=float, default=42.0)
    parser.add_argument("--local-turn-width", type=float, default=7.0)
    parser.add_argument("--sigma-q-low", type=float, default=1.2)
    parser.add_argument("--sigma-q-high", type=float, default=0.28)
    parser.add_argument("--locality-strength", type=float, default=14.0)
    parser.add_argument(
        "--cascade-epsilon",
        type=float,
        default=3.0e-4,
        help="Effective interaction four-volume multiplying each channel rate.",
    )
    parser.add_argument(
        "--implicit-epsilon",
        type=float,
        default=None,
        help="Picard interaction strength. Defaults to --cascade-epsilon.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = replace(
        AxionCascadeConfig(),
        seed=args.seed,
        n_seed=args.n_seed,
        n_pairs_per_channel=args.n_pairs_per_channel,
        kernel_a=args.kernel_a,
        kernel_b=args.kernel_b,
        m_max=args.m_max,
    )
    log_edges = np.linspace(np.log(5.0), np.log(args.m_max), 76)
    locality_params = dict(
        m_turn=args.m_local_turn,
        turn_width=args.local_turn_width,
        sigma_low=args.sigma_q_low,
        sigma_high=args.sigma_q_high,
        locality_strength=args.locality_strength,
    )

    noax = simulate_weighted_tracer(
        cfg,
        use_axion=False,
        cascade_epsilon=args.cascade_epsilon,
        kernel_params=locality_params,
    )
    axion = simulate_weighted_tracer(
        cfg,
        use_axion=True,
        cascade_epsilon=args.cascade_epsilon,
        kernel_params=locality_params,
    )
    implicit_epsilon = args.cascade_epsilon if args.implicit_epsilon is None else args.implicit_epsilon
    imp_H, imp_diag = implicit_rate_plane(
        cfg, log_edges, implicit_epsilon=implicit_epsilon
    )
    fig_path = plot_comparison(noax, axion, imp_H, cfg, log_edges)

    np.savez(
        ART_DIR / "axion_rate_plane_comparison.npz",
        noax_events=noax["events"],
        axion_events=axion["events"],
        noax_channels=noax["channels"],
        axion_channels=axion["channels"],
        implicit_H=imp_H,
        log_edges=log_edges,
        cascade_epsilon=np.asarray(args.cascade_epsilon),
        implicit_epsilon=np.asarray(implicit_epsilon),
        locality_keys=np.asarray(list(locality_params.keys())),
        locality_values=np.asarray(list(locality_params.values()), dtype=np.float64),
    )
    print(f"cascade_epsilon = {args.cascade_epsilon:.3e}")
    print(f"implicit_epsilon = {implicit_epsilon:.3e}")
    print("locality kernel:")
    for key, value in locality_params.items():
        print(f"  {key}: {value}")
    print()

    print_channel_table("No-axion weighted tracer channel rates", noax["channels"])
    print_channel_table("Axion-spin weighted tracer channel rates", axion["channels"])
    print("Implicit Picard diagnostics:")
    for key, value in imp_diag.items():
        print(f"  {key}: {value}")
    print()
    print(f"wrote {fig_path}")
    print(f"wrote {ART_DIR / 'axion_rate_plane_comparison.npz'}")


if __name__ == "__main__":
    main()
