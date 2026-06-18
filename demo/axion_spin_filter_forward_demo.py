"""Forward demo: hierarchical coagulation plus an axion spin filter.

This is a constructive toy model for the discussion in ``tmp_Zee``:

* the existing Doctor-style coagulation cascade supplies the mass ladder;
* after each retained merger remnant forms, it spends a residence time in an
  AGN-disk environment;
* quiet residents relax toward an axion Regge floor;
* high-mass, rapidly interacting residents are assigned short coherent cloud
  lifetimes, so the merger-product spin cascade survives.

The implementation is deliberately phenomenological.  It is meant to make the
parameter logic inspectable, not to replace a population fit:

    chi_out = chi_Regge(M) + (chi_birth - chi_Regge(M)) exp(-Lambda_eff)

with ``Lambda_eff`` suppressed by a mass-dependent violent-interaction channel.
The default settings are tuned to produce qualitatively GWTC-like higher-mass
features: a low-spin island near 25--35 Msun, a moderate/high-spin band near
45--60 Msun, and a high-spin tail around 70--100 Msun.  We use ``S1, S2, ...``
for single-BH/remnant generations to avoid confusion with the paper's
``G1, ..., G5`` binary mixture components.

Run:

    PYTHONPATH=src python demo/axion_spin_filter_forward_demo.py

Outputs:

    demo/figures/axion_spin_filter_hierarchy.png
    demo/artifacts/axion_spin_filter_hierarchy.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

from hierarchical_ad_pop_engine.remnants.mass import M_rem as _M_rem_jax
from hierarchical_ad_pop_engine.remnants.spin import chi_rem as _chi_rem_jax


FIG_DIR = Path(__file__).parent / "figures"
ART_DIR = Path(__file__).parent / "artifacts"
FIG_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class AxionCascadeConfig:
    """Knobs for one forward realization."""

    seed: int = 20260618
    n_seed: int = 30_000
    n_pairs_per_channel: int = 35_000
    k_max: int = 5

    mu_natal: float = 8.5
    sigma_natal: float = 1.0
    m_min: float = 4.0
    m_max: float = 140.0
    natal_spin_loc: float = 0.08
    natal_spin_scale: float = 0.08

    # Pairing K ~ (m1+m2)^a q^b.  The mass and q bias favor the high-mass,
    # near-equal branch once it exists, which makes the high-mass S3/S4/S5
    # single-remnant ladder visible.
    kernel_a: float = 2.2
    kernel_b: float = 2.8

    # Axion field parameters.  m_a controls alpha; tau0 is the stimulated
    # spin-down time near alpha_ref before environmental decoherence.
    ma_eV: float = 1.55e-13
    tau0_yr: float = 2.0e4
    alpha_ref: float = 0.035
    tau_power: float = 9.0
    residence_yr: float = 3.0e6

    # Dynamical channel.  Above m_disrupt the coherent cloud duty factor drops
    # and the inter-encounter time becomes the relevant coherence cap.
    m_disrupt: float = 40.0
    disrupt_width: float = 5.0
    disrupted_lambda_cap: float = 0.08
    t_enc_ref_yr: float = 2.0e5
    t_enc_mass_ref: float = 30.0
    t_enc_power: float = 3.0
    min_orbits: float = 4.0
    orbital_period_yr: float = 1.0e3

    # Small residual scatter after spin filtering; this stands in for a spread
    # in residence histories and cloud saturation states.
    chi_scatter: float = 0.025


def M_rem(m1, m2, chi1, chi2) -> np.ndarray:
    return np.asarray(_M_rem_jax(m1, m2, chi1, chi2), dtype=np.float64)


def chi_rem(m1, m2, chi1, chi2) -> np.ndarray:
    return np.asarray(_chi_rem_jax(m1, m2, chi1, chi2), dtype=np.float64)


def kernel_K(m1: np.ndarray, m2: np.ndarray, a: float, b: float) -> np.ndarray:
    m_big = np.maximum(m1, m2)
    m_sml = np.minimum(m1, m2)
    q = np.clip(m_sml / m_big, 1e-6, 1.0)
    return (m1 + m2) ** a * q**b


def sample_truncnorm(
    rng: np.random.Generator,
    mean: float,
    sigma: float,
    lo: float,
    hi: float,
    size: int,
) -> np.ndarray:
    a = (lo - mean) / sigma
    b = (hi - mean) / sigma
    return np.asarray(
        truncnorm.rvs(a, b, loc=mean, scale=sigma, size=size, random_state=rng),
        dtype=np.float64,
    )


def alpha_of_mass(m_msun: np.ndarray, ma_eV: float) -> np.ndarray:
    """Dimensionless gravitational fine-structure constant for a stellar BH."""
    return 0.0748 * (m_msun / 10.0) * (ma_eV / 1.0e-12)


def regge_floor_l1(m_msun: np.ndarray, ma_eV: float) -> np.ndarray:
    """Leading m=l=1 saturation curve, omega ~= mu.

    This intentionally uses the dominant illustrative trajectory rather than
    the multi-level envelope: the demo's purpose is the constructive hierarchy
    with partial relaxation and disrupted clouds.
    """
    alpha = alpha_of_mass(m_msun, ma_eV)
    chi = 4.0 * alpha / (1.0 + 4.0 * alpha**2)
    return np.clip(chi, 0.0, 0.98)


def quiet_probability(m_msun: np.ndarray, cfg: AxionCascadeConfig) -> np.ndarray:
    x = (m_msun - cfg.m_disrupt) / cfg.disrupt_width
    return 1.0 / (1.0 + np.exp(x))


def encounter_time_yr(m_msun: np.ndarray, cfg: AxionCascadeConfig) -> np.ndarray:
    t_dyn = cfg.t_enc_ref_yr * (np.maximum(m_msun, 1e-6) / cfg.t_enc_mass_ref) ** (
        -cfg.t_enc_power
    )
    t_floor = cfg.min_orbits * cfg.orbital_period_yr
    return np.maximum(t_dyn, t_floor)


def axion_tau_yr(m_msun: np.ndarray, cfg: AxionCascadeConfig) -> np.ndarray:
    alpha = np.maximum(alpha_of_mass(m_msun, cfg.ma_eV), 1e-5)
    return cfg.tau0_yr * (alpha / cfg.alpha_ref) ** (-cfg.tau_power)


def apply_axion_filter(
    mass: np.ndarray,
    chi_birth: np.ndarray,
    cfg: AxionCascadeConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return filtered spins plus diagnostic arrays.

    The cloud is quiet with probability p_quiet(M).  Quiet objects can use the
    full AGN residence time; disrupted objects are capped at the larger of a few
    SMBH orbits and the inter-encounter time, then additionally limited by
    ``disrupted_lambda_cap`` to represent repeated cloud ionization/depletion.
    """
    p_quiet = quiet_probability(mass, cfg)
    quiet = rng.random(mass.size) < p_quiet

    tau = axion_tau_yr(mass, cfg)
    t_coh_quiet = np.full_like(mass, cfg.residence_yr, dtype=np.float64)
    t_coh_disrupted = encounter_time_yr(mass, cfg)

    lambda_quiet = t_coh_quiet / tau
    lambda_disrupted = np.minimum(t_coh_disrupted / tau, cfg.disrupted_lambda_cap)
    lambda_eff = np.where(quiet, lambda_quiet, lambda_disrupted)

    chi_floor = regge_floor_l1(mass, cfg.ma_eV)
    chi_filtered = chi_floor + (chi_birth - chi_floor) * np.exp(-lambda_eff)
    if cfg.chi_scatter > 0:
        chi_filtered += rng.normal(0.0, cfg.chi_scatter, size=chi_filtered.size)
    return np.clip(chi_filtered, 0.0, 0.98), lambda_eff, p_quiet, chi_floor


def choose_weighted(
    rng: np.random.Generator, weights: np.ndarray, size: int
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return rng.integers(0, weights.size, size=size)
    return rng.choice(weights.size, size=size, replace=True, p=weights / total)


def simulate(cfg: AxionCascadeConfig):
    rng = np.random.default_rng(cfg.seed)

    m1 = sample_truncnorm(
        rng, cfg.mu_natal, cfg.sigma_natal, cfg.m_min, cfg.m_max, cfg.n_seed
    )
    chi_birth1 = sample_truncnorm(
        rng, cfg.natal_spin_loc, cfg.natal_spin_scale, 0.0, 0.98, cfg.n_seed
    )
    chi1, lam1, pq1, floor1 = apply_axion_filter(m1, chi_birth1, cfg, rng)

    populations: dict[int, dict[str, np.ndarray]] = {
        1: {
            "mass": m1,
            "chi": chi1,
            "chi_birth": chi_birth1,
            "lambda": lam1,
            "p_quiet": pq1,
            "chi_floor": floor1,
        }
    }
    events: list[np.ndarray] = []

    for gen in range(2, cfg.k_max + 1):
        pool_mass: list[np.ndarray] = []
        pool_chi_birth: list[np.ndarray] = []
        pool_weight: list[np.ndarray] = []
        event_blocks: list[np.ndarray] = []

        # Match red_team.sim_coagulation and the grid response solver:
        # n_k receives contributions from population k-1 interacting with every
        # earlier resident population, including the equal-generation branch
        # (k-1)+(k-1).  That branch is what produces the familiar
        # 8 -> 16 -> 30 -> 55 -> 100 Msun Doctor-style mass ladder.
        l = gen - 1
        for j in range(1, l + 1):
            if l not in populations:
                continue
            pop_j = populations[j]
            pop_l = populations[l]

            ia = rng.integers(0, pop_j["mass"].size, size=cfg.n_pairs_per_channel)
            ib = rng.integers(0, pop_l["mass"].size, size=cfg.n_pairs_per_channel)

            ma = pop_j["mass"][ia]
            mb = pop_l["mass"][ib]
            ca = pop_j["chi"][ia]
            cb = pop_l["chi"][ib]
            weights = kernel_K(ma, mb, cfg.kernel_a, cfg.kernel_b)
            if j == l:
                weights *= 0.5

            mass_new = M_rem(ma, mb, ca, cb)
            chi_new_birth = chi_rem(ma, mb, ca, cb)

            pool_mass.append(mass_new)
            pool_chi_birth.append(chi_new_birth)
            pool_weight.append(weights)

            swap = ma < mb
            event_blocks.append(
                np.column_stack(
                    [
                        np.where(swap, mb, ma),
                        np.where(swap, ma, mb),
                        np.where(swap, cb, ca),
                        np.where(swap, ca, cb),
                        np.full_like(ma, gen, dtype=np.float64),
                        np.where(swap, l, j).astype(np.float64),
                        np.where(swap, j, l).astype(np.float64),
                        weights,
                    ]
                )
            )

        mass_pool = np.concatenate(pool_mass)
        chi_birth_pool = np.concatenate(pool_chi_birth)
        weight_pool = np.concatenate(pool_weight)
        idx = choose_weighted(rng, weight_pool, cfg.n_seed)

        mass_k = mass_pool[idx]
        chi_birth_k = chi_birth_pool[idx]
        chi_k, lam_k, pq_k, floor_k = apply_axion_filter(mass_k, chi_birth_k, cfg, rng)
        populations[gen] = {
            "mass": mass_k,
            "chi": chi_k,
            "chi_birth": chi_birth_k,
            "lambda": lam_k,
            "p_quiet": pq_k,
            "chi_floor": floor_k,
        }
        events.append(np.concatenate(event_blocks, axis=0))

    return populations, np.concatenate(events, axis=0), cfg


def weighted_hist_density(samples: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(samples, bins=bins)
    width = np.diff(edges)
    density = counts / np.maximum(width * counts.sum(), 1e-30)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers, density


def plot(populations: dict[int, dict[str, np.ndarray]], cfg: AxionCascadeConfig) -> Path:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, cfg.k_max))

    fig = plt.figure(figsize=(10.8, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_mass = fig.add_subplot(gs[0, 0])
    ax_spin = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[1, :])

    m_bins = np.geomspace(5.0, 125.0, 90)
    chi_bins = np.linspace(0.0, 1.0, 80)

    for gen in range(1, cfg.k_max + 1):
        pop = populations[gen]
        mass = pop["mass"]
        chi = pop["chi"]
        m_centers, dndm = weighted_hist_density(mass, m_bins)
        c_centers = 0.5 * (chi_bins[:-1] + chi_bins[1:])
        c_counts, _ = np.histogram(chi, bins=chi_bins)
        dndchi = c_counts / np.maximum(np.diff(chi_bins) * c_counts.sum(), 1e-30)

        if dndm.max() > 0:
            ax_mass.semilogx(
                m_centers,
                dndm / dndm.max(),
                color=colors[gen - 1],
                lw=2.0,
                label=f"S{gen}",
            )
        if dndchi.max() > 0:
            ax_spin.plot(
                c_centers,
                dndchi / dndchi.max(),
                color=colors[gen - 1],
                lw=2.0,
                label=f"S{gen}",
            )

        # Downsample for an interpretable scatter.  Use quantiles to avoid one
        # overplotted point cloud dominating the visible hierarchy.
        rng = np.random.default_rng(cfg.seed + 100 * gen)
        take = rng.choice(mass.size, size=min(1800, mass.size), replace=False)
        ax_scatter.scatter(
            mass[take],
            chi[take],
            s=4,
            alpha=0.16,
            color=colors[gen - 1],
            rasterized=True,
        )

        med_m = float(np.median(mass))
        med_chi = float(np.median(chi))
        q16, q84 = np.quantile(chi, [0.16, 0.84])
        ax_scatter.errorbar(
            med_m,
            med_chi,
            yerr=[[med_chi - q16], [q84 - med_chi]],
            fmt="o",
            ms=5,
            color=colors[gen - 1],
            mec="white",
            mew=0.6,
            capsize=2.5,
            label=f"S{gen} median",
        )

    m_grid = np.geomspace(5.0, 125.0, 500)
    ax_scatter.plot(
        m_grid,
        regge_floor_l1(m_grid, cfg.ma_eV),
        color="black",
        lw=1.2,
        ls="--",
        label=rf"$m=\ell=1$ Regge floor, $m_a={cfg.ma_eV:.2g}$ eV",
    )

    # Eyeball target bands from the tmp_Zee paper discussion.
    for lo, hi, label in [
        (25, 35, "paper G3-like"),
        (45, 60, "paper G4-like"),
        (70, 105, "paper G5-like"),
    ]:
        ax_scatter.axvspan(lo, hi, color="0.7", alpha=0.13, lw=0)
        ax_scatter.text(
            np.sqrt(lo * hi),
            0.965,
            label,
            ha="center",
            va="top",
            fontsize=8.5,
            color="0.25",
        )

    ax_mass.set_xlabel(r"single-BH mass $m\,[M_\odot]$")
    ax_mass.set_ylabel(r"$dN_k/dm$ (peak-normalized)")
    ax_mass.set_title("Mass ladder from coagulation")
    ax_mass.set_xlim(5, 125)
    ax_mass.set_ylim(0, 1.08)
    ax_mass.grid(True, which="both", ls=":", alpha=0.35)
    ax_mass.legend(frameon=False, ncol=2)

    ax_spin.set_xlabel(r"single-BH spin $\chi$")
    ax_spin.set_ylabel(r"$dN_k/d\chi$ (peak-normalized)")
    ax_spin.set_title("Spin ladder after axion residency")
    ax_spin.set_xlim(0, 1)
    ax_spin.set_ylim(0, 1.08)
    ax_spin.grid(True, ls=":", alpha=0.35)
    ax_spin.legend(frameon=False, ncol=2)

    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel(r"single-BH mass $m\,[M_\odot]$")
    ax_scatter.set_ylabel(r"post-residency spin $\chi$")
    ax_scatter.set_title(
        "Constructive axion-filter hierarchy: quiet mid-mass spin-down, "
        "violent high-mass preservation"
    )
    ax_scatter.set_xlim(5, 125)
    ax_scatter.set_ylim(0, 1.0)
    ax_scatter.grid(True, which="both", ls=":", alpha=0.35)
    ax_scatter.legend(frameon=False, loc="lower right", ncol=2)

    fig.suptitle(
        rf"$\tau_0={cfg.tau0_yr:.1g}$ yr, "
        rf"$t_{{res}}={cfg.residence_yr:.1g}$ yr, "
        rf"$M_{{disrupt}}={cfg.m_disrupt:.0f}M_\odot$",
        y=1.02,
        fontsize=11,
    )

    out = FIG_DIR / "axion_spin_filter_hierarchy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_artifact(populations, events, cfg: AxionCascadeConfig) -> Path:
    out = ART_DIR / "axion_spin_filter_hierarchy.npz"
    payload = {"events": events}
    for gen, pop in populations.items():
        for key, value in pop.items():
            payload[f"S{gen}_{key}"] = value
    payload["config_keys"] = np.array(list(cfg.__dict__.keys()))
    payload["config_values"] = np.array([str(v) for v in cfg.__dict__.values()])
    np.savez(out, **payload)
    return out


def print_summary(populations, cfg: AxionCascadeConfig) -> None:
    print("Axion-filter hierarchical forward demo")
    print(f"  m_a = {cfg.ma_eV:.3e} eV")
    print(f"  tau0(alpha_ref={cfg.alpha_ref:g}) = {cfg.tau0_yr:.3g} yr")
    print(f"  residence = {cfg.residence_yr:.3g} yr")
    print(f"  disruption mass = {cfg.m_disrupt:.1f} Msun")
    print()
    print("Single-BH/remnant generation summaries:")
    print("  gen   med(m)   q16-q84(m)      med(chi)  q16-q84(chi)   med(Lambda)  med(p_quiet)")
    for gen in range(1, cfg.k_max + 1):
        pop = populations[gen]
        mq = np.quantile(pop["mass"], [0.16, 0.5, 0.84])
        cq = np.quantile(pop["chi"], [0.16, 0.5, 0.84])
        lam = np.median(pop["lambda"])
        pq = np.median(pop["p_quiet"])
        print(
            f"  S{gen:<2d}  {mq[1]:7.2f}  [{mq[0]:6.2f},{mq[2]:6.2f}]"
            f"    {cq[1]:7.3f}  [{cq[0]:6.3f},{cq[2]:6.3f}]"
            f"      {lam:8.3g}      {pq:7.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=AxionCascadeConfig.seed)
    parser.add_argument("--n-seed", type=int, default=AxionCascadeConfig.n_seed)
    parser.add_argument(
        "--n-pairs-per-channel",
        type=int,
        default=AxionCascadeConfig.n_pairs_per_channel,
    )
    parser.add_argument("--ma-eV", type=float, default=AxionCascadeConfig.ma_eV)
    parser.add_argument("--tau0-yr", type=float, default=AxionCascadeConfig.tau0_yr)
    parser.add_argument(
        "--residence-yr", type=float, default=AxionCascadeConfig.residence_yr
    )
    parser.add_argument(
        "--m-disrupt", type=float, default=AxionCascadeConfig.m_disrupt
    )
    parser.add_argument(
        "--disrupted-lambda-cap",
        type=float,
        default=AxionCascadeConfig.disrupted_lambda_cap,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AxionCascadeConfig(
        seed=args.seed,
        n_seed=args.n_seed,
        n_pairs_per_channel=args.n_pairs_per_channel,
        ma_eV=args.ma_eV,
        tau0_yr=args.tau0_yr,
        residence_yr=args.residence_yr,
        m_disrupt=args.m_disrupt,
        disrupted_lambda_cap=args.disrupted_lambda_cap,
    )
    populations, events, cfg = simulate(cfg)
    fig_path = plot(populations, cfg)
    art_path = save_artifact(populations, events, cfg)
    print_summary(populations, cfg)
    print()
    print(f"wrote {fig_path}")
    print(f"wrote {art_path}")


if __name__ == "__main__":
    main()
