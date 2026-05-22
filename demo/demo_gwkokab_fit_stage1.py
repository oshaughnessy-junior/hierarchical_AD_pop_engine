"""Demo 3 — Stage 1: single-Gaussian-pulse, flat-selection end-to-end fit.

The smallest possible coagulation fit. We inject a single natal Gaussian
pulse at (μ*, σ*) plus a narrow natal-spin truncated normal, run the full
generational coagulation cascade, sample a mock catalog of N_obs events,
and recover the **6** free parameters

    mu_natal, sigma_natal, natal_spin_loc, natal_spin_scale,
    kernel_b, log_rate

with SVI (and optionally NUTS).

Selection: flat. Efficiency is folded into ``log_rate`` — we fit the
detected rate.

Likelihood: inhomogeneous-Poisson on (m₁, m₂, χ₁, χ₂), rolled by hand
against ``CoagulationResponse.log_rate_density`` plus a grid quadrature
for the expected event count. We do **not** route through
``gwkokab.inference.numpyro_analytical_poisson_likelihood`` yet — that
path requires wrapping the engine as a ``ScaledMixture``, which is a
Stage 2 concern. Stage 1 is the smallest end-to-end fit.

Mock catalog: events have **continuous** ``(m, χ)`` values. We pick the
grid cell from the discrete joint density, then jitter uniformly inside
the cell (uniform in ``log m`` for mass, uniform in ``χ`` for spin).
Without this, snapping events to grid nodes broadens the apparent IMF
and stacks ~25 % of spins at the lowest ``χ`` node — visible as
inflated ``sigma_natal`` and an upward bias on ``natal_spin_loc``.
The framework itself handles continuous inputs natively
(``log_density_at`` uses ``jax.scipy.ndimage.map_coordinates`` for
bilinear interpolation in ``(log m, χ)``).

Resolution / order knobs (CLI flags below) for the methodology question
"are bilinear interp + trapezoid quadrature high-enough order?" — bump
``--n-m``, ``--n-chi``, or match ``--k-max-fit K_MAX_TRUTH`` to test it
empirically. Bilinear + trapezoid are both formally O(h²), which gives
sub-percent absolute errors at the default grid; the larger biases at
Stage 1's first pass were from discrete sampling, not interp/quadrature.

Run
---

::

    python demo/demo_gwkokab_fit_stage1.py            # SVI only
    python demo/demo_gwkokab_fit_stage1.py --nuts     # SVI + NUTS

Outputs
-------

::

    demo/figures/stage1_truth_dN_dm.png
    demo/figures/stage1_truth_dN_dchi.png
    demo/figures/stage1_mock_catalog.png
    demo/figures/stage1_svi_loss.png
    demo/figures/stage1_corner.png
    demo/figures/stage1_ppc_mass.png
    demo/figures/stage1_ppc_spin.png
    demo/artifacts/stage1_theta_star.json
    demo/artifacts/stage1_mock_catalog.npz
    demo/artifacts/stage1_posterior_svi.npz
    demo/artifacts/stage1_posterior_nuts.npz   # only with --nuts
"""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.response.coagulation import (
    CoagulationResponse,
    build_gaussian_coagulation,
)


# ---------------------------------------------------------------------------
# Paths + output dirs.
# ---------------------------------------------------------------------------
# FIG_DIR and ART_DIR are *reassigned* inside main() to per-run directories
# (``figures_<hash>/`` and ``artifacts_<hash>/``) so parallel runs with
# different settings don't overwrite each other's plots.
HERE = Path(__file__).parent
RUN_HASH = "default"
FIG_DIR = HERE / "figures"
ART_DIR = HERE / "artifacts"


# ---------------------------------------------------------------------------
# 0. Configuration.
# ---------------------------------------------------------------------------
# Truth model parameters θ* and fixed (known) hyperparameters.
THETA_STAR = dict(
    mu_natal=8.0,
    sigma_natal=1.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.10,
    kernel_b=1.0,
    log_rate=None,  # back-computed below to target N_OBS events
)

FIXED = dict(
    m_min=4.0,
    m_max=80.0,
    kernel_a=0.0,
    v_esc=1000.0,
)

# Grids: truth uses defaults; inference uses a smaller grid + smaller k_max
# to keep NUTS / SVI compile + step time reasonable on a CPU laptop.
N_OBS = 100
K_MAX_TRUTH = 4
# Match K_MAX_FIT to K_MAX_TRUTH by default so there's no cascade-depth
# mismatch confounding the recovery. Dial down with --k-max-fit 3 to make
# NUTS faster (at the cost of dropping the 4G high-mass tail).
K_MAX_FIT = 4
N_M_FIT = 64
N_CHI_FIT = 20

# Fit settings.
SVI_STEPS = 2000
SVI_LR = 5e-3
NUTS_WARMUP = 200
NUTS_SAMPLES = 500
NUTS_CHAINS = 1
N_PPC_DRAWS = 200

SEED = 20260517


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def _trapezoid_weights(x: jnp.ndarray) -> jnp.ndarray:
    """Trapezoid cell widths for a 1-D array (length-matched)."""
    dx = jnp.diff(x)
    return jnp.concatenate(
        [dx[:1] * 0.5, 0.5 * (dx[:-1] + dx[1:]), dx[-1:] * 0.5]
    )


def _grid_for_fit() -> BinaryGrid:
    return BinaryGrid.default(
        m_min=FIXED["m_min"],
        m_max=FIXED["m_max"],
        n_m=N_M_FIT,
        n_chi=N_CHI_FIT,
    )


def build_response(
    *,
    mu_natal,
    sigma_natal,
    natal_spin_loc,
    natal_spin_scale,
    kernel_b,
    log_rate,
    k_max=K_MAX_FIT,
    grid=None,
) -> CoagulationResponse:
    """Build a CoagulationResponse for the Stage 1 6-param model.

    ``log_rate`` maps onto ``log_R_env`` (we treat T_obs = 1 and absorb
    efficiency into the rate).
    """
    return build_gaussian_coagulation(
        mu=mu_natal,
        sigma=sigma_natal,
        m_min=FIXED["m_min"],
        m_max=FIXED["m_max"],
        natal_spin_loc=natal_spin_loc,
        natal_spin_scale=natal_spin_scale,
        kernel_a=FIXED["kernel_a"],
        kernel_b=kernel_b,
        log_R_env=log_rate,
        k_max=k_max,
        grid=grid,
    )


def total_log_N_grid(response: CoagulationResponse, v_esc: jnp.ndarray) -> jnp.ndarray:
    """Return ``log N(m, χ) = log Σ_k n_k`` on the response grid."""
    return response._log_total_density(jnp.asarray(v_esc))


def expected_event_count(
    response: CoagulationResponse, v_esc: jnp.ndarray
) -> jnp.ndarray:
    """Grid quadrature of ``∫ λ(m1,m2,χ1,χ2) dω`` ⇒ expected detected events.

    Selection is flat (folded into ``log_rate``), so this is

        N_exp = e^{log_R_env} · ½ · ∫∫ K(m1,m2) N_m(m1) N_m(m2) dm1 dm2

    with ``N_m(m) = ∫ N(m,χ) dχ``. Uses trapezoid weights on the response
    grid.
    """
    grid = response.grid
    log_N = total_log_N_grid(response, v_esc)  # (N_m, N_chi)
    dchi = _trapezoid_weights(grid.chi)
    N_m = jnp.einsum("mc,c->m", jnp.exp(log_N), dchi)  # (N_m,)

    # log K on the m×m grid (unnormalised). Adding 0 from kernel_a=0 is fine.
    from hierarchical_ad_pop_engine.kernels import log_pairing_kernel_grid

    log_K = log_pairing_kernel_grid(grid.m, **response.kernel_params)  # (N_m, N_m)
    K = jnp.exp(log_K)

    dm = _trapezoid_weights(grid.m)
    pair_integral = jnp.einsum("ij,i,j,i,j->", K, N_m, N_m, dm, dm)
    return 0.5 * jnp.exp(response.log_R_env) * pair_integral


# ---------------------------------------------------------------------------
# 1. Build truth model + plot 1-D truth densities.
# ---------------------------------------------------------------------------
def build_truth_model_and_calibrate_rate(
    target_N_obs: int, grid: BinaryGrid | None = None
) -> CoagulationResponse:
    """Build θ* response, then back-compute ``log_rate`` so the expected
    event count matches ``target_N_obs``.

    Parameters
    ----------
    target_N_obs
        Desired ``E[N_obs]`` at θ*; ``log_rate`` is calibrated to hit it.
    grid
        Optional override of the truth grid.  When ``None`` (the demo's
        default), the truth lives on ``BinaryGrid.default(m_min, m_max)``
        — that is, n_m=96, n_chi=32 — which is finer than the fit grid
        and is the natural choice when the demo's purpose is to
        illustrate that the inference can recover from data generated
        on a richer support.  Pass ``_grid_for_fit()`` (or any other
        ``BinaryGrid``) to force truth onto the same quadrature as the
        fit; the integration test does this to isolate inference
        behaviour from quadrature-resolution drift, since the engine's
        grid is non-uniform (``jnp.geomspace`` in mass) and the
        expected-count integral evaluated on 96×32 vs 64×20 differs
        by enough to push the credible-interval gate.  Resolution and
        robustness validation tracks (separate work) deliberately
        decouple the two grids.
    """
    truth = build_response(
        mu_natal=THETA_STAR["mu_natal"],
        sigma_natal=THETA_STAR["sigma_natal"],
        natal_spin_loc=THETA_STAR["natal_spin_loc"],
        natal_spin_scale=THETA_STAR["natal_spin_scale"],
        kernel_b=THETA_STAR["kernel_b"],
        log_rate=0.0,  # placeholder
        k_max=K_MAX_TRUTH,
        grid=grid,
    )
    # Compute expected count at log_R = 0, then rescale.
    n_exp_unit = float(expected_event_count(truth, jnp.asarray(FIXED["v_esc"])))
    log_rate_star = float(jnp.log(target_N_obs / n_exp_unit))
    THETA_STAR["log_rate"] = log_rate_star

    # Rebuild with the correct log_rate.
    return build_response(
        mu_natal=THETA_STAR["mu_natal"],
        sigma_natal=THETA_STAR["sigma_natal"],
        natal_spin_loc=THETA_STAR["natal_spin_loc"],
        natal_spin_scale=THETA_STAR["natal_spin_scale"],
        kernel_b=THETA_STAR["kernel_b"],
        log_rate=log_rate_star,
        k_max=K_MAX_TRUTH,
        grid=grid,
    )


def plot_truth_marginals(truth: CoagulationResponse) -> None:
    log_N = total_log_N_grid(truth, jnp.asarray(FIXED["v_esc"]))
    N = np.asarray(jnp.exp(log_N))
    m = np.asarray(truth.grid.m)
    chi = np.asarray(truth.grid.chi)
    dchi = np.asarray(_trapezoid_weights(jnp.asarray(chi)))
    dm = np.asarray(_trapezoid_weights(jnp.asarray(m)))
    dN_dm = N @ dchi  # ∫ N dχ
    dN_dchi = N.T @ dm  # ∫ N dm

    # Log y axis so low-rate high-generation echoes (4G near 53 M⊙, etc.)
    # are visible alongside the 1G peak. Clamp the bottom of the y range
    # to a few decades below the peak so log(0) doesn't blow up the axis.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(m, np.maximum(dN_dm, 1e-30), lw=2.0, color="C0")
    ax.set_xlabel(r"primary mass $m \;\; [M_\odot]$")
    ax.set_ylabel(r"$dN/dm$ (single-BH density, all generations)")
    ax.set_title(rf"Truth $dN/dm$ at $\theta^*$, $v_{{\rm esc}}={FIXED['v_esc']:.0f}$ km/s")
    ax.set_ylim(bottom=float(dN_dm.max()) * 1e-8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage1_truth_dN_dm.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(chi, np.maximum(dN_dchi, 1e-30), lw=2.0, color="C0")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(r"$dN/d\chi$")
    ax.set_title(r"Truth $dN/d\chi$ at $\theta^*$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=float(dN_dchi.max()) * 1e-6)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage1_truth_dN_dchi.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Mock-catalog sampler — grid inverse-CDF, factored
#       p(m1,m2,χ1,χ2) ∝ K(m1,m2) N(m1,χ1) N(m2,χ2).
# ---------------------------------------------------------------------------
def _sample_categorical(key, p, n):
    """Sample n indices into a 1-D probability array p (normalised)."""
    cdf = jnp.cumsum(p) / jnp.sum(p)
    u = jax.random.uniform(key, (n,))
    return jnp.searchsorted(cdf, u)


def _cell_edges(axis: np.ndarray) -> np.ndarray:
    """Midpoint cell edges for a 1-D node array, with half-cell extrapolation
    at the ends.

    Returns an array of length ``N+1`` such that cell ``i`` spans
    ``edges[i] .. edges[i+1]`` and is centered on ``axis[i]``.
    """
    N = axis.size
    edges = np.empty(N + 1, dtype=axis.dtype)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return edges


def _continuous_from_bin(
    axis: np.ndarray, bin_idx: np.ndarray, key: jax.Array, clip_to_axis: bool = True
) -> np.ndarray:
    """Convert bin indices to continuous values by uniform jitter within each cell.

    Cells are midpoint-bounded. The result is clipped to
    ``[axis[0], axis[-1]]`` by default so query points stay strictly inside
    the response grid (otherwise the bilinear interp returns ``-inf``).
    """
    edges = _cell_edges(axis)
    lo = edges[bin_idx]
    hi = edges[bin_idx + 1]
    u = np.asarray(jax.random.uniform(key, (bin_idx.size,)))
    out = lo + u * (hi - lo)
    if clip_to_axis:
        out = np.clip(out, axis[0], axis[-1])
    return out


def sample_mock_catalog(
    truth: CoagulationResponse, n_events: int, key: jax.Array
) -> np.ndarray:
    """Return an array of shape ``(N, 4)`` of ``(m1, m2, χ1, χ2)``.

    Pick the grid cell via the discrete joint pair density, then **continuously
    jitter within the cell** (uniform in ``log m`` for mass, uniform in ``χ``
    for spin). This gives a properly continuous catalog rather than one
    snapped to grid nodes — the apparent IMF width is otherwise broadened by
    the grid spacing and the bounded spin distribution sees a spurious pileup
    at the first ``χ`` node.
    """
    grid = truth.grid
    m_axis = np.asarray(grid.m)
    log_m_axis = np.asarray(grid.log_m)
    chi_axis = np.asarray(grid.chi)
    dm = np.asarray(_trapezoid_weights(grid.m))
    dchi = np.asarray(_trapezoid_weights(grid.chi))

    log_N = np.asarray(total_log_N_grid(truth, jnp.asarray(FIXED["v_esc"])))
    N = np.exp(log_N)  # (N_m, N_chi)
    N_m = N @ dchi  # (N_m,) — marginal in m

    from hierarchical_ad_pop_engine.kernels import log_pairing_kernel_grid

    log_K = np.asarray(log_pairing_kernel_grid(grid.m, **truth.kernel_params))
    K = np.exp(log_K)

    # Discrete joint pair density: P(i1, i2) ∝ K · N_m · N_m · dm · dm.
    pair = K * (N_m[:, None] * dm[:, None]) * (N_m[None, :] * dm[None, :])
    pair_flat = pair.reshape(-1)

    keys = jax.random.split(key, 5)
    k_pair, k_jm1, k_jm2, k_jc1, k_jc2 = keys

    idx_flat = np.asarray(_sample_categorical(k_pair, jnp.asarray(pair_flat), n_events))
    N_m_size = m_axis.size
    i1 = idx_flat // N_m_size
    i2 = idx_flat % N_m_size

    # Continuous mass: jitter uniformly within the log-m cell.
    log_m1 = _continuous_from_bin(log_m_axis, i1, k_jm1)
    log_m2 = _continuous_from_bin(log_m_axis, i2, k_jm2)
    m1 = np.exp(log_m1)
    m2 = np.exp(log_m2)

    # Conditional spin bin given mass bin: P(j | i) ∝ N(i, j) · dχ_j.
    cond1 = N[i1] * dchi[None, :]
    cond2 = N[i2] * dchi[None, :]
    cond1 = cond1 / cond1.sum(axis=1, keepdims=True)
    cond2 = cond2 / cond2.sum(axis=1, keepdims=True)
    cdf1 = np.cumsum(cond1, axis=1)
    cdf2 = np.cumsum(cond2, axis=1)
    # Vectorised bin draw via per-row searchsorted.
    u_bin1 = np.asarray(jax.random.uniform(jax.random.fold_in(k_jc1, 0), (n_events,)))
    u_bin2 = np.asarray(jax.random.uniform(jax.random.fold_in(k_jc2, 0), (n_events,)))
    j1 = np.clip(
        np.array([np.searchsorted(cdf1[k], u_bin1[k]) for k in range(n_events)]),
        0,
        chi_axis.size - 1,
    )
    j2 = np.clip(
        np.array([np.searchsorted(cdf2[k], u_bin2[k]) for k in range(n_events)]),
        0,
        chi_axis.size - 1,
    )

    # Continuous spin: jitter uniformly within the χ cell.
    chi1 = _continuous_from_bin(chi_axis, j1, jax.random.fold_in(k_jc1, 1))
    chi2 = _continuous_from_bin(chi_axis, j2, jax.random.fold_in(k_jc2, 1))

    # Enforce primary-mass convention m1 ≥ m2 (swap spins co-with).
    # The rate density is symmetric in (1↔2) so this is purely cosmetic —
    # it collapses the scatter into the upper triangle and keeps the
    # mass-ratio q = m2/m1 ∈ (0, 1] convention.
    swap = m1 < m2
    m1, m2 = np.where(swap, m2, m1), np.where(swap, m1, m2)
    chi1, chi2 = np.where(swap, chi2, chi1), np.where(swap, chi1, chi2)

    return np.stack([m1, m2, chi1, chi2], axis=1)


def plot_mock_catalog(events: np.ndarray) -> None:
    m1, m2, c1, c2 = events.T
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(m1, m2, s=10, alpha=0.6, color="C1")
    lo = min(m1.min(), m2.min())
    hi = max(m1.max(), m2.max())
    axes[0].plot([lo, hi], [lo, hi], lw=0.5, color="0.5", ls=":")
    axes[0].set_xlabel(r"$m_1 \;[M_\odot]$")
    axes[0].set_ylabel(r"$m_2 \;[M_\odot]$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("mock catalog — masses")
    axes[1].scatter(c1, c2, s=10, alpha=0.6, color="C1")
    axes[1].set_xlabel(r"$\chi_1$")
    axes[1].set_ylabel(r"$\chi_2$")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("mock catalog — spins")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage1_mock_catalog.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Numpyro inference model.
# ---------------------------------------------------------------------------
def make_inference_model(events: jnp.ndarray):
    """Return a numpyro model closing over ``events`` (shape (N, 4))."""
    m1 = events[:, 0]
    m2 = events[:, 1]
    chi1 = events[:, 2]
    chi2 = events[:, 3]
    grid = _grid_for_fit()
    env = {"v_esc": jnp.asarray(FIXED["v_esc"])}

    def model():
        mu = numpyro.sample("mu_natal", dist.Uniform(5.0, 50.0))
        sigma = numpyro.sample("sigma_natal", dist.LogUniform(0.5, 10.0))
        spin_loc = numpyro.sample("natal_spin_loc", dist.Uniform(0.0, 0.3))
        spin_scale = numpyro.sample("natal_spin_scale", dist.LogUniform(0.02, 0.3))
        kb = numpyro.sample("kernel_b", dist.Uniform(-2.0, 6.0))
        log_rate = numpyro.sample("log_rate", dist.Uniform(-2.0, 6.0))

        response = build_response(
            mu_natal=mu,
            sigma_natal=sigma,
            natal_spin_loc=spin_loc,
            natal_spin_scale=spin_scale,
            kernel_b=kb,
            log_rate=log_rate,
            k_max=K_MAX_FIT,
            grid=grid,
        )

        # Per-event log-rate.
        per_event = response.log_rate_density(m1, m2, chi1, chi2, env)  # (N,)
        # Expected count at this θ.
        n_exp = expected_event_count(response, env["v_esc"])

        # Inhomogeneous-Poisson log-likelihood (T_obs = 1):
        #   log L = Σ_i log λ(θ_i) - N_exp
        log_lik = jnp.sum(per_event) - n_exp
        # Guard against -inf / NaN bubbling out of the grid quadrature.
        log_lik = jnp.where(jnp.isfinite(log_lik), log_lik, -1e20)
        numpyro.factor("log_lik", log_lik)

    return model


# ---------------------------------------------------------------------------
# 4. SVI smoke test.
# ---------------------------------------------------------------------------
def run_svi(model, key, n_steps=SVI_STEPS, lr=SVI_LR):
    guide = AutoNormal(model)
    optimizer = optax.adam(lr)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_result = svi.run(key, n_steps)
    return svi_result, guide


def plot_svi_loss(losses) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.asarray(losses))
    ax.set_xlabel("SVI step")
    ax.set_ylabel("-ELBO")
    ax.set_title("Stage 1 SVI loss")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage1_svi_loss.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. NUTS (optional).
# ---------------------------------------------------------------------------
def run_nuts(model, key):
    kernel = NUTS(model, target_accept_prob=0.85)
    mcmc = MCMC(
        kernel,
        num_warmup=NUTS_WARMUP,
        num_samples=NUTS_SAMPLES,
        num_chains=NUTS_CHAINS,
        chain_method="sequential",
        progress_bar=True,
    )
    mcmc.run(key)
    return mcmc.get_samples()


# ---------------------------------------------------------------------------
# 6. Corner plot.
# ---------------------------------------------------------------------------
PARAM_NAMES = [
    "mu_natal",
    "sigma_natal",
    "natal_spin_loc",
    "natal_spin_scale",
    "kernel_b",
    "log_rate",
]


def _stack_samples(samples: dict) -> np.ndarray:
    return np.stack([np.asarray(samples[k]).reshape(-1) for k in PARAM_NAMES], axis=1)


def plot_corner(samples_arr: np.ndarray, truths: list[float], outpath: Path) -> None:
    try:
        import corner  # noqa: WPS433
    except ImportError:
        # Fallback: simple per-param posterior histograms.
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        for ax, name, truth in zip(axes.flat, PARAM_NAMES, truths):
            x = samples_arr[:, PARAM_NAMES.index(name)]
            ax.hist(x, bins=40, color="C0", alpha=0.7)
            ax.axvline(truth, color="r", lw=1.5, ls="--")
            ax.set_xlabel(name)
        fig.suptitle("posteriors (corner unavailable; install `corner` for full plot)")
        fig.tight_layout()
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        return

    fig = corner.corner(
        samples_arr,
        labels=PARAM_NAMES,
        truths=truths,
        show_titles=True,
        quantiles=[0.05, 0.5, 0.95],
        title_fmt=".3g",
    )
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Posterior predictive check on dN/dm and dN/dχ.
# ---------------------------------------------------------------------------
def posterior_predictive_bands(
    samples_arr: np.ndarray, n_draws: int, key: jax.Array
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (m, dN_dm_q, chi, dN_dchi_q), each q shaped (3, len)."""
    grid = _grid_for_fit()
    m = np.asarray(grid.m)
    chi = np.asarray(grid.chi)
    dm = np.asarray(_trapezoid_weights(grid.m))
    dchi = np.asarray(_trapezoid_weights(grid.chi))

    n_total = samples_arr.shape[0]
    if n_total <= n_draws:
        idx = np.arange(n_total)
    else:
        idx = np.asarray(
            jax.random.choice(key, n_total, shape=(n_draws,), replace=False)
        )

    dN_dm_stack = np.empty((len(idx), m.size))
    dN_dchi_stack = np.empty((len(idx), chi.size))
    for r, j in enumerate(idx):
        mu, sigma, sl, ss, kb, lr = samples_arr[j]
        response = build_response(
            mu_natal=float(mu),
            sigma_natal=float(sigma),
            natal_spin_loc=float(sl),
            natal_spin_scale=float(ss),
            kernel_b=float(kb),
            log_rate=float(lr),
            k_max=K_MAX_FIT,
            grid=grid,
        )
        log_N = np.asarray(total_log_N_grid(response, jnp.asarray(FIXED["v_esc"])))
        N = np.exp(log_N)
        dN_dm_stack[r] = N @ dchi
        dN_dchi_stack[r] = N.T @ dm

    qs = [0.05, 0.5, 0.95]
    dN_dm_q = np.quantile(dN_dm_stack, qs, axis=0)
    dN_dchi_q = np.quantile(dN_dchi_stack, qs, axis=0)
    return m, dN_dm_q, chi, dN_dchi_q


def plot_ppc(truth: CoagulationResponse, samples_arr: np.ndarray, key: jax.Array) -> None:
    # Truth marginals on the FIT grid (so x-axes line up).
    grid = _grid_for_fit()
    truth_on_fit = build_response(
        mu_natal=THETA_STAR["mu_natal"],
        sigma_natal=THETA_STAR["sigma_natal"],
        natal_spin_loc=THETA_STAR["natal_spin_loc"],
        natal_spin_scale=THETA_STAR["natal_spin_scale"],
        kernel_b=THETA_STAR["kernel_b"],
        log_rate=THETA_STAR["log_rate"],
        k_max=K_MAX_FIT,
        grid=grid,
    )
    dchi = np.asarray(_trapezoid_weights(grid.chi))
    dm = np.asarray(_trapezoid_weights(grid.m))
    N_truth = np.exp(np.asarray(total_log_N_grid(truth_on_fit, jnp.asarray(FIXED["v_esc"]))))
    dN_dm_truth = N_truth @ dchi
    dN_dchi_truth = N_truth.T @ dm

    m_arr, dN_dm_q, chi_arr, dN_dchi_q = posterior_predictive_bands(
        samples_arr, N_PPC_DRAWS, key
    )

    # Log-y so high-mass, high-spin "hierarchical echoes" (2G+, 3G+) are
    # visible alongside the 1G peak. Floor the band at a small positive
    # number to keep log scale well-defined.
    eps_m = float(max(dN_dm_truth.max(), dN_dm_q[2].max())) * 1e-8
    dN_dm_q_safe = np.maximum(dN_dm_q, eps_m)
    dN_dm_truth_safe = np.maximum(dN_dm_truth, eps_m)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.fill_between(m_arr, dN_dm_q_safe[0], dN_dm_q_safe[2], color="C0", alpha=0.3, label="PPC 90 %")
    ax.loglog(m_arr, dN_dm_q_safe[1], color="C0", lw=2.0, label="PPC median")
    ax.loglog(m_arr, dN_dm_truth_safe, color="k", lw=1.5, ls="--", label="truth")
    ax.set_xlabel(r"primary mass $m \;[M_\odot]$")
    ax.set_ylabel(r"$dN/dm$")
    ax.set_title("Stage 1 posterior-predictive on $dN/dm$")
    ax.set_ylim(bottom=eps_m * 10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage1_ppc_mass.png", dpi=150)
    plt.close(fig)

    eps_c = float(max(dN_dchi_truth.max(), dN_dchi_q[2].max())) * 1e-6
    dN_dchi_q_safe = np.maximum(dN_dchi_q, eps_c)
    dN_dchi_truth_safe = np.maximum(dN_dchi_truth, eps_c)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.fill_between(chi_arr, dN_dchi_q_safe[0], dN_dchi_q_safe[2], color="C0", alpha=0.3, label="PPC 90 %")
    ax.semilogy(chi_arr, dN_dchi_q_safe[1], color="C0", lw=2.0, label="PPC median")
    ax.semilogy(chi_arr, dN_dchi_truth_safe, color="k", lw=1.5, ls="--", label="truth")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(r"$dN/d\chi$")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=eps_c * 10)
    ax.set_title("Stage 1 posterior-predictive on $dN/d\\chi$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage1_ppc_spin.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main() -> None:
    # Module-level sizing constants are CLI-overridable; declare global up
    # front so argparse can read them as defaults and we can reassign below.
    global N_M_FIT, N_CHI_FIT, K_MAX_FIT, N_OBS, RUN_HASH, FIG_DIR, ART_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuts", action="store_true", help="Also run NUTS")
    parser.add_argument("--svi-steps", type=int, default=SVI_STEPS)
    parser.add_argument(
        "--n-m", type=int, default=N_M_FIT, help="N_m for the inference grid"
    )
    parser.add_argument(
        "--n-chi", type=int, default=N_CHI_FIT, help="N_chi for the inference grid"
    )
    parser.add_argument(
        "--k-max-fit",
        type=int,
        default=K_MAX_FIT,
        help="k_max for the inference model (set equal to K_MAX_TRUTH=%d "
        "to remove the truth/fit cascade-depth mismatch)" % K_MAX_TRUTH,
    )
    parser.add_argument(
        "--n-obs", type=int, default=N_OBS, help="Number of mock events"
    )
    parser.add_argument(
        "--run-hash",
        type=str,
        default=None,
        help="Output dir suffix; default: random 6-char hex. Outputs land in "
        "demo/figures_<hash>/ and demo/artifacts_<hash>/ so parallel runs "
        "with different settings don't clobber each other.",
    )
    args = parser.parse_args()

    N_M_FIT = args.n_m
    N_CHI_FIT = args.n_chi
    K_MAX_FIT = args.k_max_fit
    N_OBS = args.n_obs
    RUN_HASH = args.run_hash if args.run_hash is not None else secrets.token_hex(3)
    FIG_DIR = HERE / f"figures_{RUN_HASH}"
    ART_DIR = HERE / f"artifacts_{RUN_HASH}"
    FIG_DIR.mkdir(exist_ok=True)
    ART_DIR.mkdir(exist_ok=True)
    print(
        f"Run hash: {RUN_HASH}   "
        f"figures → {FIG_DIR.relative_to(HERE.parent)}/   "
        f"artifacts → {ART_DIR.relative_to(HERE.parent)}/"
    )
    print(
        f"Config: n_obs={N_OBS}, fit grid=({N_M_FIT},{N_CHI_FIT}), "
        f"k_max_truth={K_MAX_TRUTH}, k_max_fit={K_MAX_FIT}"
    )

    key = jax.random.PRNGKey(SEED)
    k_calib, k_mock, k_svi, k_nuts, k_ppc = jax.random.split(key, 5)

    # 1. Truth.
    print("Building truth model and calibrating rate to N_obs ≈", N_OBS, "...")
    truth = build_truth_model_and_calibrate_rate(N_OBS)
    print("  log_rate* =", THETA_STAR["log_rate"])
    print("  N_exp(θ*) =", float(expected_event_count(truth, jnp.asarray(FIXED["v_esc"]))))
    plot_truth_marginals(truth)

    with (ART_DIR / "stage1_theta_star.json").open("w") as f:
        json.dump({**THETA_STAR, **FIXED, "n_obs_target": N_OBS}, f, indent=2)

    # 2. Mock catalog.
    print("Sampling mock catalog ...")
    events_np = sample_mock_catalog(truth, N_OBS, k_mock)
    np.savez(
        ART_DIR / "stage1_mock_catalog.npz",
        events=events_np,
        theta_star=np.array(list(THETA_STAR.values()), dtype=float),
    )
    plot_mock_catalog(events_np)
    events = jnp.asarray(events_np)

    # Sanity: confirm masses and spins are continuous (not grid-snapped) and
    # that the framework accepts them — the log-rate at every event must be
    # finite.
    unique_m1 = np.unique(events_np[:, 0]).size
    unique_chi1 = np.unique(events_np[:, 2]).size
    env_dict = {"v_esc": jnp.asarray(FIXED["v_esc"])}
    per_event_truth = truth.log_rate_density(
        events[:, 0], events[:, 1], events[:, 2], events[:, 3], env_dict
    )
    n_finite = int(jnp.sum(jnp.isfinite(per_event_truth)))
    print(
        f"  catalog: unique m1 = {unique_m1}/{N_OBS}, "
        f"unique χ1 = {unique_chi1}/{N_OBS}, "
        f"finite log-rate values = {n_finite}/{N_OBS}"
    )

    # 3. SVI fit.
    print("Running SVI ...")
    model = make_inference_model(events)
    svi_result, guide = run_svi(model, k_svi, n_steps=args.svi_steps)
    plot_svi_loss(svi_result.losses)
    svi_samples = guide.sample_posterior(
        jax.random.PRNGKey(SEED + 1),
        svi_result.params,
        sample_shape=(2000,),
    )
    svi_samples_arr = _stack_samples(svi_samples)
    np.savez(ART_DIR / "stage1_posterior_svi.npz", **{k: np.asarray(v) for k, v in svi_samples.items()})

    truths = [THETA_STAR[k] for k in PARAM_NAMES]
    plot_corner(svi_samples_arr, truths, FIG_DIR / "stage1_corner.png")

    print("Running posterior-predictive on dN/dm and dN/dχ ...")
    plot_ppc(truth, svi_samples_arr, k_ppc)

    # 4. NUTS (optional).
    if args.nuts:
        print("Running NUTS ...")
        nuts_samples = run_nuts(model, k_nuts)
        np.savez(
            ART_DIR / "stage1_posterior_nuts.npz",
            **{k: np.asarray(v) for k, v in nuts_samples.items()},
        )
        nuts_arr = _stack_samples(nuts_samples)
        plot_corner(nuts_arr, truths, FIG_DIR / "stage1_corner_nuts.png")
        plot_ppc(truth, nuts_arr, jax.random.PRNGKey(SEED + 2))

    # 5. Recovery report.
    print()
    print("Recovery report (median ± 1-σ; 90 % credible interval):")
    for k_name, truth_val in zip(PARAM_NAMES, truths):
        col = svi_samples_arr[:, PARAM_NAMES.index(k_name)]
        med = np.median(col)
        lo, hi = np.quantile(col, [0.05, 0.95])
        sig = 0.5 * (np.quantile(col, 0.8413) - np.quantile(col, 0.1587))
        inside = "✓" if (lo <= truth_val <= hi) else "✗"
        print(
            f"  {k_name:>18}: truth = {truth_val:>8.4g}   "
            f"median = {med:>8.4g} ± {sig:.3g}   "
            f"[{lo:.4g}, {hi:.4g}] {inside}"
        )


if __name__ == "__main__":
    main()
