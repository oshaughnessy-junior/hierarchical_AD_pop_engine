r"""Forward demo — MC histogram vs k-NN balloon vs Gaussian KDE.

Visualises the density-estimator trade flagged in
``[[feedback-gaussian-kde-over-smooths-cascade-peaks]]``.

At Stage 1 θ* run a single ``particle_mc_jax`` cascade (J2), then plot
the surviving-pool ``dN/dm`` and ``dN/dχ`` (and the 2-D
``dN/(dm dχ)`` heatmap) three ways side-by-side:

  (1) raw histogram of the m_active / chi_active samples, weighted by
      the J2 cascade weights w_active.  This is the *truth* — no
      smoothing applied.

  (2) numpy ``MCEvaluator`` (k-NN balloon, biweight kernel, k=√n_eff).
      Adapts bandwidth per query point — resolves narrow cascade
      features.  Production-quality (Session M2).

  (3) JAX ``MCEvaluatorJax`` (Gaussian KDE, Silverman bandwidth on
      n_eff).  Global fixed bandwidth — smoother / lossier on sharp
      features but differentiable in θ_pop.  J4 SVI uses this.

Run::

    python demo/forward_modeling_mc_kde_compare.py
    # writes demo/figures_<hash>/mc_kde_compare_*.png

Per-run hash directory matches the existing demo convention so
parallel runs don't clobber each other.

This is a sanity check, not a unit test.  The intended outcome:

* (1) and (2) agree to within MC noise; k-NN resolves the cascade
  peaks the histogram shows.
* (3) is smoother than both — peaks are wider, valleys filled in.
  How much depends on n_active and the Silverman bandwidth.

If (3) looks indistinguishable from (1) or (2), great — Silverman is
fine for this regime.  If (3) over-smooths the χ cascade peaks
significantly, the J4 SVI likelihood surface will lose information
on natal_spin_scale and natal_spin_loc, and we'll revisit per
``[[feedback-gaussian-kde-over-smooths-cascade-peaks]]``.
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from red_team.backup_particle_mc import MCConfig, particle_mc_jax  # noqa: E402


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
)


def _run_cascade(n_seed: int, depletion: float, key_seed: int):
    """Run J2 particle_mc_jax at Stage 1 θ*; return (m, chi, w, n_init)."""
    cfg = MCConfig(
        n_seed=n_seed, depletion_fraction=depletion, **THETA_STAR,
    )
    out = particle_mc_jax(cfg, jax.random.PRNGKey(key_seed))
    am = out["active_mask"]
    m = np.asarray(out["m_active"][am])
    chi = np.asarray(out["chi_active"][am])
    w = np.asarray(out["w_active"][am])
    mergers = np.asarray(out["mergers"][out["step_active"]])
    return m, chi, w, mergers, n_seed


def _build_numpy_evaluator(m, chi, mergers, w):
    """numpy MCEvaluator (k-NN balloon)."""
    from hierarchical_ad_pop_engine.evaluation.mc import MCEvaluator
    return MCEvaluator(
        m, chi, mergers,
        log_R_env=0.0,
        kernel_params={"a": THETA_STAR["kernel_a"], "b": THETA_STAR["kernel_b"]},
        weights=w,
    )


def _build_jax_evaluator(m, chi, mergers, w):
    """JAX MCEvaluatorJax (Gaussian KDE)."""
    from hierarchical_ad_pop_engine.evaluation.mc_jax import MCEvaluatorJax
    return MCEvaluatorJax.from_run(
        jnp.asarray(m), jnp.asarray(chi), jnp.asarray(mergers),
        log_R_env=jnp.asarray(0.0),
        kernel_params={"a": jnp.asarray(THETA_STAR["kernel_a"]),
                       "b": jnp.asarray(THETA_STAR["kernel_b"])},
        weights=jnp.asarray(w),
    )


def _plot_dN_dm_compare(m, w, ev_np, ev_jax, out_dir: Path, m_min, m_max):
    import matplotlib.pyplot as plt
    m_grid = np.geomspace(m_min, m_max, 200)
    chi_grid = np.linspace(0.001, 0.999, 96)

    # k-NN N(m, χ) → integrate over χ for dN/dm.
    M, C = np.meshgrid(m_grid, chi_grid, indexing="ij")
    N_np = ev_np.density(M, C)
    N_jax = np.asarray(ev_jax.density(jnp.asarray(M), jnp.asarray(C)))
    dchi = np.diff(chi_grid)
    w_chi = np.concatenate([dchi[:1], 0.5 * (dchi[:-1] + dchi[1:]), dchi[-1:]])
    dN_dm_np = (N_np * w_chi[None, :]).sum(axis=1)
    dN_dm_jax = (N_jax * w_chi[None, :]).sum(axis=1)

    # Raw histogram on log-m bins.  Sums weights per bin then divides
    # by bin width to get a density.
    n_bins = 40
    log_m_edges = np.linspace(np.log(m_min), np.log(m_max), n_bins + 1)
    m_edges = np.exp(log_m_edges)
    bin_w = np.diff(m_edges)
    counts, _ = np.histogram(m, bins=m_edges, weights=w)
    dN_dm_hist = counts / bin_w
    bin_centers = 0.5 * (m_edges[:-1] + m_edges[1:])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.loglog(bin_centers, np.maximum(dN_dm_hist, 1e-30),
              drawstyle="steps-mid", color="0.25", lw=2.0,
              label=f"raw histogram (n_bins={n_bins})")
    ax.loglog(m_grid, np.maximum(dN_dm_np, 1e-30), color="C0", lw=2.0,
              label="numpy MCEvaluator (k-NN balloon)")
    ax.loglog(m_grid, np.maximum(dN_dm_jax, 1e-30), color="C3", lw=2.0, ls="--",
              label="JAX MCEvaluatorJax (Gaussian KDE)")
    ax.set_xlabel(r"single-BH mass $m \;\;[M_\odot]$")
    ax.set_ylabel(r"$dN/dm$")
    ax.set_title(
        rf"Forward demo $dN/dm$ — Stage 1 $\theta^*$, "
        rf"n_active={m.size}, $h_{{\log m}}={float(ev_jax.h_log_m):.3f}$"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    peak = max(dN_dm_hist.max(), dN_dm_np.max(), dN_dm_jax.max())
    ax.set_ylim(bottom=peak * 1e-6)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    fig.tight_layout()
    out = out_dir / "mc_kde_compare_dN_dm.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    return out


def _plot_dN_dchi_compare(chi, w, ev_np, ev_jax, out_dir: Path, m_min, m_max):
    import matplotlib.pyplot as plt
    m_grid = np.geomspace(m_min, m_max, 96)
    chi_grid = np.linspace(0.001, 0.999, 200)

    M, C = np.meshgrid(m_grid, chi_grid, indexing="ij")
    N_np = ev_np.density(M, C)
    N_jax = np.asarray(ev_jax.density(jnp.asarray(M), jnp.asarray(C)))
    log_m = np.log(m_grid)
    dlog = np.diff(log_m)
    w_log_m = np.concatenate([dlog[:1], 0.5 * (dlog[:-1] + dlog[1:]), dlog[-1:]])
    w_m = w_log_m * m_grid
    dN_dchi_np = (N_np * w_m[:, None]).sum(axis=0)
    dN_dchi_jax = (N_jax * w_m[:, None]).sum(axis=0)

    n_bins = 50
    chi_edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(chi, bins=chi_edges, weights=w)
    bin_w = np.diff(chi_edges)
    dN_dchi_hist = counts / bin_w
    centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.semilogy(centers, np.maximum(dN_dchi_hist, 1e-30),
                drawstyle="steps-mid", color="0.25", lw=2.0,
                label=f"raw histogram (n_bins={n_bins})")
    ax.semilogy(chi_grid, np.maximum(dN_dchi_np, 1e-30), color="C0", lw=2.0,
                label="numpy MCEvaluator (k-NN balloon)")
    ax.semilogy(chi_grid, np.maximum(dN_dchi_jax, 1e-30), color="C3", lw=2.0, ls="--",
                label="JAX MCEvaluatorJax (Gaussian KDE)")
    ax.axvline(0.6865, color="0.5", lw=0.8, ls=":",
               label=r"$\chi_{\rm rem}(1, 0)$ cascade target")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(r"$dN/d\chi$")
    ax.set_title(
        rf"Forward demo $dN/d\chi$ — Stage 1 $\theta^*$, "
        rf"n_active={chi.size}, $h_\chi={float(ev_jax.h_chi):.4f}$"
    )
    ax.set_xlim(0.0, 1.0)
    peak = max(dN_dchi_hist.max(), dN_dchi_np.max(), dN_dchi_jax.max())
    ax.set_ylim(bottom=peak * 1e-4)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    fig.tight_layout()
    out = out_dir / "mc_kde_compare_dN_dchi.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    return out


def _plot_2d_heatmap_compare(m, chi, w, ev_np, ev_jax, out_dir: Path, m_min, m_max):
    import matplotlib.pyplot as plt
    m_grid = np.geomspace(m_min, m_max, 80)
    chi_grid = np.linspace(0.0, 1.0, 80)
    M, C = np.meshgrid(m_grid, chi_grid, indexing="ij")
    N_np = ev_np.density(M, C)
    N_jax = np.asarray(ev_jax.density(jnp.asarray(M), jnp.asarray(C)))

    # 2-D histogram on (log m, χ).
    log_m_edges = np.linspace(np.log(m_min), np.log(m_max), 41)
    chi_edges = np.linspace(0.0, 1.0, 41)
    H, _, _ = np.histogram2d(
        np.log(np.clip(m, m_min, m_max)), chi,
        bins=[log_m_edges, chi_edges], weights=w,
    )
    # Convert count-per-cell to density via cell area (m·dlog m·dχ).
    log_m_centers = 0.5 * (log_m_edges[:-1] + log_m_edges[1:])
    chi_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    m_centers = np.exp(log_m_centers)
    dlog_m = np.diff(log_m_edges)[0]
    dchi = np.diff(chi_edges)[0]
    H_density = H / (m_centers[:, None] * dlog_m * dchi)

    # Three-panel plot — share color scale across all three.
    vmax = max(N_np.max(), N_jax.max(), H_density.max())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, data, label, x_grid, y_grid in [
        (axes[0], H_density, "raw 2-D histogram", m_centers, chi_centers),
        (axes[1], N_np, "numpy k-NN balloon", m_grid, chi_grid),
        (axes[2], N_jax, "JAX Gaussian KDE", m_grid, chi_grid),
    ]:
        # log-normal color map
        from matplotlib.colors import LogNorm
        nonzero_floor = max(vmax * 1e-4, 1e-12)
        norm = LogNorm(vmin=nonzero_floor, vmax=vmax)
        Xm, Cy = np.meshgrid(x_grid, y_grid, indexing="ij")
        im = ax.pcolormesh(
            Xm, Cy, np.maximum(data, nonzero_floor),
            shading="auto", norm=norm, cmap="viridis",
        )
        ax.set_xscale("log")
        ax.set_xlim(m_min, m_max)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$m \;\;[M_\odot]$")
        ax.set_title(label)
        ax.grid(True, ls=":", color="white", alpha=0.3, which="both")
    axes[0].set_ylabel(r"single-BH spin $\chi$")
    fig.suptitle(
        r"Forward demo: $N(m, \chi)$ — histogram vs k-NN vs Gaussian KDE"
    )
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(r"$N(m, \chi)$ (count density)")
    out = out_dir / "mc_kde_compare_2d.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-hash", default=None)
    parser.add_argument("--n-seed", type=int, default=20_000,
                        help="Initial particle count (default 20000 — "
                             "production-scale; cascade-fidelity bias "
                             "~0.2%% per project-jax-mc-j2-done.md).")
    parser.add_argument("--depletion-fraction", type=float, default=0.10,
                        help="Stage 1 mid-cascade depletion (default 0.10).")
    parser.add_argument("--key-seed", type=int, default=2026)
    args = parser.parse_args(argv)

    rh = args.run_hash or secrets.token_hex(3)
    out_dir = Path(__file__).parent / f"figures_{rh}"
    out_dir.mkdir(exist_ok=True)
    print(f"Run hash: {rh}")
    print(f"  figures → demo/figures_{rh}/")
    print(f"  cfg: n_seed={args.n_seed}, depletion={args.depletion_fraction}")

    print("Running JAX cascade...")
    m, chi, w, mergers, n_init = _run_cascade(
        args.n_seed, args.depletion_fraction, args.key_seed,
    )
    print(f"  n_init={n_init}, n_active={m.size}, "
          f"Σw_active={w.sum():.2f}, n_mergers={mergers.shape[0]}")

    print("Building numpy MCEvaluator (k-NN balloon)...")
    ev_np = _build_numpy_evaluator(m, chi, mergers, w)
    print(f"  k = {ev_np.k}, n_eff = {ev_np.n_eff:.1f}")

    print("Building JAX MCEvaluatorJax (Gaussian KDE)...")
    ev_jax = _build_jax_evaluator(m, chi, mergers, w)
    print(f"  h_log_m = {float(ev_jax.h_log_m):.4f}, "
          f"h_chi = {float(ev_jax.h_chi):.4f}, "
          f"n_eff = {float(ev_jax.n_eff):.1f}")

    print("Plotting...")
    _plot_dN_dm_compare(m, w, ev_np, ev_jax, out_dir,
                        THETA_STAR["m_min"], THETA_STAR["m_max"])
    _plot_dN_dchi_compare(chi, w, ev_np, ev_jax, out_dir,
                          THETA_STAR["m_min"], THETA_STAR["m_max"])
    _plot_2d_heatmap_compare(m, chi, w, ev_np, ev_jax, out_dir,
                             THETA_STAR["m_min"], THETA_STAR["m_max"])
    print("Done.")


if __name__ == "__main__":
    main()
