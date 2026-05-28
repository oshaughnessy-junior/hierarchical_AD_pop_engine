r"""Forward demo — new JAX MC vs old numpy MC, raw histograms only.

Direct cascade-dynamics comparison.  No KDE, no balloon, no smoothing —
just raw histograms of the surviving pool + accepted-merger record
from both implementations at the same θ*, same n_seed, same depletion.

If the two histograms agree to MC SE, the K-rejection-as-weighting
design (J2) reproduces the discrete-K-rejection dynamics (numpy
particle_mc) to within expected statistical noise, and any difference
seen in the density-estimator comparison (``forward_modeling_mc_kde_compare.py``)
is genuinely a *density-estimator* artefact — not a cascade-dynamics
artefact.

User finding 2026-05-28 (post-J4 workstation visual):
  "JAX KDE actually better than k-NN balloon on the χ ∈ [0.4, 0.6]
   gap; both smooth the deep-cascade peaks (3rd / 4th order have ~100
   counts per bin so MC noise dominates).  Probably better to test
   new MC vs old MC directly."

This script answers exactly that.

Run::

    python demo/forward_modeling_mc_vs_mc.py
    # writes demo/figures_<hash>/mc_vs_mc_{dN_dm,dN_dchi,2d}.png
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

from red_team.backup_particle_mc import (  # noqa: E402
    MCConfig,
    particle_mc,
    particle_mc_jax,
)


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


def _run_jax(n_seed: int, depletion: float, key_seed: int):
    cfg = MCConfig(n_seed=n_seed, depletion_fraction=depletion, **THETA_STAR)
    out = particle_mc_jax(cfg, jax.random.PRNGKey(key_seed))
    am = out["active_mask"]
    sa = out["step_active"]
    m = np.asarray(out["m_active"][am])
    chi = np.asarray(out["chi_active"][am])
    w = np.asarray(out["w_active"][am])
    mergers = np.asarray(out["mergers"][sa])
    w_mergers = np.asarray(out["w_mergers"][sa])
    return dict(
        m_active=m, chi_active=chi, w_active=w,
        mergers=mergers, w_mergers=w_mergers,
    )


def _run_numpy(n_seed: int, depletion: float, rng_seed: int):
    cfg = MCConfig(
        n_seed=n_seed, depletion_fraction=depletion, **THETA_STAR,
    )
    out = particle_mc(cfg, np.random.default_rng(rng_seed))
    return dict(
        m_active=out["m_active"],
        chi_active=out["chi_active"],
        w_active=out.get("w_active",
                         np.ones_like(out["m_active"])),
        mergers=out["mergers"],
        w_mergers=out.get("w_mergers",
                          np.ones(out["mergers"].shape[0])),
    )


def _plot_dN_dm(jax_run, np_run, out_dir: Path, m_min, m_max):
    import matplotlib.pyplot as plt
    n_bins = 40
    log_m_edges = np.linspace(np.log(m_min), np.log(m_max), n_bins + 1)
    m_edges = np.exp(log_m_edges)
    bin_w = np.diff(m_edges)
    centers = 0.5 * (m_edges[:-1] + m_edges[1:])

    # Active pool
    h_jax, _ = np.histogram(
        jax_run["m_active"], bins=m_edges, weights=jax_run["w_active"],
    )
    h_np, _ = np.histogram(
        np_run["m_active"], bins=m_edges, weights=np_run["w_active"],
    )
    # Merger remnants (column 7 = m_rem when retained, NaN otherwise)
    mj_rem = jax_run["mergers"][:, 7]
    keep_j = np.isfinite(mj_rem) & (mj_rem > 0)
    h_jax_rem, _ = np.histogram(
        mj_rem[keep_j], bins=m_edges, weights=jax_run["w_mergers"][keep_j],
    )
    mn_rem = np_run["mergers"][:, 7]
    keep_n = np.isfinite(mn_rem) & (mn_rem > 0)
    h_np_rem, _ = np.histogram(
        mn_rem[keep_n], bins=m_edges, weights=np_run["w_mergers"][keep_n],
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=False)
    for ax, h_j, h_n, ylabel, title in [
        (axes[0], h_jax / bin_w, h_np / bin_w,
         r"$dN/dm$ (active pool, per $M_\odot$)",
         "surviving pool"),
        (axes[1], h_jax_rem / bin_w, h_np_rem / bin_w,
         r"$dN/dm_{\rm rem}$ (per $M_\odot$)",
         "accepted mergers (retained, weighted)"),
    ]:
        ax.loglog(centers, np.maximum(h_n, 1e-30),
                  drawstyle="steps-mid", color="C0", lw=2.0,
                  label=f"numpy MC (n_acc={h_n.sum()*bin_w.mean():.0f})")
        ax.loglog(centers, np.maximum(h_j, 1e-30),
                  drawstyle="steps-mid", color="C3", lw=2.0, ls="--",
                  label=f"JAX MC (Σw={h_j.sum()*bin_w.mean():.0f})")
        ax.set_xlabel(r"$m \;\;[M_\odot]$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        peak = max(h_j.max(), h_n.max())
        ax.set_ylim(bottom=peak * 1e-5)
        ax.legend(loc="lower left", frameon=False, fontsize="small")
    fig.suptitle("MC dynamics: numpy vs JAX (raw histograms, n_bins={})".format(n_bins))
    fig.tight_layout()
    out = out_dir / "mc_vs_mc_dN_dm.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


def _plot_dN_dchi(jax_run, np_run, out_dir: Path):
    import matplotlib.pyplot as plt
    n_bins = 50
    chi_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    bin_w = np.diff(chi_edges)[0]

    h_jax, _ = np.histogram(
        jax_run["chi_active"], bins=chi_edges, weights=jax_run["w_active"],
    )
    h_np, _ = np.histogram(
        np_run["chi_active"], bins=chi_edges, weights=np_run["w_active"],
    )
    # Remnant χ — column 7 is m_rem; mergers schema doesn't carry chi_rem
    # directly except through the heavier-first chi1 column.  Use the
    # *evolved* chi_active for both (which IS the cascade output of
    # remnant χ values for everything past 1G).
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(centers, np.maximum(h_np / bin_w, 1e-30),
                drawstyle="steps-mid", color="C0", lw=2.0,
                label=f"numpy MC (n_active={np_run['m_active'].size})")
    ax.semilogy(centers, np.maximum(h_jax / bin_w, 1e-30),
                drawstyle="steps-mid", color="C3", lw=2.0, ls="--",
                label=f"JAX MC (n_active={jax_run['m_active'].size})")
    ax.axvline(0.6865, color="0.4", lw=0.8, ls=":",
               label=r"$\chi_{\rm rem}(1, 0)$ cascade target")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(r"$dN/d\chi$ (surviving pool, weighted)")
    ax.set_title(f"MC dynamics: $dN/d\\chi$ raw histograms, n_bins={n_bins}")
    ax.set_xlim(0.0, 1.0)
    peak = max(h_jax.max(), h_np.max()) / bin_w
    ax.set_ylim(bottom=peak * 1e-4)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    fig.tight_layout()
    out = out_dir / "mc_vs_mc_dN_dchi.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


def _plot_2d_compare(jax_run, np_run, out_dir: Path, m_min, m_max):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    log_m_edges = np.linspace(np.log(m_min), np.log(m_max), 41)
    chi_edges = np.linspace(0.0, 1.0, 41)
    m_centers = np.exp(0.5 * (log_m_edges[:-1] + log_m_edges[1:]))

    H_j, _, _ = np.histogram2d(
        np.log(np.clip(jax_run["m_active"], m_min, m_max)),
        jax_run["chi_active"], bins=[log_m_edges, chi_edges],
        weights=jax_run["w_active"],
    )
    H_n, _, _ = np.histogram2d(
        np.log(np.clip(np_run["m_active"], m_min, m_max)),
        np_run["chi_active"], bins=[log_m_edges, chi_edges],
        weights=np_run["w_active"],
    )

    vmax = max(H_j.max(), H_n.max())
    nonzero_floor = max(vmax * 1e-4, 1e-3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, H, title in [
        (axes[0], H_n, f"numpy MC (n={np_run['m_active'].size})"),
        (axes[1], H_j, f"JAX MC (n={jax_run['m_active'].size})"),
        (axes[2], H_j - H_n, "JAX - numpy (signed)"),
    ]:
        if title.startswith("JAX - numpy"):
            cmax = max(np.abs(H_j - H_n).max(), 1e-3)
            im = ax.pcolormesh(
                m_centers, 0.5 * (chi_edges[:-1] + chi_edges[1:]),
                (H_j - H_n).T,
                shading="auto", cmap="seismic", vmin=-cmax, vmax=cmax,
            )
        else:
            norm = LogNorm(vmin=nonzero_floor, vmax=vmax)
            im = ax.pcolormesh(
                m_centers, 0.5 * (chi_edges[:-1] + chi_edges[1:]),
                np.maximum(H, nonzero_floor).T,
                shading="auto", norm=norm, cmap="viridis",
            )
        ax.set_xscale("log")
        ax.set_xlim(m_min, m_max)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"$m \;\;[M_\odot]$")
        ax.set_title(title)
        ax.grid(True, ls=":", color="0.6", alpha=0.3, which="both")
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    axes[0].set_ylabel(r"single-BH spin $\chi$")
    fig.suptitle("MC vs MC: 2-D (m, χ) histograms of the surviving pool")
    fig.tight_layout()
    out = out_dir / "mc_vs_mc_2d.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


def _ks_summary(jax_run, np_run):
    from scipy.stats import ks_2samp
    print()
    print("Distributional parity (KS test on surviving pool):")
    for col, vals_j, vals_n in [
        ("m_active",
         jax_run["m_active"], np_run["m_active"]),
        ("chi_active",
         jax_run["chi_active"], np_run["chi_active"]),
    ]:
        ks = ks_2samp(vals_j, vals_n)
        print(f"  {col}: KS p = {ks.pvalue:.3g}   (>0.01 ⇒ "
              f"indistinguishable at 1 % level)")
    # Merger m_rem
    mj = jax_run["mergers"][:, 7]
    mn = np_run["mergers"][:, 7]
    mj = mj[np.isfinite(mj) & (mj > 0)]
    mn = mn[np.isfinite(mn) & (mn > 0)]
    if mj.size and mn.size:
        ks = ks_2samp(mj, mn)
        print(f"  m_rem (retained): KS p = {ks.pvalue:.3g}, "
              f"n_jax={mj.size}, n_np={mn.size}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-hash", default=None)
    parser.add_argument("--n-seed", type=int, default=20_000)
    parser.add_argument("--depletion-fraction", type=float, default=0.10)
    parser.add_argument("--jax-key-seed", type=int, default=2026)
    parser.add_argument("--np-rng-seed", type=int, default=20260518)
    args = parser.parse_args(argv)

    rh = args.run_hash or secrets.token_hex(3)
    out_dir = Path(__file__).parent / f"figures_{rh}"
    out_dir.mkdir(exist_ok=True)
    print(f"Run hash: {rh}")
    print(f"  figures → demo/figures_{rh}/")
    print(f"  cfg: n_seed={args.n_seed}, depletion={args.depletion_fraction}")
    print()

    print("Running numpy particle_mc (discrete K-rejection)...")
    np_run = _run_numpy(args.n_seed, args.depletion_fraction, args.np_rng_seed)
    print(f"  n_init={args.n_seed}, n_active={np_run['m_active'].size}, "
          f"n_accepted={np_run['mergers'].shape[0]}")

    print()
    print("Running JAX particle_mc_jax (K-rejection-as-weighting)...")
    jax_run = _run_jax(args.n_seed, args.depletion_fraction, args.jax_key_seed)
    print(f"  n_init={args.n_seed}, n_active={jax_run['m_active'].size}, "
          f"n_steps={jax_run['mergers'].shape[0]}, "
          f"Σw_mergers={jax_run['w_mergers'].sum():.1f} "
          f"(≈ numpy n_accepted in expectation)")

    _ks_summary(jax_run, np_run)

    print()
    print("Plotting...")
    _plot_dN_dm(jax_run, np_run, out_dir,
                THETA_STAR["m_min"], THETA_STAR["m_max"])
    _plot_dN_dchi(jax_run, np_run, out_dir)
    _plot_2d_compare(jax_run, np_run, out_dir,
                     THETA_STAR["m_min"], THETA_STAR["m_max"])
    print("Done.")


if __name__ == "__main__":
    main()
