"""Spin-results demo: clustered χ grid (believable) × Fisher-κ (interesting),
benchmarked against the particle MC (authoritative).

Two panels, both on the production ``solver="implicit"`` path, each overlaid
with the **particle-MC raw weighted histogram** — the authoritative reliable
answer.  No KDE anywhere (we've been bedeviled by KDE smoothing errors;
benchmarking against MC is by eye for now — raw histograms only).

* **Left — "believable" (reorganized χ grid).**  ``dN/dχ`` of the surviving
  single-BH density ``N*`` on the clustered grid vs a uniform grid of the
  *same* node count, vs the MC histogram.  The clustered nodes (dense at
  the natal peak χ≈0.05 and the 2G cascade peak χ≈0.69, max cell width
  capped at ``max_gap_factor ×`` uniform — the χ→1 tail-artifact guard)
  resolve the narrow cascade structure that the uniform grid blurs.

* **Right — "interesting" (Fisher-κ).**  The same ``dN/dχ`` as the
  spin-orientation concentration κ is swept Isotropic → aligned, with a
  matched MC run per κ.

MC convention (``[[feedback-mc-depletion-tracks-engine]]``): each MC run
uses ``depletion_fraction =`` the matched engine solve's converged
``f_natal_depleted`` — never hardcoded.  The MC histogram is scaled to the
engine marginal's total integral so shapes and relative normalisation are
comparable by eye.

Rendering (``--render``): ``pchip-log`` (default) draws each engine curve
as a shape-preserving monotone cubic of ``log dN/dχ`` through the node
values — the higher-order continuous recovery used for paper figures;
``hat`` draws the faithful piecewise-linear bilinear-hat representation
(exactly what the solver's basis encodes).  Node markers are drawn either
way; the MC histogram is never smoothed.

Caching: every engine solve / MC run is cached as an ``.npz`` keyed on the
full configuration hash (``--cache-dir``, default
``demo/.cache_chi_grid_kappa``).  Re-running with the same flags reuses
results — this is what lets paper-resolution figures be assembled across
multiple short sandbox commands and re-made cheaply by ``make``.

Run (workstation / JAX env)::

    python demo/forward_modeling_chi_grid_kappa.py                 # sandbox-ish
    python demo/forward_modeling_chi_grid_kappa.py \
        --n-m 64 --n-chi 32 --kappas 0,2,8,30 --mc-n-seed 200000   # paper res

Output PNG: ``<outdir>/chi_grid_kappa.png`` (default
``demo/figures_chi_grid_kappa/``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.spin_orientation import Aligned, Isotropic, Fisher
from hierarchical_ad_pop_engine.response.coagulation import build_gaussian_coagulation
from hierarchical_ad_pop_engine.response.implicit import (
    solve_implicit_fixed_point,
    _dchi_weights,
    _dm_log_weights,
)

# ``COAG_LEAN_GAIN`` (any truthy value) disables the F2 gain-side χ-out cache
# (``orientation_precompute.R_CHI_OUT_DEFAULT``), which scales ~``n_χ⁴`` and OOMs
# at the ``n_χ ≥ 64`` this κ-sweep now runs at; the gain side falls back to the
# memory-lean per-Picard-iteration orientation scan (F2 is documented
# bit-equivalent).  Same switch as ``forward_modeling_observables.py`` — see that
# script's "efficient high-resolution settings" docstring for the full recipe.
# Off by default so low-n_χ callers keep the F2 speed-up.
if os.environ.get("COAG_LEAN_GAIN"):
    import hierarchical_ad_pop_engine.orientation_precompute as _op  # noqa: E402
    import hierarchical_ad_pop_engine.response.implicit as _imp  # noqa: E402

    _op.R_CHI_OUT_DEFAULT = None
    _imp.R_CHI_OUT_DEFAULT = None


# ---------------------------------------------------------------------------
# Orientation helpers.
# ---------------------------------------------------------------------------
def _orientation_from_kappa(kappa: float, n_theta: int, n_dphi: int):
    """κ=0 → Isotropic; κ>0 → Fisher(κ); κ=inf → Aligned."""
    if kappa == 0.0:
        return Isotropic.default(n_theta=n_theta, n_dphi=n_dphi)
    if np.isinf(kappa):
        return Aligned()
    return Fisher.default(kappa=kappa, n_theta=n_theta, n_dphi=n_dphi)


def _orientation_tag(kappa: float) -> str:
    if kappa == 0.0:
        return "iso"
    if np.isinf(kappa):
        return "aligned"
    return f"fisher{kappa:g}"


# ---------------------------------------------------------------------------
# Cache plumbing — config-hash keyed npz files.
# ---------------------------------------------------------------------------
def _cfg_hash(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _cached(cache_dir: str, key: dict, compute):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key['kind']}_{_cfg_hash(key)}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    out = compute()
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------------------
# Engine solve → χ marginal of N* (+ f_natal_depleted for the MC).
# ---------------------------------------------------------------------------
def _engine_chi_marginal(grid_kind: str, kappa: float, args) -> dict:
    if grid_kind == "clustered":
        grid = BinaryGrid.clustered_chi(
            m_min=args.m_min, m_max=args.m_max, n_m=args.n_m,
            n_chi=args.n_chi, max_gap_factor=args.max_gap_factor,
        )
    else:
        grid = BinaryGrid.default(
            m_min=args.m_min, m_max=args.m_max, n_m=args.n_m, n_chi=args.n_chi,
        )
    o = _orientation_from_kappa(kappa, args.n_theta, args.n_dphi)
    resp = build_gaussian_coagulation(
        mu=args.mu, sigma=args.sigma, m_min=args.m_min, m_max=args.m_max,
        natal_spin_loc=0.0, natal_spin_scale=args.natal_spin_scale,
        kernel_b=args.kernel_b, grid=grid, solver="implicit",
        orientation=o, retention_kind="sigmoid",
    )
    res = solve_implicit_fixed_point(
        resp, jnp.asarray(args.v_esc), epsilon=args.epsilon,
        max_iter=args.max_iter, tol=1e-7, damping=0.5,
    )
    dm_w = _dm_log_weights(grid.m, grid.log_m)
    dN_dchi = jnp.einsum("ij,i->j", res.N, dm_w)
    return dict(
        chi=np.asarray(grid.chi),
        dN_dchi=np.asarray(dN_dchi),
        f_dep=np.asarray(res.f_natal_depleted),
        n_iter=np.asarray(res.n_iter),
    )


def engine_chi_marginal(grid_kind: str, kappa: float, args) -> dict:
    key = dict(
        kind="engine", grid=grid_kind, kappa=kappa,
        n_m=args.n_m, n_chi=args.n_chi, m_min=args.m_min, m_max=args.m_max,
        mu=args.mu, sigma=args.sigma, sscale=args.natal_spin_scale,
        kb=args.kernel_b, eps=args.epsilon, ve=args.v_esc,
        nth=args.n_theta, ndp=args.n_dphi, mgf=args.max_gap_factor,
        mi=args.max_iter,
    )
    return _cached(args.cache_dir, key,
                   lambda: _engine_chi_marginal(grid_kind, kappa, args))


# ---------------------------------------------------------------------------
# Particle MC (authoritative) → raw weighted χ histogram.
# ---------------------------------------------------------------------------
def _mc_chi_hist(kappa: float, f_dep: float, args) -> dict:
    from red_team.comparison_helpers import run_mc

    theta = dict(
        mu_natal=args.mu, sigma_natal=args.sigma,
        m_min=args.m_min, m_max=args.m_max,
        natal_spin_loc=0.0, natal_spin_scale=args.natal_spin_scale,
        kernel_a=0.0, kernel_b=args.kernel_b, v_esc=args.v_esc,
    )
    o = _orientation_from_kappa(kappa, args.n_theta, args.n_dphi)
    rng = np.random.default_rng(args.mc_seed)
    mc = run_mc(
        theta, args.mc_n_seed, rng,
        depletion_fraction=max(float(f_dep), 1e-3),
        orientation=None if isinstance(o, Aligned) else o,
    )
    chi_act = np.asarray(mc["chi_active"])
    w = np.asarray(mc.get("w_active", np.ones_like(chi_act)))
    edges = np.linspace(0.0, 1.0, args.mc_bins + 1)
    hist, _ = np.histogram(chi_act, bins=edges, weights=w)
    dens = hist / np.diff(edges) / max(w.sum(), 1e-30)  # unit-integral density
    return dict(edges=edges, density=dens, n_active=np.asarray(chi_act.size))


def mc_chi_hist(kappa: float, f_dep: float, args) -> dict:
    key = dict(
        kind="mc", kappa=kappa, fdep=round(float(f_dep), 6),
        n_seed=args.mc_n_seed, seed=args.mc_seed, bins=args.mc_bins,
        m_min=args.m_min, m_max=args.m_max, mu=args.mu, sigma=args.sigma,
        sscale=args.natal_spin_scale, kb=args.kernel_b, ve=args.v_esc,
        nth=args.n_theta, ndp=args.n_dphi,
    )
    return _cached(args.cache_dir, key, lambda: _mc_chi_hist(kappa, f_dep, args))


# ---------------------------------------------------------------------------
# Rendering — higher-order continuous recovery of the node marginal.
# ---------------------------------------------------------------------------
def render_curve(chi_nodes, dN, chi_fine, kind: str, floor: float = 1e-300):
    """Continuous dN/dχ through the node values.

    ``hat``       — piecewise linear (the faithful bilinear-hat basis trace).
    ``pchip-log`` — monotone shape-preserving cubic (PCHIP) of
                    ``log dN/dχ``: smooth, positive, no overshoot — the
                    higher-order recovery for paper rendering.
    """
    if kind == "hat":
        return np.interp(chi_fine, chi_nodes, dN)
    from scipy.interpolate import PchipInterpolator

    logd = np.log(np.maximum(dN, floor))
    return np.exp(PchipInterpolator(chi_nodes, logd)(chi_fine))


# ---------------------------------------------------------------------------
# Figure.
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-m", type=int, default=24)
    p.add_argument("--n-chi", type=int, default=16)
    p.add_argument("--m-min", type=float, default=4.0)
    p.add_argument("--m-max", type=float, default=80.0)
    p.add_argument("--mu", type=float, default=8.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--natal-spin-scale", type=float, default=0.05)
    p.add_argument("--kernel-b", type=float, default=2.0)
    p.add_argument("--epsilon", type=float, default=0.6)
    p.add_argument("--v-esc", type=float, default=300.0)
    p.add_argument("--kappas", type=str, default="0,2,8,30",
                   help="comma list; 0 => Isotropic, inf => Aligned")
    p.add_argument("--n-theta", type=int, default=4)
    p.add_argument("--n-dphi", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=80)
    p.add_argument("--max-gap-factor", type=float, default=2.0)
    p.add_argument("--render", choices=("pchip-log", "hat"), default="pchip-log")
    p.add_argument("--panels", choices=("both", "right", "left"), default="both",
                   help="which panel(s) to render: 'right' = the Fisher-kappa "
                        "sweep alone (a standalone PRD figure); 'both' keeps the "
                        "clustered-vs-uniform placement panel on the left.")
    p.add_argument("--mc", action=argparse.BooleanOptionalAction, default=True,
                   help="overlay the particle-MC raw histograms (authoritative)")
    p.add_argument("--mc-n-seed", type=int, default=60000)
    p.add_argument("--mc-bins", type=int, default=60)
    p.add_argument("--mc-seed", type=int, default=20260604)
    p.add_argument("--cache-dir", type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        ".cache_chi_grid_kappa"))
    p.add_argument("--outdir", type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        "figures_chi_grid_kappa"))
    p.add_argument("--solve-only", action="store_true",
                   help="fill the cache (engine+MC) and exit without plotting "
                        "— lets long configs be assembled across short runs")
    args = p.parse_args()

    kappas = [float(k) for k in args.kappas.split(",")]
    chi_fine = np.linspace(1e-3, 1.0 - 1e-3, 2001)

    # ── Engine solves (cached) ──────────────────────────────────────────
    sweep = {k: engine_chi_marginal("clustered", k, args) for k in kappas}
    if args.panels in ("both", "left"):
        # left panel: clustered-vs-uniform placement at the isotropic point.
        eng_clu_iso = engine_chi_marginal("clustered", 0.0, args)
        eng_uni_iso = engine_chi_marginal("uniform", 0.0, args)
    else:
        # right-only: reuse the isotropic sweep solve; skip the extra pair.
        eng_clu_iso = sweep.get(0.0)
        eng_uni_iso = None

    # ── MC runs (cached; depletion tracks each engine solve) ───────────
    mc_runs = {}
    if args.mc:
        if eng_clu_iso is not None:
            mc_runs[0.0] = mc_chi_hist(0.0, eng_clu_iso["f_dep"], args)
        for k in kappas:
            if k not in mc_runs:
                mc_runs[k] = mc_chi_hist(k, sweep[k]["f_dep"], args)

    if args.solve_only:
        print("cache filled:", args.cache_dir)
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _integral(chi, dN):
        d = np.diff(chi)
        w = np.concatenate([d[:1] * 0.5, 0.5 * (d[:-1] + d[1:]), d[-1:] * 0.5])
        return float(np.sum(w * dN))

    def _mc_step(ax, mc, scale, color, label=None):
        ax.stairs(mc["density"] * scale, mc["edges"], color=color, lw=1.1,
                  alpha=0.85, label=label, zorder=1)

    def _plot_left(axL):
        # clustered vs uniform vs MC (isotropic config).
        for eng, color, lab in (
            (eng_uni_iso, "0.6", f"uniform χ (n_χ={args.n_chi})"),
            (eng_clu_iso, "C3", f"clustered χ (n_χ={args.n_chi})"),
        ):
            curve = render_curve(eng["chi"], eng["dN_dchi"], chi_fine, args.render)
            axL.plot(chi_fine, curve, "-", color=color, lw=1.4, label=lab, zorder=2)
            axL.plot(eng["chi"], eng["dN_dchi"], "o", color=color, ms=3, zorder=3)
        if args.mc:
            scale = _integral(eng_clu_iso["chi"], eng_clu_iso["dN_dchi"])
            _mc_step(axL, mc_runs[0.0], scale, "k",
                     f"particle MC (n={int(mc_runs[0.0]['n_active'])}, raw hist)")
        axL.set_title("Clustered χ vs uniform vs MC (isotropic)")
        axL.set_xlabel(r"$\chi$"); axL.set_ylabel(r"$dN/d\chi$")
        axL.set_yscale("log"); axL.legend(fontsize=8, loc="upper right")
        axL.axvline(0.6865, ls=":", color="k", lw=0.8)

    def _plot_right(axR):
        # κ sweep on the clustered grid, MC overlay per κ.
        cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(kappas)))
        for k, c in zip(kappas, cmap):
            eng = sweep[k]
            lab = ("Isotropic (κ→0)" if k == 0.0
                   else ("Aligned (κ→∞)" if np.isinf(k) else f"Fisher κ={k:g}"))
            curve = render_curve(eng["chi"], eng["dN_dchi"], chi_fine, args.render)
            axR.plot(chi_fine, curve, "-", color=c, lw=1.4, label=lab, zorder=2)
            axR.plot(eng["chi"], eng["dN_dchi"], "o", color=c, ms=2.5, zorder=3)
            if args.mc:
                scale = _integral(eng["chi"], eng["dN_dchi"])
                _mc_step(axR, mc_runs[k], scale, c)
        axR.set_title(rf"Fisher-$\kappa$ orientation sweep vs particle MC "
                      rf"($n_\chi={args.n_chi}$; steps = MC)")
        axR.set_xlabel(r"$\chi$"); axR.set_ylabel(r"$dN/d\chi$")
        axR.set_yscale("log"); axR.legend(fontsize=8)
        axR.axvline(0.6865, ls=":", color="k", lw=0.8)

    if args.panels == "both":
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))
        _plot_left(axL); _plot_right(axR); axes = (axL, axR)
    elif args.panels == "right":
        fig, axR = plt.subplots(1, 1, figsize=(6.2, 4.6))
        _plot_right(axR); axes = (axR,)
    else:
        fig, axL = plt.subplots(1, 1, figsize=(6.2, 4.6))
        _plot_left(axL); axes = (axL,)

    # Common y floor: don't let log-rendering artifacts dominate the view.
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(max(lo, hi * 1e-9), hi)

    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "chi_grid_kappa.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
