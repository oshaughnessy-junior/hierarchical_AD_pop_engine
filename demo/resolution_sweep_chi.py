"""χ/m resolution convergence sweep — truncation error for forward modeling.

The paper's "believable resolution" question, answered with a *continuous*
measure of *relative* differences against the highest-resolution run:

For each resolution knob (``n_chi`` at fixed benchmark ``n_m``; ``n_m`` at
fixed benchmark ``n_chi``) this script solves the production implicit
response on the clustered χ grid, reconstructs the marginals
``dN/dχ`` and ``dN/dm`` **continuously** (the evaluate-framework recovery:
node marginals on the engine quadrature + a higher-order shape-preserving
``pchip-log`` reconstruction — or the faithful piecewise-linear ``hat``
trace via ``--render hat``), and reports

    relL1(n) = ∫ |f_n(x) − f_ref(x)| dx / ∫ f_ref(x) dx

evaluated on a fine common axis against the benchmark (= the largest value
in each sweep list).  This is a continuous functional distance — no
node-to-node comparison, no histogram binning, no KDE.

Outputs (``--outdir``):

* ``resolution_sweep_chi.png`` — two-panel convergence figure
  (relL1 vs n_chi, relL1 vs n_m; both marginals on each panel, log-log)
  plus a trace panel showing the dN/dχ reconstructions converging.
* ``resolution_sweep_summary.txt`` — the paper-ready summary table.
* ``resolution_sweep.npz`` — raw sweep data.

Each solve is cached (``--cache-dir``) keyed on the full configuration, so
paper-resolution sweeps can be assembled across short sandbox commands and
re-made cheaply from ``make``.

Defaults are sandbox-sized.  Paper resolution (workstation / Makefile)::

    python demo/resolution_sweep_chi.py \
        --n-chi-list 8,12,16,24,32,48,64 --bench-n-m 64 \
        --n-m-list 16,24,32,48,64,96 --bench-n-chi 32

User gate (2026-06-04): a "believable" χ resolution can't be lower than
n_chi=24, preferably 32, with ~64 as the benchmark — this sweep is the
quantitative version of that statement for the paper.
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
    _dm_log_weights,
)


# ---------------------------------------------------------------------------
# Continuous recovery + metric.
# ---------------------------------------------------------------------------
def render_marginal(nodes, vals, x_fine, kind: str = "pchip-log",
                    log_axis: bool = False, floor: float = 1e-300):
    """Continuous reconstruction of a node marginal on a fine axis.

    ``pchip-log``: monotone shape-preserving cubic of ``log f`` (in
    ``log x`` for mass axes) — the higher-order recovery.
    ``hat``: piecewise linear — the faithful bilinear-hat trace.
    """
    xn = np.log(nodes) if log_axis else np.asarray(nodes)
    xq = np.log(x_fine) if log_axis else np.asarray(x_fine)
    if kind == "hat":
        return np.interp(xq, xn, vals)
    from scipy.interpolate import PchipInterpolator

    logf = np.log(np.maximum(np.asarray(vals), floor))
    return np.exp(PchipInterpolator(xn, logf)(xq))


def rel_l1(f, f_ref, x):
    """Continuous relative L1 distance ∫|f−f_ref| / ∫f_ref (trapezoid)."""
    w = np.concatenate([
        np.diff(x)[:1] * 0.5,
        0.5 * (np.diff(x)[:-1] + np.diff(x)[1:]),
        np.diff(x)[-1:] * 0.5,
    ])
    return float(np.sum(w * np.abs(f - f_ref)) / max(np.sum(w * f_ref), 1e-300))


# ---------------------------------------------------------------------------
# One engine solve → both node marginals (cached).
# ---------------------------------------------------------------------------
def _cfg_hash(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def solve_marginals(n_m: int, n_chi: int, args) -> dict:
    key = dict(
        kind="res", n_m=n_m, n_chi=n_chi, m_min=args.m_min, m_max=args.m_max,
        mu=args.mu, sigma=args.sigma, sscale=args.natal_spin_scale,
        kb=args.kernel_b, eps=args.epsilon, ve=args.v_esc, kappa=args.kappa,
        nth=args.n_theta, ndp=args.n_dphi, mgf=args.max_gap_factor,
        mi=args.max_iter,
    )
    os.makedirs(args.cache_dir, exist_ok=True)
    path = os.path.join(args.cache_dir, f"res_{_cfg_hash(key)}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    grid = BinaryGrid.clustered_chi(
        m_min=args.m_min, m_max=args.m_max, n_m=n_m, n_chi=n_chi,
        max_gap_factor=args.max_gap_factor,
    )
    if args.kappa == 0.0:
        o = Isotropic.default(n_theta=args.n_theta, n_dphi=args.n_dphi)
    elif np.isinf(args.kappa):
        o = Aligned()
    else:
        o = Fisher.default(kappa=args.kappa, n_theta=args.n_theta,
                           n_dphi=args.n_dphi)
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
    d = np.diff(np.asarray(grid.chi))
    dchi_w = np.concatenate([d[:1] * 0.5, 0.5 * (d[:-1] + d[1:]), d[-1:] * 0.5])
    out = dict(
        m=np.asarray(grid.m),
        chi=np.asarray(grid.chi),
        dN_dchi=np.asarray(jnp.einsum("ij,i->j", res.N, dm_w)),
        dN_dm=np.asarray(res.N) @ dchi_w,
        f_dep=np.asarray(res.f_natal_depleted),
    )
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------------------
# Sweep driver.
# ---------------------------------------------------------------------------
def run_sweep(args) -> dict:
    chi_fine = np.linspace(1e-3, 1.0 - 1e-3, args.n_fine)
    m_fine = np.geomspace(args.m_min, args.m_max, args.n_fine)

    def _curves(sol):
        c_chi = render_marginal(sol["chi"], sol["dN_dchi"], chi_fine,
                                kind=args.render)
        c_m = render_marginal(sol["m"], sol["dN_dm"], m_fine,
                              kind=args.render, log_axis=True)
        return c_chi, c_m

    results = {"chi_sweep": [], "m_sweep": [],
               "chi_fine": chi_fine, "m_fine": m_fine}

    # ── n_chi sweep at fixed benchmark n_m ─────────────────────────────
    n_chi_list = sorted({int(x) for x in args.n_chi_list.split(",")})
    bench = solve_marginals(args.bench_n_m, n_chi_list[-1], args)
    ref_chi, ref_m = _curves(bench)
    for n in n_chi_list:
        sol = solve_marginals(args.bench_n_m, n, args)
        c_chi, c_m = _curves(sol)
        results["chi_sweep"].append(dict(
            n=n,
            rel_chi=rel_l1(c_chi, ref_chi, chi_fine),
            rel_m=rel_l1(c_m, ref_m, m_fine),
            chi=sol["chi"], dN_dchi=sol["dN_dchi"],
        ))

    # ── n_m sweep at fixed benchmark n_chi ─────────────────────────────
    n_m_list = sorted({int(x) for x in args.n_m_list.split(",")})
    bench_m = solve_marginals(n_m_list[-1], args.bench_n_chi, args)
    ref_chi_m, ref_m_m = _curves(bench_m)
    for n in n_m_list:
        sol = solve_marginals(n, args.bench_n_chi, args)
        c_chi, c_m = _curves(sol)
        results["m_sweep"].append(dict(
            n=n,
            rel_chi=rel_l1(c_chi, ref_chi_m, chi_fine),
            rel_m=rel_l1(c_m, ref_m_m, m_fine),
        ))
    return results


def summarize(results, args) -> str:
    lines = [
        "Resolution sweep — continuous rel-L1 vs highest-resolution benchmark",
        f"(render={args.render}; clustered chi grid, max_gap_factor="
        f"{args.max_gap_factor}; kappa={args.kappa}; epsilon={args.epsilon}; "
        f"v_esc={args.v_esc})",
        "",
        f"n_chi sweep @ n_m={args.bench_n_m} "
        f"(benchmark n_chi={results['chi_sweep'][-1]['n']}):",
        "  n_chi   relL1[dN/dchi]   relL1[dN/dm]",
    ]
    for r in results["chi_sweep"]:
        lines.append(f"  {r['n']:5d}   {r['rel_chi']:12.4e}   {r['rel_m']:10.4e}")
    lines += [
        "",
        f"n_m sweep @ n_chi={args.bench_n_chi} "
        f"(benchmark n_m={results['m_sweep'][-1]['n']}):",
        "  n_m     relL1[dN/dchi]   relL1[dN/dm]",
    ]
    for r in results["m_sweep"]:
        lines.append(f"  {r['n']:5d}   {r['rel_chi']:12.4e}   {r['rel_m']:10.4e}")
    lines += [
        "",
        "Note: the benchmark row reads 0 by construction; the row above it",
        "is the honest truncation estimate at that resolution.",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # physics config (Stage-1-like defaults; isotropic baseline)
    p.add_argument("--m-min", type=float, default=4.0)
    p.add_argument("--m-max", type=float, default=80.0)
    p.add_argument("--mu", type=float, default=8.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--natal-spin-scale", type=float, default=0.05)
    p.add_argument("--kernel-b", type=float, default=2.0)
    p.add_argument("--epsilon", type=float, default=0.6)
    p.add_argument("--v-esc", type=float, default=300.0)
    p.add_argument("--kappa", type=float, default=0.0,
                   help="0 => Isotropic; >0 => Fisher; inf => Aligned")
    p.add_argument("--n-theta", type=int, default=4)
    p.add_argument("--n-dphi", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=120)
    p.add_argument("--max-gap-factor", type=float, default=2.0)
    # sweep config (sandbox-sized; paper res in the paper-repo Makefile)
    p.add_argument("--n-chi-list", type=str, default="8,12,16,24")
    p.add_argument("--bench-n-m", type=int, default=24)
    p.add_argument("--n-m-list", type=str, default="12,16,24,32")
    p.add_argument("--bench-n-chi", type=int, default=16)
    p.add_argument("--n-fine", type=int, default=2001)
    p.add_argument("--render", choices=("pchip-log", "hat"), default="pchip-log")
    p.add_argument("--cache-dir", type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        ".cache_resolution_sweep"))
    p.add_argument("--outdir", type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                        "figures_resolution_sweep"))
    p.add_argument("--solve-only", action="store_true",
                   help="fill the solve cache and exit (progressive runs)")
    args = p.parse_args()

    results = run_sweep(args)
    if args.solve_only:
        print("cache filled:", args.cache_dir)
        return 0

    table = summarize(results, args)
    print(table)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "resolution_sweep_summary.txt"), "w") as fh:
        fh.write(table + "\n")
    np.savez(
        os.path.join(args.outdir, "resolution_sweep.npz"),
        chi_ns=np.asarray([r["n"] for r in results["chi_sweep"]]),
        chi_rel_chi=np.asarray([r["rel_chi"] for r in results["chi_sweep"]]),
        chi_rel_m=np.asarray([r["rel_m"] for r in results["chi_sweep"]]),
        m_ns=np.asarray([r["n"] for r in results["m_sweep"]]),
        m_rel_chi=np.asarray([r["rel_chi"] for r in results["m_sweep"]]),
        m_rel_m=np.asarray([r["rel_m"] for r in results["m_sweep"]]),
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

    cs = results["chi_sweep"]
    ns = [r["n"] for r in cs[:-1]]
    ax1.loglog(ns, [r["rel_chi"] for r in cs[:-1]], "o-", label=r"$dN/d\chi$")
    ax1.loglog(ns, [r["rel_m"] for r in cs[:-1]], "s--", label=r"$dN/dm$")
    for gate, c in ((24, "C2"), (32, "C0")):
        ax1.axvline(gate, ls=":", color=c, lw=0.9)
    ax1.set_xlabel(r"$n_\chi$"); ax1.set_ylabel("rel-L1 vs benchmark")
    ax1.set_title(f"χ-resolution truncation (n_m={args.bench_n_m})")
    ax1.legend(fontsize=8)

    ms = results["m_sweep"]
    ns = [r["n"] for r in ms[:-1]]
    ax2.loglog(ns, [r["rel_chi"] for r in ms[:-1]], "o-", label=r"$dN/d\chi$")
    ax2.loglog(ns, [r["rel_m"] for r in ms[:-1]], "s--", label=r"$dN/dm$")
    ax2.set_xlabel(r"$n_m$"); ax2.set_ylabel("rel-L1 vs benchmark")
    ax2.set_title(f"m-resolution truncation (n_χ={args.bench_n_chi})")
    ax2.legend(fontsize=8)

    chi_fine = results["chi_fine"]
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(cs)))
    for r, c in zip(cs, cmap):
        curve = render_marginal(r["chi"], r["dN_dchi"], chi_fine,
                                kind=args.render)
        ax3.semilogy(chi_fine, curve, color=c, lw=1.1,
                     label=f"$n_\\chi$={r['n']}")
    ax3.set_xlabel(r"$\chi$"); ax3.set_ylabel(r"$dN/d\chi$")
    ax3.set_title("dN/dχ reconstructions converging")
    ax3.legend(fontsize=7)
    lo, hi = ax3.get_ylim()
    ax3.set_ylim(max(lo, hi * 1e-9), hi)

    fig.tight_layout()
    out = os.path.join(args.outdir, "resolution_sweep_chi.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
