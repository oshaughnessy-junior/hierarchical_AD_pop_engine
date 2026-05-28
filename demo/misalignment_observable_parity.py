r"""Misalignment Session D — observable-space MC ↔ engine parity.

Layered on top of ``demo/forward_modeling_observables.py`` and
``red_team/observable_metrics.py``.  Runs **one resolution × three
orientations** (Aligned + Isotropic + Fisher:κ) in a single
invocation, computes per-observable JS / rel-L1 / total-rate-ratio
metrics with the engine-derived bulk mask, and writes:

* ``demo/figures_mobs_<hash>/all_observables.png`` — single summary
  figure, rows = orientations, columns = observables; engine solid
  vs MC overlay so the orientation effect is visually direct.
* ``demo/figures_mobs_<hash>/comparison_<obs>.png`` — one figure
  per observable with all three orientations side-by-side; same axes
  so cascade compression vs broadening is the eye's first read.
* ``demo/artifacts_mobs_<hash>/metrics.json`` — unified machine-
  readable summary.
* ``demo/artifacts_mobs_<hash>/orientations_<o_tag>.npz`` — per-
  orientation engine + MC density arrays for re-plotting.

Per ``demo/PLAN_misalignment_observable_parity.md``.  See also
``[[project-misalignment-session-c-done]]`` deferred-items section
for the motivation: Session C shipped the orientation pipeline + solver-
level invariants; Session D is the workstation-only observable-layer
parity gate that Session C explicitly punted to a demo.

CLI
---
::

    python demo/misalignment_observable_parity.py
    python demo/misalignment_observable_parity.py --kappa 5.0 --n-mc 50000
    python demo/misalignment_observable_parity.py --resolution 96x32

The Fisher concentration is the user-decided representative value
(κ=5 by default, sitting between κ=0 ≡ Isotropic and κ→∞ ≡ Aligned).

Constraints honoured
--------------------
* No fake physics ([[feedback-no-fake-physics-stand-ins]]) — MC and
  engine both run real vector BR + LZ + per-orientation retention.
* No frozen numerical references ([[feedback-no-frozen-test-references]])
  — figures + JSON record numbers for the human; the companion
  pytest gates on engine-vs-MC and MC-vs-MC RELATIONSHIPS only.
* No hardcoded physical scales ([[feedback-no-hardcoded-physical-scales]])
  — bulk mask is engine-derived at fixed ``tail_frac=0.01``.
* MC depletion tracks engine f_natal_depleted
  ([[feedback-mc-depletion-tracks-engine]]) — read from the implicit
  solver's converged result, never hardcoded.
* Implicit solver is production
  ([[feedback-perturbative-unreliable-implicit-production]]).
"""
from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from hierarchical_ad_pop_engine import observables as obs  # noqa: E402
from hierarchical_ad_pop_engine.spin_orientation import (  # noqa: E402
    Aligned, Fisher, Isotropic,
)

from red_team import observable_metrics as omet  # noqa: E402
from red_team import observables_mc as mc_obs  # noqa: E402
from red_team.comparison_helpers import (  # noqa: E402
    run_implicit, run_mc,
)


# ---------------------------------------------------------------------------
# Stage 1 truth point — kept in sync with forward_modeling_observables.py and
# test_redteam_observables_parity.py.
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
TAIL_FRAC_DEFAULT = 0.01


# ---------------------------------------------------------------------------
# Orientation helpers.  Mirror forward_modeling_observables.parse_orientation /
# orientation_tag so the demo + companion tests use one vocabulary.
# ---------------------------------------------------------------------------
def make_orientations(
    kappa: float,
    *,
    n_theta: int | None = None,
    n_dphi: int | None = None,
) -> list[tuple[str, object]]:
    r"""Return [(tag, model), ...] for the three Session D orientations.

    The list order is the row order in the summary figure.
    ``n_theta`` / ``n_dphi`` ``None`` ⇒ use the :mod:`spin_orientation`
    module defaults (post-2026-05-27 default is ``(4, 8) = 128``
    quadrature points; see that module's Convergence Notes for the
    smooth-integrand rationale).
    """
    extra: dict = {}
    if n_theta is not None:
        extra["n_theta"] = int(n_theta)
    if n_dphi is not None:
        extra["n_dphi"] = int(n_dphi)
    return [
        ("aligned", Aligned()),  # Aligned ignores n_theta/n_dphi (single point)
        ("isotropic", Isotropic.default(**extra)),
        (f"fisher_k{float(kappa):g}", Fisher.default(kappa=float(kappa), **extra)),
    ]


def orientation_label(tag: str) -> str:
    """Pretty axis-label form of the file-safe orientation tag."""
    if tag == "aligned":
        return "Aligned"
    if tag == "isotropic":
        return "Isotropic"
    if tag.startswith("fisher_k"):
        return rf"Fisher($\kappa={tag[len('fisher_k'):]}$)"
    return tag


# ---------------------------------------------------------------------------
# Bin-edges helper — same logic Phase C uses; centralised here.
# ---------------------------------------------------------------------------
def m_bin_edges_from_grid(m_grid: np.ndarray) -> np.ndarray:
    r"""Bin edges that put each engine grid point at a bin centre (log-half-step)."""
    log_m = np.log(m_grid)
    half = 0.5 * np.diff(log_m)
    lo = log_m[0] - half[0]
    hi = log_m[-1] + half[-1]
    inner = 0.5 * (log_m[:-1] + log_m[1:])
    return np.exp(np.concatenate([[lo], inner, [hi]]))


# ---------------------------------------------------------------------------
# Per-orientation pipeline: solve → MC × 2 seeds → observable arrays.
# ---------------------------------------------------------------------------
def solve_and_sample(
    theta: dict,
    n_m: int,
    n_chi: int,
    epsilon: float,
    n_mc: int,
    seed: int,
    orientation: object,
) -> dict:
    r"""Run the engine + two MC seeds at one orientation; return a dict
    carrying every array the metrics computation downstream needs.

    Two MC seeds is the minimum needed for an MC-vs-MC noise gauge —
    Phase C convention.  ``[[feedback-mc-depletion-tracks-engine]]``:
    the depletion fraction is the engine's converged f_natal_depleted.
    """
    t0 = time.time()
    response, result, m_grid, chi_grid = run_implicit(
        theta,
        epsilon,
        n_m=n_m,
        n_chi=n_chi,
        orientation=orientation,
    )
    t_solve = time.time() - t0

    # Engine 1-D and 2-D observables on matching axes.
    N = jnp.asarray(result.N)
    engine_total = float(
        obs.total_rate(
            N, response.grid, response.kernel_params, response.log_R_env,
        )
    )

    _, eng_dN_dm1 = obs.marginal_1d(
        N, response.grid, response.kernel_params, response.log_R_env,
        axis=obs.AXIS_M1,
    )
    eng_dN_dm1 = np.asarray(eng_dN_dm1)

    # 2-D bin grids — same edges as Phase C (16 χ_eff bins, 19 q bins).
    chi_eff_edges = np.linspace(0.0, 1.0, 17)   # 16 bins
    q_edges = np.linspace(0.05, 1.0, 20)        # 19 bins
    m_edges = m_bin_edges_from_grid(m_grid)
    m_widths = np.diff(m_edges)
    chi_eff_widths = np.diff(chi_eff_edges)
    q_widths = np.diff(q_edges)

    _, _, eng_m1_chi_eff = obs.dN_dm1_dchi_eff(
        N, response.grid, response.kernel_params, response.log_R_env,
        chi_eff_bin_edges=jnp.asarray(chi_eff_edges),
        ordered=True,
    )
    _, _, eng_m1_q = obs.dN_dm1_dq(
        N, response.grid, response.kernel_params, response.log_R_env,
        q_bin_edges=jnp.asarray(q_edges),
        ordered=True,
    )
    eng_m1_chi_eff = np.asarray(eng_m1_chi_eff)
    eng_m1_q = np.asarray(eng_m1_q)

    # 1-D χ_eff = mass-integrate the 2-D (m₁, χ_eff) engine density.
    # The 2-D density is per (m₁, χ_eff), so we sum over m₁ with the
    # m-axis bin widths to recover the 1-D χ_eff density.
    eng_dN_dchi_eff = np.sum(eng_m1_chi_eff * m_widths[:, None], axis=0)

    # MC: two seeds at the engine's converged depletion fraction.
    f_dep = max(float(result.f_natal_depleted), 1e-3)
    # Independent orientation key streams per seed.
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)
    t0 = time.time()
    mc1 = run_mc(
        theta,
        n_seed=n_mc,
        rng=rng1,
        depletion_fraction=f_dep,
        orientation=orientation,
        orientation_key_seed=seed * 7919 + 11,
        track_generations=False,
    )
    mc2 = run_mc(
        theta,
        n_seed=n_mc,
        rng=rng2,
        depletion_fraction=f_dep,
        orientation=orientation,
        orientation_key_seed=seed * 7919 + 23,
        track_generations=False,
    )
    t_mc = time.time() - t0

    mergers1 = mc1["mergers"]
    mergers2 = mc2["mergers"]
    if mergers1.shape[0] == 0 or mergers2.shape[0] == 0:
        raise RuntimeError(
            f"Empty MC mergers under orientation; cannot compute parity. "
            f"n_accepted_seed1={mergers1.shape[0]} "
            f"n_accepted_seed2={mergers2.shape[0]}"
        )

    rate_scale_1 = mc_obs.infer_rate_scale_from_total(mergers1, engine_total)
    rate_scale_2 = mc_obs.infer_rate_scale_from_total(mergers2, engine_total)

    _, mc1_dN_dm1 = mc_obs.dN_dm1_mc(mergers1, m_edges, rate_scale_1)
    _, mc2_dN_dm1 = mc_obs.dN_dm1_mc(mergers2, m_edges, rate_scale_2)
    _, _, mc1_m1_chi_eff = mc_obs.dN_dm1_dchi_eff_mc(
        mergers1, m_edges, chi_eff_edges, rate_scale_1,
    )
    _, _, mc2_m1_chi_eff = mc_obs.dN_dm1_dchi_eff_mc(
        mergers2, m_edges, chi_eff_edges, rate_scale_2,
    )
    _, _, mc1_m1_q = mc_obs.dN_dm1_dq_mc(
        mergers1, m_edges, q_edges, rate_scale_1,
    )
    _, _, mc2_m1_q = mc_obs.dN_dm1_dq_mc(
        mergers2, m_edges, q_edges, rate_scale_2,
    )

    # 1-D MC χ_eff: same mass-integration recipe applied to the MC 2-D
    # density.  Sharing the engine's m_widths keeps the two paths' axis
    # measure identical (the rate_scale already absorbs the per-event
    # weighting), so the comparison is apples-to-apples.
    mc1_dN_dchi_eff = np.sum(mc1_m1_chi_eff * m_widths[:, None], axis=0)
    mc2_dN_dchi_eff = np.sum(mc2_m1_chi_eff * m_widths[:, None], axis=0)
    chi_eff_centres = 0.5 * (chi_eff_edges[:-1] + chi_eff_edges[1:])
    q_centres = 0.5 * (q_edges[:-1] + q_edges[1:])

    # v_kicks summary — diagnostic only, mirrors forward_modeling_observables.
    v_kicks1 = mc1["v_kicks"]
    if v_kicks1.size:
        kick_summary = dict(
            v_kick_median=float(np.median(v_kicks1)),
            v_kick_p90=float(np.percentile(v_kicks1, 90)),
            v_kick_max=float(np.max(v_kicks1)),
            v_kick_frac_above_v_esc=float(np.mean(v_kicks1 > theta["v_esc"])),
        )
    else:
        kick_summary = dict(
            v_kick_median=0.0, v_kick_p90=0.0, v_kick_max=0.0,
            v_kick_frac_above_v_esc=0.0,
        )

    return dict(
        # config
        n_m=n_m, n_chi=n_chi,
        # engine
        m_grid=m_grid,
        chi_grid=chi_grid,
        m_edges=m_edges, m_widths=m_widths,
        chi_eff_edges=chi_eff_edges, chi_eff_widths=chi_eff_widths,
        q_edges=q_edges, q_widths=q_widths,
        chi_eff_centres=chi_eff_centres, q_centres=q_centres,
        engine_total_rate=engine_total,
        eng_dN_dm1=eng_dN_dm1,
        eng_dN_dchi_eff=eng_dN_dchi_eff,
        eng_m1_chi_eff=eng_m1_chi_eff,
        eng_m1_q=eng_m1_q,
        # MC
        mc1_dN_dm1=mc1_dN_dm1, mc2_dN_dm1=mc2_dN_dm1,
        mc1_dN_dchi_eff=mc1_dN_dchi_eff, mc2_dN_dchi_eff=mc2_dN_dchi_eff,
        mc1_m1_chi_eff=mc1_m1_chi_eff, mc2_m1_chi_eff=mc2_m1_chi_eff,
        mc1_m1_q=mc1_m1_q, mc2_m1_q=mc2_m1_q,
        # diagnostics
        diag=dict(
            n_iter=int(result.n_iter),
            final_diff=float(result.final_diff),
            f_natal_depleted=float(result.f_natal_depleted),
            natal_integral=float(result.natal_integral),
            loss_integral=float(result.loss_integral),
            wall_seconds_solve=t_solve,
            wall_seconds_mc=t_mc,
            mc1_n_accepted=int(mergers1.shape[0]),
            mc2_n_accepted=int(mergers2.shape[0]),
        ),
        kick_summary=kick_summary,
    )


# ---------------------------------------------------------------------------
# Metrics computation — uses the engine-derived bulk mask everywhere.
# ---------------------------------------------------------------------------
def compute_metrics(payload: dict, tail_frac: float) -> dict:
    r"""Return per-observable metrics for one orientation payload.

    For each observable we compute:

    * ``js_imp_mc`` — bulk JS between engine and MC1 (or rel-L1 for 2-D).
    * ``js_noise`` — bulk JS between MC1 and MC2 (or rel-L1 for 2-D).
    * ``ratio_imp_to_noise`` — the headline scaled metric the pytest
      gates on.  The Phase C convention: gate at JS_MULT × noise.
    * ``total_rate_ratio`` — MC1 vs engine.
    * ``bulk_coverage`` — engine bulk-mask coverage at tail_frac.
    * ``peak_engine`` / ``peak_mc`` — argmax-axis-value for human read.

    No assertions here — gating lives in the test file.
    """
    m_grid = payload["m_grid"]
    m_widths = payload["m_widths"]
    chi_eff_centres = payload["chi_eff_centres"]
    chi_eff_widths = payload["chi_eff_widths"]
    q_centres = payload["q_centres"]
    q_widths = payload["q_widths"]

    out = {}

    # --- dN/dm_1 (1-D mass) ---------------------------------------------
    eng = payload["eng_dN_dm1"]
    mc1 = payload["mc1_dN_dm1"]
    mc2 = payload["mc2_dN_dm1"]
    js_imp_mc = omet.js_on_grid_bulk(eng, mc1, m_grid, tail_frac=tail_frac)
    js_noise = omet.js_on_grid_bulk(mc1, mc2, m_grid, tail_frac=tail_frac)
    cov = omet.bulk_mask_coverage_fraction(
        eng, [m_widths], tail_frac=tail_frac,
    )
    eng_total_1d = float(np.sum(eng * m_widths))
    mc_total_1d = float(np.sum(mc1 * m_widths))
    out["dN_dm1"] = dict(
        kind="1d",
        metric="js",
        js_imp_mc=float(js_imp_mc),
        js_noise=float(js_noise),
        ratio_imp_to_noise=(
            float(js_imp_mc / js_noise)
            if np.isfinite(js_noise) and js_noise > 0
            else float("nan")
        ),
        total_rate_ratio=(
            mc_total_1d / eng_total_1d if eng_total_1d > 0 else float("nan")
        ),
        bulk_coverage=float(cov),
        peak_engine=float(omet.peak_location(eng, m_grid)),
        peak_mc=float(omet.peak_location(mc1, m_grid)),
    )

    # --- dN/dχ_eff (1-D effective spin, mass-integrated) ----------------
    eng = payload["eng_dN_dchi_eff"]
    mc1 = payload["mc1_dN_dchi_eff"]
    mc2 = payload["mc2_dN_dchi_eff"]
    js_imp_mc = omet.js_on_grid_bulk(
        eng, mc1, chi_eff_centres, tail_frac=tail_frac,
    )
    js_noise = omet.js_on_grid_bulk(
        mc1, mc2, chi_eff_centres, tail_frac=tail_frac,
    )
    cov = omet.bulk_mask_coverage_fraction(
        eng, [chi_eff_widths], tail_frac=tail_frac,
    )
    eng_total_1d = float(np.sum(eng * chi_eff_widths))
    mc_total_1d = float(np.sum(mc1 * chi_eff_widths))
    out["dN_dchi_eff"] = dict(
        kind="1d",
        metric="js",
        js_imp_mc=float(js_imp_mc),
        js_noise=float(js_noise),
        ratio_imp_to_noise=(
            float(js_imp_mc / js_noise)
            if np.isfinite(js_noise) and js_noise > 0
            else float("nan")
        ),
        total_rate_ratio=(
            mc_total_1d / eng_total_1d if eng_total_1d > 0 else float("nan")
        ),
        bulk_coverage=float(cov),
        peak_engine=float(omet.peak_location(eng, chi_eff_centres)),
        peak_mc=float(omet.peak_location(mc1, chi_eff_centres)),
    )

    # --- dN/(dm_1 dχ_eff) (2-D) ------------------------------------------
    eng = payload["eng_m1_chi_eff"]
    mc1 = payload["mc1_m1_chi_eff"]
    mc2 = payload["mc2_m1_chi_eff"]
    l1_imp_mc = omet.relative_L1_bulk(
        eng, mc1, [m_widths, chi_eff_widths], tail_frac=tail_frac,
    )
    l1_noise = omet.relative_L1_bulk(
        mc1, mc2, [m_widths, chi_eff_widths], tail_frac=tail_frac,
    )
    ratio2d = omet.total_rate_ratio(eng, mc1, [m_widths, chi_eff_widths])
    cov = omet.bulk_mask_coverage_fraction(
        eng, [m_widths, chi_eff_widths], tail_frac=tail_frac,
    )
    p_e = omet.peak_location_2d(eng, m_grid, chi_eff_centres)
    p_m = omet.peak_location_2d(mc1, m_grid, chi_eff_centres)
    out["dN_dm1_dchi_eff"] = dict(
        kind="2d",
        metric="rel_l1",
        rel_l1_imp_mc=float(l1_imp_mc),
        rel_l1_noise=float(l1_noise),
        ratio_imp_to_noise=(
            float(l1_imp_mc / l1_noise)
            if np.isfinite(l1_noise) and l1_noise > 0
            else float("nan")
        ),
        total_rate_ratio=float(ratio2d),
        bulk_coverage=float(cov),
        peak_engine=[float(p_e[0]), float(p_e[1])],
        peak_mc=[float(p_m[0]), float(p_m[1])],
    )

    # --- dN/(dm_1 dq) (2-D) ----------------------------------------------
    eng = payload["eng_m1_q"]
    mc1 = payload["mc1_m1_q"]
    mc2 = payload["mc2_m1_q"]
    l1_imp_mc = omet.relative_L1_bulk(
        eng, mc1, [m_widths, q_widths], tail_frac=tail_frac,
    )
    l1_noise = omet.relative_L1_bulk(
        mc1, mc2, [m_widths, q_widths], tail_frac=tail_frac,
    )
    ratio2d = omet.total_rate_ratio(eng, mc1, [m_widths, q_widths])
    cov = omet.bulk_mask_coverage_fraction(
        eng, [m_widths, q_widths], tail_frac=tail_frac,
    )
    p_e = omet.peak_location_2d(eng, m_grid, q_centres)
    p_m = omet.peak_location_2d(mc1, m_grid, q_centres)
    out["dN_dm1_dq"] = dict(
        kind="2d",
        metric="rel_l1",
        rel_l1_imp_mc=float(l1_imp_mc),
        rel_l1_noise=float(l1_noise),
        ratio_imp_to_noise=(
            float(l1_imp_mc / l1_noise)
            if np.isfinite(l1_noise) and l1_noise > 0
            else float("nan")
        ),
        total_rate_ratio=float(ratio2d),
        bulk_coverage=float(cov),
        peak_engine=[float(p_e[0]), float(p_e[1])],
        peak_mc=[float(p_m[0]), float(p_m[1])],
    )

    return out


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------
def _plot_1d_axis(ax, x, eng, mc1, mc2, *, label_x, title, log_x=False):
    floor = max(float(np.nanmax(eng)), 1e-300) * 1e-4
    plot = ax.loglog if log_x else ax.semilogy
    plot(x, np.maximum(eng, floor), color="C0", lw=2.0, label="engine")
    plot(x, np.maximum(mc1, floor), color="C2", lw=1.4, ls="--", label="MC seed 1")
    plot(x, np.maximum(mc2, floor), color="C3", lw=0.9, ls=":", label="MC seed 2")
    ax.set_xlabel(label_x)
    ax.set_title(title, fontsize="x-small")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.tick_params(labelsize="x-small")
    ax.set_ylim(bottom=floor)


def _plot_2d_axis(ax, x, y, eng, mc1, *, label_x, label_y, title, log_x=False):
    import matplotlib.colors as mcolors
    finite = eng[np.isfinite(eng) & (eng > 0)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "no engine support", ha="center",
                transform=ax.transAxes)
        return
    vmax = float(np.max(finite))
    vmin = vmax * 1e-3
    ax.pcolormesh(
        x, y, eng.T,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis", shading="auto", rasterized=True,
    )
    finite_mc = mc1[np.isfinite(mc1) & (mc1 > 0)]
    if finite_mc.size:
        levels = np.geomspace(vmin * 5, vmax * 0.5, 4)
        try:
            ax.contour(
                x, y, mc1.T,
                levels=levels, colors="white", linewidths=0.8, linestyles="--",
            )
        except Exception:
            pass
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(title, fontsize="x-small")
    ax.tick_params(labelsize="x-small")


def render_summary_figure(
    payloads: dict, metrics: dict, theta: dict, path: Path,
) -> None:
    r"""Single summary figure: rows = orientations, columns = observables.

    Columns: dN/dm_1, dN/dχ_eff, dN/(dm_1 dχ_eff), dN/(dm_1 dq).
    """
    import matplotlib.pyplot as plt

    tags = list(payloads.keys())
    n_rows = len(tags)
    n_cols = 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.6 * n_cols, 2.8 * n_rows + 0.4),
        squeeze=False,
    )

    for r, tag in enumerate(tags):
        p = payloads[tag]
        m_metric = metrics[tag]
        label = orientation_label(tag)
        # Col 0 — dN/dm_1
        _plot_1d_axis(
            axes[r, 0], p["m_grid"],
            p["eng_dN_dm1"], p["mc1_dN_dm1"], p["mc2_dN_dm1"],
            label_x=r"$m_1\;[M_\odot]$",
            title=(
                f"{label}  dN/dm_1   "
                f"ratio={m_metric['dN_dm1']['ratio_imp_to_noise']:.2g}"
            ),
            log_x=True,
        )
        if r == 0:
            axes[r, 0].legend(loc="upper right", fontsize="x-small", frameon=False)
        # Col 1 — dN/dχ_eff
        _plot_1d_axis(
            axes[r, 1], p["chi_eff_centres"],
            p["eng_dN_dchi_eff"], p["mc1_dN_dchi_eff"], p["mc2_dN_dchi_eff"],
            label_x=r"$\chi_{\rm eff}$",
            title=(
                f"{label}  dN/dχ_eff   "
                f"ratio={m_metric['dN_dchi_eff']['ratio_imp_to_noise']:.2g}"
            ),
            log_x=False,
        )
        # Col 2 — dN/(dm_1 dχ_eff)
        _plot_2d_axis(
            axes[r, 2], p["m_grid"], p["chi_eff_centres"],
            p["eng_m1_chi_eff"], p["mc1_m1_chi_eff"],
            label_x=r"$m_1$", label_y=r"$\chi_{\rm eff}$",
            title=(
                f"{label}  dN/(dm_1 dχ_eff)   "
                f"L1={m_metric['dN_dm1_dchi_eff']['rel_l1_imp_mc']:.2g}"
            ),
            log_x=True,
        )
        # Col 3 — dN/(dm_1 dq)
        _plot_2d_axis(
            axes[r, 3], p["m_grid"], p["q_centres"],
            p["eng_m1_q"], p["mc1_m1_q"],
            label_x=r"$m_1$", label_y=r"$q$",
            title=(
                f"{label}  dN/(dm_1 dq)   "
                f"L1={m_metric['dN_dm1_dq']['rel_l1_imp_mc']:.2g}"
            ),
            log_x=True,
        )

    fig.suptitle(
        f"Misalignment Session D — observable-space MC↔engine parity "
        f"(θ* Stage 1; v_esc={theta['v_esc']:.0f} km/s)\n"
        f"Per-panel ratio = engine-vs-MC bulk JS / MC-vs-MC bulk JS; "
        f"L1 = engine-vs-MC bulk rel-L1.",
        fontsize="small",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_per_observable_figure(
    obs_name: str, x_label: str,
    payloads: dict, metrics: dict, path: Path,
    *, is_2d: bool, log_x: bool,
) -> None:
    r"""One figure per observable, three orientations side-by-side.

    Eye-first reading: cascade-compression vs broadening as orientation
    moves Aligned → Fisher → Isotropic.
    """
    import matplotlib.pyplot as plt

    tags = list(payloads.keys())
    n_cols = len(tags)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(3.6 * n_cols, 3.0), squeeze=False,
    )

    for c, tag in enumerate(tags):
        p = payloads[tag]
        m = metrics[tag][obs_name]
        label = orientation_label(tag)
        if obs_name == "dN_dm1":
            x = p["m_grid"]
            eng, mc1, mc2 = p["eng_dN_dm1"], p["mc1_dN_dm1"], p["mc2_dN_dm1"]
            title = f"{label}\ndN/dm_1   ratio={m['ratio_imp_to_noise']:.2g}"
            _plot_1d_axis(
                axes[0, c], x, eng, mc1, mc2,
                label_x=x_label, title=title, log_x=log_x,
            )
        elif obs_name == "dN_dchi_eff":
            x = p["chi_eff_centres"]
            eng, mc1, mc2 = (
                p["eng_dN_dchi_eff"], p["mc1_dN_dchi_eff"],
                p["mc2_dN_dchi_eff"],
            )
            title = (
                f"{label}\ndN/dχ_eff   "
                f"ratio={m['ratio_imp_to_noise']:.2g}"
            )
            _plot_1d_axis(
                axes[0, c], x, eng, mc1, mc2,
                label_x=x_label, title=title, log_x=log_x,
            )
        elif obs_name == "dN_dm1_dchi_eff":
            title = (
                f"{label}\ndN/(dm_1 dχ_eff)   "
                f"L1={m['rel_l1_imp_mc']:.2g}"
            )
            _plot_2d_axis(
                axes[0, c], p["m_grid"], p["chi_eff_centres"],
                p["eng_m1_chi_eff"], p["mc1_m1_chi_eff"],
                label_x=r"$m_1$", label_y=r"$\chi_{\rm eff}$",
                title=title, log_x=log_x,
            )
        elif obs_name == "dN_dm1_dq":
            title = (
                f"{label}\ndN/(dm_1 dq)   "
                f"L1={m['rel_l1_imp_mc']:.2g}"
            )
            _plot_2d_axis(
                axes[0, c], p["m_grid"], p["q_centres"],
                p["eng_m1_q"], p["mc1_m1_q"],
                label_x=r"$m_1$", label_y=r"$q$",
                title=title, log_x=log_x,
            )
        else:
            raise ValueError(f"Unknown observable name: {obs_name}")

    fig.suptitle(
        f"{obs_name} — three orientations side-by-side", fontsize="small",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Artifact serialisation.
# ---------------------------------------------------------------------------
def save_per_orientation_npz(path: Path, payload: dict) -> None:
    r"""Per-orientation density arrays for post-hoc re-plotting."""
    np.savez(
        path,
        m_grid=payload["m_grid"],
        chi_grid=payload["chi_grid"],
        m_edges=payload["m_edges"],
        chi_eff_edges=payload["chi_eff_edges"],
        chi_eff_centres=payload["chi_eff_centres"],
        q_edges=payload["q_edges"],
        q_centres=payload["q_centres"],
        eng_dN_dm1=payload["eng_dN_dm1"],
        eng_dN_dchi_eff=payload["eng_dN_dchi_eff"],
        eng_m1_chi_eff=payload["eng_m1_chi_eff"],
        eng_m1_q=payload["eng_m1_q"],
        mc1_dN_dm1=payload["mc1_dN_dm1"],
        mc2_dN_dm1=payload["mc2_dN_dm1"],
        mc1_dN_dchi_eff=payload["mc1_dN_dchi_eff"],
        mc2_dN_dchi_eff=payload["mc2_dN_dchi_eff"],
        mc1_m1_chi_eff=payload["mc1_m1_chi_eff"],
        mc2_m1_chi_eff=payload["mc2_m1_chi_eff"],
        mc1_m1_q=payload["mc1_m1_q"],
        mc2_m1_q=payload["mc2_m1_q"],
        engine_total_rate=np.float64(payload["engine_total_rate"]),
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_resolution(s: str) -> tuple[int, int]:
    if "x" not in s:
        raise argparse.ArgumentTypeError(
            f"--resolution {s!r} must be Nm×Nχ (e.g. 64x20)"
        )
    a, b = s.split("x")
    return int(a), int(b)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resolution", type=parse_resolution, default="64x20",
        help="Engine resolution Nm×Nχ (single pair).  Three orientations "
             "× one resolution = three implicit solves per run.  "
             "Default: 64x20 (matches Phase C).",
    )
    parser.add_argument(
        "--kappa", type=float, default=5.0,
        help="Fisher concentration κ for the third orientation.  "
             "Default 5.0 sits between κ=0 ≡ Isotropic and κ→∞ ≡ "
             "Aligned.  κ ≥ 0.",
    )
    parser.add_argument(
        "--n-mc", type=int, default=50_000,
        help="Natal pool size for the particle MC (per seed; two seeds per "
             "orientation).  Default 5e4.  Bump to 1e5 if MC noise "
             "visibly dominates the engine-vs-MC parity ratio.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="Implicit-solver coupling × time.  ε ≈ 1 is a fully "
             "developed cascade.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260527,
        help="Root seed for both MC numpy RNGs and the JAX orientation "
             "key stream.",
    )
    parser.add_argument(
        "--out-hash", type=str, default=None,
        help="Output-dir suffix (default: random 6-hex per "
             "[[feedback-demo-output-conventions]]).",
    )
    parser.add_argument(
        "--n-theta", type=int, default=None,
        help="Override the per-spin Gauss-Legendre node count on "
             "(cos θ_1, cos θ_2) for Isotropic / Fisher orientations.  "
             "Default uses the spin_orientation module default "
             "(currently 4 — see that module's Convergence Notes for "
             "the smooth-integrand argument behind the cheap default).  "
             "Bump to 8 alongside --n-dphi=16 to recover the v1 "
             "1024-point quadrature accuracy under Heaviside retention.",
    )
    parser.add_argument(
        "--n-dphi", type=int, default=None,
        help="Override the periodic-trapezoidal node count on Δφ.  "
             "Default uses the spin_orientation module default "
             "(currently 8).  See --n-theta for the v1-accuracy recipe.",
    )
    parser.add_argument(
        "--tail-frac", type=float, default=TAIL_FRAC_DEFAULT,
        help=f"Engine-bulk-mask threshold (fraction of peak engine "
             f"density).  Default {TAIL_FRAC_DEFAULT} — matches Phase "
             f"C.  Per [[feedback-no-fudge-factors]] you should not "
             f"tune this per orientation; expose to the human only as "
             f"a diagnostic knob.",
    )
    args = parser.parse_args()

    if args.kappa < 0.0:
        raise SystemExit(f"--kappa must be ≥ 0 (got {args.kappa}).")

    theta = dict(THETA_STAR)

    here = Path(__file__).parent
    rh = args.out_hash or secrets.token_hex(3)
    fig_dir = here / f"figures_mobs_{rh}"
    art_dir = here / f"artifacts_mobs_{rh}"
    fig_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    print(f"misalignment_observable_parity — run hash: {rh}")
    print(f"  figures   → {fig_dir.relative_to(here.parent)}/")
    print(f"  artifacts → {art_dir.relative_to(here.parent)}/")
    print(
        f"  resolution={args.resolution}  kappa={args.kappa}  "
        f"n_mc={args.n_mc}  epsilon={args.epsilon}  seed={args.seed}  "
        f"tail_frac={args.tail_frac}"
    )

    orientations = make_orientations(
        args.kappa, n_theta=args.n_theta, n_dphi=args.n_dphi,
    )
    n_m, n_chi = args.resolution
    # Report the effective quadrature for the non-Aligned orientations
    # (Aligned is always single-point).
    _iso = orientations[1][1]
    n_theta_eff, n_dphi_eff = int(_iso.n_theta), int(_iso.n_dphi)
    print(
        f"  orientation quadrature: ({n_theta_eff}, {n_theta_eff}, "
        f"{n_dphi_eff}) = {n_theta_eff * n_theta_eff * n_dphi_eff} "
        f"points per merger cell (Aligned uses 1 point)"
    )

    payloads = {}
    metrics_all = {}
    metadata = dict(
        run_hash=rh,
        theta_star=theta,
        resolution=list(args.resolution),
        kappa=args.kappa,
        n_mc=args.n_mc,
        epsilon=args.epsilon,
        seed=args.seed,
        tail_frac=args.tail_frac,
        orientation_n_theta=n_theta_eff,
        orientation_n_dphi=n_dphi_eff,
        orientation_n_quad=n_theta_eff * n_theta_eff * n_dphi_eff,
        orientations={},
    )

    for tag, orientation in orientations:
        print(f"\n=== orientation: {tag} ===")
        payload = solve_and_sample(
            theta, n_m, n_chi, args.epsilon, args.n_mc, args.seed, orientation,
        )
        d = payload["diag"]
        print(
            f"  engine   iters={d['n_iter']}  |ΔN|∞={d['final_diff']:.2e}  "
            f"f_dep={d['f_natal_depleted']:.4g}  "
            f"solve={d['wall_seconds_solve']:.1f}s"
        )
        print(
            f"  MC       n_acc_seed1={d['mc1_n_accepted']}  "
            f"n_acc_seed2={d['mc2_n_accepted']}  "
            f"mc={d['wall_seconds_mc']:.1f}s"
        )
        print(
            f"  kicks    median={payload['kick_summary']['v_kick_median']:.1f}  "
            f"p90={payload['kick_summary']['v_kick_p90']:.1f}  "
            f"max={payload['kick_summary']['v_kick_max']:.1f}  "
            f"frac>v_esc={payload['kick_summary']['v_kick_frac_above_v_esc']:.3g}"
        )

        m = compute_metrics(payload, tail_frac=args.tail_frac)
        for obs_name, mm in m.items():
            if mm["kind"] == "1d":
                print(
                    f"  metric   {obs_name:18s}  JS_imp/MC={mm['js_imp_mc']:.3e}  "
                    f"JS_noise={mm['js_noise']:.3e}  ratio="
                    f"{mm['ratio_imp_to_noise']:.2g}  rate_ratio="
                    f"{mm['total_rate_ratio']:.3f}  bulk_cov="
                    f"{mm['bulk_coverage']:.2f}"
                )
            else:
                print(
                    f"  metric   {obs_name:18s}  L1_imp/MC={mm['rel_l1_imp_mc']:.3e}  "
                    f"L1_noise={mm['rel_l1_noise']:.3e}  ratio="
                    f"{mm['ratio_imp_to_noise']:.2g}  rate_ratio="
                    f"{mm['total_rate_ratio']:.3f}  bulk_cov="
                    f"{mm['bulk_coverage']:.2f}"
                )

        # Per-orientation npz.
        npz_path = art_dir / f"orientations_{tag}.npz"
        save_per_orientation_npz(npz_path, payload)
        print(f"  wrote {npz_path.relative_to(here.parent)}")

        payloads[tag] = payload
        metrics_all[tag] = m
        metadata["orientations"][tag] = dict(
            engine_total_rate=payload["engine_total_rate"],
            diag=d,
            kick_summary=payload["kick_summary"],
            metrics=m,
        )

    # Summary figure (one figure, all observables × orientations).
    summary_path = fig_dir / "all_observables.png"
    print(f"\n[render] {summary_path.relative_to(here.parent)}")
    render_summary_figure(payloads, metrics_all, theta, summary_path)

    # Per-observable side-by-side figures.
    obs_specs = [
        ("dN_dm1",            r"$m_1\;[M_\odot]$",   False, True),
        ("dN_dchi_eff",       r"$\chi_{\rm eff}$",   False, False),
        ("dN_dm1_dchi_eff",   "",                    True,  True),
        ("dN_dm1_dq",         "",                    True,  True),
    ]
    for obs_name, xlab, is_2d, log_x in obs_specs:
        p = fig_dir / f"comparison_{obs_name}.png"
        print(f"[render] {p.relative_to(here.parent)}")
        render_per_observable_figure(
            obs_name, xlab, payloads, metrics_all, p,
            is_2d=is_2d, log_x=log_x,
        )

    # Unified JSON.
    meta_path = art_dir / "metrics.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=float))
    print(f"  wrote {meta_path.relative_to(here.parent)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
