r"""Forward-modelling observable demo — implicit Picard ↔ particle MC.

At a single ``θ*``, compare the production implicit-Picard coagulation
solver against an independent particle-MC realisation of the same model,
across observables a LIGO PE reader cares about:

* **Source-frame 1D**  — ``dN/dm`` and ``dN/dχ`` (sanity check on the
  engine's native quadrature; matches both methods at the "no LIGO
  reweighting" baseline).
* **LIGO 1D**  — ``dN/dm₁``, ``dN/dq``, ``dN/dχ_eff``, ``dN/dM_chirp``
  under the ordered ``m₁ ≥ m₂`` convention.
* **LIGO 2D joints**  — ``dN/(dm₁ dq)`` and ``dN/(dm₁ dχ_eff)``, where
  parameter correlations live.

The MC is the simplest possible coagulation simulator (continuous
particle cloud, no grid).  Its stopping fraction is driven self-
consistently from the implicit solver's ``f_natal_depleted``, per
[[feedback-mc-depletion-tracks-engine]].

PLACEHOLDER: KDE smoothing
--------------------------
The MC 1-D plots use a **k-NN balloon estimator** with bandwidth
``h(x) = distance to k-th nearest sample at x`` (Loftsgaarden-
Quesenberry 1965, uniform kernel).  This is a placeholder for the
proper adaptive evaluator described in
``PLAN_arbitrary_point_evaluation.md``: it is location-adaptive (no
hardcoded scales — complies with [[feedback-no-hardcoded-physical-
scales]]) but uniform-kernel k-NN is noisier than the Abramson
variant the plan ultimately specifies.  The plan also covers the
implicit / basis evaluators which currently use bin-scatter +
Gaussian smoothing for the same plot.

To make the placeholder safe in the meantime:
* every panel overlays the **raw fine histogram** (no smoothing) as
  a faint step line — gold-standard reference, KDE-independent;
* the ``--knn-k`` CLI flag exposes the only knob.  Smaller k →
  sharper kernel (resolves substructure, more shot-noise wiggle);
  larger k → smoother.  If KDE disagrees with engine but raw
  histogram agrees, the disagreement is smoothing-driven; if both
  disagree, cause is elsewhere (genuine miscalibration or
  insufficient particle statistics at high generation).
* engine 1-D plots of ``dN/dq``, ``dN/dχ_eff``, ``dN/dM_chirp``
  apply a σ ≈ 1.5-bin Gaussian smoothing to the bin-scatter output
  (the raw scatter is kept in the npz artifact as ``ligo_1d_raw``).

Resolution sweep
----------------
``--resolutions 64x20,128x40,...`` adds one row per resolution to the
figure.  Each row, panel title, legend entry, and output filename
includes ``(N_m, N_χ)`` so post-hoc comparison across runs is
unambiguous.  Default: one baseline pair.

Memory cost
-----------
The engine LIGO-1D path materialises the full 4-D forward-rate tensor
``(N_m, N_χ, N_m, N_χ)`` once per resolution.  In float64 that is
``8 · (N_m · N_χ)²`` bytes — ``≈ 210 MB`` at ``(128, 40)``,
``≈ 13 GB`` at ``(256, 80)``.  Stay at or below ``(128, 40)`` on a
typical workstation; use a more memory-frugal scatter (``jax.lax.scan``
over ``m_heavy`` rows) if a 256+-point sweep is needed.

Per [[feedback-perturbative-unreliable-implicit-production]] we use
``solver="implicit"`` only; the perturbative path is not exercised.

Outputs
-------
::

    demo/figures_fmobs_<hash>/forward_modeling_observables.png
    demo/artifacts_fmobs_<hash>/metadata.json
    demo/artifacts_fmobs_<hash>/cells_<Nm>x<Nchi>.npz   (one per resolution)

Run
---
::

    python demo/forward_modeling_observables.py
    python demo/forward_modeling_observables.py --resolutions 32x10,64x20
    python demo/forward_modeling_observables.py --epsilon 0.5 --n-mc 100000
"""
from __future__ import annotations

import argparse
import json
import secrets
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (config-after-import is intentional)
import numpy as np  # noqa: E402

from hierarchical_ad_pop_engine import observables as obs  # noqa: E402
from hierarchical_ad_pop_engine.grids import BinaryGrid  # noqa: E402
from hierarchical_ad_pop_engine.response.coagulation import (  # noqa: E402
    build_gaussian_coagulation,
)
from hierarchical_ad_pop_engine.response.implicit import (  # noqa: E402
    solve_implicit_fixed_point,
)
from hierarchical_ad_pop_engine.spin_orientation import (  # noqa: E402
    Aligned, Fisher, Isotropic,
)

from red_team import backup_particle_mc as B1  # noqa: E402


# ---------------------------------------------------------------------------
# --orientation CLI parsing.
# ---------------------------------------------------------------------------
def parse_orientation(spec: str):
    r"""Parse the ``--orientation`` CLI value into an OrientationModel.

    Recognised forms (case-insensitive):

    * ``aligned``                 — :class:`Aligned`, v1 default
    * ``isotropic``               — :class:`Isotropic.default`
    * ``fisher:κ``                — :class:`Fisher.default(kappa=κ)`
      where ``κ`` is parsed as ``float`` (e.g. ``fisher:5.0``).

    ``κ`` may also be passed as ``inf`` to nominally select the
    ``Aligned`` limit — but since Fisher's quadrature loses tail
    accuracy for very large κ, the explicit ``aligned`` keyword is the
    right way to request the bit-for-bit aligned path.
    """
    s = spec.strip().lower()
    if s in ("aligned", "a"):
        return Aligned()
    if s in ("isotropic", "iso", "i"):
        return Isotropic.default()
    if s.startswith("fisher:") or s.startswith("f:"):
        _, _, kappa_s = s.partition(":")
        kappa_s = kappa_s.strip()
        if not kappa_s:
            raise argparse.ArgumentTypeError(
                "--orientation 'fisher:κ' requires a concentration value, "
                "e.g. 'fisher:5.0'"
            )
        try:
            kappa = float(kappa_s)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--orientation 'fisher:{kappa_s}' — could not parse κ "
                f"as float: {exc}"
            ) from exc
        if not (kappa >= 0.0):
            raise argparse.ArgumentTypeError(
                f"--orientation 'fisher:{kappa}' — κ must be ≥ 0 "
                "(κ = 0 ≡ isotropic, κ → ∞ ≡ aligned).  Use 'isotropic' "
                "or 'aligned' for the endpoints."
            )
        return Fisher.default(kappa=kappa)
    raise argparse.ArgumentTypeError(
        f"--orientation {spec!r} not recognised; expected 'aligned', "
        "'isotropic', or 'fisher:κ' (e.g. 'fisher:5.0')"
    )


def orientation_tag(orientation: object) -> str:
    """Short label for the orientation, suitable for filenames / titles."""
    if isinstance(orientation, Aligned):
        return "aligned"
    if isinstance(orientation, Isotropic):
        return "isotropic"
    if isinstance(orientation, Fisher):
        return f"fisher_k{float(orientation.kappa):g}"
    return type(orientation).__name__.lower()


# ---------------------------------------------------------------------------
# Stage 1 truth point — kept in sync with red_team/phase1_overlay.py and
# demo/demo_gwkokab_fit_stage1.py.
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


# ---------------------------------------------------------------------------
# MC 1-D KDE — k-NN balloon estimator (placeholder, location-adaptive).
#
# Replaces an earlier fixed-bandwidth Gaussian KDE.  User direction
# 2026-05-22:
#
#   "MC bandwidth — issue is very small mass/spin 'islands'; Abramson
#    isn't capturing these well, probably want k-NN.  Massive
#    oversmoothing obvious in spin."
#
# The k-NN balloon estimator (Loftsgaarden–Quesenberry 1965) sets the
# bandwidth *per eval-point* to the distance to the k-th nearest
# sample at that point.  In dense regions h(x) shrinks → the kernel
# resolves narrow structure; in sparse tails h(x) grows → shot noise
# is naturally suppressed.  Uniform-kernel form:
#
#       f̂(x) = k / (2 · N · h(x))
#
# where ``N`` is the original (un-reflected) sample size.  Reflection
# at hard boundaries (χ ∈ [0,1], q ∈ (0,1]) extends the sample by
# mirroring; the extended sample size enters the denominator so
# ∫_{support} f̂ dx = 1 by construction.  No hardcoded physical scales
# enter — this complies with
# [[feedback-no-hardcoded-physical-scales]].
#
# Still a placeholder: the *adaptive* k-NN bandwidth is correct in
# principle but uniform-kernel k-NN is noisier than Abramson at fixed
# computational cost.  See PLAN_arbitrary_point_evaluation.md for the
# proper evaluation API.  The raw fine histogram is overlaid in the
# figure (faint step underlay) so any KDE artefact is visible.
# ---------------------------------------------------------------------------
KNN_K_DEFAULT = 100  # k-th nearest neighbour; tune via --knn-k.

# Fine eval-grid size for the MC 1-D KDE — decoupled from the engine
# resolution so the MC plot looks identical regardless of N_m, N_χ.
N_MC_EVAL_M = 400
N_MC_EVAL_CHI = 400
N_MC_EVAL_Q = 400
N_MC_EVAL_MC = 400


def kde_knn_1d(
    samples: np.ndarray,
    eval_pts: np.ndarray,
    k: int,
    *,
    reflect_lo: float | None = None,
    reflect_hi: float | None = None,
    void_h_ratio: float = 0.5,
) -> np.ndarray:
    r"""k-NN balloon density estimator (1-D, uniform kernel).

    ``f̂(x) = k / (2 · N · h(x))`` where ``h(x)`` is the distance from
    ``x`` to its k-th nearest sample in the (optionally reflected)
    sample set, and ``N`` is the **original** (un-reflected) sample
    size.  Reflection shortens ``h(x)`` near the boundary without
    adding probability mass — the ghost samples are mirror images of
    real samples and represent the same density on the original
    support, so the denominator stays at ``N``, not ``N_ext``.

    A previous version divided by ``ext.size`` and produced densities
    2× too low on log-mass axes (one-side reflection) and 3× too low
    on χ axes (two-side reflection).  Sanity check: uniform on [0, 1]
    with ``N = 400k``, ``k = 100`` → bulk ``h ≈ k/(2N) = 1.25e-4`` →
    ``f̂ = 100 / (2 · 400000 · 1.25e-4) = 1.0`` ✓.

    Parameters
    ----------
    samples
        1-D array of MC samples.  May be unsorted.
    eval_pts
        1-D array of query points.
    k
        Number of nearest neighbours that define the local bandwidth.
        Smaller k → sharper kernel (resolves fine structure, more
        shot-noise wiggle); larger k → smoother (bias-variance trade-
        off).  ``k = 100`` is a reasonable default for ``N ≈ 4 × 10⁵``.
    reflect_lo, reflect_hi
        If set, the sample is mirrored across the corresponding
        boundary before the k-NN lookup.  Use for hard boundaries
        (χ ∈ [0, 1], q ∈ (0, 1]).  Leave ``None`` for axes that decay
        naturally (log m, log M_chirp).
    void_h_ratio
        Void gate threshold on ``h_1(x) / h_k(x)``.  In a dense
        neighbourhood the k samples surround ``x`` and ``h_1 ≪ h_k``;
        in a void the k-NN cluster sits as a tight ball *far from
        x* and ``h_1 ≈ h_k``.  Eval points with ``h_1 / h_k`` above
        this threshold are zeroed — kills the artificial density
        spread into empty regions / past hard edges that pure k-NN
        balloon produces (e.g. the "extends toward zero mass" m₁
        artefact reported 2026-05-22).  Default ``0.5``; set to 1.0
        to disable.  Scale-free — no hardcoded physical units.

    Returns
    -------
    ndarray, same shape as ``eval_pts``
        Density estimate normalised to integrate to 1 over the
        original support.  Zero outside the data cloud per the
        void gate.
    """
    from scipy.spatial import cKDTree

    samples = np.asarray(samples, dtype=np.float64)
    n_orig = samples.size
    if n_orig == 0:
        return np.zeros_like(eval_pts, dtype=np.float64)
    parts = [samples]
    if reflect_lo is not None:
        parts.append(2.0 * reflect_lo - samples)
    if reflect_hi is not None:
        parts.append(2.0 * reflect_hi - samples)
    ext = np.concatenate(parts)
    k_eff = int(min(k, ext.size))
    tree = cKDTree(ext.reshape(-1, 1))
    # ``query`` returns distances to the k_eff nearest samples, sorted.
    dists, _ = tree.query(np.asarray(eval_pts, dtype=np.float64).reshape(-1, 1), k=k_eff)
    if dists.ndim == 1:
        # k_eff == 1 (degenerate sample); h_1 == h_k by definition,
        # void gate would zero everything → bypass it in this case.
        h_k = dists
        h_1 = dists
        void_active = False
    else:
        h_k = dists[:, -1]
        h_1 = dists[:, 0]
        void_active = True
    # Guard against h == 0 (eval pt exactly on a sample).
    h_k = np.where(h_k > 0, h_k, np.finfo(np.float64).tiny)
    density = k_eff / (2.0 * n_orig * h_k)
    # Void gate (see docstring).  When ``h_1 / h_k`` is close to 1,
    # the k samples form a tight ball sitting away from ``x`` — the
    # eval point is outside the data cloud and any density estimate
    # there is fictional.
    if void_active and void_h_ratio < 1.0:
        ratio = h_1 / h_k
        density = np.where(ratio > void_h_ratio, 0.0, density)
    return density


def kde_knn_log_axis(
    samples: np.ndarray,
    eval_lin: np.ndarray,
    k: int,
    *,
    reflect_log_lo: float | None = None,
    void_h_ratio: float = 0.5,
) -> np.ndarray:
    r"""k-NN KDE in log-space, returned as a density in the *linear* axis.

    KDE is performed on ``log(samples)``; the result at
    ``log(eval_lin)`` is divided by ``eval_lin`` to apply the
    ``d(log x) = dx / x`` Jacobian.  Used for mass and M_chirp axes.
    ``void_h_ratio`` passes through to :func:`kde_knn_1d`.
    """
    if samples.size == 0:
        return np.zeros_like(eval_lin, dtype=np.float64)
    log_s = np.log(np.maximum(samples, 1e-30))
    log_e = np.log(eval_lin)
    dens_log = kde_knn_1d(
        log_s, log_e, k,
        reflect_lo=reflect_log_lo,
        void_h_ratio=void_h_ratio,
    )
    return dens_log / eval_lin


def _gaussian_smooth_1d(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    r"""1-D Gaussian smoothing with reflect-edge boundary handling.

    Used to soften the engine bin-scatter discreteness in 1-D plots of
    ``dN/dq``, ``dN/dχ_eff``, ``dN/dM_chirp``.  These engine paths
    bin-scatter a ``(N_m, N_χ, N_m, N_χ)`` rate tensor onto fine bin
    edges, which leaves visible step artefacts when the bin count
    exceeds the natural derived-axis cell resolution (e.g. at fine
    q-binning the q-tail has only a few engine cells per bin).
    Smoothing with ``σ_bins ≈ 1.5`` removes this artefact without
    smearing real cascade features (whose widths are wider than a
    few bins by construction at the bin counts in this demo).

    Smoothing is NOT applied to engine ``dN/dm₁`` (lives on the
    engine grid, not on bin edges — already smooth at engine
    resolution) or to MC outputs (whose KDE / raw histogram are
    independent of engine resolution).
    """
    if sigma_bins <= 0:
        return y
    n_kernel = max(3, int(np.ceil(6.0 * sigma_bins)) | 1)  # odd
    x = np.arange(n_kernel) - n_kernel // 2
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    n_pad = n_kernel // 2
    # Reflect-edge padding so the boundary doesn't drag the value down.
    y_padded = np.concatenate([y[n_pad:0:-1], y, y[-2:-(n_pad + 2):-1]])
    return np.convolve(y_padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Engine path — build response, run implicit solver, compute observables.
# ---------------------------------------------------------------------------
def build_response(
    theta: dict,
    n_m: int,
    n_chi: int,
    epsilon: float,
    *,
    orientation: object | None = None,
    retention_kind: str = "heaviside",
    w_over_v_esc: float = 1.0,
):
    r"""Build a :class:`CoagulationResponse` at ``θ*`` with the given grid.

    Forces ``solver="implicit"`` so any caller that uses
    ``response._log_total_density`` would route through the production
    fixed-point solver.  In this demo we call
    :func:`solve_implicit_fixed_point` directly (gives explicit
    hyperparameter control), so the ``solver`` field is informational —
    but keeping it pinned makes the build call self-describing.

    ``orientation`` defaults to ``None`` (engine falls back to Aligned()
    bit-for-bit).  Pass any concrete OrientationModel to engage the
    misalignment-mode path inside ``_step_one_generation``.
    """
    grid = BinaryGrid.default(
        m_min=float(theta["m_min"]),
        m_max=float(theta["m_max"]),
        n_m=int(n_m),
        n_chi=int(n_chi),
    )
    return build_gaussian_coagulation(
        mu=theta["mu_natal"],
        sigma=theta["sigma_natal"],
        m_min=theta["m_min"],
        m_max=theta["m_max"],
        natal_spin_loc=theta["natal_spin_loc"],
        natal_spin_scale=theta["natal_spin_scale"],
        kernel_a=theta["kernel_a"],
        kernel_b=theta["kernel_b"],
        log_R_env=0.0,
        k_max=theta["k_max"],
        grid=grid,
        solver="implicit",
        epsilon=epsilon,
        orientation=orientation,
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )


def engine_total_rate_at_v_esc(
    response, v_esc, *, epsilon: float,
):
    r"""Closed-form ``v_esc → total_rate`` map at fixed ``response``.

    Routes through the production implicit solver
    (``solve_implicit_fixed_point``) — same path as
    :func:`engine_solve` — so ``jax.grad`` cashes in the IFT custom-VJP
    plumbing already in ``response/implicit.py``.  Used for the
    Session C step-10 demo-level confirmation that
    ``∂total_rate/∂v_esc`` is finite and non-zero under any non-Aligned
    orientation + sigmoid retention (Session B sandbox verified the
    same property at the lower-level ``_solve`` surface; this is the
    end-to-end demo-level cash-in).
    """
    result = solve_implicit_fixed_point(
        response,
        v_esc,
        epsilon=epsilon,
        max_iter=200,
        tol=1e-7,
        damping=0.4,
    )
    return obs.total_rate(
        result.N, response.grid, response.kernel_params, response.log_R_env,
    )


def engine_solve(
    theta: dict,
    n_m: int,
    n_chi: int,
    epsilon: float,
    *,
    orientation: object | None = None,
    retention_kind: str = "heaviside",
    w_over_v_esc: float = 1.0,
):
    r"""Run the production implicit Picard solver.  Returns the response
    object, the converged density ``N(m, χ)``, and the solver diagnostic
    dict (``n_iter``, ``f_natal_depleted``, ...).
    """
    response = build_response(
        theta, n_m, n_chi, epsilon,
        orientation=orientation,
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )
    t0 = time.time()
    result = solve_implicit_fixed_point(
        response,
        jnp.asarray(theta["v_esc"]),
        epsilon=epsilon,
        max_iter=200,
        tol=1e-7,
        damping=0.4,
    )
    t_solve = time.time() - t0
    diag = dict(
        n_iter=int(result.n_iter),
        final_diff=float(result.final_diff),
        f_natal_depleted=float(result.f_natal_depleted),
        natal_integral=float(result.natal_integral),
        loss_integral=float(result.loss_integral),
        wall_seconds=t_solve,
    )
    return response, result.N, diag


def engine_4d_ordered(response, N_mchi):
    r"""Return the engine's 4-D forward-rate count-per-cell tensor under the
    ordered ``m₁ ≥ m₂`` convention.

    Returns a ``(N_m, N_χ, N_m, N_χ)`` numpy array whose total sum equals
    :func:`obs.total_rate`.  ``i`` indexes ``m_heavy`` (i.e. ``m_1``),
    ``k`` indexes ``m_light`` (i.e. ``m_2``); ``i > k`` cells are doubled
    to absorb the ``(i, k) ↔ (k, i)`` pair-symmetry, ``i == k`` cells
    are kept, ``i < k`` cells are zeroed.

    All four "LIGO 1D" engine histograms below are derived from this
    tensor by scattering onto a derived axis with cell-volume weights.
    This unifies the integration measure across observables — no path-
    by-path quadrature mismatches.
    """
    grid = response.grid
    kp = response.kernel_params
    log_R = response.log_R_env
    R4 = np.asarray(obs.rate_density_4d(N_mchi, grid, kp, log_R))  # (N_m, N_χ, N_m, N_χ)

    w_m, w_chi = obs.trapezoid_weights(grid)
    w_m_np = np.asarray(w_m)
    w_chi_np = np.asarray(w_chi)
    cell_vol = (
        w_m_np[:, None, None, None]
        * w_chi_np[None, :, None, None]
        * w_m_np[None, None, :, None]
        * w_chi_np[None, None, None, :]
    )

    m_np = np.asarray(grid.m)
    m_i = m_np[:, None, None, None]
    m_k = m_np[None, None, :, None]
    # ordered mask: 2× upper triangle (m_i > m_k), 1× diagonal, 0× lower.
    mask = np.where(m_i > m_k, 2.0, np.where(m_i == m_k, 1.0, 0.0))

    rate_count = R4 * cell_vol * mask  # sum = total_rate
    return rate_count


def engine_observables(
    theta: dict, n_m: int, n_chi: int, epsilon: float,
    *,
    smooth_sigma_bins: float = 0.7,
    orientation: object | None = None,
    retention_kind: str = "heaviside",
    w_over_v_esc: float = 1.0,
):
    r"""Compute all engine observables on a single grid resolution.

    Returns a dict with the structure::

        {
          "n_m", "n_chi", "diag",
          "total_rate",
          "source": {
              "m":  (m_grid, dN_dm),
              "chi": (chi_grid, dN_dchi),
          },
          "ligo_1d": {
              "m1":      (m_centres, dN_dm1),
              "q":       (q_centres, dN_dq),
              "chi_eff": (chi_eff_centres, dN_dchi_eff),
              "M_chirp": (M_chirp_centres, dN_dM_chirp),
              "bin_edges": {"m1": ..., "q": ..., "chi_eff": ..., "M_chirp": ...},
          },
          "ligo_2d": {
              "m1_q":      (m_axis, q_centres, dN),     # ordered
              "m1_chi_eff": (m_axis, chi_eff_centres, dN),
          },
        }

    All histograms returned are densities (units: per unit on each axis),
    so they integrate to ``total_rate`` over the relevant volume.
    """
    response, N_mchi, diag = engine_solve(
        theta, n_m, n_chi, epsilon,
        orientation=orientation,
        retention_kind=retention_kind,
        w_over_v_esc=w_over_v_esc,
    )
    grid = response.grid
    kp = response.kernel_params
    log_R = response.log_R_env

    total = float(obs.total_rate(N_mchi, grid, kp, log_R))

    # --- source-frame 1D marginals ---------------------------------------
    m_g, dN_dm = obs.marginal_1d(N_mchi, grid, kp, log_R, axis=obs.AXIS_M1)
    chi_g, dN_dchi = obs.marginal_1d(N_mchi, grid, kp, log_R, axis=obs.AXIS_CHI1)
    source = dict(
        m=(np.asarray(m_g), np.asarray(dN_dm)),
        chi=(np.asarray(chi_g), np.asarray(dN_dchi)),
    )

    # --- LIGO 2D joints (densities, ordered convention) ------------------
    # Bin counts here drive both the engine 2-D heatmap resolution and the
    # engine 1-D derived-axis histograms (q, χ_eff).  Set well above the
    # previous 20-bin baseline — the engine bin-scatter integrand is
    # smooth, and ~50 bins is plenty to resolve cascade structure
    # without producing visible step artefacts in the plots.
    m_min = float(theta["m_min"])
    m_max = float(theta["m_max"])
    q_bin_edges = jnp.linspace(0.05, 1.0, 51)         # 50 bins on (0.05, 1.0]
    chi_eff_bin_edges = jnp.linspace(0.0, 1.0, 61)    # 60 bins on [0, 1]

    m_ax, q_ax, dC_m1_q = obs.dN_dm1_dq(
        N_mchi, grid, kp, log_R,
        q_bin_edges=q_bin_edges,
        ordered=True,
    )
    _, chi_eff_ax, dC_m1_chieff = obs.dN_dm1_dchi_eff(
        N_mchi, grid, kp, log_R,
        chi_eff_bin_edges=chi_eff_bin_edges,
        ordered=True,
    )

    ligo_2d = dict(
        m1_q=(np.asarray(m_ax), np.asarray(q_ax), np.asarray(dC_m1_q)),
        m1_chi_eff=(
            np.asarray(m_ax), np.asarray(chi_eff_ax), np.asarray(dC_m1_chieff),
        ),
    )

    # --- LIGO 1D (engine): unified scatter from 4-D ordered tensor -------
    rate_count = engine_4d_ordered(response, N_mchi)  # sum = total
    m_np = np.asarray(grid.m)
    chi_np = np.asarray(grid.chi)
    w_m_np, w_chi_np = obs.trapezoid_weights(grid)
    w_m_np = np.asarray(w_m_np)
    w_chi_np = np.asarray(w_chi_np)
    m1g = m_np[:, None, None, None]
    m2g = m_np[None, None, :, None]
    chi1g = chi_np[None, :, None, None]
    chi2g = chi_np[None, None, None, :]

    # M₁ axis: histogram bin edges built around the engine grid centres
    # (geomspace edges).  These are used only for *MC histogramming*; the
    # density divisor on the m-axis is the engine's symmetric-trapezoid
    # weight ``w_m = m · d(log m)`` for both engine and MC, so both paths
    # share the same integration measure as ``obs.marginal_1d`` and
    # ``obs.marginal_2d``.  This avoids a boundary-cell mismatch that
    # would otherwise appear because the trapezoid weights are half-
    # weight at the endpoints while the bin widths are full-half-bin.
    log_m_np = np.log(m_np)
    log_edges = np.concatenate([
        [log_m_np[0] - 0.5 * (log_m_np[1] - log_m_np[0])],
        0.5 * (log_m_np[:-1] + log_m_np[1:]),
        [log_m_np[-1] + 0.5 * (log_m_np[-1] - log_m_np[-2])],
    ])
    m_bin_edges_eng = np.exp(log_edges)
    # rate_count is per-cell count under the ordered fold, with the
    # cell measure ``w_m[i] w_chi[j] w_m[k] w_chi[l]`` already absorbed.
    # Summing over (χ₁, m₂, χ₂) and dividing by ``w_m[i]`` gives the
    # ordered marginal density at ``m_i`` — same convention as
    # ``obs.marginal_1d`` on the source-frame plot.
    counts_m1 = rate_count.sum(axis=(1, 2, 3))  # (N_m,)
    dN_dm1 = counts_m1 / w_m_np

    # q: scatter all (i, j, k, l) cells onto a q histogram.
    q_4d = np.minimum(m1g, m2g) / np.maximum(m1g, m2g)
    q_bin_edges_np = np.asarray(q_bin_edges)
    counts_q, _ = np.histogram(
        np.broadcast_to(q_4d, rate_count.shape).reshape(-1),
        bins=q_bin_edges_np,
        weights=rate_count.reshape(-1),
    )
    q_widths = np.diff(q_bin_edges_np)
    dN_dq = counts_q / q_widths
    q_centres = 0.5 * (q_bin_edges_np[:-1] + q_bin_edges_np[1:])

    # χ_eff: (m₁·χ₁ + m₂·χ₂) / (m₁ + m₂).
    chi_eff_4d = (m1g * chi1g + m2g * chi2g) / (m1g + m2g)
    chi_eff_bin_edges_np = np.asarray(chi_eff_bin_edges)
    counts_chi_eff, _ = np.histogram(
        np.broadcast_to(chi_eff_4d, rate_count.shape).reshape(-1),
        bins=chi_eff_bin_edges_np,
        weights=rate_count.reshape(-1),
    )
    chi_eff_widths = np.diff(chi_eff_bin_edges_np)
    dN_dchi_eff = counts_chi_eff / chi_eff_widths
    chi_eff_centres = 0.5 * (chi_eff_bin_edges_np[:-1] + chi_eff_bin_edges_np[1:])

    # M_chirp: (m₁·m₂)^(3/5) / (m₁+m₂)^(1/5).  Range bracketing: at θ*,
    # smallest pairing is (m_min, m_min) ≈ 4 → M_chirp ≈ 3.5; heaviest is
    # (m_max, m_max) → ≈ 70.  Use a 24-bin geomspace.
    M_chirp_4d = (m1g * m2g) ** 0.6 / (m1g + m2g) ** 0.2
    M_chirp_lo = float(0.95 * np.min(M_chirp_4d))
    M_chirp_hi = float(1.05 * np.max(M_chirp_4d))
    M_chirp_bin_edges = np.geomspace(M_chirp_lo, M_chirp_hi, 61)  # 60 log-spaced bins
    counts_Mc, _ = np.histogram(
        np.broadcast_to(M_chirp_4d, rate_count.shape).reshape(-1),
        bins=M_chirp_bin_edges,
        weights=rate_count.reshape(-1),
    )
    M_chirp_widths = np.diff(M_chirp_bin_edges)
    dN_dM_chirp = counts_Mc / M_chirp_widths
    M_chirp_centres = np.sqrt(M_chirp_bin_edges[:-1] * M_chirp_bin_edges[1:])

    # Post-smooth the derived-axis 1-D arrays for display.  The engine
    # paths use a hard bin-scatter from a finite-cell rate tensor; at
    # fine bin counts the few-cells-per-bin tails (e.g. small q at θ*)
    # show ±1-bin step artefacts that are NOT physical features.
    # ``smooth_sigma_bins`` defaults to 0.7 (just enough to soften the
    # ±1-bin discreteness — earlier 1.5 oversmoothed narrow LIGO
    # features, especially χ_eff at high gen).  Pass 0 to disable
    # smoothing entirely.  ``dN_dm1`` is on the engine grid (not
    # bin-scattered) and is left unsmoothed.
    dN_dq_smooth = _gaussian_smooth_1d(dN_dq, smooth_sigma_bins)
    dN_dchi_eff_smooth = _gaussian_smooth_1d(dN_dchi_eff, smooth_sigma_bins)
    dN_dM_chirp_smooth = _gaussian_smooth_1d(dN_dM_chirp, smooth_sigma_bins)

    ligo_1d = dict(
        m1=(m_np, dN_dm1),
        q=(q_centres, dN_dq_smooth),
        chi_eff=(chi_eff_centres, dN_dchi_eff_smooth),
        M_chirp=(M_chirp_centres, dN_dM_chirp_smooth),
        # Keep the raw (un-smoothed) arrays in the artifact for
        # post-hoc inspection — caller can re-plot without smoothing
        # if they want to verify the smoothing isn't hiding something.
        ligo_1d_raw=dict(
            q=dN_dq, chi_eff=dN_dchi_eff, M_chirp=dN_dM_chirp,
        ),
        bin_edges=dict(
            m1=m_bin_edges_eng,
            q=q_bin_edges_np,
            chi_eff=chi_eff_bin_edges_np,
            M_chirp=M_chirp_bin_edges,
        ),
        smooth_sigma_bins=smooth_sigma_bins,
    )

    return dict(
        n_m=n_m,
        n_chi=n_chi,
        diag=diag,
        total_rate=total,
        source=source,
        ligo_1d=ligo_1d,
        ligo_2d=ligo_2d,
        # Trapezoid weights — the unifying integration measure for both
        # engine and MC mass / spin histograms (see mc_observables).
        weights=dict(w_m=w_m_np, w_chi=w_chi_np),
    )


# ---------------------------------------------------------------------------
# MC path — particle cascade, histograms on matching axes.
# ---------------------------------------------------------------------------
def run_mc(
    theta: dict, n_seed: int, depletion_fraction: float, seed: int,
    *,
    orientation: object | None = None,
    orientation_key_seed: int = 0,
) -> tuple[dict, np.ndarray, np.ndarray]:
    r"""Run the particle-MC cascade.  Returns ``(stats, mergers, v_kicks)``.

    The cascade stops when the active pool has shrunk by
    ``depletion_fraction``; ``depletion_fraction`` is taken from the
    implicit solver's ``f_natal_depleted`` per
    [[feedback-mc-depletion-tracks-engine]].

    ``orientation`` defaults to ``None`` → Aligned bit-for-bit
    (existing behaviour).  Any other concrete OrientationModel triggers
    the per-event vector-physics path in
    :func:`red_team.backup_particle_mc.particle_mc`.
    """
    cfg = B1.MCConfig(
        n_seed=int(n_seed),
        depletion_fraction=float(depletion_fraction),
        mu_natal=theta["mu_natal"],
        sigma_natal=theta["sigma_natal"],
        m_min=theta["m_min"],
        m_max=theta["m_max"],
        natal_spin_loc=theta["natal_spin_loc"],
        natal_spin_scale=theta["natal_spin_scale"],
        kernel_a=theta["kernel_a"],
        kernel_b=theta["kernel_b"],
        v_esc=theta["v_esc"],
        orientation=orientation,
        orientation_key_seed=int(orientation_key_seed),
    )
    rng = np.random.default_rng(seed)
    t0 = time.time()
    result = B1.particle_mc(cfg, rng, track_generations=False)
    t_mc = time.time() - t0
    stats = dict(result["stats"])
    stats["wall_seconds"] = t_mc
    return stats, result["mergers"], result["v_kicks"]


def _hist_density(
    samples: np.ndarray, edges: np.ndarray, rate_scale: float,
) -> np.ndarray:
    r"""Histogram into ``edges``, convert to density via ``count·scale / Δedge``.

    Returned shape: ``(len(edges) - 1,)``.  ``∫ density d(axis) =
    N_samples · rate_scale``.
    """
    counts, _ = np.histogram(samples, bins=edges)
    return counts.astype(np.float64) * rate_scale / np.diff(edges)


def mc_observables(
    mergers: np.ndarray,
    engine_obs: dict,
    rate_scale_ordered: float,
    *,
    knn_k: int = KNN_K_DEFAULT,
    void_h_ratio: float = 0.5,
    theta: dict | None = None,
) -> dict:
    r"""Compute MC observables — k-NN balloon KDE on a fine eval grid
    AND a raw fine histogram — for every 1-D axis.

    Two scales:
      - ``rate_scale_ordered`` = engine_total_rate / N_events.  Used for
        any "one entry per merger event" histogram (LIGO 1D, 2D).
      - ``rate_scale_unordered`` = rate_scale_ordered / 2.  Used for the
        source-frame ``dN/dm`` and ``dN/dχ`` plots, where each merger
        event contributes both its progenitor masses (or spins) to the
        histogram, so the integrated count is ``2 N_events`` and the
        ½ factor brings it back to the engine's unordered total.

    Bandwidth selection is **k-NN**, NOT a hardcoded scale: at each
    eval point the local bandwidth is the distance to the ``knn_k``-th
    nearest sample.  Dense regions get narrow kernels (resolve
    substructure); sparse tails get wide ones (suppress shot-noise).
    See [[feedback-no-hardcoded-physical-scales]].

    Returns
    -------
    dict
        Per-observable: ``out[group][axis]`` is itself a dict::

            {
              "x":  ndarray,   # x positions for KDE evaluation
              "kde": ndarray,  # KDE density at those positions
              "x_raw": ndarray,    # bin centres of the raw fine histogram
              "raw": ndarray,      # density value per bin (no smoothing)
            }

        The raw fine histogram is plotted as a faint underlay so the user
        can see the un-smoothed MC realisation alongside the KDE.  This
        is the safety net against the k-NN choice smearing real cascade
        structure at high mass / sparse-sample tails.
    """
    assert theta is not None, "theta required for evaluation-grid construction"
    m_min = float(theta["m_min"])
    m_max = float(theta["m_max"])

    # _split_mergers already gives heavier-first (m1, m2, χ1, χ2).
    m1, m2, chi1, chi2 = (
        mergers[:, 0].astype(np.float64),
        mergers[:, 1].astype(np.float64),
        mergers[:, 2].astype(np.float64),
        mergers[:, 3].astype(np.float64),
    )

    rate_scale_unordered = 0.5 * rate_scale_ordered
    n_events = int(mergers.shape[0])

    out = dict(
        n_events=n_events,
        knn_k=int(knn_k),
        void_h_ratio=float(void_h_ratio),
    )

    # Fine eval grids — decoupled from engine resolution.  Wide mass
    # span ([m_min, M_rem(m_max, m_max) ≈ 2·m_max]) so the cascade tail
    # is fully visible.
    m_eval = np.geomspace(m_min, m_max * 2.5, N_MC_EVAL_M)
    chi_eval = np.linspace(1e-3, 1.0 - 1e-3, N_MC_EVAL_CHI)
    q_eval = np.linspace(1e-3, 1.0 - 1e-3, N_MC_EVAL_Q)
    chi_eff_eval = np.linspace(1e-3, 1.0 - 1e-3, N_MC_EVAL_CHI)
    # M_chirp span: from (m_min, m_min) ≈ m_min · 0.871 to (M_rem(m_max, m_max),
    # M_rem(m_max, m_max)) at maxgen.  Conservative bracket.
    Mc_lo = m_min * 0.85
    Mc_hi = m_max * 2.5
    Mc_eval = np.geomspace(Mc_lo, Mc_hi, N_MC_EVAL_MC)

    # Raw fine histogram edges (independent of engine, independent of
    # the KDE eval grid; used for the underlay).  Bin counts chosen so
    # that shot noise per bin in the 1G/2G bulk is ~few %, while still
    # being fine enough to NOT smear cascade peaks.
    raw_m_edges = np.geomspace(m_min, m_max * 2.5, 121)        # 120 log bins
    raw_chi_edges = np.linspace(0.0, 1.0, 121)
    raw_q_edges = np.linspace(0.0, 1.0, 121)
    raw_chi_eff_edges = np.linspace(0.0, 1.0, 121)
    raw_Mc_edges = np.geomspace(Mc_lo, Mc_hi, 121)
    raw_m_centres = np.sqrt(raw_m_edges[:-1] * raw_m_edges[1:])
    raw_chi_centres = 0.5 * (raw_chi_edges[:-1] + raw_chi_edges[1:])
    raw_q_centres = 0.5 * (raw_q_edges[:-1] + raw_q_edges[1:])
    raw_chi_eff_centres = 0.5 * (raw_chi_eff_edges[:-1] + raw_chi_eff_edges[1:])
    raw_Mc_centres = np.sqrt(raw_Mc_edges[:-1] * raw_Mc_edges[1:])

    # --- source-frame 1D (stack m1+m2, χ1+χ2) ----------------------------
    m_stack = np.concatenate([m1, m2])  # 2·N_events entries
    chi_stack = np.concatenate([chi1, chi2])

    # k-NN KDE: ∫ KDE = 1; multiply by (2·N_events)·rate_scale_unordered = total_rate.
    src_m_kde = kde_knn_log_axis(
        m_stack, m_eval, knn_k,
        reflect_log_lo=np.log(m_min), void_h_ratio=void_h_ratio,
    ) * (2.0 * n_events) * rate_scale_unordered
    src_chi_kde = kde_knn_1d(
        chi_stack, chi_eval, knn_k,
        reflect_lo=0.0, reflect_hi=1.0, void_h_ratio=void_h_ratio,
    ) * (2.0 * n_events) * rate_scale_unordered

    out["source"] = dict(
        m=dict(
            x=m_eval, kde=src_m_kde,
            x_raw=raw_m_centres,
            raw=_hist_density(m_stack, raw_m_edges, rate_scale_unordered),
        ),
        chi=dict(
            x=chi_eval, kde=src_chi_kde,
            x_raw=raw_chi_centres,
            raw=_hist_density(chi_stack, raw_chi_edges, rate_scale_unordered),
        ),
    )

    # --- LIGO 1D ---------------------------------------------------------
    # χ_eff and M_chirp on heavier-first columns.
    chi_eff = (m1 * chi1 + m2 * chi2) / (m1 + m2)
    M_chirp = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    q = m2 / m1  # m1 ≥ m2 already, q ∈ (0, 1]

    # KDE: each event contributes one entry → ∫ KDE = 1 →
    #   density · N_events · rate_scale_ordered = total_rate
    m1_kde = kde_knn_log_axis(
        m1, m_eval, knn_k,
        reflect_log_lo=np.log(m_min), void_h_ratio=void_h_ratio,
    ) * n_events * rate_scale_ordered
    q_kde = kde_knn_1d(
        q, q_eval, knn_k,
        reflect_lo=0.0, reflect_hi=1.0, void_h_ratio=void_h_ratio,
    ) * n_events * rate_scale_ordered
    chi_eff_kde = kde_knn_1d(
        chi_eff, chi_eff_eval, knn_k,
        reflect_lo=0.0, reflect_hi=1.0, void_h_ratio=void_h_ratio,
    ) * n_events * rate_scale_ordered
    Mc_kde = kde_knn_log_axis(
        M_chirp, Mc_eval, knn_k,
        reflect_log_lo=np.log(Mc_lo), void_h_ratio=void_h_ratio,
    ) * n_events * rate_scale_ordered

    out["ligo_1d"] = dict(
        m1=dict(
            x=m_eval, kde=m1_kde,
            x_raw=raw_m_centres,
            raw=_hist_density(m1, raw_m_edges, rate_scale_ordered),
        ),
        q=dict(
            x=q_eval, kde=q_kde,
            x_raw=raw_q_centres,
            raw=_hist_density(q, raw_q_edges, rate_scale_ordered),
        ),
        chi_eff=dict(
            x=chi_eff_eval, kde=chi_eff_kde,
            x_raw=raw_chi_eff_centres,
            raw=_hist_density(chi_eff, raw_chi_eff_edges, rate_scale_ordered),
        ),
        M_chirp=dict(
            x=Mc_eval, kde=Mc_kde,
            x_raw=raw_Mc_centres,
            raw=_hist_density(M_chirp, raw_Mc_edges, rate_scale_ordered),
        ),
    )

    # Engine bin edges still drive the LIGO 2-D histograms (engine 2-D
    # density lives on the engine grid × derived-axis bin centres, so
    # we keep MC 2-D on the same edges for direct comparison).
    eb = engine_obs["ligo_1d"]["bin_edges"]
    w_m_np = engine_obs["weights"]["w_m"]

    # --- LIGO 2D joints --------------------------------------------------
    m_ax = engine_obs["ligo_2d"]["m1_q"][0]
    q_centres_2d = engine_obs["ligo_2d"]["m1_q"][1]
    chi_eff_centres_2d = engine_obs["ligo_2d"]["m1_chi_eff"][1]
    # 2D engine density was divided by ``w_m[:, None] * bin_widths[None, :]``
    # inside ``obs.dN_dm1_dq`` / ``obs.dN_dm1_dchi_eff``; mirror that here
    # so the MC heatmap shares the engine's integration measure.
    counts_2d_q, _, _ = np.histogram2d(
        m1, q, bins=[eb["m1"], eb["q"]],
    )
    counts_2d_ce, _, _ = np.histogram2d(
        m1, chi_eff, bins=[eb["m1"], eb["chi_eff"]],
    )
    out["ligo_2d"] = dict(
        m1_q=(
            m_ax, q_centres_2d,
            counts_2d_q * rate_scale_ordered
            / (w_m_np[:, None] * np.diff(eb["q"])[None, :]),
        ),
        m1_chi_eff=(
            m_ax, chi_eff_centres_2d,
            counts_2d_ce * rate_scale_ordered
            / (w_m_np[:, None] * np.diff(eb["chi_eff"])[None, :]),
        ),
    )

    return out


# ---------------------------------------------------------------------------
# Plotting — one row per resolution, columns = observables.
# ---------------------------------------------------------------------------
# Each row carries (Nm, Nχ) explicitly in the panel titles so the figure
# is unambiguous post-hoc.
N_COLS = 8  # source-m, source-chi, m1, q, chi_eff, M_chirp, m1-q, m1-chi_eff
PANEL_W = 3.3
PANEL_H = 2.8


def _plot_1d(
    ax, x_eng, y_eng, mc_entry, *,
    log_x=False, log_y=True, label_x="", title="",
):
    r"""Plot engine (solid) + MC KDE (dashed) + MC raw fine histogram
    (faint step underlay) on one axes.

    ``mc_entry`` is the dict-shaped output from ``mc_observables``::

        {"x", "kde", "x_raw", "raw"}

    The raw fine histogram is the un-smoothed MC realisation — its
    presence on the figure is the safety net against the (placeholder,
    fixed-bandwidth) KDE over-smoothing real cascade structure.
    """
    y_kde = mc_entry["kde"]
    y_raw = mc_entry["raw"]
    eng_max = max(float(np.nanmax(y_eng)), 1e-300)
    kde_max = max(float(np.nanmax(y_kde)), 1e-300)
    raw_max = max(float(np.nanmax(y_raw)), 1e-300)
    floor = max(eng_max, kde_max, raw_max) * 1e-4
    plot_fn = ax.loglog if (log_x and log_y) else (ax.semilogy if log_y else ax.plot)
    # Faint raw histogram underneath — step style so it's clearly a histogram.
    plot_fn(
        mc_entry["x_raw"], np.maximum(y_raw, floor),
        color="C2", lw=0.6, ls="-", alpha=0.35,
        drawstyle="steps-mid",
        label="MC raw hist",
    )
    plot_fn(x_eng, np.maximum(y_eng, floor),
            color="C0", lw=2.0, label="engine")
    plot_fn(mc_entry["x"], np.maximum(y_kde, floor),
            color="C2", lw=1.4, ls="--", label="MC KDE")
    ax.set_xlabel(label_x)
    ax.set_title(title, fontsize="x-small")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.tick_params(labelsize="x-small")
    ax.set_ylim(bottom=floor)


def _plot_2d(ax, x, y, dens_eng, dens_mc, *, log_x=False, label_x="", label_y="", title=""):
    """Side-by-side log-density: engine pcolormesh + MC contour overlay."""
    import matplotlib.colors as mcolors

    finite = dens_eng[np.isfinite(dens_eng) & (dens_eng > 0)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "no engine support", ha="center", transform=ax.transAxes)
        return
    vmax = float(np.max(finite))
    vmin = vmax * 1e-3
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    # pcolormesh: x rows = m₁ grid (along axis 0), y cols = derived axis.
    pcm = ax.pcolormesh(
        x, y, dens_eng.T,
        norm=norm, cmap="viridis", shading="auto", rasterized=True,
    )
    # Overlay MC as contours at the engine's iso-density levels.
    mc_finite = dens_mc[np.isfinite(dens_mc) & (dens_mc > 0)]
    if mc_finite.size > 0:
        levels = np.geomspace(vmin * 5, vmax * 0.5, 4)
        try:
            ax.contour(
                x, y, dens_mc.T,
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


def build_figure(rows: list[dict], theta: dict, fig_path: Path):
    r"""Build the multi-row figure.  Each ``row`` is a dict with keys
    ``"n_m"``, ``"n_chi"``, ``"engine"``, ``"mc"``.
    """
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(PANEL_W * N_COLS, PANEL_H * n_rows + 0.4),
        squeeze=False,
    )

    for r_idx, row in enumerate(rows):
        eng = row["engine"]
        mc = row["mc"]
        nm = row["n_m"]
        nc = row["n_chi"]
        resn = f"({nm}×{nc})"

        # Col 0 — source-frame dN/dm.
        x_eng, y_eng = eng["source"]["m"]
        _plot_1d(
            axes[r_idx, 0], x_eng, y_eng, mc["source"]["m"],
            log_x=True, log_y=True,
            label_x=r"$m\;[M_\odot]$",
            title=f"source dN/dm  {resn}",
        )
        if r_idx == 0:
            axes[r_idx, 0].set_ylabel(r"$dN/dm$")
            axes[r_idx, 0].legend(loc="upper right", fontsize="x-small", frameon=False)

        # Col 1 — source-frame dN/dχ.
        x_eng, y_eng = eng["source"]["chi"]
        _plot_1d(
            axes[r_idx, 1], x_eng, y_eng, mc["source"]["chi"],
            log_x=False, log_y=True,
            label_x=r"$\chi$",
            title=f"source dN/dχ  {resn}",
        )
        if r_idx == 0:
            axes[r_idx, 1].set_ylabel(r"$dN/d\chi$")

        # Col 2 — LIGO dN/dm₁.
        x_eng, y_eng = eng["ligo_1d"]["m1"]
        _plot_1d(
            axes[r_idx, 2], x_eng, y_eng, mc["ligo_1d"]["m1"],
            log_x=True, log_y=True,
            label_x=r"$m_1\;[M_\odot]$",
            title=f"LIGO dN/dm₁  {resn}",
        )
        if r_idx == 0:
            axes[r_idx, 2].set_ylabel(r"$dN/dm_1$")

        # Col 3 — LIGO dN/dq.
        x_eng, y_eng = eng["ligo_1d"]["q"]
        _plot_1d(
            axes[r_idx, 3], x_eng, y_eng, mc["ligo_1d"]["q"],
            log_x=False, log_y=True,
            label_x=r"$q = m_2/m_1$",
            title=f"LIGO dN/dq  {resn}",
        )
        if r_idx == 0:
            axes[r_idx, 3].set_ylabel(r"$dN/dq$")

        # Col 4 — LIGO dN/dχ_eff.
        x_eng, y_eng = eng["ligo_1d"]["chi_eff"]
        _plot_1d(
            axes[r_idx, 4], x_eng, y_eng, mc["ligo_1d"]["chi_eff"],
            log_x=False, log_y=True,
            label_x=r"$\chi_{\rm eff}$",
            title=f"LIGO dN/dχ_eff  {resn}",
        )
        if r_idx == 0:
            axes[r_idx, 4].set_ylabel(r"$dN/d\chi_{\rm eff}$")

        # Col 5 — LIGO dN/dM_chirp.
        x_eng, y_eng = eng["ligo_1d"]["M_chirp"]
        _plot_1d(
            axes[r_idx, 5], x_eng, y_eng, mc["ligo_1d"]["M_chirp"],
            log_x=True, log_y=True,
            label_x=r"$\mathcal{M}_c\;[M_\odot]$",
            title=f"LIGO dN/dM_chirp  {resn}",
        )
        if r_idx == 0:
            axes[r_idx, 5].set_ylabel(r"$dN/d\mathcal{M}_c$")

        # Col 6 — 2D (m₁, q).
        x_ax, y_ax, dens_e = eng["ligo_2d"]["m1_q"]
        _, _, dens_m = mc["ligo_2d"]["m1_q"]
        _plot_2d(
            axes[r_idx, 6], x_ax, y_ax, dens_e, dens_m,
            log_x=True,
            label_x=r"$m_1$", label_y=r"$q$",
            title=f"dN/(dm₁ dq)  {resn}",
        )

        # Col 7 — 2D (m₁, χ_eff).
        x_ax, y_ax, dens_e = eng["ligo_2d"]["m1_chi_eff"]
        _, _, dens_m = mc["ligo_2d"]["m1_chi_eff"]
        _plot_2d(
            axes[r_idx, 7], x_ax, y_ax, dens_e, dens_m,
            log_x=True,
            label_x=r"$m_1$", label_y=r"$\chi_{\rm eff}$",
            title=f"dN/(dm₁ dχ_eff)  {resn}",
        )

    fig.suptitle(
        f"Forward-modelling observables — implicit Picard ↔ particle MC "
        f"(θ* Stage 1; v_esc={theta['v_esc']:.0f} km/s)\n"
        f"MC: k-NN balloon KDE (placeholder, normalised; "
        f"--knn-k to tune; faint step = raw fine histogram, un-smoothed).  "
        f"Engine 1-D q/χ_eff/M_chirp: σ-bin Gaussian on bin-scatter "
        f"(--engine-smooth-sigma; 0 disables).  "
        f"See PLAN_arbitrary_point_evaluation.md.",
        fontsize="small",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_path, dpi=140)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Artifact serialisation.
# ---------------------------------------------------------------------------
def save_cell_npz(path: Path, engine_obs: dict, mc_obs: dict) -> None:
    r"""Serialise every panel-cell array for post-hoc re-plotting.

    Engine 1-D entries are ``(x, y)`` tuples; MC 1-D entries are dicts
    with ``{"x", "kde", "x_raw", "raw"}``.  Both forms are preserved so
    a downstream notebook can re-render with any smoothing the user
    prefers — the raw fine histogram is the gold-standard reference
    independent of any KDE bandwidth choice.
    """
    src_e = engine_obs["source"]
    src_m = mc_obs["source"]
    l1_e = engine_obs["ligo_1d"]
    l1_m = mc_obs["ligo_1d"]
    l2_e = engine_obs["ligo_2d"]
    l2_m = mc_obs["ligo_2d"]

    def _kv_1d(prefix_eng, eng_tuple, prefix_mc, mc_dict):
        x_e, y_e = eng_tuple
        return {
            f"{prefix_eng}_x": np.asarray(x_e),
            f"{prefix_eng}_eng": np.asarray(y_e),
            f"{prefix_mc}_kde_x": np.asarray(mc_dict["x"]),
            f"{prefix_mc}_kde": np.asarray(mc_dict["kde"]),
            f"{prefix_mc}_raw_x": np.asarray(mc_dict["x_raw"]),
            f"{prefix_mc}_raw": np.asarray(mc_dict["raw"]),
        }

    kv = {}
    kv.update(_kv_1d("src_m", src_e["m"], "src_m_mc", src_m["m"]))
    kv.update(_kv_1d("src_chi", src_e["chi"], "src_chi_mc", src_m["chi"]))
    kv.update(_kv_1d("m1", l1_e["m1"], "m1_mc", l1_m["m1"]))
    kv.update(_kv_1d("q", l1_e["q"], "q_mc", l1_m["q"]))
    kv.update(_kv_1d("chi_eff", l1_e["chi_eff"], "chi_eff_mc", l1_m["chi_eff"]))
    kv.update(_kv_1d("Mc", l1_e["M_chirp"], "Mc_mc", l1_m["M_chirp"]))
    np.savez(
        path,
        # 1-D engine + MC (KDE + raw)
        **kv,
        # bin edges (engine LIGO 1-D scatter)
        m1_edges=l1_e["bin_edges"]["m1"],
        q_edges=l1_e["bin_edges"]["q"],
        chi_eff_edges=l1_e["bin_edges"]["chi_eff"],
        Mc_edges=l1_e["bin_edges"]["M_chirp"],
        # LIGO 2-D
        m1_q_x=l2_e["m1_q"][0], m1_q_y=l2_e["m1_q"][1],
        m1_q_eng=l2_e["m1_q"][2], m1_q_mc=l2_m["m1_q"][2],
        m1_chi_eff_x=l2_e["m1_chi_eff"][0], m1_chi_eff_y=l2_e["m1_chi_eff"][1],
        m1_chi_eff_eng=l2_e["m1_chi_eff"][2], m1_chi_eff_mc=l2_m["m1_chi_eff"][2],
        # totals
        engine_total_rate=np.float64(engine_obs["total_rate"]),
        n_mc_events=np.int64(mc_obs["n_events"]),
        knn_k=np.int64(mc_obs["knn_k"]),
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_resolutions(s: str) -> list[tuple[int, int]]:
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" not in chunk:
            raise argparse.ArgumentTypeError(
                f"--resolutions entry {chunk!r} must be Nm×Nchi (e.g. 64x20)"
            )
        a, b = chunk.split("x")
        out.append((int(a), int(b)))
    if not out:
        raise argparse.ArgumentTypeError("--resolutions must list at least one pair")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--resolutions", type=parse_resolutions, default="64x20",
        help="Comma-separated list of Nm×Nχ pairs.  E.g. 32x10,64x20,128x40. "
             "Each pair becomes one row of panels.  Default: 64x20.",
    )
    parser.add_argument(
        "--n-mc", type=int, default=50_000,
        help="Natal pool size for the particle MC.  Default 5e4.  "
             "Bump to 1e5 if MC noise visibly dominates.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260521,
        help="Deterministic RNG root for the particle MC.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="Implicit-solver coupling × time.  ε ≈ 1 is a fully-developed "
             "cascade.  ε ≪ 1 leaves the natal pool intact.",
    )
    parser.add_argument(
        "--v-esc", type=float, default=None,
        help="Per-environment escape velocity (km/s).  Default uses "
             "THETA_STAR['v_esc'].",
    )
    parser.add_argument(
        "--out-hash", type=str, default=None,
        help="Output-dir suffix (default: random 6-hex per "
             "[[feedback-demo-output-conventions]]).",
    )
    parser.add_argument(
        "--knn-k", type=int, default=KNN_K_DEFAULT,
        help=f"k for the k-NN balloon-estimator MC KDE.  Smaller k = "
             f"sharper kernel (resolves fine substructure, more shot-"
             f"noise wiggle); larger k = smoother.  Default "
             f"{KNN_K_DEFAULT} (good for N_events ~ 4×10⁵).  "
             f"Placeholder pending PLAN_arbitrary_point_evaluation.md.",
    )
    parser.add_argument(
        "--engine-smooth-sigma", type=float, default=0.7,
        help="Gaussian-smoothing σ (in bin units) applied to the "
             "engine's 1-D dN/dq, dN/dχ_eff, dN/dM_chirp arrays for "
             "display only.  Default 0.7 — just enough to soften ±1-"
             "bin bin-scatter discreteness without smearing physics. "
             "Pass 0 to disable.  Raw arrays preserved in npz "
             "(ligo_1d_raw).",
    )
    parser.add_argument(
        "--void-h-ratio", type=float, default=0.5,
        help="Void-gate threshold for the MC k-NN KDE.  Zero density "
             "wherever h_1(x)/h_k(x) > threshold — i.e. the k samples "
             "form a tight ball far from x (eval point in a void).  "
             "Default 0.5 kills the boundary-spread artefacts (e.g. "
             "the m1 distribution extending toward zero mass).  Pass "
             "1.0 to disable the gate (recovers the unmodified k-NN "
             "balloon estimator).  Scale-free.",
    )
    parser.add_argument(
        "--orientation", type=parse_orientation, default="aligned",
        help="Spin-orientation model.  'aligned' (default) routes both "
             "engine and MC through the v1 bit-for-bit scalar path.  "
             "'isotropic' draws each spin uniformly on the 2-sphere.  "
             "'fisher:κ' uses von Mises-Fisher concentrated on L̂ with "
             "concentration κ ≥ 0 (κ → 0 == isotropic, κ → ∞ == aligned).",
    )
    parser.add_argument(
        "--retention-kind", type=str, default="heaviside",
        choices=["heaviside", "sigmoid"],
        help="Per-orientation retention model under non-Aligned "
             "orientations (ignored if --orientation is 'aligned'): "
             "'heaviside' (default; preserves the v1 hard-cut at each "
             "orientation quadrature point) or 'sigmoid' (soft cut, "
             "recovers ∂/∂v_esc through JAX).",
    )
    parser.add_argument(
        "--w-over-v-esc", type=float, default=1.0,
        help="Sigmoid sharpness fraction (ignored unless "
             "--retention-kind=sigmoid).  Default 1.0 → soft transition "
             "of width v_esc around v_esc.",
    )
    args = parser.parse_args()

    theta = dict(THETA_STAR)
    if args.v_esc is not None:
        theta["v_esc"] = float(args.v_esc)

    orientation = args.orientation
    o_tag = orientation_tag(orientation)

    here = Path(__file__).parent
    rh = args.out_hash or secrets.token_hex(3)
    # Tag the run-hash with the orientation choice so per-orientation
    # runs at the same θ* and seed don't clobber each other's figures.
    # Per [[feedback-demo-output-conventions]] each run gets its own
    # figures_/artifacts_ dir.
    rh_full = f"{rh}_{o_tag}"
    fig_dir = here / f"figures_fmobs_{rh_full}"
    art_dir = here / f"artifacts_fmobs_{rh_full}"
    fig_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    print(f"forward_modeling_observables — run hash: {rh_full}")
    print(f"  figures   → {fig_dir.relative_to(here.parent)}/")
    print(f"  artifacts → {art_dir.relative_to(here.parent)}/")
    print(f"  resolutions: {args.resolutions}")
    print(f"  ε={args.epsilon}  v_esc={theta['v_esc']}  n_mc={args.n_mc}  seed={args.seed}")
    print(
        f"  orientation={o_tag}  retention_kind={args.retention_kind}"
        + (f"  w/v_esc={args.w_over_v_esc}" if args.retention_kind == "sigmoid" else "")
    )

    rows = []
    metadata = dict(
        run_hash=rh_full,
        theta_star=theta,
        epsilon=args.epsilon,
        seed=args.seed,
        n_mc=args.n_mc,
        knn_k=args.knn_k,
        void_h_ratio=args.void_h_ratio,
        engine_smooth_sigma_bins=args.engine_smooth_sigma,
        mc_kde_scheme="knn_balloon_uniform_kernel_with_void_gate",
        orientation=o_tag,
        retention_kind=args.retention_kind,
        w_over_v_esc=args.w_over_v_esc,
        resolutions=[list(r) for r in args.resolutions],
        per_resolution=[],
    )

    for n_m, n_chi in args.resolutions:
        print(f"\n=== resolution {n_m}×{n_chi} ===")
        print(f"[1/3] engine implicit solver ...")
        engine_obs = engine_observables(
            theta, n_m, n_chi, args.epsilon,
            smooth_sigma_bins=args.engine_smooth_sigma,
            orientation=orientation,
            retention_kind=args.retention_kind,
            w_over_v_esc=args.w_over_v_esc,
        )

        # Session C step 10 — demo-level confirmation that
        # ``∂total_rate/∂v_esc`` is finite and non-zero under any non-
        # Aligned orientation + sigmoid retention.  Under Aligned +
        # Heaviside the v1 path returns *exactly* 0 (the engine docstring
        # documents this; gradients flow through every other parameter).
        # See [[project-misalignment-session-b-done]] for the lower-
        # level ``_solve``-surface verification.
        v_esc_grad = None
        if not isinstance(orientation, Aligned):
            grad_response = build_response(
                theta, n_m, n_chi, args.epsilon,
                orientation=orientation,
                retention_kind=args.retention_kind,
                w_over_v_esc=args.w_over_v_esc,
            )
            grad_fn = jax.grad(
                lambda v: engine_total_rate_at_v_esc(
                    grad_response, v, epsilon=args.epsilon,
                )
            )
            v_esc_jax = jnp.asarray(theta["v_esc"])
            t0 = time.time()
            v_esc_grad = float(grad_fn(v_esc_jax))
            t_grad = time.time() - t0
            sign = "+" if v_esc_grad >= 0 else "-"
            print(
                f"      ∂(total_rate)/∂(v_esc) = {sign}{abs(v_esc_grad):.4e}"
                f"  (wall={t_grad:.1f}s, [{args.retention_kind}])"
            )
            if v_esc_grad == 0.0:
                print(
                    "      WARNING: v_esc gradient is exactly zero — "
                    "expected non-zero under non-Aligned orientation.  "
                    "Check retention_kind / sigmoid sharpness."
                )
        d = engine_obs["diag"]
        print(
            f"      converged in {d['n_iter']} iters; "
            f"|ΔN|∞={d['final_diff']:.2e}; "
            f"f_natal_depleted={d['f_natal_depleted']:.4g}; "
            f"wall={d['wall_seconds']:.1f}s"
        )
        print(f"      total per-env rate = {engine_obs['total_rate']:.4e}")

        # MC stopping fraction tracks the engine's f_natal_depleted.
        f_dep = max(d["f_natal_depleted"], 1e-3)
        print(f"[2/3] particle MC (n_seed={args.n_mc}, depletion={f_dep:.4g}) ...")
        stats, mergers, v_kicks = run_mc(
            theta,
            n_seed=args.n_mc,
            depletion_fraction=f_dep,
            seed=args.seed + n_m * 1009 + n_chi,
            orientation=orientation,
            # Deterministic per-resolution orientation key — keeps the
            # run reproducible without colliding the orientation stream
            # against the numpy seed used for the active-pool dynamics.
            orientation_key_seed=args.seed + n_m * 7919 + n_chi * 31,
        )
        print(
            f"      n_accepted={stats['n_accepted']}  "
            f"n_retained={stats['n_retained']}  "
            f"stop={stats['stop_reason']}  "
            f"wall={stats['wall_seconds']:.1f}s"
        )

        # Rate scale: matches engine total to MC count (one event per row).
        if mergers.shape[0] == 0:
            print("      WARNING: zero accepted mergers — MC panels will be empty")
            rate_scale_ordered = 0.0
        else:
            rate_scale_ordered = engine_obs["total_rate"] / float(mergers.shape[0])

        print(
            f"[3/3] MC observables (placeholder k-NN KDE, knn_k={args.knn_k}, "
            f"void_h_ratio={args.void_h_ratio}) ..."
        )
        mc_obs = mc_observables(
            mergers, engine_obs, rate_scale_ordered,
            knn_k=args.knn_k,
            void_h_ratio=args.void_h_ratio,
            theta=theta,
        )

        # Save per-resolution npz.
        npz_path = art_dir / f"cells_{n_m}x{n_chi}.npz"
        save_cell_npz(npz_path, engine_obs, mc_obs)
        print(f"      wrote {npz_path.relative_to(here.parent)}")

        rows.append(dict(
            n_m=n_m, n_chi=n_chi, engine=engine_obs, mc=mc_obs,
        ))
        # Per-resolution kick-distribution summary — Session C diagnostic.
        # Under Aligned: v_kicks are the scalar v_kick_aligned values
        # (small, O(10) km/s at θ*).  Under Isotropic / Fisher: the per-
        # event vector-LZ kicks include the v_⊥ "superkick" branch and
        # can reach O(few×10³) km/s — this is what activates retention.
        if v_kicks.size:
            kick_summary = dict(
                v_kick_median=float(np.median(v_kicks)),
                v_kick_p90=float(np.percentile(v_kicks, 90)),
                v_kick_max=float(np.max(v_kicks)),
                v_kick_frac_above_v_esc=float(
                    np.mean(v_kicks > theta["v_esc"])
                ),
            )
        else:
            kick_summary = dict(
                v_kick_median=0.0, v_kick_p90=0.0, v_kick_max=0.0,
                v_kick_frac_above_v_esc=0.0,
            )
        metadata["per_resolution"].append(dict(
            n_m=n_m, n_chi=n_chi,
            engine_diag=d,
            engine_total_rate=engine_obs["total_rate"],
            v_esc_grad=v_esc_grad,
            mc_stats=stats,
            mc_kicks=kick_summary,
            rate_scale_ordered=rate_scale_ordered,
        ))

    fig_path = fig_dir / "forward_modeling_observables.png"
    print(f"\n[render] {fig_path.relative_to(here.parent)}")
    build_figure(rows, theta, fig_path)
    print(f"  wrote {fig_path.relative_to(here.parent)}")

    meta_path = art_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=float))
    print(f"  wrote {meta_path.relative_to(here.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
