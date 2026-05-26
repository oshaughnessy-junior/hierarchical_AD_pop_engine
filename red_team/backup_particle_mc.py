"""Backup implementation 1 — pure particle Monte Carlo.

The simplest possible coagulation simulator. No grids, no per-generation
densities, no truncated Volterra series. Just a particle list whose
members merge with each other, depleting the pool as the cascade runs.

Why this implementation cannot be wrong by construction:

1. State is a list of continuous ``(m, χ)`` floats. No discretisation.
2. There is no "generation" abstraction in the dynamics — the cascade is
   a sequence of pair-mergers; generation labels are optional diagnostics.
3. The pool depletes when particles merge: two singles become one
   remnant (if retained) or zero (if not). No "non-depleting source"
   approximation.
4. Each accepted pair corresponds to *one physical merger event*. There
   is no abstraction between density and rate.

If this disagrees with the engine, the engine has a bug. If it agrees
with the Doctor-perturbative-with-loss backup (Backup 2), we have two
independent witnesses to the "correct" total ``N(m, χ)``.

Misalignment mode — Session C wiring
------------------------------------
``MCConfig.orientation`` accepts any concrete :mod:`spin_orientation`
model (``Aligned`` / ``Isotropic`` / ``Fisher(κ)``).  Defaults to
``None`` which is treated as ``Aligned()``: the per-merger code path is
the **bit-for-bit existing** scalar ``chi_rem`` / ``v_kick_aligned`` /
``retention_aligned`` triple.  Under any non-``Aligned`` model the MC
draws one orientation realisation per accepted pair from
``orientation.sample(key, 1)``, promotes the natal spin magnitudes to
vectors, evaluates the full vector Barausse-Rezzolla 2009 remnant spin
and Lousto-Zlochower 2008/2011 kick, and applies a per-event Heaviside
retention against the per-event ``v_kick`` (no MC-side marginalisation
needed; each accepted pair is a single physical event).  The
``v_kicks`` 1-D array on the return dict carries the per-event kick
magnitude alongside the existing ``mergers`` 8-column schema, which is
left untouched for downstream-reader compatibility.

The natal ``chi`` distribution is sampled in ``[0, 1]`` and interpreted
as **signed χ** under ``Aligned`` (v1 semantics) and as ``|χ|`` under
non-``Aligned`` (the orientation distribution carries the sign / tilt).
Same numerical values, different interpretation — no schema change.

Run::

    python -m red_team.backup_particle_mc       # smoke at Stage 1 θ*
"""

from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import truncnorm

# Allowed engine imports — physics primitives only. Red-team rule is the
# same as for sim_coagulation.py: may not import response.* (the thing
# under test) but may import the lowest-level physics fits.
from hierarchical_ad_pop_engine.remnants.mass import M_rem as _M_rem_jax
from hierarchical_ad_pop_engine.remnants.spin import chi_rem as _chi_rem_jax
from hierarchical_ad_pop_engine.remnants.kick import (
    retention_aligned as _retention_aligned_jax,
    v_kick_aligned as _v_kick_aligned_jax,
)
from hierarchical_ad_pop_engine.remnants.marginal import (
    chi_rem_at_orientation as _chi_rem_at_orientation_jax,
    v_kick_at_orientation as _v_kick_at_orientation_jax,
)
from hierarchical_ad_pop_engine.spin_orientation import Aligned as _Aligned

from red_team.sim_coagulation import make_marginal


# ---------------------------------------------------------------------------
# Numpy wrappers — same convention as sim_coagulation. Vectorised because
# we call them on small batches in the hot loop.
# ---------------------------------------------------------------------------
def M_rem(m1, m2, chi1, chi2):
    return np.asarray(_M_rem_jax(m1, m2, chi1, chi2))


def chi_rem_(m1, m2, chi1, chi2):
    return np.asarray(_chi_rem_jax(m1, m2, chi1, chi2))


def retention_aligned(m1, m2, chi1, chi2, v_esc):
    return np.asarray(_retention_aligned_jax(m1, m2, chi1, chi2, v_esc))


def kernel_K(m1, m2, a, b):
    """Same kernel as ``sim_coagulation.kernel_K`` — duplicated to keep this
    backup importable on its own."""
    m_big = np.maximum(m1, m2)
    m_sml = np.minimum(m1, m2)
    q = m_sml / m_big
    return (m1 + m2) ** a * q ** b


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
@dataclass
class MCConfig:
    mu_natal: float
    sigma_natal: float
    m_min: float
    m_max: float
    natal_spin_loc: float
    natal_spin_scale: float
    kernel_a: float
    kernel_b: float
    v_esc: float
    n_seed: int = 20_000
    # Maximum number of merger *attempts* (rejected + accepted). The cascade
    # naturally terminates when the active pool can't form more accepted
    # pairs, but this caps runtime in pathological cases.
    n_max_attempts: int = 2_000_000
    # Stopping criterion 1 — hard cap on accepted mergers. If ``None`` the
    # depletion-fraction criterion takes over.
    n_max_mergers: Optional[int] = None
    # Stopping criterion 2 — stop once the active pool has shrunk by this
    # fraction relative to ``n_seed``. Defaults to 0.05 (5 %) which places
    # the cascade in the engine's "small effective ε" regime — see
    # `compare_all` output, where engine ≈ Backup 2 implicit at ε ≈ 0.025.
    # Running to full depletion (the original default) leaves only a
    # handful of high-generation remnants and the KDE looks featureless.
    #
    # Per `feedback-mc-depletion-tracks-engine` this is **not** a free
    # knob in production fits: the demo / acceptance harnesses set it
    # from the engine's ``f_natal_depleted`` so the two paths sample the
    # same regime.  Left as a config field here because the standalone
    # MC smoke (this module's ``__main__``) doesn't have an engine in
    # the loop.
    depletion_fraction: float = 0.05
    # Sample budget on each call to the K-rejection batch — bigger means
    # less Python overhead but more wasted work if acceptance is high.
    rejection_batch: int = 1024
    # Misalignment mode — Session C.  ``None`` is treated as
    # :class:`~hierarchical_ad_pop_engine.spin_orientation.Aligned` and
    # routes every accepted pair through the existing scalar v1
    # ``chi_rem`` / ``v_kick_aligned`` / ``retention_aligned`` code path
    # bit-for-bit.  Any other concrete ``OrientationModel`` activates the
    # per-event vector-physics path (see
    # ``demo/PLAN_misalignment_mode.md`` § Session C, step 8).
    orientation: object = None
    # Master jax PRNG seed used to derive per-event sampling keys under
    # non-``Aligned`` orientations.  Independent of the numpy ``rng`` so
    # that switching orientation models does not perturb the pair-
    # rejection sequence (Aligned bit-for-bit invariant relies on the
    # numpy ``rng`` driving everything that touches the active pool).
    orientation_key_seed: int = 0


def _orientation_is_aligned(orientation: object) -> bool:
    """``True`` if ``orientation`` should route to the v1 scalar path.

    ``None`` is treated as ``Aligned()`` so callers that pre-date Session
    C continue to land on the bit-for-bit existing code path.
    """
    return orientation is None or isinstance(orientation, _Aligned)


def _per_event_remnant_misaligned(
    key: jax.Array,
    m1: float,
    m2: float,
    chi1_mag: float,
    chi2_mag: float,
    v_esc: float,
    orientation: object,
) -> tuple[float, float, float, bool]:
    """One MC-event remnant evaluation under a non-``Aligned`` orientation.

    Samples ``(θ_1, φ_1, θ_2, φ_2)`` from ``orientation.sample(key, 1)``,
    promotes the scalar progenitor magnitudes to vector spins via the
    convention ``φ_1 = 0``, ``φ_2 = Δφ`` (azimuthal-symmetry collapse:
    ``|χ_rem|`` and ``v_kick`` are invariant under rigid rotation about
    ``L̂``), and returns the scalar ``(m_rem, |χ_rem|, v_kick, retain)``
    for this event.

    ``m_rem`` uses the parallel projection ``χ_z = χ_mag · cos θ`` —
    matches the convention the engine's ``_step_one_generation``
    marginalised path uses for ``M_rem``'s ``chi_eff`` input (see
    ``project-misalignment-session-b-done``).

    Retention is **per-event Heaviside** ``v_kick < v_esc``.  No
    orientation-side marginalisation: each accepted pair is one physical
    event with a definite outcome.
    """
    theta_1, phi_1, theta_2, phi_2 = orientation.sample(key, n_events=1)
    cos_t1 = float(jnp.cos(theta_1[0]))
    cos_t2 = float(jnp.cos(theta_2[0]))
    dphi = float(phi_2[0] - phi_1[0])

    # Parallel projection feeds M_rem's chi_eff input.
    chi_z_1 = chi1_mag * cos_t1
    chi_z_2 = chi2_mag * cos_t2
    m_rem = float(_M_rem_jax(m1, m2, chi_z_1, chi_z_2))

    chi_rem_mag = float(
        _chi_rem_at_orientation_jax(
            m1, m2, chi1_mag, chi2_mag, cos_t1, cos_t2, dphi
        )
    )
    v_kick = float(
        _v_kick_at_orientation_jax(
            m1, m2, chi1_mag, chi2_mag, cos_t1, cos_t2, dphi
        )
    )
    retain = v_kick < float(v_esc)
    return m_rem, chi_rem_mag, v_kick, retain


# ---------------------------------------------------------------------------
# Particle MC.
# ---------------------------------------------------------------------------
def sample_natals(cfg: MCConfig, rng: np.random.Generator):
    a_m = (cfg.m_min - cfg.mu_natal) / cfg.sigma_natal
    b_m = (cfg.m_max - cfg.mu_natal) / cfg.sigma_natal
    m = truncnorm.rvs(
        a_m, b_m, loc=cfg.mu_natal, scale=cfg.sigma_natal,
        size=cfg.n_seed, random_state=rng,
    ).astype(np.float64)
    a_c = (0.0 - cfg.natal_spin_loc) / cfg.natal_spin_scale
    b_c = (1.0 - cfg.natal_spin_loc) / cfg.natal_spin_scale
    chi = truncnorm.rvs(
        a_c, b_c, loc=cfg.natal_spin_loc, scale=cfg.natal_spin_scale,
        size=cfg.n_seed, random_state=rng,
    ).astype(np.float64)
    return m, chi


def _kernel_upper_bound(m_active: np.ndarray, a: float, b: float) -> float:
    """Conservative K_max over the active pool."""
    mlo = float(m_active.min())
    mhi = float(m_active.max())
    candidates = [
        kernel_K(np.array(2 * mhi), np.array(2 * mhi), a, b),  # equal-mass heavy
        kernel_K(np.array(2 * mlo), np.array(2 * mlo), a, b),  # equal-mass light
        kernel_K(np.array(mhi + mlo), np.array(mhi + mlo), a, b),  # asymmetric (q=mlo/mhi)
    ]
    return float(max(candidates)) * 1.05


def particle_mc(
    cfg: MCConfig,
    rng: np.random.Generator,
    *,
    track_generations: bool = True,
) -> dict:
    """Run the cascade to natural termination (or budget exhaustion).

    Returns a dict with keys:
        ``m_active``, ``chi_active``, ``gen_active`` — surviving particles.
        ``mergers``  : structured array shape ``(N_accept, 8)`` of accepted
            merger events: (m1, m2, χ1, χ2, g1, g2, retained, m_remnant_or_nan).
            Heavier first.  Under non-``Aligned`` orientation the χ columns
            are the **progenitor magnitudes** ``|χ_1|, |χ_2|`` (the
            orientation distribution carries the sign / tilt).
        ``v_kicks`` : 1-D array shape ``(N_accept,)`` of the per-event
            recoil ``v_kick`` (km/s).  Under ``Aligned`` this is the
            scalar :func:`v_kick_aligned`; under any other orientation
            the per-event sample.  Added in Session C — older readers
            that only consume ``mergers`` are unaffected.
        ``stats`` : dict with n_attempts, n_accepted, n_retained, n_active_final.

    Notes
    -----
    The hot loop is intentionally Python-level because the bookkeeping
    (active-mask updates, pair removals) doesn't vectorise cleanly. We
    batch the *K-rejection step* (drawing many candidate pairs at once)
    because that's where ~all the wallclock goes — typically O(1/⟨K⟩)
    rejections per accepted pair.

    Aligned bit-for-bit invariant: when ``cfg.orientation is None`` or
    ``Aligned()`` the only numerical change versus pre-Session-C code is
    the new ``v_kicks`` array on the return dict — every other quantity
    (active particles, mergers list, stats counters, rng draws) is
    literally unchanged.  The Heaviside retention is what was already
    routed; the new ``v_kick_aligned`` evaluation on each accepted pair
    is read-only and does not perturb the numpy ``rng`` sequence.
    """
    aligned_path = _orientation_is_aligned(cfg.orientation)
    orient_key_state = jax.random.PRNGKey(int(cfg.orientation_key_seed))
    m_arr, chi_arr = sample_natals(cfg, rng)
    n_init = m_arr.size
    gen_arr = np.ones(n_init, dtype=np.int32) if track_generations else None
    active = np.ones(n_init, dtype=bool)

    # Resolve stopping criterion. If the user passed an explicit
    # n_max_mergers we use that; otherwise compute one from
    # depletion_fraction. We also stop on n_max_attempts as a safety net.
    if cfg.n_max_mergers is not None:
        n_max_mergers = cfg.n_max_mergers
    else:
        # Each merger removes ≈1 (retained) or 2 (non-retained) from the
        # pool; for high-retention regimes ≈ 1 per merger. Stopping at
        # depletion_fraction means roughly that many mergers.
        n_max_mergers = max(1, int(cfg.depletion_fraction * n_init))
    n_active_floor = max(2, int(n_init * (1.0 - cfg.depletion_fraction))) if cfg.n_max_mergers is None else 2

    mergers_records: list[np.ndarray] = []
    # Per-event v_kick (km/s).  Aligned: v_kick_aligned scalar.
    # Non-Aligned: per-event vector-LZ sample.  Indexed in lockstep with
    # ``mergers_records`` so the i-th entry corresponds to the i-th
    # accepted merger.
    v_kicks_records: list[float] = []

    n_attempts = 0
    n_accepted = 0
    n_retained = 0
    no_progress_streak = 0
    max_no_progress = 5  # how many full rejection batches we tolerate
    stop_reason = "n_max_mergers"  # updated as we go

    while n_accepted < n_max_mergers and n_attempts < cfg.n_max_attempts:
        if int(active.sum()) < n_active_floor:
            stop_reason = "depletion_fraction"
            break
        active_idx = np.flatnonzero(active)
        n_active = active_idx.size
        if n_active < 2:
            stop_reason = "pool_exhausted"
            break

        # Update K_max from current pool — pool shrinks and broadens, so the
        # static bound from natals would become loose over time.
        K_max = _kernel_upper_bound(m_arr[active_idx], cfg.kernel_a, cfg.kernel_b)
        if not np.isfinite(K_max) or K_max <= 0:
            stop_reason = "K_max_invalid"
            break

        # Batch K-rejection. Draw `rejection_batch` candidate pairs from the
        # active list; accept by K_ij / K_max.
        batch = min(cfg.rejection_batch, n_active * (n_active - 1) // 2 or 1)
        ia_in_active = rng.integers(0, n_active, size=batch)
        ib_in_active = rng.integers(0, n_active, size=batch)
        # Disallow self-pairs by resampling i_b for collisions.
        same = ia_in_active == ib_in_active
        while same.any():
            ib_in_active[same] = rng.integers(0, n_active, size=int(same.sum()))
            same = ia_in_active == ib_in_active

        ia = active_idx[ia_in_active]
        ib = active_idx[ib_in_active]
        m_a = m_arr[ia]; m_b = m_arr[ib]
        chi_a = chi_arr[ia]; chi_b = chi_arr[ib]
        K_vals = kernel_K(m_a, m_b, cfg.kernel_a, cfg.kernel_b)
        u = rng.random(size=batch)
        K_accept = u < (K_vals / K_max)

        n_attempts += batch
        if not K_accept.any():
            no_progress_streak += 1
            if no_progress_streak > max_no_progress:
                stop_reason = "no_K_acceptance"
                break
            continue
        no_progress_streak = 0

        # We process accepted pairs *one at a time* because each merger
        # mutates the active set, which would invalidate later pair indices
        # in the batch. (We accept the Python-loop cost here; the dominant
        # work is the K-rejection above which IS vectorised.)
        accept_idx = np.flatnonzero(K_accept)
        for k in accept_idx:
            i_a = ia[k]; i_b = ib[k]
            # Reject silently if either particle has already merged earlier
            # in this same batch.
            if not (active[i_a] and active[i_b]):
                continue
            n_accepted += 1
            ma = m_arr[i_a]; mb = m_arr[i_b]
            ca = chi_arr[i_a]; cb = chi_arr[i_b]
            ga = (gen_arr[i_a] if gen_arr is not None else 1)
            gb = (gen_arr[i_b] if gen_arr is not None else 1)

            if aligned_path:
                # ---------- Aligned bit-for-bit existing v1 path. ----------
                retain = bool(retention_aligned(ma, mb, ca, cb, cfg.v_esc) > 0.5)
                if retain:
                    m_new = float(M_rem(ma, mb, ca, cb))
                    chi_new = float(chi_rem_(ma, mb, ca, cb))
                # v_kick for the v_kicks return array — read-only, does
                # not affect the active-pool sequence.  Evaluating both
                # branches keeps the per-event work uniform regardless of
                # whether the pair was retained.
                v_kick_evt = float(_v_kick_aligned_jax(ma, mb, ca, cb))
            else:
                # ---------- Non-Aligned: per-event vector physics. ---------
                # Advance the orientation key stream.  Independent of the
                # numpy ``rng`` so the active-pool sequence stays in lock-
                # step with the Aligned baseline up to the retention
                # decision.
                orient_key_state, evt_key = jax.random.split(orient_key_state)
                m_new, chi_new, v_kick_evt, retain = (
                    _per_event_remnant_misaligned(
                        evt_key, float(ma), float(mb),
                        float(ca), float(cb),
                        cfg.v_esc, cfg.orientation,
                    )
                )

            if retain:
                # Replace heavier-slot with the remnant; deactivate the
                # other. Heavier slot first so the bookkeeping conserves
                # m1 ≥ m2 cosmetically.
                if ma >= mb:
                    m_arr[i_a] = m_new; chi_arr[i_a] = chi_new
                    if gen_arr is not None: gen_arr[i_a] = max(ga, gb) + 1
                    active[i_b] = False
                else:
                    m_arr[i_b] = m_new; chi_arr[i_b] = chi_new
                    if gen_arr is not None: gen_arr[i_b] = max(ga, gb) + 1
                    active[i_a] = False
                n_retained += 1
                event_rem = m_new
            else:
                active[i_a] = False; active[i_b] = False
                event_rem = np.nan

            # Heavier-first event record.
            if ma >= mb:
                rec = (ma, mb, ca, cb, ga, gb, float(retain), event_rem)
            else:
                rec = (mb, ma, cb, ca, gb, ga, float(retain), event_rem)
            mergers_records.append(rec)
            v_kicks_records.append(v_kick_evt)

            if n_accepted >= n_max_mergers:
                break

    out_idx = np.flatnonzero(active)
    return dict(
        m_active=m_arr[out_idx].copy(),
        chi_active=chi_arr[out_idx].copy(),
        gen_active=(gen_arr[out_idx].copy() if gen_arr is not None else None),
        mergers=(
            np.asarray(mergers_records, dtype=np.float64)
            if mergers_records else np.empty((0, 8), dtype=np.float64)
        ),
        v_kicks=(
            np.asarray(v_kicks_records, dtype=np.float64)
            if v_kicks_records else np.empty((0,), dtype=np.float64)
        ),
        stats=dict(
            n_attempts=n_attempts,
            n_accepted=n_accepted,
            n_retained=n_retained,
            n_active_final=int(active.sum()),
            n_init=n_init,
            stop_reason=stop_reason,
            depletion=1.0 - (int(active.sum()) / max(n_init, 1)),
            orientation=type(cfg.orientation).__name__ if cfg.orientation is not None else "Aligned",
        ),
    )


# ---------------------------------------------------------------------------
# Smoke driver — Stage 1 θ*, KDE plots.
# ---------------------------------------------------------------------------
_STAGE1 = dict(
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


def _plot_active_marginals(result: dict, cfg: MCConfig, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    m_active = result["m_active"]
    chi_active = result["chi_active"]
    gen_active = result["gen_active"]

    m_eval = np.geomspace(cfg.m_min, cfg.m_max * 2, 400)
    chi_eval = np.linspace(0.001, 0.999, 400)

    # Total (all surviving particles).
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if m_active.size >= 2:
        dN_dm = make_marginal(m_active, m_eval, kind="kde",
                              bounds=(cfg.m_min, cfg.m_max * 2), log_space=True)
        ax.loglog(m_eval, np.maximum(dN_dm, 1e-30), lw=2.0, color="C3",
                  label=f"all surviving (N={m_active.size})")

    # Per-generation breakdown if tracked.
    if gen_active is not None:
        G = int(gen_active.max()) if gen_active.size else 0
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(G, 1)))
        for g in range(1, G + 1):
            mask = gen_active == g
            n_g = int(mask.sum())
            if n_g < 2:
                continue
            dN = make_marginal(m_active[mask], m_eval, kind="kde",
                               bounds=(cfg.m_min, cfg.m_max * 2), log_space=True)
            ax.loglog(m_eval, np.maximum(dN, 1e-30), lw=1.4, ls="--",
                      color=colors[g - 1], label=f"{g}G (N={n_g})")
    ax.set_xlabel(r"single-BH mass $m \;[M_\odot]$")
    ax.set_ylabel(r"$dN/dm$ (KDE on surviving particles)")
    ax.set_title("Backup 1 — pure particle MC, surviving population $dN/dm$")
    if m_active.size:
        ax.set_ylim(bottom=float(make_marginal(m_active, m_eval, kind="kde",
                    bounds=(cfg.m_min, cfg.m_max * 2), log_space=True).max()) * 1e-6)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    fig.tight_layout()
    out = out_dir / "backup1_dN_dm.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")

    # Spin.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if chi_active.size >= 2:
        dN_dchi = make_marginal(chi_active, chi_eval, kind="kde",
                                bounds=(0.0, 1.0))
        ax.semilogy(chi_eval, np.maximum(dN_dchi, 1e-30), lw=2.0, color="C3",
                    label=f"all surviving (N={chi_active.size})")
    if gen_active is not None:
        G = int(gen_active.max()) if gen_active.size else 0
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(G, 1)))
        for g in range(1, G + 1):
            mask = gen_active == g
            n_g = int(mask.sum())
            if n_g < 2:
                continue
            dN = make_marginal(chi_active[mask], chi_eval, kind="kde",
                               bounds=(0.0, 1.0))
            ax.semilogy(chi_eval, np.maximum(dN, 1e-30), lw=1.4, ls="--",
                        color=colors[g - 1], label=f"{g}G (N={n_g})")
    ax.axvline(0.6865, color="0.4", lw=0.7, ls=":", label=r"$\chi_{\rm rem}(1,0)$")
    ax.set_xlabel(r"single-BH spin $\chi$")
    ax.set_ylabel(r"$dN/d\chi$ (KDE on surviving particles)")
    ax.set_title("Backup 1 — pure particle MC, surviving population $dN/d\\chi$")
    ax.set_xlim(0, 1)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    fig.tight_layout()
    out = out_dir / "backup1_dN_dchi.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _smoke_main(
    run_hash: Optional[str] = None,
    n_seed: int = 20_000,
    n_max_mergers: Optional[int] = None,
    depletion_fraction: float = 0.05,
) -> None:
    here = Path(__file__).parent
    rh = run_hash or secrets.token_hex(3)
    fig_dir = here / f"figures_{rh}"
    art_dir = here / f"artifacts_{rh}"
    fig_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    print(f"Run hash: {rh}")
    print(f"  figures → {fig_dir.relative_to(here.parent)}/")
    print(f"  artifacts → {art_dir.relative_to(here.parent)}/")

    cfg = MCConfig(
        n_seed=n_seed,
        n_max_mergers=n_max_mergers,
        depletion_fraction=depletion_fraction,
        **_STAGE1,
    )
    rng = np.random.default_rng(20260518)
    print(f"Config: {cfg}")

    print("Running pure particle MC ...")
    result = particle_mc(cfg, rng, track_generations=True)
    s = result["stats"]
    print()
    print("Stats:")
    print(f"  n_init             = {s['n_init']}")
    print(f"  n_attempts         = {s['n_attempts']}")
    print(f"  n_accepted         = {s['n_accepted']}")
    print(f"  n_retained         = {s['n_retained']}")
    print(f"  n_active_final     = {s['n_active_final']}")
    print(f"  depletion fraction = {s['depletion']:.4f}")
    print(f"  retain fraction    = {s['n_retained']/max(s['n_accepted'],1):.3f}")
    print(f"  stop reason        = {s['stop_reason']}")

    gen_active = result["gen_active"]
    if gen_active is not None and gen_active.size:
        print()
        print("Surviving population by generation:")
        for g in range(1, int(gen_active.max()) + 1):
            n_g = int((gen_active == g).sum())
            print(f"  {g}G surviving: N = {n_g}")

    np.savez(
        art_dir / "backup1_mc_result.npz",
        m_active=result["m_active"],
        chi_active=result["chi_active"],
        gen_active=(
            result["gen_active"] if result["gen_active"] is not None
            else np.zeros(0)
        ),
        mergers=result["mergers"],
    )
    print(f"  wrote {art_dir / 'backup1_mc_result.npz'}")

    _plot_active_marginals(result, cfg, fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-hash", default=None)
    parser.add_argument("--n-seed", type=int, default=20_000)
    parser.add_argument(
        "--n-max-mergers", type=int, default=None,
        help="Hard cap on accepted mergers. If given, overrides "
        "--depletion-fraction.",
    )
    parser.add_argument(
        "--depletion-fraction", type=float, default=0.05,
        help="Stop when the active pool has shrunk by this fraction "
        "relative to n_seed. Defaults to 0.05 (early-cascade regime "
        "matching engine ≈ Backup 2 implicit at ε ≈ 0.025). Set to 0.5 "
        "for deeper cascade; 1.0 for full exhaustion.",
    )
    args = parser.parse_args()
    _smoke_main(
        run_hash=args.run_hash,
        n_seed=args.n_seed,
        n_max_mergers=args.n_max_mergers,
        depletion_fraction=args.depletion_fraction,
    )
