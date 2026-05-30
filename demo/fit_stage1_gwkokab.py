r"""Stage-1 single-pulse identifiability + recovery through the gwkokab path.

The "lowest-hanging fruit" check: with a multi-sample mock catalogue drawn from
a known truth ``θ*``, can the **gwkokab analytical-Poisson likelihood** — fed by
the coagulation engine via the new ``inference/stage1_wiring`` + ``selection`` —
recover the natal-population parameters and the interaction (pairing) model, and
is the origin *unique* (a single likelihood peak at truth) or degenerate?

Rather than a full SVI/NUTS run (whose ``AutoNormal`` width underestimate +
Gaussian-KDE χ-oversmoothing make the M4 6-param coverage a known xfail), this
script answers identifiability *directly*:

1. **Profile** the log-likelihood along each parameter axis through ``θ*``
   (others held at truth).  A clear peak at the truth value ⇒ that parameter is
   locally identified; a flat profile ⇒ unconstrained by the data.
2. **MAP**: gradient-ascend the log-likelihood from a perturbed start (the AD
   payoff — gradients flow through the whole cascade) and compare the recovered
   point to ``θ*``.

Everything routes through the REAL ``gwkokab.inference.poissonlikelihood_utils.
analytical_poisson_likelihood_fn`` (so it exercises the actual fit contract),
with the engine built via ``make_coagulation_dist_fn`` (``solver="mc_jax"``) and
the flat-selection ``coagulation_poisson_mean_estimator`` by default
(``--selection vt`` swaps in the ``M_chirp^{15/6}`` VT estimator).

Run (needs the full engine+gwkokab stack; in the Cowork VM use the RIFT stub —
see HANDOFF_PORTABLE §4.2):

    python3 demo/fit_stage1_gwkokab.py --n-obs 40 --n-samples 5 \
        --n-seed 400 --k-max 3 --map-steps 150

NB the absolute ``log_rate`` rides on ``MCResponseJax.total_rate``, whose
metric bias at small ``n_seed`` is the open §4.3 thread; we calibrate ``θ*``'s
``log_rate`` against that SAME metric so its identifiability is self-consistent
(the rate scale cancels), and focus the headline on the SHAPE parameters.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_DEMO = Path(__file__).resolve().parent
_REPO = _DEMO.parent
for p in (str(_DEMO), str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_gwkokab_fit_stage1 as demo  # noqa: E402
from mock_catalog import make_mock_posteriors  # noqa: E402

from hierarchical_ad_pop_engine.inference.gwkokab_bridge import (  # noqa: E402
    coagulation_poisson_mean_estimator,
    make_coagulation_dist_fn,
)
from hierarchical_ad_pop_engine.inference.selection import (  # noqa: E402
    make_vt_poisson_mean_estimator,
    vt_powerlaw,
)

PARAM_NAMES = ["mu_natal", "sigma_natal", "natal_spin_loc",
               "natal_spin_scale", "kernel_b", "log_rate"]


def _theta_vec(d):
    return jnp.asarray([float(d[k]) for k in PARAM_NAMES])


def _unpack(theta):
    return {k: theta[i] for i, k in enumerate(PARAM_NAMES)}


def sample_from_model(model, n_events, key, *, m_lo, m_hi, n_sm=20, n_sc=8):
    r"""Self-consistent mock: inverse-CDF draw of ``(N,4)`` true events from the
    FIT model's own pair intensity ``exp(model.log_prob)`` on a coarse
    ``m₁×m₂×χ₁×χ₂`` grid (geomspaced mass, linear χ), jittered within cells.

    Generating the data from the SAME model we fit removes the
    implicit-vs-mc_jax solver mismatch, so a profile peak away from θ* is then a
    genuine identifiability statement, not a forward-model discrepancy.
    """
    log_m = jnp.linspace(jnp.log(m_lo), jnp.log(m_hi), n_sm)
    m_grid = jnp.exp(log_m)
    chi = jnp.linspace(0.0, 1.0, n_sc)
    M1, M2, C1, C2 = jnp.meshgrid(m_grid, m_grid, chi, chi, indexing="ij")
    nodes = jnp.stack([M1, M2, C1, C2], axis=-1)
    logp = model.log_prob(nodes)
    p = jnp.exp(logp - jnp.max(logp)).reshape(-1)
    p = p / jnp.sum(p)

    k_idx, k_jm1, k_jm2, k_jc1, k_jc2 = jax.random.split(key, 5)
    idx = jax.random.choice(k_idx, p.size, shape=(n_events,), p=p)
    i1, i2, j1, j2 = jnp.unravel_index(idx, (n_sm, n_sm, n_sc, n_sc))
    dlog = float(log_m[1] - log_m[0])
    dchi = float(chi[1] - chi[0])
    # jitter: uniform in log-m within the mass cell, linear within the χ cell.
    m1 = jnp.exp(log_m[i1] + dlog * (jax.random.uniform(k_jm1, (n_events,)) - 0.5))
    m2 = jnp.exp(log_m[i2] + dlog * (jax.random.uniform(k_jm2, (n_events,)) - 0.5))
    c1 = jnp.clip(chi[j1] + dchi * (jax.random.uniform(k_jc1, (n_events,)) - 0.5), 0.0, 1.0)
    c2 = jnp.clip(chi[j2] + dchi * (jax.random.uniform(k_jc2, (n_events,)) - 0.5), 0.0, 1.0)
    hi = m1 >= m2
    return jnp.stack([jnp.where(hi, m1, m2), jnp.where(hi, m2, m1),
                      jnp.where(hi, c1, c2), jnp.where(hi, c2, c1)], axis=-1)


def build_logL(events, ln_offsets, dist_fn, estimator, *, T_obs, n_quad):
    r"""θ-vector → scalar gwkokab analytical-Poisson log-likelihood (jit-able)."""
    from gwkokab.inference.poissonlikelihood_utils import (
        analytical_poisson_likelihood_fn,
    )
    pmean = {"T_obs": float(T_obs), "n_quad": int(n_quad)}

    def logL(theta):
        model = dist_fn(**_unpack(theta))
        return analytical_poisson_likelihood_fn(
            model, estimator, events, ln_offsets, pmean, None)

    return logL


# --- MAP reparametrisation: unconstrained u -> physical θ (engine-safe) ------
def _to_physical(u):
    mu = 5.0 + 45.0 * jax.nn.sigmoid(u[0])           # (5, 50)
    sigma = jnp.clip(jnp.exp(u[1]), 0.1, 20.0)        # > 0
    spin_loc = 0.3 * jax.nn.sigmoid(u[2])             # (0, 0.3)
    spin_scale = jnp.clip(jnp.exp(u[3]), 0.01, 0.5)   # > 0
    kernel_b = u[4]                                   # free
    log_rate = u[5]                                   # free
    return jnp.stack([mu, sigma, spin_loc, spin_scale, kernel_b, log_rate])


def _from_physical(theta):
    r"""Invert :func:`_to_physical` (so the MAP can start near a physical θ)."""
    mu, sigma, sl, ss, kb, lr = [float(theta[i]) for i in range(6)]
    logit = lambda x: float(np.log(x / (1.0 - x)))
    return jnp.asarray([
        logit((mu - 5.0) / 45.0), float(np.log(sigma)),
        logit(sl / 0.3), float(np.log(ss)), kb, lr,
    ])


def _profile_ranges(theta_star):
    mu, sg, sl, ss, kb, lr = [float(theta_star[i]) for i in range(6)]
    return {
        "mu_natal": np.linspace(max(5.0, mu - 4), mu + 4, 13),
        "sigma_natal": np.linspace(0.5, 3.0, 13),
        "natal_spin_loc": np.linspace(0.0, 0.25, 13),
        "natal_spin_scale": np.linspace(0.03, 0.25, 13),
        "kernel_b": np.linspace(kb - 3, kb + 3, 13),
        "log_rate": np.linspace(lr - 2.0, lr + 2.0, 13),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-seed", type=int, default=400)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--n-quad", type=int, default=32)
    ap.add_argument("--map-steps", type=int, default=150)
    ap.add_argument("--map-lr", type=float, default=0.05)
    ap.add_argument("--selection", choices=["flat", "vt"], default="flat")
    ap.add_argument("--gen", choices=["self", "implicit"], default="self",
                    help="mock-event generator: 'self' = the fit mc_jax model "
                         "(clean identifiability), 'implicit' = the implicit "
                         "oracle (cross-solver recovery).")
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--no-map", action="store_true")
    args = ap.parse_args()

    fixed = demo.FIXED
    print(f"[cfg] n_obs={args.n_obs} n_samples={args.n_samples} "
          f"n_seed={args.n_seed} k_max={args.k_max} selection={args.selection}")

    # --- 1. truth (implicit oracle for event positions) + θ* shape -----------
    fit_grid = demo._grid_for_fit()
    truth_implicit = demo.build_truth_model_and_calibrate_rate(
        args.n_obs, grid=fit_grid)
    demo.THETA_STAR["natal_spin_loc"] = 0.05  # boundary-corrected (M4 rationale)
    theta_star = dict(demo.THETA_STAR)

    # --- 2. dist_fn + estimator (the new wiring) -----------------------------
    key = jax.random.PRNGKey(args.seed)
    k_mc, k_cat, k_map = jax.random.split(key, 3)
    dist_fn = make_coagulation_dist_fn(
        v_esc=fixed["v_esc"], m_min=fixed["m_min"], m_max=fixed["m_max"],
        kernel_a=fixed["kernel_a"], k_max=args.k_max, n_seed=args.n_seed,
        depletion_fraction=0.10, gate="soft", grid=fit_grid, mc_jax_key=k_mc)
    if args.selection == "vt":
        estimator = make_vt_poisson_mean_estimator(
            vt_powerlaw(), m_min=fixed["m_min"], m_max=fixed["m_max"])
    else:
        estimator = coagulation_poisson_mean_estimator

    # --- 3. self-consistent log_rate calibration against the FIT's μ metric --
    probe = dist_fn(**{**theta_star, "log_rate": 0.0})
    mu_unit, _ = estimator(probe, T_obs=1.0, n_quad=args.n_quad)
    log_rate_star = float(jnp.log(args.n_obs / mu_unit))
    theta_star["log_rate"] = log_rate_star
    print(f"[cal] μ(logR=0)={float(mu_unit):.4g} → log_rate* = {log_rate_star:.3f} "
          f"(targets {args.n_obs} detected)")

    # --- 4. multi-sample mock catalogue --------------------------------------
    m_hi = fixed["m_max"] * 2.5
    if args.gen == "self":
        from mock_catalog import scatter_to_posteriors
        truth_model = dist_fn(**theta_star)
        k_true, k_scat = jax.random.split(k_cat, 2)
        true_pts = sample_from_model(
            truth_model, args.n_obs, k_true, m_lo=fixed["m_min"], m_hi=m_hi)
        events = scatter_to_posteriors(true_pts, args.n_samples, k_scat, m_hi=m_hi)
        ln_offsets = jnp.zeros((args.n_obs, args.n_samples))
    else:
        events, ln_offsets = make_mock_posteriors(
            truth_implicit, args.n_obs, args.n_samples, k_cat, m_hi=m_hi)
    print(f"[cat] gen={args.gen}  events {tuple(events.shape)}  "
          f"ln_offsets {tuple(ln_offsets.shape)}")

    # --- 5. jitted log-likelihood -------------------------------------------
    logL = build_logL(events, ln_offsets, dist_fn, estimator,
                      T_obs=1.0, n_quad=args.n_quad)
    logL_j = jax.jit(logL)
    theta0 = _theta_vec(theta_star)
    ll_star = float(logL_j(theta0))
    print(f"[ll ] logL(θ*) = {ll_star:.3f}")

    # --- 6. profiles ---------------------------------------------------------
    print("\n=== 1-D LOG-LIKELIHOOD PROFILES (through θ*) ===")
    ranges = _profile_ranges(theta0)
    profiles = {}
    ident = {}
    for i, name in enumerate(PARAM_NAMES):
        xs = ranges[name]
        ys = []
        for x in xs:
            th = theta0.at[i].set(float(x))
            ys.append(float(logL_j(th)))
        ys = np.asarray(ys)
        profiles[name] = (xs, ys)
        peak = float(xs[int(np.argmax(ys))])
        truth = float(theta0[i])
        span = float(ys.max() - ys.min())
        # curvature proxy: drop from the peak across the scanned window
        ident[name] = (peak, truth, span)
        flag = "IDENTIFIED" if span > 2.0 else "weak/flat"
        print(f"  {name:18s} truth={truth:+.3f}  peak={peak:+.3f}  "
              f"Δ(logL) over scan={span:6.2f}  [{flag}]")

    # --- 7. MAP (gradient ascent from a perturbed start) ---------------------
    map_recovered = None
    if not args.no_map:
        print(f"\n=== MAP (gradient ascent, {args.map_steps} steps) ===")
        try:
            import optax
        except Exception as exc:  # pragma: no cover
            print(f"  [skip MAP] optax unavailable: {exc}")
        else:
            # init at a MILD perturbation of θ* (a far start lands in a
            # −inf region — the intensity underflows where no data sit).
            u0 = _from_physical(theta0) + jnp.asarray(
                [0.4, 0.25, 0.4, 0.25, 0.6, 0.4])
            neg = lambda u: -logL(_to_physical(u))
            negj = jax.jit(jax.value_and_grad(neg))
            opt = optax.adam(args.map_lr)
            st = opt.init(u0)
            u = u0
            for step in range(args.map_steps):
                val, g = negj(u)
                upd, st = opt.update(g, st)
                u = optax.apply_updates(u, upd)
                if step % 25 == 0 or step == args.map_steps - 1:
                    print(f"   step {step:4d}  -logL = {float(val):.3f}")
            map_recovered = np.asarray(_to_physical(u))
            print("\n   param              truth      MAP       Δ")
            for i, name in enumerate(PARAM_NAMES):
                t = float(theta0[i]); m = float(map_recovered[i])
                print(f"   {name:18s} {t:+8.3f} {m:+8.3f} {m - t:+8.3f}")

    # --- 8. figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, name in zip(axes.ravel(), PARAM_NAMES):
            xs, ys = profiles[name]
            ax.plot(xs, ys - ys.max(), "o-", color="C0")
            ax.axvline(float(theta0[PARAM_NAMES.index(name)]), color="C3",
                       ls="--", label="truth")
            ax.set_title(name)
            ax.set_ylabel(r"$\log L - \max$")
            ax.set_ylim(-15, 1)
            ax.grid(True, ls=":", alpha=0.4)
        axes.ravel()[0].legend()
        fig.suptitle("Stage-1 single-pulse identifiability — gwkokab "
                     f"analytical-Poisson ({args.selection} selection, "
                     f"N={args.n_obs}, S={args.n_samples})")
        fig.tight_layout()
        out = _DEMO / "figures" / "stage1_identifiability_profiles.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out, dpi=140)
        print(f"\n[fig] {out}")
    except Exception as exc:  # pragma: no cover
        print(f"[fig] skipped ({exc})")

    print("\n=== SUMMARY ===")
    strong = [n for n in PARAM_NAMES if ident[n][2] > 2.0]
    weak = [n for n in PARAM_NAMES if ident[n][2] <= 2.0]
    print(f"  identified (peaked logL): {', '.join(strong) or '—'}")
    print(f"  weak / flat:              {', '.join(weak) or '—'}")
    return profiles, ident, map_recovered


if __name__ == "__main__":
    main()
