#!/usr/bin/env python3
r"""H1-H5 validation for the hybrid analytic-1G + KDE-(2G+) density.

Implements ``demo/PLAN_hybrid_analytic_1g_density.md`` § Validation.

Run from the repo root after sourcing the JAX sandbox::

    source demo/bootstrap_jax_sandbox.sh
    python3 demo/validate_hybrid_1g_density.py

H1  pulse-1 FWHM recovery   hybrid ≈ truth (2.355·σ₁); KDE-only 2-3× wide
H2  convention              ∫∫ N dm dχ ≈ W_total to ~2%
H3  AD                      d(dN/dm at peak)/dσ₁  AD-vs-FD agree
H4  regression              hybrid_1g=False ≡ legacy KDE density bit-for-bit
H5  end-to-end              build_two_pulse_coagulation(solver='mc_jax')
                            → total_rate finite + log_rate_density finite

Self-contained: H1-H4 build a synthetic gen-tagged pool directly so they do
not need the cascade; H5 exercises the full wiring.

Memory note: the Gaussian KDE materialises a dense ``(Q_query × S_sample)``
kernel matrix inside ``density``.  To stay within a small machine's physical
memory this script (a) uses a modest synthetic pool and (b) evaluates the
density in query-point CHUNKS via :func:`_density_chunked` so peak memory is
``chunk × S × 8 bytes`` regardless of grid resolution.  Tune ``CHUNK`` /
``N1`` / ``N2`` down further if needed.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from hierarchical_ad_pop_engine.evaluation.mc_jax import MCEvaluatorJax

# ---- two-pulse truth parameters (methods-paper defaults) ----------------- #
MU1, SIG1 = 10.0, 1.0
MU2, SIG2 = 30.0, 2.0
LAM2 = 0.4
M_MIN, M_MAX = 2.0, 200.0
SPIN_LOC, SPIN_SCALE = 0.0, 0.05
KERNEL = {"a": 0.0, "b": 2.0}
LOG_R_ENV = 0.0

NATAL_MASS_PARAMS = dict(
    mu_1=MU1, sigma_1=SIG1, mu_2=MU2, sigma_2=SIG2,
    lambda_2=LAM2, m_min=M_MIN, m_max=M_MAX,
)
NATAL_SPIN_PARAMS = dict(loc=SPIN_LOC, scale=SPIN_SCALE)

# Memory-frugality knobs (small-machine friendly).
N1, N2 = 1500, 600   # synthetic pool sizes (gen==1 / gen>=2)
CHUNK = 256          # query points per KDE block; peak mem ≈ CHUNK·S·8 bytes


def _density_chunked(ev, m_flat, chi_flat, chunk=CHUNK):
    """Evaluate ``ev.density`` over flat query arrays in memory-bounded blocks.

    ``density`` builds a dense ``(Q × S)`` kernel matrix; looping over Q in
    chunks caps peak memory at ``chunk × S × 8`` bytes.
    """
    m_flat = jnp.asarray(m_flat).ravel()
    chi_flat = jnp.asarray(chi_flat).ravel()
    out = []
    for i in range(0, m_flat.shape[0], chunk):
        out.append(ev.density(m_flat[i:i + chunk], chi_flat[i:i + chunk]))
    return jnp.concatenate(out)


def _synthetic_pool(seed=0, n1=N1, n2=N2):
    """Gen-tagged pool: n1 natal (gen=1) + n2 cascade-remnant (gen=2)."""
    rng = np.random.default_rng(seed)
    # gen==1: two-Gaussian natal mixture, truncated to [m_min, m_max].
    comp = rng.random(n1) < (1.0 - LAM2)
    m1 = np.where(comp, rng.normal(MU1, SIG1, n1), rng.normal(MU2, SIG2, n1))
    m1 = np.clip(m1, M_MIN + 1e-3, M_MAX - 1e-3)
    chi1 = np.clip(np.abs(rng.normal(SPIN_LOC, SPIN_SCALE, n1)), 0.0, 1.0)
    # gen>=2: broad cascade remnants (merger products) ~ around 20 & 40 M_sun
    #         with substantially larger spins (BR ~0.7).
    m2 = np.clip(rng.normal(25.0, 8.0, n2), M_MIN + 1e-3, M_MAX - 1e-3)
    chi2 = np.clip(rng.normal(0.68, 0.06, n2), 0.0, 1.0)

    m = np.concatenate([m1, m2])
    chi = np.concatenate([chi1, chi2])
    gen = np.concatenate([np.ones(n1, int), np.full(n2, 2, int)])
    w = np.ones(m.size)
    mergers = np.zeros((1, 8))
    return (
        jnp.asarray(m), jnp.asarray(chi), jnp.asarray(gen),
        jnp.asarray(w), jnp.asarray(mergers),
    )


def _build(hybrid, m, chi, gen, w, mergers):
    return MCEvaluatorJax.from_run(
        m, chi, mergers,
        log_R_env=LOG_R_ENV, kernel_params=KERNEL, weights=w,
        gen_active=gen if hybrid else None,
        hybrid_1g=hybrid,
        natal_mass_kind="two_gaussian" if hybrid else None,
        natal_mass_params=NATAL_MASS_PARAMS if hybrid else None,
        natal_spin_params=NATAL_SPIN_PARAMS if hybrid else None,
    )


def _dN_dm(ev, m_grid, n_chi=80):
    chi_grid = jnp.linspace(0.0, 1.0, n_chi)
    M, C = jnp.meshgrid(m_grid, chi_grid, indexing="ij")
    N = _density_chunked(ev, M, C).reshape(M.shape)  # (n_m, n_chi)
    return jnp.trapezoid(N, chi_grid, axis=1)        # ∫ dχ → dN/dm


def _fwhm(m_grid, y, m_lo, m_hi):
    m = np.asarray(m_grid); y = np.asarray(y)
    sel = (m >= m_lo) & (m <= m_hi)
    ms, ys = m[sel], y[sel]
    pk = ys.max(); half = pk / 2.0
    i = ys.argmax()
    # left crossing
    li = np.where(ys[:i] <= half)[0]
    ri = np.where(ys[i:] <= half)[0]
    if li.size == 0 or ri.size == 0:
        return np.nan
    lo = np.interp(half, [ys[li[-1]], ys[li[-1] + 1]], [ms[li[-1]], ms[li[-1] + 1]])
    j = i + ri[0]
    hi = np.interp(half, [ys[j], ys[j - 1]], [ms[j], ms[j - 1]])
    return hi - lo


def main():
    m, chi, gen, w, mergers = _synthetic_pool()
    ev_h = _build(True, m, chi, gen, w, mergers)
    ev_k = _build(False, m, chi, gen, w, mergers)
    W_total = float(jnp.sum(w))
    results = {}

    # ---- H1 width recovery --------------------------------------------- #
    m_grid = jnp.linspace(6.0, 16.0, 200)
    fwhm_h = _fwhm(m_grid, _dN_dm(ev_h, m_grid), 6.0, 16.0)
    fwhm_k = _fwhm(m_grid, _dN_dm(ev_k, m_grid), 6.0, 16.0)
    truth = 2.3548 * SIG1
    results["H1"] = (
        abs(fwhm_h - truth) / truth < 0.15 and fwhm_k > 1.4 * truth,
        f"truth={truth:.2f}  hybrid={fwhm_h:.2f}  KDE-only={fwhm_k:.2f} M_sun",
    )

    # ---- H2 convention ∫∫N = W_total ----------------------------------- #
    mg = jnp.linspace(M_MIN, 80.0, 300)
    cg = jnp.linspace(0.0, 1.0, 100)
    M, C = jnp.meshgrid(mg, cg, indexing="ij")
    N_h = _density_chunked(ev_h, M, C).reshape(M.shape)
    integral = float(jnp.trapezoid(jnp.trapezoid(N_h, cg, axis=1), mg))
    results["H2"] = (
        abs(integral - W_total) / W_total < 0.03,
        f"∫∫N={integral:.1f}  W_total={W_total:.1f}  rel={abs(integral-W_total)/W_total:.3%}",
    )

    # ---- H3 AD vs FD on d(dN/dm peak)/dσ₁ ------------------------------ #
    def peak_density(sig1):
        params = dict(NATAL_MASS_PARAMS); params["sigma_1"] = sig1
        ev = MCEvaluatorJax.from_run(
            m, chi, mergers, log_R_env=LOG_R_ENV, kernel_params=KERNEL,
            weights=w, gen_active=gen, hybrid_1g=True,
            natal_mass_kind="two_gaussian",
            natal_mass_params=params, natal_spin_params=NATAL_SPIN_PARAMS,
        )
        return ev.density(jnp.asarray(MU1), jnp.asarray(SPIN_LOC))
    g_ad = float(jax.grad(peak_density)(jnp.asarray(SIG1)))
    h = 1e-4
    g_fd = float((peak_density(jnp.asarray(SIG1 + h)) - peak_density(jnp.asarray(SIG1 - h))) / (2 * h))
    rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
    results["H3"] = (rel < 1e-3, f"AD={g_ad:.4e}  FD={g_fd:.4e}  rel={rel:.2e}")

    # ---- H4 regression: hybrid off ≡ no-hybrid-args construction ------- #
    ev_legacy = MCEvaluatorJax.from_run(
        m, chi, mergers, log_R_env=LOG_R_ENV, kernel_params=KERNEL, weights=w,
    )
    q_m = jnp.linspace(5.0, 50.0, 50)
    q_c = jnp.linspace(0.0, 1.0, 50)
    d_off = ev_k.density(q_m, q_c)
    d_legacy = ev_legacy.density(q_m, q_c)
    maxdiff = float(jnp.max(jnp.abs(d_off - d_legacy)))
    results["H4"] = (maxdiff == 0.0, f"max|hybrid_off - legacy| = {maxdiff:.3e} (want 0)")

    # ---- H5 end-to-end through build_two_pulse_coagulation ------------- #
    try:
        from hierarchical_ad_pop_engine.response.coagulation import (
            build_two_pulse_coagulation,
        )
        resp = build_two_pulse_coagulation(
            mu_1=MU1, sigma_1=SIG1, mu_2=MU2, sigma_2=SIG2, lambda_2=LAM2,
            m_min=M_MIN, m_max=M_MAX,
            natal_spin_loc=SPIN_LOC, natal_spin_scale=SPIN_SCALE,
            kernel_a=KERNEL["a"], kernel_b=KERNEL["b"], log_R_env=LOG_R_ENV,
            solver="mc_jax", mc_v_esc=300.0, mc_n_seed=800,
            mc_jax_key=jax.random.PRNGKey(0), mc_jax_gate="soft",
        )
        tr = float(resp.mc_jax_response.total_rate(n_quad=32))
        lrd = float(resp.mc_jax_response.log_rate_density(
            jnp.asarray(MU1), jnp.asarray(MU2),
            jnp.asarray(0.0), jnp.asarray(0.0), {},
        ))
        ok5 = np.isfinite(tr) and tr > 0 and np.isfinite(lrd)
        results["H5"] = (ok5, f"total_rate={tr:.4e}  log_rate_density={lrd:.3f}")
    except Exception as e:  # noqa: BLE001
        results["H5"] = (False, f"end-to-end raised: {type(e).__name__}: {e}")

    print("\n=== Hybrid analytic-1G density validation ===")
    allok = True
    for k in ("H1", "H2", "H3", "H4", "H5"):
        ok, msg = results[k]
        allok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {msg}")
    print("=== {} ===".format("ALL PASS" if allok else "SOME FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
