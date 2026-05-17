"""Demo 1 — impulse response of the coagulation operator.

We exercise the engine in its cleanest pedagogical regime: a narrow Gaussian
natal IMF centred at 8 M⊙, low natal spin, and a high-retention environment
(v_esc = 1000 km/s) so essentially every remnant survives.

The forward map is then a coagulation impulse response. Successive generations
should pile up as separate, increasingly broad peaks at the expected
"merger product of two k-th gen BHs" masses:

    1G:  m ≈ 8                        (natal Gaussian)
    2G:  m ≈ M_rem(8, 8, χ=0, χ=0)             ≈ 15  M⊙
    3G:  m ≈ M_rem(2G, 2G, χ≈0.69)              ≈ 28  M⊙
    4G:  m ≈ M_rem(3G, 3G, χ≈0.85)              ≈ 53  M⊙

(`M_rem` is the Lousto-Zlochower aligned-spin remnant-mass fit; the canonical
χ_rem ≈ 0.6865 for an equal-mass, nonspinning merger is the Barausse-Rezzolla
value.)

The spin spectrum tells the matched story: 1G is the natal half-normal near
0, 2G accumulates near the canonical 0.69, and higher generations climb past
0.8 — the standard "spin cascade" of e.g. Yang et al. 2019 (arXiv:1911.04424).

Run:

    python demo/demo_response_doctor2020.py

Outputs:

    demo/figures/impulse_response_mass_per_generation.png
    demo/figures/impulse_response_spin_per_generation.png
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from hierarchical_ad_pop_engine.response.coagulation import build_gaussian_coagulation


FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Build the impulse-response model.
# ---------------------------------------------------------------------------
#
# Natal IMF: truncated Gaussian at μ = 8 M⊙, σ = 1 M⊙, on [4, 80] M⊙.
# Implemented via `jax.scipy.stats.truncnorm.logpdf` — the same primitive
# gwkokab's own ``GaussianPrimaryMassRatio`` uses internally.
#
# Natal spin: narrow half-normal near zero (scale 0.05). 1G is effectively
# nonspinning, so 2G remnants land at the canonical Barausse-Rezzolla value
# χ_rem(q=1, χ=0) ≈ 0.6865.
#
# Environment: v_esc = 1000 km/s — high enough that even spinning, asymmetric
# remnant kicks fail to escape and we retain through k_max generations.
response = build_gaussian_coagulation(
    mu=8.0,
    sigma=1.0,
    m_min=4.0,
    m_max=80.0,
    natal_spin_loc=0.0,
    natal_spin_scale=0.05,
    kernel_a=0.0,
    kernel_b=2.0,
    log_R_env=0.0,
    k_max=4,
)

v_esc = jnp.asarray(1000.0)

# log n_k(m, χ) for k = 1..k_max; shape (K, N_m, N_chi).
log_n_k = response._solve(v_esc)
n_k = jnp.exp(log_n_k)  # linear, (K, N_m, N_chi)

grid = response.grid
m = np.asarray(grid.m)          # (N_m,)
chi = np.asarray(grid.chi)      # (N_chi,)
K = log_n_k.shape[0]


# ---------------------------------------------------------------------------
# 2. Marginalise over χ → mass spectrum per generation.
# ---------------------------------------------------------------------------
dchi_edges = np.diff(chi)
dchi_full = np.concatenate(
    [dchi_edges[:1], 0.5 * (dchi_edges[:-1] + dchi_edges[1:]), dchi_edges[-1:]]
)
dN_dm = np.einsum("kmc,c->km", np.asarray(n_k), dchi_full)  # (K, N_m)
dN_dm_total = dN_dm.sum(axis=0)


# ---------------------------------------------------------------------------
# 3. Plot per-generation mass spectrum.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.0))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, K))

# Expected impulse-response locations (rough; from M_rem table above) — drawn
# as light vertical guides so the user can see the alignment instantly.
expected_peaks = [8.0, 15.0, 28.0, 53.0]
for k, mp in enumerate(expected_peaks[:K]):
    ax.axvline(mp, color=colors[k], lw=0.7, alpha=0.4, ls=":")

for k in range(K):
    y = dN_dm[k]
    if y.max() < 1e-25:
        continue
    # Peak-normalise each generation so the cascade is visible — the absolute
    # weights drop by orders of magnitude per generation.
    ax.semilogx(m, y / y.max(), color=colors[k], lw=2.0, label=f"{k+1}G")

ax.set_xlabel(r"primary mass $m \;\; [M_\odot]$")
ax.set_ylabel(r"$dN_k/dm$ (peak-normalised per generation)")
ax.set_title(
    rf"Coagulation impulse response from Gaussian natal at 8 $M_\odot$, "
    rf"$v_{{\rm esc}}={float(v_esc):.0f}$ km/s"
    "\n"
    "(narrow natal IMF → cascade of peaks at $\\sim$8, 15, 28, 53 $M_\\odot$)"
)
ax.set_xlim(m.min(), m.max())
ax.set_ylim(0.0, 1.08)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="upper right", frameon=False, title="Generation")

fig.tight_layout()
out_mass = FIG_DIR / "impulse_response_mass_per_generation.png"
fig.savefig(out_mass, dpi=150)
plt.close(fig)
print(f"wrote {out_mass}")


# ---------------------------------------------------------------------------
# 4. Companion: spin spectrum per generation (marginalised over m).
# ---------------------------------------------------------------------------
dlog_m_edges = np.diff(np.asarray(grid.log_m))
dlog_m_full = np.concatenate(
    [dlog_m_edges[:1], 0.5 * (dlog_m_edges[:-1] + dlog_m_edges[1:]), dlog_m_edges[-1:]]
)
dm_full = m * dlog_m_full

dN_dchi = np.einsum("kmc,m->kc", np.asarray(n_k), dm_full)  # (K, N_chi)

fig, ax = plt.subplots(figsize=(7.5, 5.0))
ax.axvline(0.6865, color="0.4", lw=0.7, ls=":", label=r"$\chi_{\rm rem}(q{=}1, \chi{=}0)\!=\!0.6865$")
for k in range(K):
    y = dN_dchi[k]
    if y.max() < 1e-25:
        continue
    ax.plot(chi, y / y.max(), color=colors[k], lw=2.0, label=f"{k+1}G")

ax.set_xlabel(r"single-BH spin $\chi$")
ax.set_ylabel(r"$dN_k/d\chi$ (peak-normalised)")
ax.set_title(
    rf"Spin cascade, $v_{{\rm esc}}={float(v_esc):.0f}$ km/s"
)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.08)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="upper right", frameon=False, title="Generation")

fig.tight_layout()
out_spin = FIG_DIR / "impulse_response_spin_per_generation.png"
fig.savefig(out_spin, dpi=150)
plt.close(fig)
print(f"wrote {out_spin}")


# ---------------------------------------------------------------------------
# 5. Per-generation number budget on stdout.
# ---------------------------------------------------------------------------
N_per_gen = (dN_dm * dm_full[None, :]).sum(axis=1)
N_total = N_per_gen.sum()
print()
print(f"v_esc                  = {float(v_esc):.1f} km/s")
print(f"k_max                  = {K}")
print(f"grid                   = (N_m={m.size}, N_chi={chi.size})")
print()
print("Generation-by-generation number budget (arb. units):")
for k in range(K):
    frac = N_per_gen[k] / N_total if N_total > 0 else 0.0
    print(f"  {k+1}G: N = {N_per_gen[k]:.3e}  ({100*frac:5.2f}%)")
print(f"  Σ : N = {N_total:.3e}")
