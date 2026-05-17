"""Demo 1 — bare response function R(m, χ) at fixed v_esc.

The simplest possible end-to-end exercise of the engine: no formation-rate
convolution, no redshift dependence, no formation amplitude. We just build the
generational coagulation response in one dense environment and look at the
per-generation primary-mass spectrum.

This is the analogue of Fig. 2 of Doctor et al. (2020, ApJ 893:35): a stack
of 1G, 2G, 3G, ... mass-function curves that broaden and shift to higher mass
with each generation, with the high-mass tail set by retention against the
chosen escape velocity.

Run:

    python demo/demo_response_doctor2020.py

Outputs:

    demo/figures/response_mass_per_generation.png
    demo/figures/response_spin_per_generation.png
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from hierarchical_ad_pop_engine.response.coagulation import build_default_coagulation


FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Build the response function.
# ---------------------------------------------------------------------------
#
# Defaults are an AGN-disk-flavoured single-environment model:
#   - Talbot-Thrane PL+Peak natal IMF (alpha=3, m_min=5, m_max=100, Gaussian
#     peak at 33 M⊙ to mimic the GWTC-3 "bump"),
#   - TruncatedNormal natal spin centred at 0.1 with width 0.15,
#   - Pairing kernel q^b with b=2 (mass-ratio biased),
#   - Tracking up to k_max=4 generations.
response = build_default_coagulation(k_max=4)

# Fixed environment: a dense AGN disk inner region, v_esc = 500 km/s.
# Doctor+2020 used a range; we pick one value here for the introductory plot.
v_esc = jnp.asarray(500.0)

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
# n_k(m, χ) is a density in (m, χ); integrating over χ gives dN_k/dm.
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

for k in range(K):
    y = dN_dm[k]
    # Skip generations that are essentially empty (avoid log(0) artefacts).
    if y.max() < 1e-25:
        continue
    ax.loglog(m, y, color=colors[k], lw=2.0, label=f"{k+1}G")

ax.loglog(m, dN_dm_total, color="k", lw=1.5, ls="--", label="Total (Σ generations)")

ax.set_xlabel(r"primary mass $m \;\; [M_\odot]$")
ax.set_ylabel(r"$dN_k/dm$ (per env, arb. norm.)")
ax.set_title(
    rf"Coagulation response, single environment, $v_{{\rm esc}}={float(v_esc):.0f}$ km/s"
    "\n"
    "(cf. Doctor+2020 Fig. 2)"
)
ax.set_xlim(m.min(), m.max())
ymax = dN_dm_total.max()
ax.set_ylim(ymax * 1e-6, ymax * 3)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="upper right", frameon=False)

fig.tight_layout()
out_mass = FIG_DIR / "response_mass_per_generation.png"
fig.savefig(out_mass, dpi=150)
plt.close(fig)
print(f"wrote {out_mass}")


# ---------------------------------------------------------------------------
# 4. Companion: spin spectrum per generation (marginalised over m).
# ---------------------------------------------------------------------------
# Integrate over m using d(log m): dN_k/dχ = ∫ n_k(m, χ) m d(log m).
dlog_m_edges = np.diff(np.asarray(grid.log_m))
dlog_m_full = np.concatenate(
    [dlog_m_edges[:1], 0.5 * (dlog_m_edges[:-1] + dlog_m_edges[1:]), dlog_m_edges[-1:]]
)
dm_full = m * dlog_m_full

dN_dchi = np.einsum("kmc,m->kc", np.asarray(n_k), dm_full)  # (K, N_chi)
dN_dchi_total = dN_dchi.sum(axis=0)

fig, ax = plt.subplots(figsize=(7.5, 5.0))
for k in range(K):
    y = dN_dchi[k]
    # Normalise each generation independently so the shape is comparable
    # — the absolute weights are wildly different across k.
    if y.max() < 1e-25:
        continue
    ax.plot(chi, y / y.max(), color=colors[k], lw=2.0, label=f"{k+1}G")

ax.plot(
    chi,
    dN_dchi_total / dN_dchi_total.max(),
    color="k",
    lw=1.5,
    ls="--",
    label="Total (Σ generations)",
)

ax.set_xlabel(r"single-BH spin $\chi$")
ax.set_ylabel(r"$dN_k/d\chi$ (peak-normalised)")
ax.set_title(
    rf"Coagulation response spin spectrum, $v_{{\rm esc}}={float(v_esc):.0f}$ km/s"
)
ax.set_xlim(0.0, 1.0)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(loc="upper right", frameon=False)

fig.tight_layout()
out_spin = FIG_DIR / "response_spin_per_generation.png"
fig.savefig(out_spin, dpi=150)
plt.close(fig)
print(f"wrote {out_spin}")


# ---------------------------------------------------------------------------
# 5. Numerical summary so the user sees something on stdout.
# ---------------------------------------------------------------------------
print()
print(f"v_esc                  = {float(v_esc):.1f} km/s")
print(f"k_max                  = {K}")
print(f"grid                   = (N_m={m.size}, N_chi={chi.size})")
total_per_gen = dN_dm.sum(axis=1) * dm_full.sum() / dm_full.sum()  # ≈ ∫ dN_k/dm dm
# More carefully: ∫ dN_k/dm dm using d log m weights.
N_per_gen = (dN_dm * dm_full[None, :]).sum(axis=1)
N_total = N_per_gen.sum()
print()
print("Generation-by-generation number budget (arb. units):")
for k in range(K):
    frac = N_per_gen[k] / N_total if N_total > 0 else 0.0
    print(f"  {k+1}G: N = {N_per_gen[k]:.3e}  ({100*frac:5.2f}%)")
print(f"  Σ : N = {N_total:.3e}")
