r"""Histogram head-to-head: discrete K-rejection vs B.2 soft gate (numpy).

Runs in-VM (no JAX).  This is the SHAPE-parity view the workstation's
`demo/forward_modeling_mc_vs_mc.py --gate soft` produces against the
actual JAX particle_mc_jax_soft.  Here both cascades are numpy and share
the simplified physics, isolating the gate mechanism.  HISTOGRAMS ONLY —
no KDE.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_soft_gate_derivation import (
    CFG, A, B, V_ESC, sample_natal, cascade_discrete, cascade_soft,
)

N_INIT = 20000
D = 0.10
N_STEPS = int(3 * D * N_INIT)

rng_d = np.random.default_rng(20260529)
m0, chi0 = sample_natal(rng_d, N_INIT, **CFG)
disc = cascade_discrete(rng_d, m0, chi0, A, B, V_ESC, D)

rng_s = np.random.default_rng(20260529)
m0s, chi0s = sample_natal(rng_s, N_INIT, **CFG)
soft = cascade_soft(rng_s, m0s, chi0s, A, B, V_ESC, D, N_STEPS)

# --- surviving m and chi histograms ---
m_edges = np.geomspace(CFG["m_min"], CFG["m_max"] * 2, 41)
m_ctr = np.sqrt(m_edges[:-1] * m_edges[1:])
m_bw = np.diff(m_edges)
chi_edges = np.linspace(0, 1, 51)
chi_ctr = 0.5 * (chi_edges[:-1] + chi_edges[1:])
chi_bw = np.diff(chi_edges)[0]

wd = np.ones(disc["m_active"].size)
ws = soft["occ_active"] * soft["w_active"]

hmd, _ = np.histogram(disc["m_active"], bins=m_edges, weights=wd)
hms, _ = np.histogram(soft["m_active"], bins=m_edges, weights=ws)
hcd, _ = np.histogram(disc["chi_active"], bins=chi_edges, weights=wd)
hcs, _ = np.histogram(soft["chi_active"], bins=chi_edges, weights=ws)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.loglog(m_ctr, np.maximum(hmd / m_bw, 1e-30), drawstyle="steps-mid",
          color="C0", lw=2.0, label=f"discrete K-reject (N={disc['m_active'].size})")
ax.loglog(m_ctr, np.maximum(hms / m_bw, 1e-30), drawstyle="steps-mid",
          color="C3", lw=2.0, ls="--", label=f"B.2 soft gate (Σwocc={ws.sum():.0f})")
ax.set_xlabel(r"single-BH mass $m\;[M_\odot]$")
ax.set_ylabel(r"$dN/dm$ (raw histogram, per $M_\odot$)")
ax.set_title("Surviving pool $dN/dm$")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.set_ylim(bottom=max(hmd.max(), hms.max()) / m_bw.mean() * 1e-5)
ax.legend(frameon=False, fontsize="small")

ax = axes[1]
ax.semilogy(chi_ctr, np.maximum(hcd / chi_bw, 1e-30), drawstyle="steps-mid",
            color="C0", lw=2.0, label="discrete K-reject")
ax.semilogy(chi_ctr, np.maximum(hcs / chi_bw, 1e-30), drawstyle="steps-mid",
            color="C3", lw=2.0, ls="--", label="B.2 soft gate")
ax.set_xlabel(r"single-BH spin $\chi$")
ax.set_ylabel(r"$dN/d\chi$ (raw histogram)")
ax.set_title("Surviving pool $dN/d\\chi$")
ax.set_xlim(0, 1)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(frameon=False, fontsize="small")

fig.suptitle(
    f"B.2 soft gate vs discrete K-rejection — numpy shape parity "
    f"(n_init={N_INIT}, D={D})\n"
    f"depletion: discrete={disc['depletion']:.4f} soft={soft['depletion']:.4f}  |  "
    f"W_total: discrete={disc['W_total']:.0f} soft={soft['W_total']:.0f}  "
    f"(raw histograms, no KDE)"
)
fig.tight_layout()
out = Path(__file__).resolve().parent / "b2_histogram_headtohead.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
print(f"depletion discrete={disc['depletion']:.4f} soft={soft['depletion']:.4f}")
print(f"W_total   discrete={disc['W_total']:.0f}   soft={soft['W_total']:.0f}")
