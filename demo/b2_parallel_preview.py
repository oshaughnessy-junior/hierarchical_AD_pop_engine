r"""Numpy preview of the parallel plot demo (runs in-VM, no JAX).

Mirrors demo/forward_modeling_mc_parallel.py's layout but with the two
NUMPY cascade algorithms that ARE the B.1/B.2 distinction:
  * discrete K-rejection  = B.1's forward dynamics in expectation
  * soft occupancy gate   = B.2
Overlay (dN/dm, dN/dchi) + fractional residual vs the discrete oracle.
Histograms only.  The workstation 3-way (numpy + actual JAX B.1 + JAX
B.2) is demo/forward_modeling_mc_parallel.py.
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

N_INIT, D = 20000, 0.10
N_STEPS = int(3 * D * N_INIT)

rng = np.random.default_rng(20260529)
m0, chi0 = sample_natal(rng, N_INIT, **CFG)
disc = cascade_discrete(rng, m0, chi0, A, B, V_ESC, D)
rng2 = np.random.default_rng(20260529)
m0s, chi0s = sample_natal(rng2, N_INIT, **CFG)
soft = cascade_soft(rng2, m0s, chi0s, A, B, V_ESC, D, N_STEPS)

ws = soft["occ_active"] * soft["w_active"]
wd = np.ones(disc["m_active"].size)

m_edges = np.geomspace(CFG["m_min"], CFG["m_max"] * 2, 41)
m_ctr = np.sqrt(m_edges[:-1] * m_edges[1:])
chi_edges = np.linspace(0, 1, 51)
chi_ctr = 0.5 * (chi_edges[:-1] + chi_edges[1:])


def hm(x, w):
    h, _ = np.histogram(x, bins=m_edges, weights=w)
    return h / np.diff(m_edges)


def hc(x, w):
    h, _ = np.histogram(x, bins=chi_edges, weights=w)
    return h / np.diff(chi_edges)[0]


hmd, hms = hm(disc["m_active"], wd), hm(soft["m_active"], ws)
hcd, hcs = hc(disc["chi_active"], wd), hc(soft["chi_active"], ws)

fig, axes = plt.subplots(2, 2, figsize=(15, 9),
                         gridspec_kw={"height_ratios": [3, 1]})
ax = axes[0, 0]
ax.loglog(m_ctr, np.maximum(hmd, 1e-30), drawstyle="steps-mid", color="C0",
          lw=2.0, label="discrete K-reject  (= B.1 in expectation)")
ax.loglog(m_ctr, np.maximum(hms, 1e-30), drawstyle="steps-mid", color="C3",
          ls="--", lw=2.0, label="B.2 soft occupancy gate")
ax.set_ylabel(r"$dN/dm$ (per $M_\odot$)")
ax.set_title("Surviving pool $dN/dm$ — raw histograms")
ax.set_ylim(bottom=max(hmd.max(), hms.max()) * 1e-5)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(frameon=False, fontsize="small")
ax = axes[1, 0]
ax.semilogx(m_ctr, (hms - hmd) / np.maximum(hmd, 1e-30), drawstyle="steps-mid",
            color="C3", ls="--", lw=1.6, label="(B.2 − discrete)/discrete")
ax.axhline(0, color="C0", lw=1.0)
ax.set_xlabel(r"$m\;[M_\odot]$"); ax.set_ylabel("frac. resid.")
ax.set_ylim(-1, 1); ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(frameon=False, fontsize="x-small")

ax = axes[0, 1]
ax.semilogy(chi_ctr, np.maximum(hcd, 1e-30), drawstyle="steps-mid", color="C0",
            lw=2.0, label="discrete K-reject")
ax.semilogy(chi_ctr, np.maximum(hcs, 1e-30), drawstyle="steps-mid", color="C3",
            ls="--", lw=2.0, label="B.2 soft gate")
ax.axvline(0.6865, color="0.4", lw=0.8, ls=":", label=r"$\chi_{\rm rem}(1,0)$")
ax.set_ylabel(r"$dN/d\chi$"); ax.set_title(r"Surviving pool $dN/d\chi$")
ax.set_xlim(0, 1); ax.set_ylim(bottom=max(hcd.max(), hcs.max()) * 1e-4)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(frameon=False, fontsize="small")
ax = axes[1, 1]
ax.plot(chi_ctr, (hcs - hcd) / np.maximum(hcd, 1e-30), drawstyle="steps-mid",
        color="C3", ls="--", lw=1.6, label="(B.2 − discrete)/discrete")
ax.axhline(0, color="C0", lw=1.0)
ax.set_xlabel(r"single-BH spin $\chi$"); ax.set_ylabel("frac. resid.")
ax.set_xlim(0, 1); ax.set_ylim(-1, 1)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(frameon=False, fontsize="x-small")

fig.suptitle(
    f"Parallel MC preview (numpy): discrete vs B.2 soft  "
    f"(n_init={N_INIT}, D={D})\n"
    f"depletion {disc['depletion']:.4f}/{soft['depletion']:.4f}  |  "
    f"W_total {disc['W_total']:.0f}/{soft['W_total']:.0f}  |  "
    f"workstation 3-way (incl. actual JAX B.1) = "
    f"demo/forward_modeling_mc_parallel.py"
)
fig.tight_layout()
out = Path(__file__).resolve().parent / "b2_parallel_preview.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
print(f"depletion discrete={disc['depletion']:.4f} soft={soft['depletion']:.4f}")
print(f"W_total   discrete={disc['W_total']:.0f}   soft={soft['W_total']:.0f}")
