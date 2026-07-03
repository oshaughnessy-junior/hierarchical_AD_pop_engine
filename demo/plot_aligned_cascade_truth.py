"""Aligned single-pulse cascade truth marginals (dN/dm, dN/dchi) with an
INDEPENDENT particle-MC overlay — the one figure that shows the aligned fractal
cascade, so the MC must confirm the engine's spikes are real (not grid ringing).

Engine = implicit aligned solve (self-consistent with Fig 8/9/10); MC = run_mc
aligned at the solve's converged f_natal_depleted, RAW weighted histogram (no
KDE, so the spikes survive)."""
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

import hierarchical_ad_pop_engine.orientation_precompute as _op
import hierarchical_ad_pop_engine.response.implicit as _imp
_op.R_CHI_OUT_DEFAULT = None
_imp.R_CHI_OUT_DEFAULT = None

import os as _os
DEMO = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, DEMO)
import demo_gwkokab_fit_stage1 as s1
from hierarchical_ad_pop_engine.grids import BinaryGrid
from hierarchical_ad_pop_engine.spin_orientation import Aligned
from hierarchical_ad_pop_engine.response.coagulation import build_gaussian_coagulation
from hierarchical_ad_pop_engine.response.implicit import solve_implicit_fixed_point, _dm_log_weights
from red_team.comparison_helpers import run_mc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def argval(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


n_m = argval("--nm", 96, int)
n_chi = argval("--nchi", 128, int)
n_seed = argval("--nseed", 1_000_000, int)
eps = argval("--eps", 1.0, float)
outdir = argval("--out", "figures_stage1_truth_n128", str)

TH, FX = s1.THETA_STAR, s1.FIXED
grid = BinaryGrid.default(m_min=FX["m_min"], m_max=FX["m_max"], n_m=n_m, n_chi=n_chi)
resp = build_gaussian_coagulation(
    mu=TH["mu_natal"], sigma=TH["sigma_natal"], m_min=FX["m_min"], m_max=FX["m_max"],
    natal_spin_loc=TH["natal_spin_loc"], natal_spin_scale=TH["natal_spin_scale"],
    kernel_a=FX["kernel_a"], kernel_b=TH["kernel_b"], grid=grid, solver="implicit",
    orientation=Aligned(), retention_kind="sigmoid",
)
res = solve_implicit_fixed_point(resp, jnp.asarray(FX["v_esc"]), epsilon=eps,
                                 max_iter=200, tol=1e-7, damping=0.5)
res.N.block_until_ready()
f_dep = float(res.f_natal_depleted)
m = np.asarray(grid.m); chi = np.asarray(grid.chi)
dm_w = np.asarray(_dm_log_weights(grid.m, grid.log_m))
d = np.diff(chi); dchi_w = np.concatenate([d[:1] * 0.5, 0.5 * (d[:-1] + d[1:]), d[-1:] * 0.5])
eng_dN_dm = np.asarray(res.N) @ dchi_w
eng_dN_dchi = np.asarray(jnp.einsum("ij,i->j", res.N, dm_w))
print(f"engine {n_m}x{n_chi} aligned: f_dep={f_dep:.4f}", flush=True)

theta = dict(mu_natal=TH["mu_natal"], sigma_natal=TH["sigma_natal"], m_min=FX["m_min"],
             m_max=FX["m_max"], natal_spin_loc=TH["natal_spin_loc"],
             natal_spin_scale=TH["natal_spin_scale"], kernel_a=FX["kernel_a"],
             kernel_b=TH["kernel_b"], v_esc=FX["v_esc"])
rng = np.random.default_rng(20260703)
mc = run_mc(theta, n_seed, rng, depletion_fraction=max(f_dep, 1e-3), orientation=None)
m_act = np.asarray(mc["m_active"]); chi_act = np.asarray(mc["chi_active"])
w = np.asarray(mc["w_active"])
print(f"MC actives={m_act.size}  chi_max={chi_act.max():.3f}", flush=True)

mb = np.geomspace(FX["m_min"], FX["m_max"], 61)
cb = np.linspace(0.0, 1.0, 81)
mc_dN_dm = np.histogram(m_act, bins=mb, weights=w)[0] / np.diff(mb)
mc_dN_dchi = np.histogram(chi_act, bins=cb, weights=w)[0] / np.diff(cb)
# scale MC to the engine integral so amplitudes overlay
mc_dN_dm *= float(np.sum(eng_dN_dm * dm_w)) / max(float(np.sum(mc_dN_dm * np.diff(mb))), 1e-30)
mc_dN_dchi *= float(np.sum(eng_dN_dchi * dchi_w)) / max(float(np.sum(mc_dN_dchi * np.diff(cb))), 1e-30)

FIG = Path(DEMO) / outdir
FIG.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.set_xscale("log"); ax.set_yscale("log")
ax.stairs(np.maximum(mc_dN_dm, 1e-30), mb, color="C3", lw=1.2, label="particle MC (raw)", zorder=2)
ax.plot(m, np.maximum(eng_dN_dm, 1e-30), lw=2.0, color="C0", label="implicit engine", zorder=3)
ax.set_xlabel(r"primary mass $m\;\;[M_\odot]$")
ax.set_ylabel(r"$dN/dm$ (single-BH density, all generations)")
ax.set_title(rf"Aligned single-pulse cascade: $dN/dm$ (${n_m}\times{n_chi}$)")
ax.set_ylim(bottom=float(eng_dN_dm.max()) * 1e-8)
ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "stage1_truth_dN_dm.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.set_yscale("log")
ax.stairs(np.maximum(mc_dN_dchi, 1e-30), cb, color="C3", lw=1.2, label="particle MC (raw)", zorder=2)
ax.plot(chi, np.maximum(eng_dN_dchi, 1e-30), lw=2.0, color="C0", label="implicit engine", zorder=3)
ax.axvline(0.6865, ls=":", color="k", lw=0.8)
ax.set_xlabel(r"single-BH spin $\chi$"); ax.set_ylabel(r"$dN/d\chi$")
ax.set_title(rf"Aligned single-pulse cascade: $dN/d\chi$ (${n_m}\times{n_chi}$)")
ax.set_ylim(bottom=float(eng_dN_dchi.max()) * 1e-6)
ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "stage1_truth_dN_dchi.png", dpi=150); plt.close(fig)
print(f"wrote {FIG}/stage1_truth_dN_dm.png + _dN_dchi.png", flush=True)
