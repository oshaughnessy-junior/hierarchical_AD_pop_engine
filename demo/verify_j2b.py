"""J2b verification gates - run on CPU x64."""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from red_team.backup_particle_mc import (
    MCConfig,
    particle_mc_jax_soft,
    _chi_rem_at_orientation_jax,
    _v_kick_at_orientation_jax,
)
from hierarchical_ad_pop_engine.spin_orientation import Aligned, Isotropic, Fisher

RETAIN_TAU = 50.0


def base_cfg(orientation=None, seed=0):
    return MCConfig(
        mu_natal=30.0, sigma_natal=5.0, m_min=5.0, m_max=80.0,
        natal_spin_loc=0.3, natal_spin_scale=0.1,
        kernel_a=1.0, kernel_b=0.0, v_esc=500.0,
        n_seed=400, depletion_fraction=0.05,
        orientation=orientation, orientation_key_seed=seed,
    )


def run_soft(cfg, key_seed=1, n_steps=40):
    key = jax.random.PRNGKey(key_seed)
    return particle_mc_jax_soft(cfg, key, n_steps=n_steps)


def gate1_aligned_parity():
    print("=" * 60)
    print("GATE 1: Aligned-limit parity")
    out_none = run_soft(base_cfg(orientation=None))
    out_aln = run_soft(base_cfg(orientation=Aligned()))
    W_none = float(out_none["stats"]["W_total"]); W_aln = float(out_aln["stats"]["W_total"])
    occ_none = np.asarray(out_none["occupancy"]); occ_aln = np.asarray(out_aln["occupancy"])
    chi_none = np.asarray(out_none["chi_active"]); chi_aln = np.asarray(out_aln["chi_active"])
    net_none = float(out_none["stats"]["net_removed"]); net_aln = float(out_aln["stats"]["net_removed"])
    dW = abs(W_none - W_aln); docc = float(np.max(np.abs(occ_none - occ_aln)))
    dchi = float(np.max(np.abs(chi_none - chi_aln))); dnet = abs(net_none - net_aln)
    print(f"  W_total    None={W_none:.12f}  Aligned={W_aln:.12f}  |d|={dW:.2e}")
    print(f"  net_removed None={net_none:.12f} Aligned={net_aln:.12f} |d|={dnet:.2e}")
    print(f"  max|d occupancy| = {docc:.2e}")
    print(f"  max|d chi_active|= {dchi:.2e}")
    ok = max(dW, docc, dchi, dnet) < 1e-12
    print(f"  GATE 1 {'PASSED' if ok else 'FAILED'} (tol 1e-12)")
    return ok


def synthetic_pairs(n=2000, key_seed=7):
    key = jax.random.PRNGKey(key_seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    m1 = jax.random.uniform(k1, (n,), minval=10.0, maxval=50.0)
    m2 = jax.random.uniform(k2, (n,), minval=10.0, maxval=50.0)
    chi1 = jax.random.uniform(k3, (n,), minval=0.0, maxval=1.0)
    chi2 = jax.random.uniform(k4, (n,), minval=0.0, maxval=1.0)
    return m1, m2, chi1, chi2


def mean_retained_chi(orientation, n=2000, key_seed=11):
    m1, m2, chi1, chi2 = synthetic_pairs(n)
    key = jax.random.PRNGKey(key_seed)
    th1, ph1, th2, ph2 = orientation.sample(key, n)
    ct1 = jnp.cos(th1); ct2 = jnp.cos(th2); dphi = ph2 - ph1
    chi_rem = _chi_rem_at_orientation_jax(m1, m2, chi1, chi2, ct1, ct2, dphi)
    return float(jnp.mean(chi_rem)), float(jnp.mean(ct1))


def gate2_large_kappa():
    print("=" * 60)
    print("GATE 2: Large-kappa asymptote -> Aligned limit")
    # The physical invariant: Fisher(kappa->inf) mean|chi_rem| must converge
    # to the ALIGNED value on the SAME synthetic pairs.  (The plan's literal
    # 0.7725 was the value for its own de-risk synthetic sampler; the mass/
    # spin distribution here differs, so the invariant is compared against
    # the Aligned reference computed on identical pairs, not the raw number.)
    m1, m2, chi1, chi2 = synthetic_pairs(2000)
    ct1a = jnp.ones(2000); ct2a = jnp.ones(2000); dphia = jnp.zeros(2000)
    chi_rem_aln = _chi_rem_at_orientation_jax(m1, m2, chi1, chi2, ct1a, ct2a, dphia)
    aln_ref = float(jnp.mean(chi_rem_aln))
    mchi, mct = mean_retained_chi(Fisher.default(2000.0))
    print(f"  Aligned reference mean|chi_rem| = {aln_ref:.5f}")
    print(f"  Fisher(k=2000):   mean|chi_rem| = {mchi:.5f}   <cos>={mct:.5f}")
    print(f"  |Fisher(2000) - Aligned| = {abs(mchi - aln_ref):.2e}")
    ok = abs(mchi - aln_ref) < 1e-3
    print(f"  GATE 2 {'PASSED' if ok else 'FAILED'} (converges to Aligned, tol 1e-3)")
    return ok, mchi


def gate3_iso_equiv():
    print("=" * 60)
    print("GATE 3: k~0 == Isotropic; <cos>~0; retention drops vs Aligned")
    mchi_f0, mct_f0 = mean_retained_chi(Fisher.default(1e-8))
    mchi_iso, mct_iso = mean_retained_chi(Isotropic.default())
    print(f"  Fisher(k~0): mean|chi_rem|={mchi_f0:.4f}  <cos>={mct_f0:.4f}")
    print(f"  Isotropic  : mean|chi_rem|={mchi_iso:.4f}  <cos>={mct_iso:.4f}")
    equiv = abs(mchi_f0 - mchi_iso) < 0.05 and abs(mct_f0) < 0.05 and abs(mct_iso) < 0.05
    m1, m2, chi1, chi2 = synthetic_pairs(2000)
    v_esc = 500.0
    def retfrac(orientation, key_seed=11):
        key = jax.random.PRNGKey(key_seed)
        th1, ph1, th2, ph2 = orientation.sample(key, 2000)
        ct1 = jnp.cos(th1); ct2 = jnp.cos(th2); dphi = ph2 - ph1
        vk = _v_kick_at_orientation_jax(m1, m2, chi1, chi2, ct1, ct2, dphi)
        return float(jnp.mean(jax.nn.sigmoid((v_esc - vk) / RETAIN_TAU)))
    ct1a = jnp.ones(2000); ct2a = jnp.ones(2000); dphia = jnp.zeros(2000)
    vka = _v_kick_at_orientation_jax(m1, m2, chi1, chi2, ct1a, ct2a, dphia)
    ret_aln = float(jnp.mean(jax.nn.sigmoid((v_esc - vka) / RETAIN_TAU)))
    ret_iso = retfrac(Isotropic.default()); ret_f0 = retfrac(Fisher.default(1e-8))
    print(f"  retention frac Aligned  = {ret_aln:.4f}")
    print(f"  retention frac Isotropic= {ret_iso:.4f}")
    print(f"  retention frac Fisher k0= {ret_f0:.4f}")
    drops = ret_iso < ret_aln
    print(f"  <cos>~0 & Fisher(k0)~Isotropic: {equiv}")
    print(f"  retention drops vs Aligned: {drops}")
    ok = equiv and drops
    print(f"  GATE 3 {'PASSED' if ok else 'FAILED'}")
    return ok, ret_aln, ret_iso, ret_f0


def gate4_ad_fd():
    print("=" * 60)
    print("GATE 4: AD-vs-FD gradients through Fisher-sampled vector step")
    m1, m2, chi1, chi2 = synthetic_pairs(2000)
    v_esc0 = 500.0
    key = jax.random.PRNGKey(11)
    k_ct1, k_ct2, k_p1, k_p2 = jax.random.split(key, 4)
    u1 = jax.random.uniform(k_ct1, (2000,)); u2 = jax.random.uniform(k_ct2, (2000,))
    phi1 = jax.random.uniform(k_p1, (2000,), minval=0.0, maxval=2*jnp.pi)
    phi2 = jax.random.uniform(k_p2, (2000,), minval=0.0, maxval=2*jnp.pi)
    dphi = phi2 - phi1
    def cos_theta_of(u, kappa):
        ks = jnp.where(jnp.abs(kappa) < 1e-6, 1.0, kappa)
        return 1.0 + (1.0 / ks) * jnp.log(u + (1.0 - u) * jnp.exp(-2.0 * ks))
    def summary_kappa(kappa):
        ct1 = jnp.clip(cos_theta_of(u1, kappa), -1.0, 1.0)
        ct2 = jnp.clip(cos_theta_of(u2, kappa), -1.0, 1.0)
        chi_rem = _chi_rem_at_orientation_jax(m1, m2, chi1, chi2, ct1, ct2, dphi)
        vk = _v_kick_at_orientation_jax(m1, m2, chi1, chi2, ct1, ct2, dphi)
        retain = jax.nn.sigmoid((v_esc0 - vk) / RETAIN_TAU)
        return jnp.mean(retain * chi_rem)
    kappa0 = 5.0
    ad_k = float(jax.grad(summary_kappa)(kappa0)); h = 1e-5
    fd_k = float((summary_kappa(kappa0 + h) - summary_kappa(kappa0 - h)) / (2 * h))
    rel_k = abs(ad_k - fd_k) / max(abs(fd_k), 1e-30)
    print(f"  d/dk    AD={ad_k:.10e}  FD={fd_k:.10e}  rel={rel_k:.2e}")
    def summary_vesc(v_esc):
        ct1 = jnp.clip(cos_theta_of(u1, kappa0), -1.0, 1.0)
        ct2 = jnp.clip(cos_theta_of(u2, kappa0), -1.0, 1.0)
        chi_rem = _chi_rem_at_orientation_jax(m1, m2, chi1, chi2, ct1, ct2, dphi)
        vk = _v_kick_at_orientation_jax(m1, m2, chi1, chi2, ct1, ct2, dphi)
        retain = jax.nn.sigmoid((v_esc - vk) / RETAIN_TAU)
        return jnp.mean(retain * chi_rem)
    ad_v = float(jax.grad(summary_vesc)(v_esc0)); hv = 1e-3
    fd_v = float((summary_vesc(v_esc0 + hv) - summary_vesc(v_esc0 - hv)) / (2 * hv))
    rel_v = abs(ad_v - fd_v) / max(abs(fd_v), 1e-30)
    print(f"  d/dvesc AD={ad_v:.10e}  FD={fd_v:.10e}  rel={rel_v:.2e}")
    ok = rel_k < 1e-6 and rel_v < 1e-6
    print(f"  GATE 4 {'PASSED' if ok else 'FAILED'} (both rel < 1e-6)")
    return ok, rel_k, rel_v


def gate5_nan_backward():
    print("=" * 60)
    print("GATE 5: NaN-in-backward through the budget clip")
    def scalar_summary(v_esc, kappa):
        cfg = MCConfig(
            mu_natal=30.0, sigma_natal=5.0, m_min=5.0, m_max=80.0,
            natal_spin_loc=0.3, natal_spin_scale=0.1,
            kernel_a=1.0, kernel_b=0.0, v_esc=v_esc,
            n_seed=300, depletion_fraction=0.1,
            orientation=Fisher.default(kappa), orientation_key_seed=0,
        )
        out = particle_mc_jax_soft(cfg, jax.random.PRNGKey(3), n_steps=60)
        return out["stats"]["W_total"]
    g = jax.grad(scalar_summary, argnums=(0, 1))(500.0, 5.0)
    gv, gk = float(g[0]), float(g[1])
    print(f"  d(W_total)/d(v_esc) = {gv:.6e}")
    print(f"  d(W_total)/d(k)     = {gk:.6e}")
    finite = np.isfinite(gv) and np.isfinite(gk)
    print(f"  GATE 5 {'PASSED' if finite else 'FAILED'} (both finite)")
    return finite


if __name__ == "__main__":
    r1 = gate1_aligned_parity()
    r2, _ = gate2_large_kappa()
    r3, *_ = gate3_iso_equiv()
    r4, _, _ = gate4_ad_fd()
    r5 = gate5_nan_backward()
    print("=" * 60); print("SUMMARY:")
    for name, r in [("Gate1 parity", r1), ("Gate2 large-k", r2),
                    ("Gate3 iso", r3), ("Gate4 AD-FD", r4), ("Gate5 NaN-bwd", r5)]:
        print(f"  {name}: {'PASS' if r else 'FAIL'}")
    print("ALL PASS" if all([r1, r2, r3, r4, r5]) else "SOME FAILED")
