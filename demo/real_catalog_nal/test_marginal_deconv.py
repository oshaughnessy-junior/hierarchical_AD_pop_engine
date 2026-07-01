"""Test the CORRECT deconvolution: divide by the marginal source-frame prior
pi(m1_s,m2_s,z) = (1+z)^2 [uniform detector masses] * p_z(z), NOT the full-D
log_prior. recent-O4 p_z(z) ~ dVc/dz * 1/(1+z) (comoving vol per source time).
ESS should recover vs the naive full-log_prior division (which gave ~1)."""
import h5py, numpy as np
from astropy.cosmology import Planck15
import astropy.units as u

def dVcdz_over_1pz(z):  # comoving volume per source-frame time (Zee's vt convention)
    return 4*np.pi*Planck15.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value/(1+z)

def load(f, prefer=("C01:Mixed","C00:Mixed","C01:IMRPhenomXPHM","C00:IMRPhenomXPHM-SpinTaylor")):
    with h5py.File(f,"r") as h:
        grps=[k for k in h if isinstance(h.get(k),h5py.Group) and "posterior_samples" in h[k]]
        g=next((p for p in prefer if p in grps), grps[0]); ps=h[g]["posterior_samples"]; nm=ps.dtype.names
        col=lambda *c: next((np.asarray(ps[x]) for x in c if x in nm), None)
        m1=col("mass_1_source"); m2=col("mass_2_source"); z=col("redshift")
        if m1 is None: m1=col("mass_1")/(1+z); m2=col("mass_2")/(1+z)
        chie=col("chi_eff"); lp=col("log_prior")
        return m1,m2,(np.zeros_like(m1) if chie is None else chie),z,lp,g

def ess(w): w=w/w.sum(); return 1.0/np.sum(w**2)
def wstat(x,w): w=w/w.sum(); m=np.sum(w*x); return m, np.sqrt(np.sum(w*(x-m)**2))

for tag,f in [("O4 S240519b","/home/richard.oshaughnessy/GWTC5-HLV/releases/v6/S240519b/C00-Combined.h5"),
              ("O4 S240413p","/home/richard.oshaughnessy/GWTC5-HLV/releases/v6/S240413p/C00-Combined.h5"),
              ("O3 S190521a","/home/pe.o3/o3a_final/run_directories/S190521a/cosmo_reweight_combined.h5")]:
    m1,m2,chie,z,lp,g=load(f); Mc=(m1*m2)**0.6/(m1+m2)**0.2
    # marginal source-frame prior (recent-O4 comoving-per-source-time)
    pi_marg = (1+z)**2 * dVcdz_over_1pz(z)
    w = 1.0/pi_marg
    mcm,mcs=wstat(Mc,w); chm,chs=wstat(chie,w)
    # naive full-prior for comparison
    if lp is not None:
        wn=1.0/np.exp(lp-lp.max()); essn=ess(wn)
    else: essn=float('nan')
    print(f"{tag} [{g}] z~{np.median(z):.2f}")
    print(f"   RAW post : Mc {Mc.mean():6.1f}+/-{Mc.std():4.2f}")
    print(f"   MARGINAL : Mc {mcm:6.1f}+/-{mcs:4.2f}  chi_eff {chm:+.2f}+/-{chs:.2f}   ESS {ess(w):6.0f}/{len(Mc)}  (naive full-logprior ESS {essn:.0f})")
