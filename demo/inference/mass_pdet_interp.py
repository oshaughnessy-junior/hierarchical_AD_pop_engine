"""JAX-differentiable real per-source detection probability pdet_fn(samples) from
the redshift-averaged xi(m1,m2) in [0,1] (planning_o4_mixed/build_mass_pdet.py).

This is a PROBABILITY (a pdet), NOT a VT -- xi(m1,m2)=int P_det(m,z) p_z(z) dz with
p_z normalized; ~Mc^(15/6) at low mass, saturating at 1 (gwkokab arXiv:2509.13638).
Drop-in for the toy pdet_fn (Mc^15/6-capped): samples[:, :2] = source-frame (m1, m2).
Bilinear interp on the log10-mass grid (jax.scipy.ndimage.map_coordinates, AD-safe).
"""
import numpy as np
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

DEFAULT_GRID = "/home/richard.oshaughnessy/coag_container/vt_data/mass_pdet_grid.npz"


def make_real_pdet_fn(grid_npz=DEFAULT_GRID, floor=1e-4):
    g = np.load(grid_npz)
    cent = np.asarray(g["log10_m_centers"])
    pdet = np.asarray(g["pdet"])                     # already in [0,1]
    cmin = float(cent[0]); dc = float(cent[1] - cent[0]); ng = len(cent)
    P = jnp.asarray(pdet); floor = float(floor)

    def pdet_fn(samples):
        m1 = samples[:, 0]; m2 = samples[:, 1]
        lm1 = jnp.log10(jnp.clip(m1, 10.0 ** cmin, 10.0 ** cent[-1]))
        lm2 = jnp.log10(jnp.clip(m2, 10.0 ** cmin, 10.0 ** cent[-1]))
        i1 = jnp.clip((lm1 - cmin) / dc, 0.0, ng - 1.0)
        i2 = jnp.clip((lm2 - cmin) / dc, 0.0, ng - 1.0)
        p = map_coordinates(P, [i1, i2], order=1, mode="nearest")
        return jnp.clip(p, floor, 1.0)

    return pdet_fn


if __name__ == "__main__":
    pdet = make_real_pdet_fn()
    ms = np.array([5, 8, 12, 18, 25, 35, 50, 70])
    samp = np.stack([ms, ms, np.zeros_like(ms), np.zeros_like(ms)], 1).astype(float)
    pv = np.asarray(pdet(jnp.asarray(samp)))
    mc = ms / 2 ** 0.2
    toy = np.minimum((mc / 35.0) ** (15 / 6), 1.0)
    print(" m1=m2   Mc    real_pdet   toy(Mc^15/6 cap)")
    for a, b, c, d in zip(ms, mc, pv, toy):
        print(f"  {a:4.0f}  {b:6.1f}    {c:.4f}      {d:.4f}")
