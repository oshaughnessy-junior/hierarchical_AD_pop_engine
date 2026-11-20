"""Check ZSelection.xi_kappa against RebootAuto's numpy trilinear loader integrated over p(z).

Brute force: pdet_eval(m1, m2, z) on a 4001-node trapezoid in z times p(z|kappa).
Also reports the old z-averaged grid (kappa 2.7, zmax 2.07) at the same masses.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path.home() / '20261120-hier-reassess'
sys.path[:0] = [str(Path(__file__).resolve().parent), '/home/oshaughn/20260613-Me-RebootAuto/data/gwtc5']
import jax.numpy as jnp  # noqa: E402
import native4d_population as NP  # noqa: E402
from combined_pdet_grid_loader import pdet_eval  # noqa: E402

grid = ROOT / 'inputs/combined_pdet_grid_mmin2_refinedz.npz'
z = np.load(grid, allow_pickle=True)
g = dict(m1_grid=z['m1_grid'], m2_grid=z['m2_grid'], z_grid=z['z_grid'], pdet=z['pdet'])
sel = NP.ZSelection(grid)
rng = np.random.default_rng(1); n = 400
m1 = np.exp(rng.uniform(np.log(2), np.log(130), n)); m2 = np.maximum(m1 * rng.uniform(0.05, 1, n), 2.0)
out = {}
for kappa in [0.0, 2.7, 5.0]:
    zz = np.linspace(0, 2.5, 4001); pz = np.exp(np.asarray(NP.log_pz(jnp.asarray(zz), kappa)))
    P = pdet_eval(g, np.repeat(m1, len(zz)), np.repeat(m2, len(zz)), np.tile(zz, n)).reshape(n, len(zz))
    ref = np.trapezoid(P * pz, zz, axis=1)
    mine = np.asarray(sel.xi(jnp.asarray(np.stack([m1, m2], 1)), kappa))
    out[str(kappa)] = dict(norm_pz=float(np.trapezoid(pz, zz)), max_abs_diff=float(np.abs(ref - mine).max()),
                           max_rel_diff_where_xi_gt_1e3=float(np.max(np.abs(ref - mine)[ref > 1e-3] / ref[ref > 1e-3])),
                           sum_ck=float(sel.zweights(kappa).sum()))
print(json.dumps(out, indent=1))
(ROOT / 'runs/A_rescore').mkdir(parents=True, exist_ok=True)
(ROOT / 'runs/A_rescore/selection_check.json').write_text(json.dumps(out, indent=1))
