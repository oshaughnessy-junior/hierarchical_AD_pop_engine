"""Gate for the lowered m_max_A floor: at m_max_A = 40 the floor-20 build must give
the same lnL as the floor-40 build at a fixed key, for every B-series model.
Also checks that only the m_max_A bound changed. Writes runs/BS_gate/mmax20.json."""
import json, os, sys, types
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
WS = os.path.expanduser("~/20261120-hier-reassess")
sys.path.insert(0, os.path.join(WS, "engine-bseries/demo/inference"))
import hier_map_laplace as H

args = types.SimpleNamespace(
    catalog=os.path.join(WS, "inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz"),
    pdet_grid=os.path.join(WS, "inputs/mass_pdet_grid_kappa2.7.npz"), likelihood="nal3d", f_dep=0.4)
out = {}
for model in sys.argv[1:]:
    r = json.load(open(os.path.join(WS, f"runs/BS_scan/{model}_f0.4/result.json")))
    th = dict(r["theta_map"]); th["m_max_A"] = 40.0
    prop = H.prop_ref_from(model, th)
    res = {}
    for lo in (40.0, 20.0):
        H.M_MAX_LO = lo
        B = H.build(args, model, 11, 3000, 3000, prop)
        x = jnp.asarray([th[k] for k in B["names"]])
        L, (mu, _) = jax.jit(B["parts"])(x)
        res[lo] = dict(lnL=float(L), mu=float(mu), bounds={k: list(v) for k, v in B["bounds"].items()})
    diff_b = sorted(k for k in res[40.0]["bounds"] if res[40.0]["bounds"][k] != res[20.0]["bounds"][k])
    out[model] = dict(lnL40=res[40.0]["lnL"], lnL20=res[20.0]["lnL"], identical=res[40.0]["lnL"] == res[20.0]["lnL"],
                      changed_bounds=diff_b, m_max_bounds=[res[40.0]["bounds"]["m_max_A"], res[20.0]["bounds"]["m_max_A"]])
    print(model, out[model], flush=True)
os.makedirs(os.path.join(WS, "runs/BS_gate"), exist_ok=True)
fn = os.path.join(WS, "runs/BS_gate/mmax20_" + "_".join(sys.argv[1:]) + ".json")
json.dump(out, open(fn, "w"), indent=1)
