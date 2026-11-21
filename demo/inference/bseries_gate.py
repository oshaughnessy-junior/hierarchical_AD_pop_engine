"""Gate for one B-series member (reassessment 2026-11-20).

1. committed path: the driver's build(prop=None) must reproduce the population
   (samples, log-weights) that two_channel_pl_demo.py's own CLI constructs from the
   member's launch-script flags, bit for bit, at the same sampler key and theta.
2. committed -> exact lnL shift at the launch init point, keys 11 and 202.
Writes runs/BS_gate/<model>.json.
"""
import argparse, json, os, sys, time, types
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

WS = os.path.expanduser("~/20261120-hier-reassess")
E = os.path.join(WS, "engine-bseries/demo/inference")
sys.path.insert(0, E)
import hier_map_laplace as H

COMMON = ["--mode", "check", "--fix-alpha-a", "0.0", "--log-r-width", "8",
          "--n-seed", "3000", "--n-b", "3000", "--n-c", "2500", "--v-esc", "1000"]
CLI = {
    "NULL": ["--f-dep", "F", "--exchange-mode", "1", "--exchange-c", "1.0"],
    "BB": ["--f-dep", "F", "--exchange-mode", "1", "--exchange-c", "1.0",
           "--channel-c", "alignedseed", "--as-f-dep", "0.5", "--as-mu", "9.5", "--as-sig", "1.0",
           "--as-chimu", "0.10", "--as-chisig", "0.05"],
    "BC": ["--f-dep", "F", "--exchange-mode", "1", "--exchange-c", "1.0",
           "--channel-c", "pl", "--c-mlo", "20", "--c-mhi", "45", "--c-alpha", "2.0",
           "--c-qa", "4", "--c-qb", "1", "--c-chi-sig", "0.2"],
    "BD": ["--f-dep", "F", "--exchange-mode", "1", "--exchange-c", "1.0",
           "--channel-c", "gauss", "--c-mu", "21.6", "--c-sig", "10.2", "--c-qa", "6",
           "--c-qb", "1", "--c-chi-sig", "0.1"],
    "BMIX": ["--f-dep", "F", "--exchange-mode", "1", "--exchange-c", "1.0",
             "--channel-b-mix", "--sig-b-min", "0.2"],
    "NULLG": ["--f-dep", "F", "--exchange-mode", "0", "--kernel-geta", "--kernel-geta-form", "lor",
              "--kernel-geta-eta0", "0.23", "--kernel-geta-w", "0.03"],
}


def cli_population(model, f_dep, theta):
    pl = H.load_pl()
    cap = {}
    orig = pl.make_combined_pl

    def wrap(n_seed_A, n_b):
        comb, nA = orig(n_seed_A, n_b)
        cap["comb"], cap["nA"] = comb, nA
        return comb, nA
    pl.make_combined_pl = wrap
    argv = [a if a != "F" else str(f_dep) for a in CLI[model]]
    sys.argv = ["two_channel_pl_demo.py"] + COMMON + argv
    pl.main()
    pl.M.SAMPLER_KEY = jax.random.PRNGKey(11)
    s, lw = jax.jit(lambda: cap["comb"](**theta))()
    return np.asarray(s), np.asarray(lw), cap["nA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--f-dep", type=float, default=0.9)
    a = ap.parse_args()
    args = types.SimpleNamespace(
        catalog=os.path.join(WS, "inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz"),
        pdet_grid=os.path.join(WS, "inputs/mass_pdet_grid_kappa2.7.npz"),
        likelihood="nal3d", f_dep=a.f_dep)
    B0 = H.build(args, a.model, 11, 3000, 3000, None, dry=True)
    init = H.default_init(a.model, B0["logr0"])
    names = B0["names"]
    theta = {k: init[k] for k in names}
    theta.update(B0["consts"])
    out = dict(model=a.model, f_dep=a.f_dep, theta=theta)
    t0 = time.time()
    s_cli, lw_cli, nA_cli = cli_population(a.model, a.f_dep, theta)
    Bc = H.build(args, a.model, 11, 3000, 3000, None)
    s_drv, lw_drv = jax.jit(lambda: Bc["comb"](**theta))()
    s_drv, lw_drv = np.asarray(s_drv), np.asarray(lw_drv)
    same_shape = s_cli.shape == s_drv.shape
    out["gate"] = dict(
        shapes=[list(s_cli.shape), list(s_drv.shape)], nA=[nA_cli, Bc["nA"]],
        bitwise=bool(same_shape and np.array_equal(s_cli, s_drv, equal_nan=True)
                     and np.array_equal(lw_cli, lw_drv, equal_nan=True)),
        max_abs_ds=float(np.nanmax(np.abs(s_cli - s_drv))) if same_shape else None,
        max_abs_dlw=float(np.nanmax(np.abs(lw_cli - lw_drv))) if same_shape else None)
    print("gate", out["gate"], f"{time.time()-t0:.0f}s", flush=True)
    th = jnp.asarray([init[k] for k in names])
    prop = H.prop_ref_from(a.model, init)
    rows = []
    for key in (11, 202):
        for mode, pr in (("committed", None), ("exact", prop)):
            B = Bc if (key == 11 and pr is None) else H.build(args, a.model, key, 3000, 3000, pr)
            L, (mu, muA) = jax.jit(B["parts"])(th)
            Lp, _ = jax.jit(B["prof"])(th)
            rows.append(dict(key=key, mode=mode, lnL=float(L), lnL_prof=float(Lp), mu=float(mu), mu_A=float(muA)))
            print(rows[-1], flush=True)
    out["shift"] = rows
    c = {r["key"]: r for r in rows if r["mode"] == "committed"}
    e = {r["key"]: r for r in rows if r["mode"] == "exact"}
    out["dlnLprof_exact_minus_committed"] = {k: e[k]["lnL_prof"] - c[k]["lnL_prof"] for k in c}
    os.makedirs(os.path.join(WS, "runs/BS_gate"), exist_ok=True)
    with open(os.path.join(WS, f"runs/BS_gate/{a.model}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("DONE", out["dlnLprof_exact_minus_committed"], f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
