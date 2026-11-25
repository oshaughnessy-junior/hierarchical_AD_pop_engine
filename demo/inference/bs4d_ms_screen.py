"""Multi-start screening for one member / data set / f_dep (O4-only injection selection, free kappa).
Starts: the member's own O4 optimum (B_b: both basins), every other member's O4 optimum mapped onto its parameters, the 3D BS_rank
optimum (log_R shifted -0.6365 and into O4 units; kappa 2.2), and 3 dispersed starts that have support (no masked event at keys
11/202/3003). Each start runs 20 Adam steps on lnL_prof (lr 0.08 to 0.01, n_seed 3000, key 11), then lnL at held-out keys 202 and 3003.
Appends one JSON line per start to OUT/<member>_<dataset>_f<fd>.jsonl; finished starts are skipped (resumable).
Usage: bs4d_ms_screen.py OUT MEMBER DATASET FDEP"""
import json, math, os, sys, time, argparse
import numpy as np, jax, jax.numpy as jnp
import hier_map_laplace as H

OUT, MEM, DS, FD = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
WS = os.environ["HIER_WS"]; os.makedirs(OUT, exist_ok=True); dst = f"{OUT}/{MEM}_{DS}_f{FD:.1f}.jsonl"
done = {json.loads(l)["start"] for l in open(dst)} if os.path.exists(dst) else set()
H.M_MAX_LO = 20.0; H.FREE_KAPPA = (-1.0, 7.0); H.KAPPA_INIT = 2.7; H.LOG_R_RANGE = (-8.0, 12.0); H.FIELD_SEED = None
N3 = 3000
EXCL = ["S241011k"] if DS == "NO241011" else None
args = argparse.Namespace(catalog=f"{WS}/inputs/full_o3_o4a_o4b_catalog_NAL_exNSBH_etafix.npz", pdet_grid=f"{WS}/inputs/mass_pdet_grid_kappa2.7.npz",
    pdet_z_grid=f"{WS}/inputs/combined_pdet_grid_mmin2_refinedz.npz", likelihood="native4d", kappa=2.7, native4d_tables=f"{WS}/runs/A_tables/native4d_events.npz",
    native4d_tiers="primary", native4d_nodes=32, native4d_window=False, selection="injections", injection_file=f"{WS}/inputs/derived/gwtc5_o4ab_found.npz",
    event_subset="o4", f_dep=FD, exclude_events=EXCL)
SHIFT3D = -0.6365 + math.log(168 / 236.0) + math.log(3.0456693791669833 / (52962816.0 / (365.25 * 86400.0)))
SC, SN = f"{WS}/runs/BS4D_O4/scan", f"{WS}/runs/BS4D_O4/scan_n8000"
OWN = {"NULL": [("own", f"{SC}/NULL_f0.3")], "BB": [("own_old", f"{SC}/BB_f0.4"), ("own_new", f"{SN}/BB_f0.3_sBC")], "BC": [("own", f"{SC}/BC_f0.3_e2m")],
       "BD": [("own", f"{SC}/BD_f0.3")], "BMIX": [("own", f"{SC}/BMIX_f0.3_e1m")], "NULLG": [("own", f"{SC}/NULLG_f0.2_e1m")], "REFRC": [("own", f"{SC}/REFRC_f0.3")]}
D3 = {"NULL": "BS_scan/NULL_f0.4", "BB": "BS_scan_mmax20/BB_f0.3", "BC": "BS_scan_mmax20/BC_f0.4", "BD": "BS_scan_mmax20/BD_f0.5",
      "BMIX": "BS_scan_mmax20/BMIX_f0.5", "NULLG": "BS_scan_mmax20/NULLG_f0.3_s25", "REFRC": "BS_scan/REFRC_f0.4"}
th_of = lambda d: json.load(open(f"{d}/result.json"))["theta_map"]
dry = H.build(args, MEM, 11, N3, N3, dry=True); names = dry["names"]; bd = dry["bounds"]
lo = np.array([bd[n][0] for n in names]); hi = np.array([bd[n][1] for n in names])
own_th = th_of(OWN[MEM][-1][1])
mapped = lambda src: {n: float(src.get(n, own_th.get(n))) for n in names}
clip = lambda th: {n: float(np.clip(th[n], bd[n][0] + 1e-3 * (bd[n][1] - bd[n][0]), bd[n][1] - 1e-3 * (bd[n][1] - bd[n][0]))) for n in names}

starts = [(f"own:{lab}", mapped(th_of(d))) for lab, d in OWN[MEM]]
for m, lst in OWN.items():
    if m == MEM: continue
    for lab, d in lst: starts.append((f"from:{m}_{lab}", mapped(th_of(d))))
d3 = json.load(open(f"{WS}/runs/diag_src/{D3[MEM]}/result.json"))["theta_map"]
starts.append(("3D", mapped({k: (v + SHIFT3D if k.startswith("log_R_") else v) for k, v in {**d3, "kappa": 2.2}.items()})))


def build(prop, key): return H.build(args, MEM, key, N3, N3, prop)
def fwd(th, key):
    B = build(H.prop_ref_from(MEM, th), key); return float(jax.jit(B["parts"])(jnp.asarray([th[n] for n in names]))[0])


# dispersed starts with support at all three keys (kept in a side file so a resume reuses the same ones)
dfile = dst.replace(".jsonl", ".dispersed.json")
if os.path.exists(dfile): disp = json.load(open(dfile))
else:
    rng = np.random.default_rng(20261126 + sum(ord(c) for c in MEM)); disp, k_try = [], 0
    own_R = {n: own_th[n] for n in names if n.startswith("log_R_")}
    while len(disp) < 3 and k_try < 15:
        k_try += 1; th = {}
        for n in names:
            if n.startswith("log_R_"): th[n] = own_R[n] + rng.uniform(-1, 1)
            else: th[n] = bd[n][0] + (0.15 + 0.7 * rng.random()) * (bd[n][1] - bd[n][0])
        th = clip(th)
        try: ok = all(fwd(th, k) > -1e9 for k in (11, 202, 3003))
        except Exception as e: ok = False; print("support check failed", repr(e)[:80], flush=True)
        if ok: disp.append(dict(theta=th, attempts=k_try)); print("dispersed start", len(disp), "attempts", k_try, flush=True)
    json.dump(disp, open(dfile, "w"))
for i, d in enumerate(disp): starts.append((f"D{i + 1}", d["theta"]))

for label, th in starts:
    if label in done: continue
    t0 = time.time(); th = clip(th); prop = H.prop_ref_from(MEM, th)
    B = build(prop, 11); vgp = jax.jit(jax.value_and_grad(B["prof"], has_aux=True))
    z = H.to_z(np.array([th[n] for n in names]), lo, hi); s0 = []
    best_th, best_L, z = H.adam(vgp, z, lo, hi, 20, 0.08, 0.01, s0.append, lambda *a, **k: None)
    if best_th is None: best_th, best_L = np.array([th[n] for n in names]), float("nan")
    ho = [fwd(dict(zip(names, best_th.tolist())), k) for k in (202, 3003)]
    rec = dict(member=MEM, dataset=DS, f_dep=FD, start=label, start_theta=th, theta=dict(zip(names, best_th.tolist())), lnL_key11=float(best_L), lnL_ho=ho, lnL_ho_mean=float(np.mean(ho)), sup=bool(min(ho) > -1e9), wall_s=time.time() - t0)
    with open(dst, "a") as f: f.write(json.dumps(rec) + "\n")
    print(MEM, DS, FD, label, "start lnL11", round(float(best_L), 1), "heldout", [round(x, 1) for x in ho], f"{rec['wall_s']:.0f}s", flush=True)
open(dst.replace(".jsonl", ".done"), "w").write("ok")
print("DONE", dst, flush=True)
