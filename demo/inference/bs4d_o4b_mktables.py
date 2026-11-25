import csv, glob, json, os, sys
import numpy as np
LAB = {"NULL": "null", "BB": "B_b", "BC": "B_c", "BD": "B_d", "BMIX": "Bmix", "NULLG": "null + g(eta)", "REFRC": "combined (REFRC)"}
def res(t):
    for d in ("res/scan_n8000", "res/scan"):
        if os.path.exists(f"{d}/{t}/result.json"): return json.load(open(f"{d}/{t}/result.json"))
# n24000 / n8000 six-key lnL
Z = {}
for p in ("data/scatter_n8000.npz", "data/scatter_n8000b.npz", "data/scatter_n24000.npz"):
    z = np.load(p, allow_pickle=True)
    for k in z.files: Z[k] = z[k]
def lnLk(t, ns):
    ks = sorted({int(k.split("|")[2]) for k in Z if k.startswith(f"{t}|{ns}|") and k.endswith("|lnL")}); v = np.array([float(Z[f"{t}|{ns}|{k}|lnL"]) for k in ks])
    return (len(v), float(v.mean()), float(v.std(ddof=1))) if len(v) > 1 else (0, None, None)
table = {r["tag"]: r for r in csv.DictReader(open("laplace_n8000.csv"))}
role = {"BMIX_f0.3_e1m": "table", "BC_f0.3_e2m": "table", "NULL_f0.3": "table", "BD_f0.3": "table", "BD_f0.2_e1p": "best lnZ (n3000)", "BD_f0.1": "extension", "REFRC_f0.3": "table",
        "REFRC_f0.4": "best lnZ (n3000)", "REFRC_f0.5": "extension", "NULLG_f0.2_e1m": "table", "NULLG_f0.3": "best lnZ (n3000)", "BB_f0.4": "table", "BB_f0.3": "best lnZ (n3000, old basin)",
        "BB_f0.2_sBC": "new basin 0.2 (non-PD fit)", "BBsBC_f0.2_x1p": "new basin 0.2 (escape)", "BB_f0.3_sBC": "new basin 0.3", "BB_f0.4_sBC": "new basin 0.4 (non-PD fit)", "BBsBC_f0.4_x1m": "new basin 0.4 (escape)", "BB_f0.5_sBC": "new basin 0.5"}
cols = ["member", "tag", "role", "f_dep", "rails", "n_keys", "n_pd", "lnL_n8000", "lnL_sd_keys", "lnZ_bdy", "lnZ_bdy_sd_keys", "lnZ_box", "lnZ_boxg", "lnZ_fix_rail", "eig0_min", "newton_decrement", "dlnL_vs_n3000k11", "dvol_vs_n3000k11", "lnZ_bdy_n3000_k11", "n24000_keys", "lnL_n24000", "lnL_sd_n24000", "neff_min", "note"]
out = []
for t, r in table.items():
    n, m, sd = lnLk(t, 24000); rr = res(t)
    out.append(dict(member=LAB[r["model"]], tag=t, role=role.get(t, ""), f_dep=r["f_dep"], rails=r["rails"], n_keys=r["n_keys"], n_pd=r["n_pd"], lnL_n8000=r["lnL_n8000"], lnL_sd_keys=r["lnL_sd_keys"], lnZ_bdy=r["lnZ_bdy"] or "", lnZ_bdy_sd_keys=r["lnZ_bdy_sd_keys"] or "",
        lnZ_box=r["lnZ_box"] or "", lnZ_boxg=r["lnZ_boxg"] or "", lnZ_fix_rail=r["lnZ_fix"] or "", eig0_min=r["eig0_min"], newton_decrement=r["newton_decrement"] or "", dlnL_vs_n3000k11=r["dlnL_part"], dvol_vs_n3000k11=r["dvol_part"] or "", lnZ_bdy_n3000_k11=r["lnZ_bdy_n3000_k11"],
        n24000_keys=n or "", lnL_n24000=m if m is not None else "", lnL_sd_n24000=sd if sd is not None else "", neff_min=r["neff_min"], note=r["note"]))
with open("table_n8000.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); [w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in out]
st = []
for fd in ("0.2", "0.3", "0.4", "0.5"):
    for s, nm in (("", "original (v3 O4 start)"), ("_sBC", "B_c O4 optimum mapped"), ("_sBD", "B_d O4 optimum mapped"), ("_s3D", "3D BS_rank B_b optimum"), ("_sD1", "dispersed 1"), ("_sD2", "dispersed 2, seed 20261125")):
        t = f"BB_f{fd}{s}"; r = res(t)
        if not r: continue
        bad = r["lnL_heldout_mean"] < -1e9
        st.append(dict(f_dep=float(fd), tag=t, start=nm, status="no support for some event at some key (masked likelihood)" if bad else "ok", lnL_heldout_mean=r["lnL_heldout_mean"], lnL_heldout_sd=r["lnL_heldout_sd"], lnL_map_n3000k11=r["lnL"], lnZ_bdy_n3000k11=r["lnZ_laplace_bdy"], negH_pd=r["negH_pd"], eig0=min(r["negH_eigs"]), kappa=r["theta_map"]["kappa"], mu_B=r["theta_map"]["mu_B"], sig_B=r["theta_map"]["sig_B"], mu_chiB=r["theta_map"]["mu_chiB"], sig_chiB=r["theta_map"]["sig_chiB"], spin_chi=r["theta_map"]["spin_chi"], rails="/".join(r["railing"])))
with open("bb_starts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(st[0])); w.writeheader(); [w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}) for r in st]
print(len(out), "table rows;", len(st), "B_b start rows")
for r in out: print(r["tag"], r["lnL_n8000"], r["lnL_sd_keys"], "n24000", r["lnL_n24000"], r["lnL_sd_n24000"])
