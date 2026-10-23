"""RT K at the UE panel alone (the runner's P_nlos_panel), for runs that need no generator, e.g. the without-diffuse runs.
LoS UEs (los_probability > 0.5, P_nlos > 0), pooled and the TR 38.811 mean of K in dB.
Usage: python post/rt_panel_k.py RUNS SCENARIO BAND   ->  RUNS/figures/rt_panel_k_SCENARIO_BAND.csv"""
import csv, os, pickle, sys
import numpy as np

runs, scen, band = sys.argv[1:4]
out = f"{runs}/figures/rt_panel_k_{scen}_{band}.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["scenario", "elevation_deg", "n_ue", "K_panel_RT_dB", "K_panel_RT_mean_dB"])
    for e in range(10, 100, 10):
        f = f"{runs}/{scen}_{band}/results_elev{e:03d}deg.pkl"
        if not os.path.exists(f): continue
        r = pickle.load(open(f, "rb")); v = {k: np.asarray(r[k], float) for k in ("P_los", "P_nlos", "P_nlos_panel", "los_probability")}; del r
        i = (v["los_probability"] > 0.5) & (v["P_nlos"] > 0); pl, pp = v["P_los"][i], v["P_nlos_panel"][i]
        w.writerow([scen, e, int(i.sum()), round(10 * np.log10(pl.sum() / pp.sum()), 3), round(float(np.mean(10 * np.log10(pl / pp))), 3)])
print("wrote", out)
