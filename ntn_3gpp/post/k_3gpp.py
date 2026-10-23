"""LoS K in the TR 38.811 definition: each LoS UE's K = P_direct / P_nondirect in dB (S: V+H ports added; Ku/Ka: the RHCP
member), then the mean over the LoS UEs, mu_K. Bars: 25th-75th percentile over the 10 scenes of the per-scene mean.
Usage: python post/k_3gpp.py RUNS SCENARIO [BAND=S]   ->  RUNS/figures/k_3gpp_SCENARIO[_BAND].csv"""
import csv, os, pickle, sys
import numpy as np

runs, scen = sys.argv[1:3]; band = sys.argv[3] if len(sys.argv) > 3 else "S"
out = f"{runs}/figures/k_3gpp_{scen}{'' if band == 'S' else '_' + band}.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["scenario", "elevation_deg", "mu_K_dB", "sigma_K_dB", "q25", "q75", "n_ue"])
    for e in range(10, 100, 10):
        fs = ([f"{runs}/{scen}_S_{p}/results_elev{e:03d}deg.pkl" for p in "VH"] if band == "S"
              else [f"{runs}/{scen}_{band}/results_elev{e:03d}deg.pkl"])
        if not all(os.path.exists(f) for f in fs):
            continue
        pd = pn = 0
        for f in fs:
            r = pickle.load(open(f, "rb"))
            pd = pd + np.asarray(r["P_los"], float); pn = pn + np.asarray(r["P_nlos"], float)
            los = np.asarray(r["los_probability"], float) > 0.5; seeds = np.asarray(r["scene_seeds"])
            del r
        k = los & (pn > 0); kdb = 10 * np.log10(pd[k] / pn[k]); sk = seeds[k]
        q = np.percentile([kdb[sk == s].mean() for s in np.unique(sk)], [25, 75])
        w.writerow([scen, e, round(kdb.mean(), 3), round(kdb.std(), 3), round(q[0], 3), round(q[1], 3), int(k.sum())])
        print(scen, band, e, f"mu_K {kdb.mean():.2f} dB, sigma {kdb.std():.2f} dB ({k.sum()} LoS UEs)", flush=True)
print("wrote", out)
