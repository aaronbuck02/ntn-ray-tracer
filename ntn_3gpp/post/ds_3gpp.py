"""Delay spread in the TR 38.811 definition (S: V+H ports; Ku/Ka: the one RHCP member): each UE's own RMS delay spread, then mu_lgDS = mean over UEs of
log10(DS / 1 s), per state. Two per-UE DS versions:
  all paths   V+H combined from the runner's per-UE path moments (every path, no floor)
  40 dB floor the runner's per-UE DS_ns (1 ns bins within 40 dB of the strongest, paths within 2 us), per port; the
              UE's lgDS is the mean of its V and H values
Bars: 25th-75th percentile over the 10 scenes of the per-scene mu_lgDS.
Usage: python post/ds_3gpp.py RUNS SCENARIO [BAND=S]   ->  RUNS/figures/ds_3gpp_SCENARIO.csv (S) or ds_3gpp_SCENARIO_BAND.csv"""
import csv, os, pickle, sys
import numpy as np

runs, scen = sys.argv[1:3]
band = sys.argv[3] if len(sys.argv) > 3 else "S"
out = f"{runs}/figures/ds_3gpp_{scen}{'' if band == 'S' else '_' + band}.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["scenario", "version", "state", "elevation_deg", "mu_lgDS", "sigma_lgDS", "q25", "q75", "n_ue"])
    for e in range(10, 100, 10):
        fs = [f"{runs}/{scen}_S_{p}/results_elev{e:03d}deg.pkl" for p in "VH"] if band == "S" else [f"{runs}/{scen}_{band}/results_elev{e:03d}deg.pkl"]
        if not all(os.path.exists(f) for f in fs):
            continue
        M, lgf = 0, []
        for f in fs:
            r = pickle.load(open(f, "rb"))
            t0 = np.asarray(r["t0_s"], float)
            m = np.asarray(r["path_moments"], float).sum(1)                     # (UE, 3): sum over classes
            s = np.where(np.isfinite(t0), t0, 0.0)[:, None]                     # shift each port to an absolute time
            M = M + np.c_[m[:, 0], m[:, 1] + s[:, 0] * m[:, 0], m[:, 2] + 2 * s[:, 0] * m[:, 1] + s[:, 0] ** 2 * m[:, 0]]
            lgf.append(np.log10(np.asarray(r["DS_ns"], float) * 1e-9))
            los = np.asarray(r["los_probability"], float) > 0.5; seeds = np.asarray(r["scene_seeds"])
            del r
        with np.errstate(all="ignore"):
            ds_all = np.sqrt(np.maximum(M[:, 2] / M[:, 0] - (M[:, 1] / M[:, 0]) ** 2, 0))
            lg = {"all paths": np.log10(ds_all), "40 dB floor": np.mean(lgf, 0)}
        for version, x in lg.items():
            for state, sel in (("LoS", los), ("NLoS", ~los)):
                k = sel & np.isfinite(x)
                if k.sum() < 2:
                    continue
                per = [x[k & (seeds == sd)].mean() for sd in np.unique(seeds) if (k & (seeds == sd)).any()]
                q = np.percentile(per, [25, 75])
                w.writerow([scen, version, state, e, round(x[k].mean(), 4), round(x[k].std(), 4), round(q[0], 4), round(q[1], 4), int(k.sum())])
        print(scen, e, "done", flush=True)
print("wrote", out)
