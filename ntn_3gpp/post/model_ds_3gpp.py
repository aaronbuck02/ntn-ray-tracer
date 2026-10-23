"""The geometry-stochastic model's DS in the TR 38.811 definition, for the matched scenes (2 GHz): one drawn city
per UE with ray-traced visibility (Model.paths(..., geometric_blockage=True)); the state is the city's own LoS event.
Each UE's DS follows the RT's two rules: "all paths" = every path, no floor, no delay cut, no binning (as the RT's
per-UE path moments); "40 dB floor" = the runner's rule (sim/run_stochastic_sim.py _ds): paths more than 2 us after the
first arrival dropped, 1 ns bins, bins 40 dB below the strongest dropped, RMS over bin centres. A UE whose DS is 0 (one bin) has no log and is counted, not averaged, as in RT.
mu_lgDS = mean over the UEs of a state of log10(DS). The same cities at every elevation (as eval29), in CHUNK-city
chunks on parallel processes.
Usage: python post/model_ds_3gpp.py OUT_CSV [N_URBAN N_RURAL [ELEVATIONS]]   (default 40000 / 80000 cities, 10..80 deg;
ELEVATIONS e.g. 89 -- the model's pass reaches 90 deg only at one instant, so 89 stands for zenith as in eval29)"""
import csv, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../chiu_roy_model/model"))
import physical_model as pm

SCENES = {"urban": "urban_ours", "rural": "rural_ours_earth"}
ELEV = [float(e) for e in sys.argv[4].split(",")] if len(sys.argv) > 4 else [float(e) for e in range(10, 90, 10)]
CHUNK = 1000
KEYS = [(v, s) for v in ("all paths", "40 dB floor") for s in ("LoS", "NLoS")]


def ue_ds(tau, w):
    """(all paths, 40 dB floor) DS of one UE; tau from the direct path (LoS) or any reference (NLoS)."""
    tau = tau - tau.min()
    m = (w * tau).sum() / w.sum(); ds_all = np.sqrt(max((w * tau * tau).sum() / w.sum() - m * m, 0.0))
    ok = tau <= 2e-6; tau, w = tau[ok], w[ok]
    pdp = np.bincount(np.floor(tau * 1e9).astype(int), w); k = pdp >= pdp.max() * 1e-4
    c = np.arange(len(pdp))[k] + 0.5; q = pdp[k]; m = (q * c).sum() / q.sum()
    return ds_all, np.sqrt(max((q * c * c).sum() / q.sum() - m * m, 0.0)) * 1e-9


def job(arg):
    scen, chunk = arg
    M = pm.Model(SCENES[scen]); ts = [pm.reference_time(e) for e in ELEV]
    acc = {(e, k): [0, 0.0, 0.0, 0] for e in ELEV for k in KEYS}           # n, sum lg, sum lg^2, n with DS = 0
    rng = np.random.default_rng([2026, chunk])
    for _ in range(CHUNK):
        city = M.draw_city(rng)
        for e, t in zip(ELEV, ts):
            p = M.paths(t, city, geometric_blockage=True)
            los = bool(p["visible"][-1]); keep = p["visible"].copy(); keep[-1] = los
            tau, w = p["tau"][keep], p["amp"][keep]
            if los:
                tau, w = np.r_[0.0, tau], np.r_[1.0, w]
            if len(w) == 0:
                continue
            st = "LoS" if los else "NLoS"
            for v, ds in zip(("all paths", "40 dB floor"), ue_ds(tau, w)):
                a = acc[(e, (v, st))]
                if ds > 0:
                    a[0] += 1; a[1] += np.log10(ds); a[2] += np.log10(ds) ** 2
                else:
                    a[3] += 1
    return scen, acc


if __name__ == "__main__":
    n = {"urban": int(sys.argv[2]) if len(sys.argv) > 2 else 40000, "rural": int(sys.argv[3]) if len(sys.argv) > 3 else 80000}
    jobs = [(s, c) for s in SCENES for c in range(n[s] // CHUNK)]
    tot = {s: {(e, k): np.zeros(4) for e in ELEV for k in KEYS} for s in SCENES}
    with Pool(40) as pool:
        for i, (scen, acc) in enumerate(pool.imap_unordered(job, jobs)):
            for key, a in acc.items():
                tot[scen][key] += a
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(jobs)} chunks", flush=True)
    with open(sys.argv[1], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "version", "state", "elevation_deg", "mu_lgDS", "sigma_lgDS", "n_ue", "n_zero_ds", "n_cities"])
        for scen in SCENES:
            for (e, (v, s)), (k, s1, s2, z) in tot[scen].items():
                if k > 1:
                    mu = s1 / k
                    w.writerow([scen, v, s, int(e), round(mu, 4), round(np.sqrt(max(s2 / k - mu * mu, 0)), 4), int(k), int(z), n[scen]])
    print("wrote", sys.argv[1])
