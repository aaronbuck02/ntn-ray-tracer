"""Summary-table rows beyond K and DS, per state: mu/sigma of lgASA and lgZSA (per-UE values of the run; S: mean
of the V and H ports' lg), excess loss L_ex = -10 lg(P_los + P_nlos) - FSPL (S: ports summed) -> CL = NLoS mean, sigma_SF
= SD per state, LoS mean too; sigma_nu (geometric Doppler spread, sigma_nu_Hz; S: mean of the ports); cluster count by the
repo's clustering rule (Ray_Tracing/src/utils/cluster_id.py: KPowerMeans in delay-angle space, -25 dB pruning, direct path
excluded) on the raw 1-in-50 sample (~200 UEs, below the note's 300-UE rule; S: port powers summed per path); mean,
its standard error, 5th / 95th percentile.
Usage: python post/table2_extra.py RUNS SCENARIO BAND [ELEVS=30]   ->  RUNS/figures/table2_extra_SCENARIO_BAND.csv"""
import csv, os, pickle, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Ray_Tracing.src.utils.cluster_id import cluster_id_rx

runs, scen, band = sys.argv[1:4]; elevs = [int(e) for e in (sys.argv[4] if len(sys.argv) > 4 else "30").split(",")]


def one(e):
    fs = [f"{runs}/{scen}_S_{p}/results_elev{e:03d}deg.pkl" for p in "VH"] if band == "S" else [f"{runs}/{scen}_{band}/results_elev{e:03d}deg.pkl"]
    if not all(os.path.exists(f) for f in fs): return [], f"{scen} {band} {e}: no results"
    rs = [pickle.load(open(f, "rb")) for f in fs]
    g = lambda k: [np.asarray(r[k], float) for r in rs]
    P = sum(g("P_los")) + sum(g("P_nlos"))
    fspl = 20 * np.log10(4 * np.pi * np.asarray(rs[0]["slant_range_m"], float) * rs[0]["frequency"] / 299792458.0)
    lex = -10 * np.log10(P) - fspl
    lgasa, lgzsa = (np.mean([np.log10(v) for v in g(k)], 0) for k in ("ASA_deg", "ZSA_deg"))
    snu = np.mean(g("sigma_nu_Hz"), 0); los = np.asarray(rs[0]["los_probability"], float) > 0.5
    raws = [r["raw_results"] for r in rs]; del rs
    ncl = {True: [], False: []}; rows = []
    for i, x in enumerate(raws[0]):
        p = sum(np.abs(np.asarray(rr[i]["amps"])) ** 2 for rr in raws)         # ports summed: same paths in V and H
        assert all(len(rr[i]["taus"]) == len(x["taus"]) for rr in raws)
        t = np.asarray(x["taus"], float)
        cl, _ = cluster_id_rx({"dtau_s": t - t.min(), "amp": np.sqrt(p), "prim_type": x["inter_d0"], "phi_r": x["phi_r"],
                               "theta_r": x["theta_r"], "has_los": x["has_los"]})
        ncl[bool(x["has_los"])].append(len(cl))
    for state, sel in (("LoS", los), ("NLoS", ~los)):
        k = sel & np.isfinite(lex); c = np.array(ncl[state == "LoS"])
        st = lambda v: (round(float(np.nanmean(v[k])), 4), round(float(np.nanstd(v[k])), 4))
        rows.append([scen, band, e, state, int(k.sum()), *st(lgasa), *st(lgzsa), *st(lex),
                     round(float(np.nanmean(snu[k])), 4), round(float(np.nanmedian(snu[k])), 4), len(c),
                     *(np.round([c.mean(), c.std(ddof=1) / np.sqrt(len(c)) if len(c) > 1 else np.nan, *np.percentile(c, [5, 95])], 2)
                       if len(c) else ["", "", "", ""])])
    return rows, (f"{scen} {band} {e} LoS L_ex mean {np.nanmean(lex[los]):.2f} dB, NLoS {np.nanmean(lex[~los]):.2f} dB, "
          f"clusters LoS {np.mean(ncl[True]):.1f}, NLoS {np.mean(ncl[False]) if ncl[False] else float('nan'):.1f}")


if __name__ == "__main__":
    out = f"{runs}/figures/table2_extra_{scen}_{band}.csv"
    with Pool(3) as pool, open(out, "w", newline="") as fh:              # 3 elevations at a time: each holds a ~1 GB results file
        w = csv.writer(fh)
        w.writerow(["scenario", "band", "elevation_deg", "state", "n_ue", "mu_lgASA", "sigma_lgASA", "mu_lgZSA", "sigma_lgZSA", "Lex_mean_dB",
                    "Lex_sd_dB", "sigma_nu_mean_Hz", "sigma_nu_median_Hz", "n_raw", "clusters_mean", "clusters_se", "clusters_p5", "clusters_p95"])
        for rows, msg in pool.imap(one, elevs):
            w.writerows(rows); print(msg, flush=True)
    print("wrote", out)
