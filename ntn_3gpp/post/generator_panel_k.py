"""Does a TR 38.811-style generator, fed with our isotropic RT parameters, reproduce the RT K at the UE panel?
Per LoS UE of the RT run (isotropic): its own K, ASA, ZSA drive one 38.901 Sec. 7.5 cluster draw (LoS case, as ns-3
implements it: cluster count, r_tau, c_ASA, c_ZSA and per-cluster shadowing from the 38.811 row of the band; at Ku the
S-band row); the n x n panel (post/metrics_band.py panel_gain_db, boresight on the satellite) weights every ray. Pooled
K_panel = sum P_los / sum (P_nlos * mean ray gain) is compared with the RT K at the panel (sum P_los / sum P_nlos_panel).
Cluster powers do not depend on DS (tau / DS cancels), so DS is not needed.
Two more generator cases:
  38.811 inputs   the table's own K, ASA, ZSA means (Ka row at Ka, S row at Ku): what an evaluator gets today.
  proposed        as the RT-input case, with every cluster's zenith moved from the LoS zenith by dZOA (azimuth, the
                  standard all-path ASA / ZSA and the LoS ray unchanged). dZOA: one value per elevation, the mean over the
                  raw 1-in-50 LoS UEs of (power-weighted circular mean zenith of the non-direct paths - LoS zenith).
Usage: python post/generator_panel_k.py RUNS SCENARIO BAND   ->  RUNS/figures/generator_panel_k_SCENARIO_BAND.csv"""
import csv, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Ray_Tracing/src/utils"))
from metrics_band import panel_gain_db
import ns3_reference

C_PHI = {2: 0.501, 3: 0.680, 4: 0.779, 5: 0.860, 8: 1.018, 10: 1.090, 12: 1.146}       # 38.901 Table 7.5-2 (+ NTN N=2,3,4, ns-3)
C_THETA = {2: 0.430, 3: 0.594, 4: 0.697, 8: 0.889, 10: 0.957, 12: 1.104}                # 38.901 Table 7.5-4 (+ NTN)
ALPHA = np.array([0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195, 2.1551])
ALPHA = np.r_[ALPHA, -ALPHA]                                                           # 20 ray offsets, 38.901 Table 7.5-3
N_UE = 10000                                                   # all LoS UEs (at most 10000)


def mean_ray_gain(k_lin, asa, zsa, row, elev, n_panel, rng, dzoa=0.0):
    """Power-weighted mean panel gain (re. boresight) of the NLoS rays of one 38.901 LoS draw per UE (vectorized).
    dzoa [deg]: zenith offset of the clusters from the LoS (0 = 38.901)."""
    U, N, rt, zeta = len(k_lin), int(row["numOfCluster"]), row["rTau"], row["perClusterShadowingStd"]
    tau = -rt * np.log(rng.random((U, N))); tau = np.sort(tau - tau.min(1, keepdims=True), 1)     # in units of DS
    p = np.exp(-tau * (rt - 1) / rt) * 10 ** (-rng.normal(0, zeta, (U, N)) / 10)
    p /= p.sum(1, keepdims=True)
    p = np.where(p < 10 ** -2.5 * p.max(1, keepdims=True), 0.0, p); p /= p.sum(1, keepdims=True)  # drop < -25 dB clusters
    kdb = 10 * np.log10(k_lin)[:, None]
    pa = p / (k_lin[:, None] + 1); pa[:, 0] += k_lin / (k_lin + 1)                       # powers used for the angles
    lg = -np.log(np.maximum(pa, 1e-300) / pa.max(1, keepdims=True))
    cphi = C_PHI[N] * (1.1035 - 0.028 * kdb - 2e-3 * kdb ** 2 + 1e-4 * kdb ** 3)
    cth = C_THETA[N] * (1.3086 + 0.0339 * kdb - 0.0077 * kdb ** 2 + 2e-4 * kdb ** 3)
    th_los, ph_los = 90.0 - elev, 0.0
    x = np.where(rng.random((U, N)) < 0.5, -1.0, 1.0)
    ph = x * 2 * (asa[:, None] / 1.4) * np.sqrt(lg) / cphi + rng.normal(0, 1, (U, N)) * asa[:, None] / 7
    th = x * zsa[:, None] * lg / cth + rng.normal(0, 1, (U, N)) * zsa[:, None] / 7
    ph = ph - (ph[:, :1] - ph_los); th = th - (th[:, :1] - th_los - dzoa)               # first cluster on the LoS (7.5-12, 7.5-17)
    ph = ph[:, :, None] + row["cASA"] * ALPHA; th = th[:, :, None] + row["cZSA"] * ALPHA   # rays
    th = np.mod(th, 360.0); th = np.where(th > 180, 360 - th, th)
    t, f = np.radians(th), np.radians(ph)
    u = np.stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)], -1).reshape(-1, 3)
    b = np.array([np.sin(np.radians(th_los)), 0.0, np.cos(np.radians(th_los))])
    g = 10 ** ((panel_gain_db(u, b, n_panel) - panel_gain_db(b[None], b, n_panel)[0]) / 10).reshape(U, N, len(ALPHA))
    return (p[:, :, None] / len(ALPHA) * g).sum((1, 2))


if __name__ == "__main__":
    runs, scen, band = sys.argv[1:4]
    table = ns3_reference.parse_ntn_table(f"NTN{scen.capitalize()}LOS")
    rng = np.random.default_rng(1)
    out = f"{runs}/figures/generator_panel_k_{scen}_{band}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "elevation_deg", "n_ue", "K_iso_dB", "K_panel_RT_dB", "K_panel_generator_dB", "K_38811_dB",
                    "ASA_deg_median", "ZSA_deg_median", "n_clusters", "K_panel_generator_38811_dB",
                    "n_raw", "dZOA_deg", "K_panel_proposed_dB",
                    # TR 38.811 definition: mean over UEs of each UE's K in dB
                    "K_iso_mean_dB", "K_panel_RT_mean_dB", "K_panel_generator_mean_dB", "K_panel_generator_38811_mean_dB",
                    "K_panel_proposed_mean_dB"])
        for e in range(10, 100, 10):
            f = f"{runs}/{scen}_{band}/results_elev{e:03d}deg.pkl"
            if not os.path.exists(f):
                continue
            r = pickle.load(open(f, "rb"))
            n_panel = int(r["cfg_snapshot"]["UE_PANEL_N"])
            v = {k: np.asarray(r[k], float) for k in ("P_los", "P_nlos", "P_nlos_panel", "ASA_deg", "ZSA_deg", "los_probability")}
            raw = [x for x in r["raw_results"] if x["has_los"] and np.isfinite(x["ASA_deg"] + x["ZSA_deg"]) and x["P_nlos"] > 0]
            del r
            dz = []                                                  # raw sample: zenith offset of the non-direct power from the LoS
            for x in raw:
                nd = np.asarray(x["inter_d0"]) != 0; th = np.asarray(x["theta_r"], float)
                dz.append(np.angle((np.abs(np.asarray(x["amps"])[nd]) ** 2 * np.exp(1j * th[nd])).sum()) - th[~nd][0])
            dzoa = float(np.degrees(np.mean(dz)))
            ok = (v["los_probability"] > 0.5) & (v["P_nlos"] > 0) & np.isfinite(v["ASA_deg"] + v["ZSA_deg"])
            i = rng.permutation(np.flatnonzero(ok))[:N_UE]
            row = table[("Ka" if band == "Ka" else "S", e)]
            gbar = mean_ray_gain(v["P_los"][i] / v["P_nlos"][i], v["ASA_deg"][i], v["ZSA_deg"][i], row, e, n_panel, rng)
            pl, pn, pp = v["P_los"][i].sum(), v["P_nlos"][i].sum(), v["P_nlos_panel"][i].sum()
            # 38.811's own means, one value per drop (no LSP spread drawn)
            k38 = np.full(N_UE, 10 ** (row["uK"] / 10))
            g38 = mean_ray_gain(k38, np.full(N_UE, 10 ** row["uLgASA"]), np.full(N_UE, 10 ** row["uLgZSA"]), row, e, n_panel, rng)
            gz = mean_ray_gain(v["P_los"][i] / v["P_nlos"][i], v["ASA_deg"][i], v["ZSA_deg"][i], row, e, n_panel, rng, dzoa)
            vals = [10 * np.log10(pl / pn), 10 * np.log10(pl / pp), 10 * np.log10(pl / (v["P_nlos"][i] * gbar).sum()), row["uK"],
                    np.median(v["ASA_deg"][i]), np.median(v["ZSA_deg"][i]), int(row["numOfCluster"]),
                    row["uK"] + 10 * np.log10(1 / g38.mean()), len(raw), dzoa, 10 * np.log10(pl / (v["P_nlos"][i] * gz).sum())]
            mdb = lambda k: float(np.mean(10 * np.log10(k)))
            vals += [mdb(v["P_los"][i] / v["P_nlos"][i]), mdb(v["P_los"][i] / v["P_nlos_panel"][i]), mdb(v["P_los"][i] / (v["P_nlos"][i] * gbar)),
                     mdb(k38 / g38), mdb(v["P_los"][i] / (v["P_nlos"][i] * gz))]
            w.writerow([scen, e, len(i)] + [round(float(x), 2) for x in vals])
            print(f"{scen} {band} {e:2d} deg ({len(i)} LoS UEs): K isotropic {vals[0]:5.1f} dB | at the {n_panel}x{n_panel} panel: "
                  f"RT {vals[1]:5.1f} dB, generator {vals[2]:5.1f} dB, generator with 38.811 inputs {vals[7]:5.1f} dB, "
                  f"proposed (dZOA {dzoa:5.1f} deg from {len(raw)} UEs) {vals[10]:5.1f} dB", flush=True)
    print("wrote", out)
