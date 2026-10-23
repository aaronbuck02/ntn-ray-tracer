"""S-band per-elevation metrics for the comparison figures, so only this small CSV needs to move between PCs.
Usage: python post/metrics.py RUNS SCENARIO   ->  RUNS/figures/metrics_SCENARIO.csv
Lines: S_VH (as run: P_LoS and K from all 10000 UEs, DS from the raw 1-in-50 sample, or from all UEs when the results
hold per-UE path_moments), S_VH_nodiff (same sample with
diffuse scattering removed), S_VH_sample (raw sample as run: sampling check). V and H ports summed per UE.
Each value also gets its 25th-75th percentile over scenes (the same quantity computed per scene).
DS follows the geometry-stochastic model's rule: the UEs' delay profiles pooled (powers as received, which share one
slant range), one RMS spread from the pool, per state and for all UEs together. Delays run from each UE's direct-path
time, blocked or not (the model's reference): the runner's geometric t_direct_s when saved for every UE, else for the
raw sample the UE's own direct path, or for an NLoS UE the distance to its scene's satellite position (rebuilt from the
scene's LoS UEs) / c.
"Without diffuse": diffuse paths dropped, specular reflections restored to full Fresnel strength (Sionna scales them by
sqrt(1 - S^2); ground S from each scene's roughness, walls/roofs S from the manifest)."""
import csv, os, pickle, sys
import numpy as np
C = 299792458.0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_moments import raw_moments

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = ("P_LoS", "K_LoS_dB", "DS_LoS_ns", "DS_NLoS_ns", "DS_all_ns")
# *_q25 / *_q75: 25th / 75th percentile over the 10 scenes of the same quantity computed scene by scene
COLS = (*MET, "n_LoS_sample", "n_NLoS_sample", *(f"{m}_{q}" for m in MET for q in ("q25", "q75")))


def ds(M):
    return np.sqrt(max(M[2] / M[0] - (M[1] / M[0]) ** 2, 0.0)) * 1e9 if M[0] > 0 else np.nan


def direct_times(raw):
    """Direct-path time of each raw-sample UE: its own direct path when LoS, else |S - rx| / c with S the satellite
    position of its scene (median of rx + c*tau*u over the scene's LoS UEs, u the direct arrival direction); NaN when the
    scene has no LoS UE in the sample."""
    def direct(x):
        d = np.asarray(x["inter_d0"]) == 0
        return (float(x["taus"][d][0]), float(x["theta_r"][d][0]), float(x["phi_r"][d][0])) if x["has_los"] and d.any() else None
    sat = {}
    for x in raw:
        if (dr := direct(x)):
            t, th, ph = dr
            u = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
            sat.setdefault(x["seed"], []).append(np.asarray(x["rx_nominal"], float) + C * t * u)
    sat = {sd: np.median(v, 0) for sd, v in sat.items()}
    return [direct(x)[0] if direct(x) else
            np.linalg.norm(sat[x["seed"]] - np.asarray(x["rx_nominal"], float)) / C if x["seed"] in sat else np.nan
            for x in raw]


def sample_metrics(mv, mh, weights):
    """Raw sample, V+H per UE, class powers scaled by weights(seed): K over LoS UEs, pooled DS per state and overall."""
    pool = {True: np.zeros(3), False: np.zeros(3)}
    num = den = np.float64(0.0)                                  # numpy: a scene with no LoS UE gives NaN, not an error
    n = {True: 0, False: 0}
    for a, b in zip(mv, mh):
        assert a["seed"] == b["seed"] and a["rx_nominal"] == b["rx_nominal"]
        # delays from the UE's direct-path time (the model's reference); without it, from the UE's first arrival
        t0 = a["t_dir"] if np.isfinite(a.get("t_dir", np.nan)) else min(a["t0"], b["t0"])
        m = np.zeros((6, 3))
        for x in (a, b):                                          # shift each port's moments to the common t0
            s = x["t0"] - t0
            m0, m1, m2 = x["mom"].T
            m += np.c_[m0, m1 + s * m0, m2 + 2 * s * m1 + s * s * m0]
        m *= weights(a["seed"])[:, None]
        pool[a["has_los"]] += m.sum(0)
        n[a["has_los"]] += 1
        if a["has_los"]:
            num += m[0, 0]
            den += m[1:, 0].sum()
    with np.errstate(all="ignore"):
        return {"K_LoS_dB": 10 * np.log10(num / den), "DS_LoS_ns": ds(pool[True]), "DS_NLoS_ns": ds(pool[False]),
                "DS_all_ns": ds(pool[True] + pool[False]), "n_LoS_sample": n[True], "n_NLoS_sample": n[False]}


if __name__ == "__main__":
    runs, scen = sys.argv[1:3]
    man = pickle.load(open(f"{KIT}/runs/{scen}_S_V/manifest.pkl", "rb"))
    sigma = {e["seed"]: float(e["cfg_overrides"]["GROUND_ROUGHNESS_M"]) for e in man["scenes"]}
    s_wall = float(man["wall_params"]["scattering_coefficient"])
    k = 2 * np.pi * 2e9 / 299792458.0
    os.makedirs(f"{runs}/figures", exist_ok=True)
    out = f"{runs}/figures/metrics_{scen}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "line", "elevation_deg", *COLS])
        for e in range(10, 100, 10):
            fs = [f"{runs}/{scen}_S_{p}/results_elev{e:03d}deg.pkl" for p in "VH"]
            if not all(os.path.exists(f) for f in fs):
                continue
            ue, mom = [], []
            for f in fs:
                r = pickle.load(open(f, "rb"))
                ue.append({c: np.asarray(r[c], float) for c in ("los_probability", "P_los", "P_nlos", "scene_seeds")})
                if "path_moments" in r:          # runner saved moments for every UE: use all of them, not the 1-in-50 sample
                    # a UE with no paths has t0 = NaN and zero moments: any t0 keeps it out of the pool
                    mom.append([dict(seed=sd, rx_nominal=list(rx), has_los=lp > 0.5, t0=t0 if np.isfinite(t0) else 0.0,
                                     t_dir=td, mom=np.asarray(m, float))
                                for sd, rx, lp, t0, td, m in zip(r["scene_seeds"], r["nominal_rx_positions"], r["los_probability"],
                                                                 r["t0_s"], r.get("t_direct_s", np.full(len(r["t0_s"]), np.nan)),
                                                                 r["path_moments"])])
                elif "raw_results" in r:
                    mom.append([{**raw_moments(x), "t_dir": td} for x, td in zip(r["raw_results"], direct_times(r["raw_results"]))])
                else:                    # slimmed file: moments only, no direct-path time -> delays from the first arrival
                    print(f"  {f}: no direct-path times, NLoS delays from the first arrival", flush=True)
                    mom.append(r["raw_moments"])
                del r
            los = ue[0]["los_probability"] > 0.5
            pd, pn = ue[0]["P_los"] + ue[1]["P_los"], ue[0]["P_nlos"] + ue[1]["P_nlos"]   # V+H: powers add exactly
            full_of = lambda sel: {"P_LoS": los[sel].mean(), "K_LoS_dB": 10 * np.log10(pd[sel & los].sum() / pn[sel & los].sum())}
            full = full_of(np.ones(len(los), bool))
            seeds = ue[0]["scene_seeds"]
            by_scene = lambda ms, sd: [x for x in ms if x["seed"] == sd]
            sg2 = lambda seed: 1 - np.exp(-(2 * k * sigma[seed] * np.sin(np.radians(e))) ** 2)
            # ground specular reflection kept as 1 - S_g^2 of its power; below 0.1 % it cannot be scaled back (rural, 20 deg up): NaN
            nodiff = lambda seed: np.array([1, 1 / (1 - sg2(seed)) if 1 - sg2(seed) > 1e-3 else np.nan,
                                            1 / (1 - s_wall ** 2), 0, 1, 1])
            asrun = sample_metrics(*mom, lambda seed: np.ones(6))
            rows = {"S_VH": {**asrun, **full}, "S_VH_nodiff": {**sample_metrics(*mom, nodiff), "P_LoS": full["P_LoS"]},
                    "S_VH_sample": {**asrun, "P_LoS": np.nan}}
            per = {line: {m: [] for m in MET} for line in rows}          # per-scene values -> quartiles
            with np.errstate(all="ignore"):
                for sd in np.unique(seeds):
                    f_s = full_of(seeds == sd)
                    a_s = sample_metrics(by_scene(mom[0], sd), by_scene(mom[1], sd), lambda seed: np.ones(6))
                    n_s = sample_metrics(by_scene(mom[0], sd), by_scene(mom[1], sd), nodiff)
                    for line, v in (("S_VH", {**a_s, **f_s}), ("S_VH_nodiff", {**n_s, "P_LoS": f_s["P_LoS"]}),
                                    ("S_VH_sample", {**a_s, "P_LoS": np.nan})):
                        for m in MET:
                            per[line][m].append(v[m])
            for line, v in rows.items():
                for m in MET:
                    x = np.asarray(per[line][m], float)
                    x = x[np.isfinite(x)]
                    v[f"{m}_q25"], v[f"{m}_q75"] = np.percentile(x, [25, 75]) if len(x) else (np.nan, np.nan)
            for line, v in rows.items():
                w.writerow([scen, line, e] + [round(float(v[c]), 4) for c in COLS])
            print(scen, e, f"P_LoS {full['P_LoS']:.3f}  K {full['K_LoS_dB']:.2f} dB  DS LoS/NLoS {asrun['DS_LoS_ns']:.1f}/{asrun['DS_NLoS_ns']:.1f} ns",
                  f"(sample {asrun['n_LoS_sample']}/{asrun['n_NLoS_sample']} UEs)", flush=True)
    print("wrote", out)
