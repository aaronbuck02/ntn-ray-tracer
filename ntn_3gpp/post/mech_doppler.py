"""S-band mechanism split and Doppler spread from the raw 1-in-50 path sample.
Per scenario, elevation, state and variant (with diffuse as run / without: diffuse dropped, specular reflections restored
as in metrics.py), pooled over UEs, V+H summed:
  - non-direct power and path count per mechanism: diffuse (a diffuse bounce at any depth when the sample keeps every
    bounce's type, else a diffuse first bounce), ground (specular reflection arriving from below), wall (specular reflection
    arriving from above), and diffraction split by Keller's cone: corner (a
    vertical edge sends its diffracted rays down at the satellite's own elevation: arrival within 1 deg of it), roof (all
    other diffraction: roof edges, and diffraction followed by more bounces); counts are paths per UE.
  - Doppler moments of the non-direct paths, Doppler relative to the UE's own direct path (blocked or not), as the
    geometry-stochastic model defines it: sum p, sum p*dnu, sum p*dnu^2.
Path Doppler nu = v_sat . u_dep / lambda (u_dep from the departure angles, as the runner computes it). The satellite
position of each scene comes from a LoS UE (rx - c*tau_direct*u_direct); its velocity from the runner's orbit (circular,
through the zenith, moving toward it).
Usage: python post/mech_doppler.py RUNS SCENARIO   ->  RUNS/figures/mech_doppler_SCENARIO.csv"""
import csv, os, pickle, sys
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C, F = 299792458.0, 2e9
R_EARTH, MU = 6371.0e3, 3.986004418e14          # as the runner
MECH = ("ground", "wall", "roof", "corner", "diffuse")


def unit(th, ph):
    th, ph = np.asarray(th, float), np.asarray(ph, float)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)


def u_dep_geo(S, rx, tau, a):
    """Departure direction of single-interaction paths from geometry: the interaction point lies along the arrival
    direction a (unit vectors, one row per path) at the distance that makes the two legs add up to c*tau."""
    D, L = S - rx, C * np.asarray(tau, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = rx + ((L ** 2 - D @ D) / (2 * (L - a @ D)))[:, None] * a
        return (P - S) / np.linalg.norm(P - S, axis=1, keepdims=True)


def sat_velocity(s):
    """The runner's _sat_velocity: circular orbit through the scene zenith, moving toward the zenith."""
    r = s - np.array([0.0, 0.0, -R_EARTH])
    up = np.array([0.0, 0.0, 1.0])
    t = up - r * (up @ r) / (r @ r)
    t = t if np.linalg.norm(t) > 1e-9 else np.array([1.0, 0.0, 0.0])
    return np.sqrt(MU / np.linalg.norm(r)) * t / np.linalg.norm(t)


if __name__ == "__main__":
    runs, scen = sys.argv[1:3]
    man = pickle.load(open(f"{KIT}/runs/{scen}_S_V/manifest.pkl", "rb"))
    sigma = {e["seed"]: float(e["cfg_overrides"]["GROUND_ROUGHNESS_M"]) for e in man["scenes"]}
    s_wall = float(man["wall_params"]["scattering_coefficient"])
    k = 2 * np.pi * F / C
    out = f"{runs}/figures/mech_doppler_{scen}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "variant", "state", "elevation_deg", *(f"P_{m}" for m in MECH), *(f"N_{m}" for m in MECH),
                    "dop_m0", "dop_m1", "dop_m2", "n_UEs"])
        for e in range(10, 100, 10):
            fs = [f"{runs}/{scen}_S_{p}/results_elev{e:03d}deg.pkl" for p in "VH"]
            if not all(os.path.exists(f) for f in fs):
                continue
            rv, rh = (pickle.load(open(f, "rb"))["raw_results"] for f in fs)
            sat = {}                                                  # scene seed -> satellite position estimates
            for x in rv:
                d = np.asarray(x["inter_d0"]) == 0
                if x["has_los"] and d.any():
                    u = unit(x["theta_t"][d][0], x["phi_t"][d][0])
                    sat.setdefault(x["seed"], []).append(np.asarray(x["rx_nominal"], float) - C * float(x["taus"][d][0]) * u)
            sat = {sd: np.median(v, 0) for sd, v in sat.items()}
            acc = {}
            for a, b in zip(rv, rh):
                assert a["seed"] == b["seed"] and a["rx_nominal"] == b["rx_nominal"]
                if a["seed"] not in sat:                              # scene without a LoS UE in the sample
                    continue
                S = sat[a["seed"]]
                v = sat_velocity(S)
                rx = np.asarray(a["rx_nominal"], float)
                nu0 = v @ ((rx - S) / np.linalg.norm(rx - S)) * F / C
                ty, below = np.asarray(a["inter_d0"]), np.asarray(a["theta_r"]) > np.pi / 2
                u_dep = unit(a["theta_t"], a["phi_t"])
                # satellite within 1 deg of the zenith: the saved float32 departure angles sit next to the pole, where they
                # are off by ~1e-4 rad (several Hz); rebuild the directions from geometry instead.
                # single-bounce geometry; paths with 2-3 bounces (< 1 % of the power at 90 deg) come out wrong
                if np.pi - np.median(a["theta_t"]) < np.radians(1):
                    u_dep = u_dep_geo(S, rx, a["taus"], unit(a["theta_r"], a["phi_r"]))
                dnu = u_dep @ v * F / C - nu0
                p = np.abs(np.asarray(a["amps"])) ** 2 + np.abs(np.asarray(b["amps"])) ** 2   # same paths in both ports
                corner = (ty == 8) & (np.abs(90 - np.degrees(np.asarray(a["theta_r"], float)) - e) < 1.0)
                # newer raw samples keep every bounce's type: a later diffuse bounce makes the path diffuse
                dif = (np.asarray(a["inter_all"]) == 2).any(0) if "inter_all" in a else ty == 2
                mech = np.select([dif, (ty == 1) & below, ty == 1, corner, ty == 8], [4, 0, 1, 3, 2], -1)
                sg2 = 1 - np.exp(-(2 * k * sigma[a["seed"]] * np.sin(np.radians(e))) ** 2)
                w_nod = np.select([dif, (ty == 1) & below, ty == 1],
                                  [0.0, 1 / (1 - sg2) if 1 - sg2 > 1e-3 else np.nan, 1 / (1 - s_wall ** 2)], 1.0)
                state = "LoS" if a["has_los"] else "NLoS"
                for var, wt in (("with", np.ones(len(p))), ("without", w_nod)):
                    q = p * wt
                    r = acc.setdefault((var, state), np.zeros(14))
                    for i in range(5):
                        sel = mech == i
                        r[i] += q[sel].sum()
                        r[5 + i] += np.count_nonzero(sel & (wt != 0))
                    nd = mech >= 0
                    r[10:13] += [q[nd].sum(), (q[nd] * dnu[nd]).sum(), (q[nd] * dnu[nd] ** 2).sum()]
                    r[13] += 1
            for (var, state), r in sorted(acc.items()):
                w.writerow([scen, var, state, e, *[f"{x:.6e}" for x in r[:5]], *[round(x / r[13], 3) for x in r[5:10]],
                            *[f"{x:.6e}" for x in r[10:13]], int(r[13])])
            print(scen, e, {f"{v}/{s}": int(r[13]) for (v, s), r in acc.items()}, flush=True)
    print("wrote", out)
