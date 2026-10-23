"""S-band normalized non-direct power-delay profiles per state (as the geometry-stochastic model's
eval29 PDPs): raw 1-in-50 sample, V+H summed, every UE's non-direct paths pooled (powers as received), 33 ns bins of
excess delay over the direct path, normalized to the state's total non-direct power. Results files that hold
the runner's per-UE binned PDP (pdp_class, path_moments) use all UEs instead of the raw sample. With diffuse as run and without
it (diffuse paths dropped, specular reflections restored, as in metrics.py).
For NLoS UEs the direct path is blocked; its delay is predicted from a LoS UE of the same scene (plane wave:
t_ref - u.(r - r_ref)/c, u = that UE's direct arrival direction; < 1 ns median error on LoS UEs).
Usage: python post/pdp.py RUNS SCENARIO   ->  RUNS/figures/pdp_SCENARIO.csv"""
import csv, os, pickle, sys
import numpy as np

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELEV = (10, 20, 40, 80)                         # the model's PDP elevations
EDGES = np.arange(201) * 33e-9                  # the model's grid: 200 bins of 33 ns
C = 299792458.0


def unit(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


if __name__ == "__main__":
    runs, scen = sys.argv[1:3]
    man = pickle.load(open(f"{KIT}/runs/{scen}_S_V/manifest.pkl", "rb"))
    sigma = {e["seed"]: float(e["cfg_overrides"]["GROUND_ROUGHNESS_M"]) for e in man["scenes"]}
    s_wall = float(man["wall_params"]["scattering_coefficient"])
    k = 2 * np.pi * 2e9 / C
    out = f"{runs}/figures/pdp_{scen}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "variant", "state", "elevation_deg", "delay_ns", "p", "n_UEs"])
        for e in ELEV:
            fs = [f"{runs}/{scen}_S_{p}/results_elev{e:03d}deg.pkl" for p in "VH"]
            if not all(os.path.exists(f) for f in fs):
                continue
            R = [pickle.load(open(f, "rb")) for f in fs]
            if "pdp_class" in R[0]:              # runner saved the binned PDP of every UE: use all UEs
                sg2 = 1 - np.exp(-(2 * k * np.array([sigma[sd] for sd in R[0]["scene_seeds"]]) * np.sin(np.radians(e))) ** 2)
                los = np.asarray(R[0]["los_probability"]) > 0.5
                pdp = np.asarray(R[0]["pdp_class"], float) + np.asarray(R[1]["pdp_class"], float)        # [UE, class, bin]
                m0 = np.asarray(R[0]["path_moments"])[:, :, 0] + np.asarray(R[1]["path_moments"])[:, :, 0]  # [UE, 6 classes]
                ok = 1 - sg2 > 1e-3
                wg = np.where(ok, 1 / np.maximum(1 - sg2, 1e-300), np.nan)
                for v, wp, wm in (("with", np.ones((len(los), 4)), np.ones((len(los), 6))),
                                  ("without", np.c_[wg, np.full(len(los), 1 / (1 - s_wall ** 2)), np.zeros(len(los)), np.ones(len(los))],
                                   np.c_[np.ones(len(los)), wg, np.full(len(los), 1 / (1 - s_wall ** 2)), np.zeros(len(los)),
                                         np.ones(len(los)), np.ones(len(los))])):
                    for s_name, sel in (("LoS", los), ("NLoS", ~los)):
                        if not sel.any():
                            continue
                        h = (pdp[sel] * wp[sel][:, :, None]).sum((0, 1))
                        tot = (m0[sel, 1:] * wm[sel, 1:]).sum()       # all non-direct power, beyond the grid too
                        if np.isfinite(tot) and tot > 0:
                            for c, val in zip((EDGES[:-1] + EDGES[1:]) / 2 * 1e9, h / tot):
                                if val > 0:
                                    w.writerow([scen, v, s_name, e, round(c, 1), f"{val:.4e}", int(sel.sum())])
                print(scen, e, "all UEs: LoS", int(los.sum()), "NLoS", int((~los).sum()), flush=True)
                continue
            rv, rh = (r["raw_results"] for r in R)
            ref = {}                                                  # scene seed -> (direct delay, position, -u)
            for x in rv:
                d = np.asarray(x["inter_d0"]) == 0
                if x["has_los"] and d.any() and x["seed"] not in ref:
                    ref[x["seed"]] = (float(x["taus"][d][0]), np.asarray(x["rx_nominal"], float),
                                      -unit(x["theta_r"][d][0], x["phi_r"][d][0]))
            acc = {(v, s): [np.zeros(len(EDGES) - 1), 0.0, 0] for v in ("with", "without") for s in ("LoS", "NLoS")}
            for a, b in zip(rv, rh):
                assert a["seed"] == b["seed"] and a["rx_nominal"] == b["rx_nominal"]
                ty = np.r_[a["inter_d0"], b["inter_d0"]]
                t = np.r_[a["taus"], b["taus"]].astype(float)
                p = np.abs(np.r_[a["amps"], b["amps"]]) ** 2
                below = np.r_[a["theta_r"], b["theta_r"]] > np.pi / 2
                if a["has_los"]:
                    t_dir = float(t[ty == 0][0])
                elif a["seed"] in ref:
                    t_ref, r_ref, u = ref[a["seed"]]
                    t_dir = t_ref + u @ (np.asarray(a["rx_nominal"], float) - r_ref) / C
                else:                                                 # no LoS UE in this scene's sample
                    t_dir = float(t.min())
                x = np.maximum(t - t_dir, 0.0)
                nd = ty != 0
                sg2 = 1 - np.exp(-(2 * k * sigma[a["seed"]] * np.sin(np.radians(e))) ** 2)
                dif = (np.c_[a["inter_all"], b["inter_all"]] == 2).any(0) if "inter_all" in a else ty == 2
                w_nodiff = np.select([dif, (ty == 1) & below, ty == 1],
                                     [0.0, 1 / (1 - sg2) if 1 - sg2 > 1e-3 else np.nan, 1 / (1 - s_wall ** 2)], 1.0)
                state = "LoS" if a["has_los"] else "NLoS"
                # ground specular reflection below 0.1 % of its power (rural, 20 deg up) is absent from the output: no "without"
                for v, wt in (("with", 1.0), ("without", w_nodiff if 1 - sg2 > 1e-3 else np.nan)):
                    q = (p * wt)[nd]
                    h, tot, n = acc[(v, state)]
                    acc[(v, state)] = [h + np.histogram(x[nd], EDGES, weights=q)[0], tot + q.sum(), n + 1]
            for (v, s), (h, tot, n) in acc.items():
                if n == 0 or not np.isfinite(tot) or tot <= 0:
                    continue
                for c, val in zip((EDGES[:-1] + EDGES[1:]) / 2 * 1e9, h / tot):
                    if val > 0:
                        w.writerow([scen, v, s, e, round(c, 1), f"{val:.4e}", n])
            print(scen, e, {f"{v}/{s}": n for (v, s), (_, _, n) in acc.items()}, flush=True)
    print("wrote", out)
