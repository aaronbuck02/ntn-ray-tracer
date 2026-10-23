"""The geometry-stochastic model's LoS K at the UE panel (post/plot_generator_panel_k.py). The model gives every
path its geometry, so the n x n panel (post/metrics_band.py panel_gain_db, boresight on the satellite) weights each path
by its arrival direction: walls / roof edges / lit-side edges from their interaction point, the ground reflection from the
image direction (elevation -beta). LoS state as the model's run_v2_all.py: expected weights amp * visibility probability
* wall weight over drawn cities, ground W_g, no blocking-wall path. K = 1 / sum(w g), against K_iso = 1 / sum(w).
Polarization average by default; PM_POL=RHCP keeps only the RHCP part of each reflection, as the RT UE at Ku/Ka.
Carrier from PM_FC_HZ (set it for Ku/Ka).
Usage: PM_FC_HZ=20e9 [PM_POL=RHCP] python post/model_panel_k.py OUT_CSV N_PANEL [N_CITIES=3000]"""
import csv, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../chiu_roy_model/model"))
import physical_model as pm
from metrics_band import panel_gain_db

SCENES = {"urban": ("urban_ours", 5), "rural": ("rural_ours_earth", 6)}          # scene, the seed run_v2_all uses
ELEV = [float(e) for e in range(10, 90, 10)] + [89.0]


def job(arg):
    scen, e, npanel, nc = arg
    name, seed = SCENES[scen]
    M = pm.Model(name); t = pm.reference_time(e); s = pm.satellite(t)
    b = (s - pm.UE) / np.linalg.norm(s - pm.UE); h = np.hypot(b[0], b[1])
    img = np.array([b[0], b[1], -b[2]])                                          # ground image direction
    g0 = panel_gain_db(b[None], b, npanel)[0]
    rng = np.random.default_rng(seed); P = Pg = 0.0; Wg = None
    for _ in range(nc):
        p = M.paths(t, M.draw_city(rng)); mech = p["mechanism"]
        los = np.isin(mech, (1, 2, 5)); w = (p["amp"] * p["prob"] * p["weight"])[los]
        d = p["points"][:, los] - pm.UE[:, None]; u = (d / np.linalg.norm(d, axis=0)).T
        g = 10 ** ((panel_gain_db(u, b, npanel) - g0) / 10) if len(w) else np.zeros(0)
        P += w.sum(); Pg += (w * g).sum(); Wg = p["amp"][mech == 3][0]
    P /= nc; Pg /= nc
    gg = 10 ** ((panel_gain_db(img[None], b, npanel)[0] - g0) / 10)
    # TR 38.811 definition: one drawn city per UE with ray-tested visibility, each LoS UE's K in dB, then the mean
    rng = np.random.default_rng(seed + 1000); ki, kp = [], []
    for _ in range(nc):
        p = M.paths(t, M.draw_city(rng), geometric_blockage=True)
        if not p["visible"][-1]: continue
        mech = p["mechanism"]; k = p["visible"] & np.isin(mech, (1, 2, 5)); w = p["amp"][k]
        d = p["points"][:, k] - pm.UE[:, None]; u = (d / np.linalg.norm(d, axis=0)).T
        g = 10 ** ((panel_gain_db(u, b, npanel) - g0) / 10) if len(w) else np.zeros(0)
        wg = p["amp"][mech == 3][0]
        ki.append(10 * np.log10(1 / (w.sum() + wg))); kp.append(10 * np.log10(1 / ((w * g).sum() + wg * gg)))
    return scen, e, 10 * np.log10(1 / (P + Wg)), 10 * np.log10(1 / (Pg + Wg * gg)), float(np.mean(ki)), float(np.mean(kp))


if __name__ == "__main__":
    out, npanel = sys.argv[1], int(sys.argv[2]); nc = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    with Pool(18) as pool, open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["scenario", "elevation_deg", "K_iso_dB", "K_panel_dB", "fc_GHz", "n_cities", "K_iso_mean_dB", "K_panel_mean_dB"])
        for scen, e, k0, k1, m0, m1 in pool.imap(job, [(s, e, npanel, nc) for s in SCENES for e in ELEV]):
            w.writerow([scen, int(e), round(k0, 2), round(k1, 2), pm.FC / 1e9, nc, round(m0, 2), round(m1, 2)])
            print(f"{scen} {e:g} deg: model K isotropic {k0:5.1f} dB, at the {npanel}x{npanel} panel {k1:5.1f} dB", flush=True)
