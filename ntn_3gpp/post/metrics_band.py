"""Ku/Ka per-elevation metrics in the layout of post/metrics.py, for post/plot_vs_elevation.py --band BAND.
Usage: python post/metrics_band.py RUNS SCENARIO BAND   ->  RUNS/figures/metrics_SCENARIO_BAND.csv
Lines (one RHCP member, no port sum):
  BAND        as run, isotropic UE: P_LoS and K from all 10000 UEs, DS pooled from every UE's path moments (the rule
              of post/metrics.py: delay profiles pooled per state, delays from the geometric direct-path time).
  BAND_panel  at the UE antenna (n x n panel on the satellite, gain relative to boresight, as the runner applies it):
              K from all UEs (P_nlos_panel); DS pooled from the raw 1-in-50 sample with each path weighted by the
              panel gain (the runner keeps per-UE panel DS only, which cannot be pooled).
Bars: 25th-75th percentile over the 10 scenes, as in post/metrics.py."""
import csv, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import COLS, MET, direct_times, sample_metrics
from raw_moments import raw_moments
C = 299792458.0


def unit(theta, phi):
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)


def panel_gain_db(u, x_ax, n):          # same as sim/run_stochastic_sim.py _panel_gain_db
    y_ax = np.cross([0.0, 0.0, 1.0], x_ax)
    y_ax = y_ax / np.linalg.norm(y_ax) if np.linalg.norm(y_ax) > 1e-9 else np.array([0.0, 1.0, 0.0])
    z_ax = np.cross(x_ax, y_ax)
    uy, uz = u @ y_ax, u @ z_ax
    th = np.degrees(np.arccos(np.clip(uz, -1.0, 1.0)))
    ph = np.degrees(np.arctan2(uy, u @ x_ax))
    a_e = -np.minimum(np.minimum(12 * ((th - 90) / 90) ** 2, 30) + np.minimum(12 * (ph / 90) ** 2, 30), 30)
    k = np.pi * np.arange(n)
    af = np.abs(np.exp(1j * np.outer(uy, k)).sum(1)) ** 2 * np.abs(np.exp(1j * np.outer(uz, k)).sum(1)) ** 2 / n ** 2
    return 2.0 + a_e + 10 * np.log10(af + 1e-30)


def panel_moments(raw, n):
    """Raw-sample moments with every path weighted by the panel gain re. boresight (boresight = satellite direction:
    the UE's direct path, or for an NLoS UE its scene's satellite position rebuilt from the scene's LoS UEs)."""
    sat = {}
    for x in raw:
        d = np.asarray(x["inter_d0"]) == 0
        if x["has_los"] and d.any():
            sat.setdefault(x["seed"], []).append(np.asarray(x["rx_nominal"], float) + C * float(x["taus"][d][0])
                                                 * unit(float(x["theta_r"][d][0]), float(x["phi_r"][d][0])))
    sat = {sd: np.median(v, 0) for sd, v in sat.items()}
    out = []
    for x, td in zip(raw, direct_times(raw)):
        if x["seed"] not in sat:
            continue
        b = sat[x["seed"]] - np.asarray(x["rx_nominal"], float); b /= np.linalg.norm(b)
        g = 10 ** ((panel_gain_db(unit(np.asarray(x["theta_r"], float), np.asarray(x["phi_r"], float)), b, n)
                    - panel_gain_db(b[None], b, n)[0]) / 10)
        out.append({**raw_moments({**x, "amps": np.asarray(x["amps"]) * np.sqrt(g)}), "t_dir": td})
    return out


if __name__ == "__main__":
    runs, scen, band = sys.argv[1:4]
    out = f"{runs}/figures/metrics_{scen}_{band}.csv"
    os.makedirs(f"{runs}/figures", exist_ok=True)
    zero = lambda ms: [{**x, "mom": np.zeros((6, 3))} for x in ms]            # second port of sample_metrics: none
    one = lambda seed: np.ones(6)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "line", "elevation_deg", *COLS])
        for e in range(10, 100, 10):
            f = f"{runs}/{scen}_{band}/results_elev{e:03d}deg.pkl"
            if not os.path.exists(f):
                continue
            r = pickle.load(open(f, "rb"))
            n = int(r["cfg_snapshot"]["UE_PANEL_N"])
            los = np.asarray(r["los_probability"], float) > 0.5
            seeds = np.asarray(r["scene_seeds"])
            pd, pn, pp = (np.asarray(r[k], float) for k in ("P_los", "P_nlos", "P_nlos_panel"))
            mom = [dict(seed=sd, rx_nominal=list(rx), has_los=lp > 0.5, t0=t0 if np.isfinite(t0) else 0.0, t_dir=td,
                        mom=np.asarray(m, float))
                   for sd, rx, lp, t0, td, m in zip(seeds, r["nominal_rx_positions"], r["los_probability"], r["t0_s"],
                                                    r["t_direct_s"], r["path_moments"])]
            pan = panel_moments(r["raw_results"], n)
            del r
            full = lambda sel, nd: {"P_LoS": los[sel].mean(), "K_LoS_dB": 10 * np.log10(pd[sel & los].sum() / nd[sel & los].sum())}
            lines = {band: (mom, pn), f"{band}_panel": (pan, pp)}
            rows, per = {}, {}
            with np.errstate(all="ignore"):
                for line, (ms, nd) in lines.items():
                    rows[line] = {**sample_metrics(ms, zero(ms), one), **full(np.ones(len(los), bool), nd)}
                    per[line] = {m: [] for m in MET}
                    for sd in np.unique(seeds):
                        sub = [x for x in ms if x["seed"] == sd]
                        v = {**sample_metrics(sub, zero(sub), one), **full(seeds == sd, nd)}
                        for m in MET:
                            per[line][m].append(v[m])
            for line, v in rows.items():
                for m in MET:
                    x = np.asarray(per[line][m], float); x = x[np.isfinite(x)]
                    v[f"{m}_q25"], v[f"{m}_q75"] = np.percentile(x, [25, 75]) if len(x) else (np.nan, np.nan)
                w.writerow([scen, line, e] + [round(float(v[c]), 4) for c in COLS])
                print(scen, band, line, e, f"K {v['K_LoS_dB']:.2f} dB  DS LoS/NLoS {v['DS_LoS_ns']:.1f}/{v['DS_NLoS_ns']:.1f} ns"
                      f" ({v['n_LoS_sample']}/{v['n_NLoS_sample']} UEs)", flush=True)
    print("wrote", out)
