"""Export Sionna-side pipeline inputs and baseline results to .mat for MATLAB.

Two jobs:
  1. UE positions + occupancy-grid metadata, so the MATLAB run can be driven by
     bit-identical receivers (the apples-to-apples requirement) and so the
     ported grid builder can be checked against the Python one.
  2. The baseline per-elevation metric sweeps, for the comparison figures.

Usage: python export_matlab_inputs.py
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.io import savemat

SRC = Path("/path/to/rician_urban_sim")
OUT = Path(__file__).resolve().parent.parent / "data"

RX_FILES = {"sieg": "rx_positions_sieg.pkl", "wharf_du": "rx_positions_wharf_du.pkl"}
RESULT_DIRS = ["results_sieg_2GHz", "results_sieg_20GHz", "results_sieg_11GHz",
               "results_wharf_du_2GHz", "results_wharf_du_20GHz"]

# Per-RX vectors carried across for the CDF overlays.
PER_RX = ["K_power_ratio", "tau_rms_mean_s", "los_probability", "ASD_deg", "ASA_deg",
          "ZSD_deg", "ZSA_deg", "P_los", "P_specular", "P_diffuse", "P_diffraction",
          "P_refraction", "P_nlos", "doppler_mean_hz", "doppler_rms_hz"]
SCALARS = ["elevation_deg", "frequency", "slant_dist_m", "n_rx_positions", "perturb_half_m"]


def export_rx(tag, fname):
    d = pickle.load(open(SRC / fname, "rb"))
    g = d["meta"]["grid_meta"]
    # Wharf used build_occupancy_grid (absolute-height) rather than the terrain
    # variant, so terrain-only keys such as free_z are absent there.
    m = {
        "rxPositions": d["rx_positions"],
        "rngSeed": d["meta"].get("rng_seed", 42),
        # occupied_xy is the grid builder's direct output and is what
        # +geom/occupancyGrid.m is validated against.
        "occupiedXY": g["occupied_xy"],
        "gridRes": g["grid_res"], "margin": g["margin"],
        "sceneBounds": [g["scene_xmin"], g["scene_xmax"], g["scene_ymin"], g["scene_ymax"]],
        "nTotal": g["n_total"], "nFree": g["n_free"], "nOccupied": g["n_occupied"],
        "gridKind": "terrain" if "free_z" in g else "absolute",
    }
    if "free_z" in g:
        m["freeZ"] = np.asarray(g["free_z"]).reshape(-1, 1)
    if "valid_xmin" in g:
        m["validBounds"] = [g["valid_xmin"], g["valid_xmax"], g["valid_ymin"], g["valid_ymax"]]
    if "ground_z_range" in g:
        m["groundZRange"] = list(g["ground_z_range"])

    savemat(OUT / f"rx_positions_{tag}.mat", m, do_compression=True)
    print(f"rx_positions_{tag}: {len(d['rx_positions'])} UEs, {g['n_free']} free / "
          f"{g['n_occupied']} occupied cells @ {g['grid_res']} m [{m['gridKind']}]")


def export_results(dirname):
    files = sorted((SRC / dirname).glob("results_elev*.pkl"))
    if not files:
        return
    out, elevs = {}, []
    for f in files:
        r = pickle.load(open(f, "rb"))
        e = int(round(r["elevation_deg"]))
        elevs.append(e)
        for k in PER_RX:
            if k in r and r[k] is not None:
                out.setdefault(k, {})[f"e{e:03d}"] = np.asarray(r[k], float).reshape(-1, 1)
        for k in SCALARS:
            out.setdefault(k, {})[f"e{e:03d}"] = float(r[k])
        out.setdefault("tx_pos", {})[f"e{e:03d}"] = np.asarray(r["tx_pos"], float).reshape(1, -1)

    out["elevations"] = np.asarray(sorted(elevs), float).reshape(-1, 1)
    out["sourceDir"] = dirname
    savemat(OUT / f"sionna_{dirname.replace('results_', '')}.mat", out, do_compression=True)
    print(f"{dirname}: {len(files)} elevations {sorted(elevs)}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, fn in RX_FILES.items():
        export_rx(tag, fn)
    for d in RESULT_DIRS:
        export_results(d)
