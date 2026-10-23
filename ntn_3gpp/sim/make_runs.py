"""Scene lists, configs and solve groups for the main sweep.

    python sim/make_runs.py OUT_DIR [--smoke] [--scenario urban|rural] [--bands S Ku Ka]

Writes OUT_DIR/<scenario>_<band>[_<pol>]/{config.json, manifest.pkl} and
OUT_DIR/groups/<scenario>_<band>.json (one solve group per scenario and band;
the S-band group reads V and H ports out of one ray trace).
"""
import argparse
import copy
import json
import os

import numpy as np

from Ray_Tracing.src.utils.variant_manifest import virtual_manifest, lhs_manifest, save_manifest

BASE_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "base_config.json")   # the 2 GHz sensitivity baseline


def p527_soil(f, mv=0.207, sand=41.96, clay=8.53, silt=49.51, rho_s=2.70, T=23.0):
    """ITU-R P.527-6 §5.2 soil (default: Table 2 loam), Eqs. (57)-(70) -> (eps_r, sigma S/m).
    mv = 0.207 fits P.2040-3's medium dry ground over 1-10 GHz within ~10%."""
    rho_b = 1.07256 + 0.078886 * np.log(sand) + 0.038753 * np.log(clay) + 0.032732 * np.log(silt)
    th = 300 / (T + 273.15) - 1
    es, einf = 77.66 + 103.3 * th, 3.52 - 7.52 * th
    e1 = 0.0671 * es
    f1 = 20.20 - 146.4 * th + 316 * th ** 2
    f2 = 39.8 * f1
    s1 = 0.0467 + 0.2204 * rho_b - 0.004111 * sand - 0.006614 * clay
    s2 = -1.645 + 1.939 * rho_b - 0.0225622 * sand + 0.01594 * clay
    se1 = (f / 1.35) * (s2 - s1) / (1 + (f / 1.35) ** 2)
    se2 = s2 + (s1 - s2) / (1 + (f / 1.35) ** 2)
    efw1 = ((es - e1) / (1 + (f / f1) ** 2) + (e1 - einf) / (1 + (f / f2) ** 2) + einf
            + 18 * se1 * (rho_s - rho_b) / (f * rho_s * mv))
    efw2 = ((f / f1) * (es - e1) / (1 + (f / f1) ** 2) + (f / f2) * (e1 - einf) / (1 + (f / f2) ** 2)
            + 18 * se2 * (rho_s - rho_b) / (f * rho_s * mv))
    a = 0.65
    bp = 1.2748 - 0.00519 * sand - 0.00152 * clay
    bpp = 1.33797 - 0.00603 * sand - 0.00166 * clay
    esm = (1.01 + 0.44 * rho_s) ** 2 - 0.062
    er = (1 + rho_b / rho_s * (esm ** a - 1) + mv ** bp * efw1 ** a - mv) ** (1 / a)
    ei = (mv ** bpp * efw2 ** a) ** (1 / a)
    return float(er), float(ei * f / 18)


def mat(a, b, c, d, s=0.4, alpha_r=10):
    # diffuse lobe around the mirror direction, alpha_R = 10 (earlier runs used "lambertian")
    return dict(a=a, b=b, c=c, d=d, scattering_coefficient=s, xpd_coefficient=0.0,
                thickness=2.0, scattering_pattern="directive", alpha_r=alpha_r)


CONCRETE = (5.24, 0.0, 0.0462, 0.7822)     # P.2040-3 Table 3
WOOD = (1.99, 0.0, 0.0047, 1.0718)

# gamma keeps the truncated mean at 18.8 m / 7 m.
SCENARIOS = {
    "urban": dict(alpha=0.30, beta=500.0, gamma=14.52, h_min=4.0, rough=0.01, seed=1000, rx_chunk=25,
                  wall=CONCRETE, roof=CONCRETE, ground=lambda f: CONCRETE),
    "rural": dict(alpha=0.08, beta=525.0, gamma=4.90, h_min=3.0, rough=0.10, seed=2000, rx_chunk=15,    # rough ground: ~0.5M diffuse paths per UE; a full candidate buffer halves the chunk
                  wall=WOOD, roof=WOOD,     # ground per band as (a, b, c, d) = (eps_r, 0, sigma, 0)
                  ground=lambda f: (p527_soil(f)[0], 0.0, p527_soil(f)[1], 0.0)),
}

BANDS = {"S": (2e9, 0, ["V", "H"]), "Ku": (12e9, 9, ["RHCP"]), "Ka": (20e9, 16, ["RHCP"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--smoke", action="store_true", help="1 scene, 50 UEs")
    ap.add_argument("--scenario", choices=list(SCENARIOS), help="only this scenario (default: both)")
    ap.add_argument("--bands", nargs="+", choices=list(BANDS), default=list(BANDS))
    args = ap.parse_args()
    n_scenes, n_rx = (1, 50) if args.smoke else (10, 1000)
    with open(BASE_CFG) as f:
        base_cfg = json.load(f)
    os.makedirs(os.path.join(args.out, "groups"), exist_ok=True)

    for sname, s in SCENARIOS.items():
        if args.scenario and sname != args.scenario:
            continue
        sk = dict(Lx=2000.0, Ly=2000.0, cell_res=40.0, alpha=s["alpha"], points_per_cell=4,
                  gamma=s["gamma"], h_min=s["h_min"], h_max=120.0, max_fill=0.8, min_side=5.0)
        base = virtual_manifest(n_scenes, sk, base_seed=s["seed"], primitive="block",
                                building_material="itu_concrete" if sname == "urban" else "itu_wood",
                                roof_material="itu_brick", ground_material="itu_plasterboard",
                                wall_params=mat(*s["wall"]), roof_params=mat(*s["roof"]),
                                ground_params=mat(*s["ground"](2.0), s=0.0))
        lhs = lhs_manifest(base, {
            "geo.alpha": (0.95 * s["alpha"], 1.05 * s["alpha"]),
            "geo.beta": (0.95 * s["beta"], 1.05 * s["beta"]),
            "geo.gamma": (0.95 * s["gamma"], 1.05 * s["gamma"]),
            "cfg.GROUND_ROUGHNESS_M": (0.95 * s["rough"], 1.05 * s["rough"]),
            "cfg.SCENE_EAST_HEADING_DEG": (0.0, 90.0),
        }, seed=s["seed"])
        for bname, (freq, panel, pols) in BANDS.items():
            if bname not in args.bands:
                continue
            man = copy.deepcopy(lhs)
            man["ground_params"] = mat(*s["ground"](freq / 1e9), s=0.0)
            members = []
            for pol in pols:
                name = f"{sname}_{bname}" + (f"_{pol}" if len(pols) > 1 else "")
                d = os.path.join(os.path.abspath(args.out), name)
                os.makedirs(d, exist_ok=True)
                cfg = dict(base_cfg)
                cfg.update(FREQUENCY=freq, SWEEP_ALTITUDE_KM=600.0, SCENE_RADIUS=850.0,
                           N_RX_PER_SCENE=n_rx, PPP_RX_MARGIN=500.0, PPP_RX_CLEARANCE=0.5,
                           RX_CHUNK=s["rx_chunk"], TX_POL_TYPE="RHCP", RX_POL_TYPE=pol, TX_AZIMUTH_DEG=0.0,
                           IONOSPHERE_ENABLED=False, ATMOSPHERE_ENABLED=False, DOPPLER_ENABLED=False,
                           RAW_SAVE_INTERVAL=50, CHECKPOINT_INTERVAL=1, GROUND_RETRY_MAX=0,
                           UE_PANEL_N=panel, DS_FLOOR_DB=40.0, MAX_EXCESS_NS=2000.0,
                           # 2026-10-15: roof edges diffract (walls + roof in one mesh); the 0.5 m stand-off holds
                           # for the final UE position (no UE inside a building)
                           CLOSED_BUILDINGS=True, PPP_RX_EXACT_CLEARANCE=True)
                with open(os.path.join(d, "config.json"), "w") as f:
                    json.dump(cfg, f, indent=2)
                save_manifest(man, os.path.join(d, "manifest.pkl"))
                members.append(dict(name=name, dir=d, config=os.path.join(d, "config.json"),
                                    manifest=os.path.join(d, "manifest.pkl")))
            with open(os.path.join(args.out, "groups", f"{sname}_{bname}.json"), "w") as f:
                json.dump({"members": members}, f, indent=2)
            g = man["ground_params"]
            print(f"{sname}_{bname}: {len(members)} member(s), ground eps_r {g['a']:.2f} sigma {g['c']:.3f}")


if __name__ == "__main__":
    main()
