"""Convert results_elev*.pkl into an ns-3 NTN elevation-table CSV.

Loss convention is -10*log10(P_los + P_nlos); lsp_stats reuses it exactly so the
two exported tables can never disagree. See docs/analysis.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from Ray_Tracing.src.utils.checkpoint_utils import load_all_results

C_LIGHT = 299_792_458.0  # m/s

# Circular-orbit speed for a ~600 km LEO reference (matches this project's
# other reference orbit defaults, e.g. orbital_utils.make_reference_satrec_leo).
DEFAULT_SAT_VELOCITY_KM_S = 7.56

FIELDS = [
    "elevation_deg", "los_probability_pct", "k_factor_db",
    "slant_range_km", "carrier_freq_hz", "fspl_db",
    "path_loss_los_db", "path_loss_nlos_db",
    "excess_loss_los_db", "excess_loss_nlos_db",
    "shadow_fading_std_los_db", "shadow_fading_std_nlos_db",
    "tau_rms_ns", "delay_s",
    "doppler_mean_hz", "doppler_rms_hz", "doppler_bound_hz",
    "n_rx_los", "n_rx_nlos", "scenario",
]


def free_space_path_loss_db(distance_m: float, freq_hz: float) -> float:
    """Analytic FSPL(d, f) = 20*log10(4*pi*d*f/c), in dB."""
    return 20.0 * np.log10(4.0 * np.pi * distance_m * freq_hz / C_LIGHT)


def loss_db_from_power(total_power):
    """Per-RX path loss [dB]. Receivers that received no power are NaN, not floored.

    Flooring with `maximum(P, 1e-300)` turns "solver found no paths" into a
    3000 dB sample. Shadow fading is a residual about the state mean, so even a
    couple of those (2 of 1918 NLOS RX at 30 deg on the UW campus) take sigma_SF
    from 5.7 dB to 91.4 dB. NaN keeps them out of every downstream statistic.
    """
    total_power = np.asarray(total_power, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total_power > 0, -10.0 * np.log10(total_power), np.nan)


def elevation_stats_row(elev_deg: float, res: dict, scenario: str,
                         sat_velocity_km_s: float) -> dict:
    """Collapse one elevation angle's RX ensemble (as saved by
`checkpoint_utils.save_results`) into the full row of NTN channel
statistics described in the module docstring.
    """
    P_los  = np.asarray(res["P_los"])
    P_nlos = np.asarray(res["P_nlos"])
    is_los = np.asarray(res["los_probability"]) >= 0.5  # per-RX boolean flag

    total_power = P_los + P_nlos
    loss_db_per_rx = loss_db_from_power(total_power)

    n_los  = int(np.sum(is_los))
    n_nlos = int(np.sum(~is_los))

    def _mean_std(mask):
        vals = loss_db_per_rx[mask]
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    path_loss_los_db,  sf_std_los_db  = _mean_std(is_los)
    path_loss_nlos_db, sf_std_nlos_db = _mean_std(~is_los)

    # K-factor over the LOS-classified subset only (physically: K is a
    # property of the LOS state, not the whole ensemble). Falls back to
    # NaN if this elevation had no LOS receivers.
    if n_los > 0:
        mean_Plos_los  = float(np.mean(P_los[is_los]))
        mean_Pnlos_los = float(np.mean(P_nlos[is_los]))
        k_lin = mean_Plos_los / (mean_Pnlos_los + 1e-300)
        k_factor_db = 10.0 * np.log10(k_lin + 1e-300)
    else:
        k_factor_db = float("nan")

    los_probability_pct = float(np.mean(is_los)) * 100.0
    tau_rms_ns = float(np.mean(res["tau_rms_mean_s"])) * 1e9

    slant_range_m = float(res["slant_dist_m"])
    carrier_freq_hz = float(res["frequency"])
    fspl_db = free_space_path_loss_db(slant_range_m, carrier_freq_hz)

    delay_s = slant_range_m / C_LIGHT
    doppler_bound_hz = abs(carrier_freq_hz / C_LIGHT * sat_velocity_km_s * 1e3)

    # Real per-pass Doppler (waypoint-mode runs only; zero-filled otherwise --
    # see checkpoint_utils.save_results / run_elevation_sim.py DOPPLER_ENABLED).
    _dmean = res.get("doppler_mean_hz")
    _drms  = res.get("doppler_rms_hz")
    doppler_mean_hz = float(np.mean(_dmean)) if _dmean else 0.0
    doppler_rms_hz  = float(np.mean(_drms))  if _drms  else 0.0

    return {
        "elevation_deg":              float(elev_deg),
        "los_probability_pct":        los_probability_pct,
        "k_factor_db":                k_factor_db,
        "slant_range_km":             slant_range_m / 1e3,
        "carrier_freq_hz":            carrier_freq_hz,
        "fspl_db":                    fspl_db,
        "path_loss_los_db":           path_loss_los_db,
        "path_loss_nlos_db":          path_loss_nlos_db,
        "excess_loss_los_db":         path_loss_los_db - fspl_db if n_los  else float("nan"),
        "excess_loss_nlos_db":        path_loss_nlos_db - fspl_db if n_nlos else float("nan"),
        "shadow_fading_std_los_db":   sf_std_los_db,
        "shadow_fading_std_nlos_db":  sf_std_nlos_db,
        "tau_rms_ns":                 tau_rms_ns,
        "delay_s":                    delay_s,
        "doppler_mean_hz":            doppler_mean_hz,
        "doppler_rms_hz":             doppler_rms_hz,
        "doppler_bound_hz":           doppler_bound_hz,
        "n_rx_los":                   n_los,
        "n_rx_nlos":                  n_nlos,
        "scenario":                   scenario,
    }


def build_table(results_dir: str, elevation_angles: list[float],
                 scenario: str, sat_velocity_km_s: float) -> list[dict]:
    """Load every available results_elev*.pkl and reduce each to one row."""
    all_results = load_all_results(results_dir, elevation_angles)
    if not all_results:
        raise FileNotFoundError(
            f"No results_elev*.pkl found in {results_dir!r} for elevations "
            f"{elevation_angles}. Run run_elevation_sim.py for each angle first."
        )
    rows = [elevation_stats_row(elev, res, scenario, sat_velocity_km_s)
            for elev, res in sorted(all_results.items())]
    return rows


def write_csv(rows: list[dict], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})
    return out_path


def write_meta(rows: list[dict], scenario: str, sat_velocity_km_s: float,
               out_path: str) -> str:
    meta_path = out_path + ".meta.json"
    freqs = sorted({row["carrier_freq_hz"] for row in rows})
    meta = {
        "scenario": scenario,
        "sat_velocity_km_s_assumed": sat_velocity_km_s,
        "carrier_freq_hz_values": freqs,
        "n_elevation_bins": len(rows),
        "elevation_deg_range": [rows[0]["elevation_deg"], rows[-1]["elevation_deg"]] if rows else None,
        "note": (
            "Doppler here is a worst-case bound (fc/c * sat_velocity), not a "
            "per-elevation instantaneous value -- instantaneous Doppler needs "
            "pass geometry; see build_ntn_channel_trace.py."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True,
                     help="Directory containing results_elev*.pkl from run_elevation_sim.py")
    ap.add_argument("--elevations", required=True, type=float, nargs="+",
                     help="Elevation angles [deg] that were simulated, e.g. --elevations 10 20 30 ... 90")
    ap.add_argument("--scenario", default="unspecified",
                     help="Scenario label for metadata only (e.g. dense_urban, urban, suburban, rural)")
    ap.add_argument("--sat_velocity_km_s", type=float, default=DEFAULT_SAT_VELOCITY_KM_S,
                     help=f"Satellite speed used only for the Doppler bound column (default: {DEFAULT_SAT_VELOCITY_KM_S} km/s, ~600 km circular LEO)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    rows = build_table(args.results_dir, args.elevations, args.scenario, args.sat_velocity_km_s)
    csv_path = write_csv(rows, args.out)
    meta_path = write_meta(rows, args.scenario, args.sat_velocity_km_s, args.out)
    print(f"Wrote {len(rows)} elevation rows -> {csv_path}")
    print(f"Wrote run metadata -> {meta_path}")
    for row in rows:
        print(
            f"  elev={row['elevation_deg']:5.1f}  "
            f"P(LoS)={row['los_probability_pct']:5.1f}%  "
            f"K={row['k_factor_db']:6.2f} dB  "
            f"FSPL={row['fspl_db']:7.2f} dB  "
            f"loss(LOS)={row['path_loss_los_db']:7.2f} dB  "
            f"loss(NLOS)={row['path_loss_nlos_db']:7.2f} dB  "
            f"SF_std(LOS)={row['shadow_fading_std_los_db']:5.2f}  "
            f"SF_std(NLOS)={row['shadow_fading_std_nlos_db']:5.2f}  "
            f"tau_rms={row['tau_rms_ns']:6.2f} ns  "
            f"delay={row['delay_s']*1e3:6.2f} ms  "
            f"doppler_mean={row['doppler_mean_hz']:8.1f} Hz  "
            f"doppler_rms={row['doppler_rms_hz']:7.1f} Hz  "
            f"n_los={row['n_rx_los']}  n_nlos={row['n_rx_nlos']}"
        )


if __name__ == "__main__":
    main()
