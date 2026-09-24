"""Checkpoint and results I/O for the elevation-sweep and stochastic runs.

Each elevation writes checkpoint_elev{angle:03d}deg.pkl (resumable) and
results_elev{angle:03d}deg.pkl (final) into its own output directory, both
carrying the same per-RX arrays plus a full config snapshot.

K_moments is written but never populated; raw_results["P_scat"] duplicates
P_nlos and exists only so older result files still load.
See docs/pipelines.md for the full schema.
"""

import os
import pickle
import time


def checkpoint_path(output_dir: str, elevation_deg: float) -> str:
    """Path of the resumable checkpoint file for one elevation angle."""
    return os.path.join(output_dir, f"checkpoint_elev{int(elevation_deg):03d}deg.pkl")


def results_path(output_dir: str, elevation_deg: float) -> str:
    """Path of the final results file for one elevation angle."""
    return os.path.join(output_dir, f"results_elev{int(elevation_deg):03d}deg.pkl")


# ─── Save ─────────────────────────────────────────────────────────────────────

def save_checkpoint(
    output_dir:     str,
    elevation_deg:  float,
    completed_rx:   int,
    total_rx:       int,
    batch_size:     int,
    frequency:      float,
    tx_pos:         list,
    slant_dist_m:   float,
    cfg:            dict,
    K_power_ratio:  list,
    K_moments:      list,
    tau_rms_mean_s: list,
    los_probability: list,
    raw_results:    list,
    P_los:          list = None,
    P_specular:     list = None,
    P_diffuse:      list = None,
    P_diffraction:  list = None,
    P_refraction:   list = None,
    P_nlos:         list = None,
    omega_F:        list = None,
    vtec_sample:    list = None,
    A_gas_db:       list = None,
    R_sample_mm_h:  list = None,
    A_rain_db:      list = None,
    scint_db:       list = None,
    atm_total_db:   list = None,
    doppler_mean_hz: list = None,
    doppler_rms_hz:  list = None,
    doppler_mean_hz_td: list = None,
    doppler_rms_hz_td:  list = None,
    ASD_deg:        list = None,
    ASA_deg:        list = None,
    ZSD_deg:        list = None,
    ZSA_deg:        list = None,
    scene_seeds:    list = None,
    cluster_samples: list = None,
    nominal_rx_positions: list = None,
) -> str:
    """Atomically write a checkpoint file for the given elevation angle.

The file is first written to a temp path then renamed so that a crash
mid-write never leaves a corrupt checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = checkpoint_path(output_dir, elevation_deg)
    tmp  = path + ".tmp"

    checkpoint = {
        "elevation_deg"    : elevation_deg,
        "completed_rx"     : completed_rx,
        "total_rx"         : total_rx,
        "batch_size"       : batch_size,
        "timestamp"        : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frequency"        : frequency,
        "tx_pos"           : tx_pos,
        "slant_dist_m"     : slant_dist_m,
        "cfg_snapshot"     : dict(cfg),
        "K_power_ratio"    : list(K_power_ratio),
        "K_moments"        : list(K_moments),
        "tau_rms_mean_s"   : list(tau_rms_mean_s),
        "los_probability"  : list(los_probability),
        # Directly stored coherent LoS power (for ensemble-K computation)
        "P_los"            : list(P_los)         if P_los         is not None else [0.0] * completed_rx,
        # Per-mechanism NLoS power (None-safe: store zeros if not provided)
        # NOTE: stored as INCOHERENT sums (Σ|a_k|²) so they are additive.
        "P_specular"       : list(P_specular)    if P_specular    is not None else [0.0] * completed_rx,
        "P_diffuse"        : list(P_diffuse)     if P_diffuse     is not None else [0.0] * completed_rx,
        "P_diffraction"    : list(P_diffraction) if P_diffraction is not None else [0.0] * completed_rx,
        "P_refraction"     : list(P_refraction)  if P_refraction  is not None else [0.0] * completed_rx,
        "P_nlos"           : list(P_nlos)        if P_nlos        is not None else [0.0] * completed_rx,
        # Ionospheric Monte Carlo: per-receiver Faraday angle and TEC sample
        "omega_F"          : list(omega_F)       if omega_F       is not None else [0.0] * completed_rx,
        "vtec_sample"      : list(vtec_sample)   if vtec_sample   is not None else [0.0] * completed_rx,
        # Atmospheric attenuation (gaseous, rain, scintillation)
        "A_gas_db"         : list(A_gas_db)      if A_gas_db      is not None else [0.0] * completed_rx,
        "R_sample_mm_h"    : list(R_sample_mm_h) if R_sample_mm_h is not None else [0.0] * completed_rx,
        "A_rain_db"        : list(A_rain_db)     if A_rain_db     is not None else [0.0] * completed_rx,
        "scint_db"         : list(scint_db)      if scint_db      is not None else [0.0] * completed_rx,
        "atm_total_db"     : list(atm_total_db)  if atm_total_db  is not None else [0.0] * completed_rx,
        # Doppler (SGP4 waypoint mobility)
        "doppler_mean_hz"  : list(doppler_mean_hz) if doppler_mean_hz is not None else [0.0] * completed_rx,
        "doppler_rms_hz"   : list(doppler_rms_hz)  if doppler_rms_hz  is not None else [0.0] * completed_rx,
        "doppler_mean_hz_td": list(doppler_mean_hz_td) if doppler_mean_hz_td is not None else [0.0] * completed_rx,
        "doppler_rms_hz_td" : list(doppler_rms_hz_td)  if doppler_rms_hz_td  is not None else [0.0] * completed_rx,
        # NTN large-scale angular spreads (3GPP TR 38.901 Sec 7.5 / TR 38.811) --
        # the ASD/ASA/ZSD/ZSA of ns-3's 7-parameter NTN LOS statistic set
        # [SF, K, DS, ASD, ASA, ZSD, ZSA]. None-safe like the stats above.
        "ASD_deg"          : list(ASD_deg)       if ASD_deg       is not None else [0.0] * completed_rx,
        "ASA_deg"          : list(ASA_deg)       if ASA_deg       is not None else [0.0] * completed_rx,
        "ZSD_deg"          : list(ZSD_deg)       if ZSD_deg       is not None else [0.0] * completed_rx,
        "ZSA_deg"          : list(ZSA_deg)       if ZSA_deg       is not None else [0.0] * completed_rx,
        # PPP scene seed per receiver (stochastic_simulation.ipynb only; empty
        # for the site pipeline, which has one fixed scene for the whole run).
        "scene_seeds"      : list(scene_seeds)   if scene_seeds   is not None else [],
        "cluster_samples"  : list(cluster_samples) if cluster_samples is not None else [],
        "nominal_rx_positions": list(nominal_rx_positions) if nominal_rx_positions is not None else [],
        "raw_results"      : list(raw_results),
    }

    with open(tmp, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)   # atomic on POSIX

    return path


# ─── Load ─────────────────────────────────────────────────────────────────────

def load_checkpoint(output_dir: str, elevation_deg: float) -> dict | None:
    """Try to load a checkpoint for the given elevation angle."""
    path = checkpoint_path(output_dir, elevation_deg)

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        ck = pickle.load(f)

    print(
        f"[checkpoint] Loaded  elev={ck['elevation_deg']}°  "
        f"completed={ck['completed_rx']}/{ck['total_rx']}  "
        f"saved={ck['timestamp']}"
    )
    return ck




# ─── Results (final) ──────────────────────────────────────────────────────────

def save_results(
    output_dir:     str,
    elevation_deg:  float,
    nominal_rx_positions,
    K_power_ratio:  list,
    K_moments:      list,
    tau_rms_mean_s: list,
    los_probability: list,
    raw_results:    list,
    cfg:            dict,
    tx_pos:         list,
    slant_dist_m:   float = None,
                                    # cfg["SLANT_DIST_M"] for legacy fixed-elevation
                                    # callers that don't have a real waypoint range
    P_los:          list = None,
    P_specular:     list = None,
    P_diffuse:      list = None,
    P_diffraction:  list = None,
    P_refraction:   list = None,
    P_nlos:         list = None,
    omega_F:        list = None,
    vtec_sample:    list = None,
    A_gas_db:       list = None,
    R_sample_mm_h:  list = None,
    A_rain_db:      list = None,
    scint_db:       list = None,
    atm_total_db:   list = None,
    doppler_mean_hz: list = None,
    doppler_rms_hz:  list = None,
    doppler_mean_hz_td: list = None,
    doppler_rms_hz_td:  list = None,
    ASD_deg:        list = None,
    ASA_deg:        list = None,
    ZSD_deg:        list = None,
    ZSA_deg:        list = None,
    scene_seeds:    list = None,
    cluster_samples: list = None,
) -> str:
    """Save the final per-angle results (called at end of simulation)."""
    os.makedirs(output_dir, exist_ok=True)
    path = results_path(output_dir, elevation_deg)

    n = len(nominal_rx_positions)
    results = {
        # Metadata
        "elevation_deg"        : elevation_deg,
        "frequency"            : cfg["FREQUENCY"],
        "tx_pos"               : tx_pos,
        "slant_dist_m"         : slant_dist_m if slant_dist_m is not None else cfg["SLANT_DIST_M"],
        "n_rx_positions"       : n,
        "perturb_half_m"       : cfg.get("PERTURB_HALF", 0.0),
        "cfg_snapshot"         : dict(cfg),
        "timestamp"            : time.strftime("%Y-%m-%dT%H:%M:%S"),
        # RX positions
        "nominal_rx_positions" : nominal_rx_positions,
        # Per-RX statistics
        "K_power_ratio"        : list(K_power_ratio),
        "K_moments"            : list(K_moments),
        "tau_rms_mean_s"       : list(tau_rms_mean_s),
        "los_probability"      : list(los_probability),
        # Directly stored coherent LoS power (enables ensemble-K computation)
        "P_los"                : list(P_los)         if P_los         is not None else [0.0] * n,
        # Per-mechanism NLoS power lists (length == n_rx_positions)
        # NOTE: stored as INCOHERENT sums (Σ|a_k|²) so they are additive.
        "P_specular"           : list(P_specular)    if P_specular    is not None else [0.0] * n,
        "P_diffuse"            : list(P_diffuse)     if P_diffuse     is not None else [0.0] * n,
        "P_diffraction"        : list(P_diffraction) if P_diffraction is not None else [0.0] * n,
        "P_refraction"         : list(P_refraction)  if P_refraction  is not None else [0.0] * n,
        "P_nlos"               : list(P_nlos)        if P_nlos        is not None else [0.0] * n,
        # Ionospheric Monte Carlo: per-receiver Faraday angle and TEC sample
        "omega_F"              : list(omega_F)       if omega_F       is not None else [0.0] * n,
        "vtec_sample"          : list(vtec_sample)   if vtec_sample   is not None else [0.0] * n,
        # Atmospheric attenuation (gaseous, rain, scintillation)
        "A_gas_db"             : list(A_gas_db)      if A_gas_db      is not None else [0.0] * n,
        "R_sample_mm_h"        : list(R_sample_mm_h) if R_sample_mm_h is not None else [0.0] * n,
        "A_rain_db"            : list(A_rain_db)     if A_rain_db     is not None else [0.0] * n,
        "scint_db"             : list(scint_db)      if scint_db      is not None else [0.0] * n,
        "atm_total_db"         : list(atm_total_db)  if atm_total_db  is not None else [0.0] * n,
        # Doppler (SGP4 waypoint mobility)
        "doppler_mean_hz"      : list(doppler_mean_hz) if doppler_mean_hz is not None else [0.0] * n,
        "doppler_rms_hz"       : list(doppler_rms_hz)  if doppler_rms_hz  is not None else [0.0] * n,
        "doppler_mean_hz_td"   : list(doppler_mean_hz_td) if doppler_mean_hz_td is not None else [0.0] * n,
        "doppler_rms_hz_td"    : list(doppler_rms_hz_td)  if doppler_rms_hz_td  is not None else [0.0] * n,
        # NTN large-scale angular spreads (3GPP TR 38.901 Sec 7.5 / TR 38.811) --
        # the ASD/ASA/ZSD/ZSA of ns-3's 7-parameter NTN LOS statistic set
        # [SF, K, DS, ASD, ASA, ZSD, ZSA]. None-safe like the stats above.
        "ASD_deg"              : list(ASD_deg)       if ASD_deg       is not None else [0.0] * n,
        "ASA_deg"              : list(ASA_deg)       if ASA_deg       is not None else [0.0] * n,
        "ZSD_deg"              : list(ZSD_deg)       if ZSD_deg       is not None else [0.0] * n,
        "ZSA_deg"              : list(ZSA_deg)       if ZSA_deg       is not None else [0.0] * n,
        # PPP scene seed per receiver (stochastic_simulation.ipynb only; empty
        # for the site pipeline, which has one fixed scene for the whole run).
        "scene_seeds"          : list(scene_seeds)   if scene_seeds   is not None else [],
        "cluster_samples"      : list(cluster_samples) if cluster_samples is not None else [],
        # Raw data (sampled subset)
        "raw_results"          : list(raw_results),
    }

    with open(path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[checkpoint] Results saved → {path}")
    return path


def load_results(output_dir: str, elevation_deg: float) -> dict | None:
    """
    Load final results for one elevation angle.

    Returns None if the file does not exist.
    """
    path = results_path(output_dir, elevation_deg)

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        res = pickle.load(f)

    print(
        f"[checkpoint] Loaded results  elev={res['elevation_deg']}°  "
        f"n_rx={res['n_rx_positions']}"
    )
    return res


def load_all_results(output_dir: str, elevation_angles: list) -> dict:
    """Load results for every elevation angle, returning a dict keyed by angle."""
    all_results = {}
    for angle in elevation_angles:
        res = load_results(output_dir, angle)
        if res is not None:
            all_results[float(angle)] = res
        else:
            print(f"[checkpoint] No results file found for elev={angle}°  (skipping)")
    return all_results


# ─── Status helpers ───────────────────────────────────────────────────────────

def simulation_status(output_dir: str, elevation_angles: list) -> None:
    """Print a status table for all elevation angles."""
    print(f"\n{'─'*65}")
    print(f"  {'Elev [°]':>8}  {'Status':>12}  {'Progress':>16}  {'Saved':>20}")
    print(f"{'─'*65}")
    for angle in elevation_angles:
        res_path = results_path(output_dir, angle)
        ck_path  = checkpoint_path(output_dir, angle)
        if os.path.exists(res_path):
            res = load_results(output_dir, angle)
            print(
                f"  {angle:>8.0f}  {'COMPLETE':>12}  "
                f"{'':>16}  {res.get('timestamp',''):>20}"
            )
        elif os.path.exists(ck_path):
            ck = load_checkpoint(output_dir, angle)
            pct = 100 * ck["completed_rx"] / max(ck["total_rx"], 1)
            prog = f"{ck['completed_rx']}/{ck['total_rx']} ({pct:.0f}%)"
            print(
                f"  {angle:>8.0f}  {'IN PROGRESS':>12}  "
                f"{prog:>16}  {ck.get('timestamp',''):>20}"
            )
        else:
            print(f"  {angle:>8.0f}  {'NOT STARTED':>12}  {'':>16}  {'':>20}")
    print(f"{'─'*65}\n")
