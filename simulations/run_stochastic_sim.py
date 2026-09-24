#!/usr/bin/env python3
"""One elevation angle over a pre-generated PPP scene ensemble.

Stochastic-scene variant of run_elevation_sim.py: instead of one fixed scene it
walks an ensemble and places N_RX_PER_SCENE receivers in each, so the sampling
varies over both morphology and position. Launched as a subprocess, one per GPU.

See docs/pipelines.md.
"""

# CRITICAL: set CUDA_VISIBLE_DEVICES *before* any drjit / mitsuba / tf import.
# The launcher (notebook) passes CUDA_VISIBLE_DEVICES via the subprocess env.
# We also honour a --gpu_id arg for documentation/logging purposes.
import os
import sys
from pathlib import Path
import argparse
import json
import time
import gc
import pickle

parser = argparse.ArgumentParser(description="Stochastic PPP-scene NTN sim — single elevation angle")
parser.add_argument("--elevation", type=float, default=None,
                    help="Elevation angle [degrees]. Used directly (with the legacy "
                         "fixed-azimuth TX placement) unless --waypoints_file is given.")
parser.add_argument("--waypoints_file", type=str, default=None,
                    help="Pickle produced by generate_pass_waypoints.py. When given, "
                         "TX position/velocity/elevation all come from one real SGP4 "
                         "pass sample instead of --elevation + assumed azimuth.")
parser.add_argument("--waypoint_index", type=int, default=None,
                    help="Index into waypoints_file['waypoints'] to run. Required if "
                         "--waypoints_file is given.")
parser.add_argument("--gpu_id",    type=int,   default=0,
                    help="Logical GPU index (used for CUDA_VISIBLE_DEVICES if not already set)")
parser.add_argument("--config",    type=str,   required=True,
                    help="Path to sim_config.json")
parser.add_argument("--scenes_manifest", type=str, required=True,
                    help="Path to scenes_manifest.pkl (generated once by "
                         "scene_gen_ppp.generate_scene_ensemble, shared across "
                         "every elevation angle in this sweep)")
parser.add_argument("--output",    type=str,   required=True,
                    help="Output directory for checkpoints and results")
parser.add_argument("--resume",    action="store_true", default=False,
                    help="Resume from checkpoint if one exists")
parser.add_argument("--no-resume", action="store_true", default=False,
                    dest="no_resume",
                    help="Force a fresh start, ignoring any existing checkpoint "
                         "(overrides --resume if both are passed)")
args = parser.parse_args()

if args.waypoints_file is not None:
    if args.waypoint_index is None:
        parser.error("--waypoint_index is required when --waypoints_file is given")
else:
    if args.elevation is None:
        parser.error("either --elevation or (--waypoints_file + --waypoint_index) is required")

# Set GPU visibility before any GPU library imports
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# GPU-side imports (after CUDA_VISIBLE_DEVICES is locked in)
DRJIT_LIBLLVM = os.environ.get(
    "DRJIT_LIBLLVM_PATH", "/usr/lib/x86_64-linux-gnu/libLLVM-20.so"
)
os.environ.setdefault("DRJIT_LIBLLVM_PATH", DRJIT_LIBLLVM)

import numpy as np
import drjit

from sionna.rt import PathSolver

# Local helpers (assumed to be in the same directory)
# Import root is the directory containing Ray_Tracing/, so the
# Ray_Tracing.src.* package paths resolve however this is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Ray_Tracing.src.utils.channel_utils  import (rms_delay_spread, rms_angular_spread, complex_channel,
                              split_by_interaction_type,
                              tx_position_from_elevation,
                              tx_position_from_pass_sample, tx_velocity_scene_frame,
                              doppler_mean_rms_from_paths,
                              rms_doppler_spread_hz)
from Ray_Tracing.src.utils.scene_utils    import build_scene_batch, sample_rx_positions
from Ray_Tracing.src.utils.orbital_utils  import slant_range_m
import Ray_Tracing.src.utils.scene_gen_ppp as sg
from Ray_Tracing.src.utils.checkpoint_utils import (save_checkpoint, load_checkpoint,
                               save_results)
from Ray_Tracing.src.utils.ionospheric_utils import (apply_ionospheric_corrections,
                                sample_faraday_angle,
                                ionospheric_summary)
from Ray_Tracing.src.utils.atmospheric_utils import (compute_atmospheric_effects,
                                apply_atmospheric_corrections,
                                atmospheric_deterministic_summary)

# Load configuration

with open(args.config, "r") as f:
    cfg = json.load(f)

# Verify the patched Sionna is loaded and record it in the results, so every
# run file carries its own provenance. scene_utils re-checks at solve time.
import Ray_Tracing.src.sionna_patch_check as sionna_patch_check
cfg["_SIONNA_PATCH"] = sionna_patch_check.require("run_stochastic_sim")
print(f"  {sionna_patch_check.summary_line(cfg['_SIONNA_PATCH'])}")

OUTPUT_DIR              = args.output

# ── Waypoint (real SGP4 pass sample) vs. legacy fixed-elevation mode ─────────
USE_WAYPOINT = args.waypoints_file is not None
if USE_WAYPOINT:
    with open(args.waypoints_file, "rb") as f:
        _wp_data = pickle.load(f)
    WAYPOINT      = _wp_data["waypoints"][args.waypoint_index]
    ELEVATION_DEG = float(WAYPOINT["matched_elev_deg"])
else:
    WAYPOINT      = None
    ELEVATION_DEG = float(args.elevation)
# Priority: --no-resume (CLI) > --resume (CLI) > RESUME_FROM_CHECKPOINT in config > False
_cfg_resume = bool(cfg.get("RESUME_FROM_CHECKPOINT", False))
if args.no_resume:
    RESUME_FROM_CHECKPOINT = False          # explicit CLI override → always fresh
elif args.resume:
    RESUME_FROM_CHECKPOINT = True           # explicit CLI override → always resume
else:
    RESUME_FROM_CHECKPOINT = _cfg_resume    # use value from sim_config.json

# Unpack common config values
FREQUENCY               = float(cfg["FREQUENCY"])
SLANT_DIST_M            = float(cfg["SLANT_DIST_M"]) if cfg.get("SLANT_DIST_M") is not None else None
SCENE_CENTER            = cfg.get("SCENE_CENTER", [0.0, 0.0])
TX_AZIMUTH_DEG          = float(cfg["TX_AZIMUTH_DEG"]) if cfg.get("TX_AZIMUTH_DEG") is not None else 180.0
RX_HEIGHT                = float(cfg.get("RX_HEIGHT", 1.5))
BASE_SAMPLES_PER_SRC    = int(cfg["BASE_SAMPLES_PER_SRC"])
BASE_MAX_PATHS          = int(cfg["BASE_MAX_PATHS"])
RNG_SEED                = int(cfg.get("RNG_SEED", 42))
CHECKPOINT_INTERVAL     = int(cfg.get("CHECKPOINT_INTERVAL", 50))
RAW_SAVE_INTERVAL       = int(cfg.get("RAW_SAVE_INTERVAL", 200))

# ── Receivers per scene ──────────────────────────────────────────────────────
# N_RX_PER_SCENE UEs are drawn uniformly from each scene's free space (analytic
# footprint mask, 1 m wall clearance by default). Default 1 keeps the receiver
# COUNT identical to the original one-per-scene behaviour -- but note the
# position is now a uniform random free point, not the scene centre, because a
# fixed centre UE is a biased sample: generate_scene_ensemble's origin gate
# guarantees the centre is outdoors, which inflates LoS probability.
N_RX_PER_SCENE          = int(cfg.get("N_RX_PER_SCENE", 1))
RX_GRID_RES             = float(cfg.get("PPP_RX_GRID_RES", 2.0))
RX_MARGIN               = float(cfg.get("PPP_RX_MARGIN", 20.0))
RX_CLEARANCE            = float(cfg.get("PPP_RX_CLEARANCE", 1.0))
# Pin every UE at (0, 0, RX_HEIGHT) instead of sampling free space -- the
# Chiu & Roy wall ensemble, where the terminal is the scene process's own
# reference point. Only sensible with N_RX_PER_SCENE = 1.
RX_AT_ORIGIN            = bool(cfg.get("PPP_RX_AT_ORIGIN", False))
# Bank per-path interaction points in raw_results (ground-vs-wall reflection
# split). Off by default: it is the bulkiest thing in the raw payload.
SAVE_VERTICES           = bool(cfg.get("SAVE_PATH_VERTICES", False))
# Retries for the Sionna ground-path dropout (see the retry block below).
# 0 disables the workaround entirely.
GROUND_RETRY_MAX        = int(cfg.get("GROUND_RETRY_MAX", 0))
GROUND_RETRY_JITTER_M   = float(cfg.get("GROUND_RETRY_JITTER_M", 0.05))
GROUND_Z_TOL_M          = float(cfg.get("GROUND_Z_TOL_M", 0.05))


def _has_los_and_ground(paths) -> bool:
    """
    True unless this solve is missing the ground reflection while the direct
    path is present -- the dropout signature. A ground path is a first-order
    interaction whose interaction point sits on the ground plane (|z| < tol);
    wall reflections and roof-edge diffractions sit above it.
    """
    a, _ = paths.cir(normalize_delays=False, out_type="numpy")
    a = np.asarray(a).reshape(-1)
    it = paths.interactions.numpy()[0].reshape(-1)
    valid = np.abs(a) > 0.0
    if not np.any(valid & (it == 0)):
        return True                     # no LoS: NLoS state, nothing to repair
    try:
        vz = paths.vertices.numpy()[0].reshape(-1, 3)[:, 2]
    except AttributeError:
        return True                     # cannot tell; do not retry blindly
    return bool(np.any(valid & (it != 0) & (np.abs(vz) < GROUND_Z_TOL_M)))
if RX_AT_ORIGIN and N_RX_PER_SCENE != 1:
    raise ValueError("PPP_RX_AT_ORIGIN pins one fixed position, so "
                     f"N_RX_PER_SCENE must be 1, got {N_RX_PER_SCENE}.")
if N_RX_PER_SCENE < 1:
    raise ValueError(f"N_RX_PER_SCENE must be >= 1, got {N_RX_PER_SCENE}")

# Polarization (actually applied in scene_utils.build_scene_batch per scene;
# read again here only so the run header/log reports what's in effect).
TX_POL_TYPE             = cfg.get("TX_POL_TYPE",      "V")
RX_POL_TYPE             = cfg.get("RX_POL_TYPE",      "V")
POL_BASE_PATTERN        = cfg.get("POL_BASE_PATTERN", "iso")

# Propagation flags
MAX_DEPTH               = int(cfg.get("MAX_DEPTH", 3))
SIM_LOS                 = bool(cfg.get("LOS", True))
SIM_SPECULAR            = bool(cfg.get("SPECULAR_REFLECTION", True))
SIM_DIFFUSE             = bool(cfg.get("DIFFUSE_REFLECTION", True))
SIM_DIFFRACTION         = bool(cfg.get("DIFFRACTION", True))
SIM_EDGE_DIFFRACTION    = bool(cfg.get("EDGE_DIFFRACTION", False))
SIM_REFRACTION          = bool(cfg.get("REFRACTION", False))
BEAMFORMING             = bool(cfg.get("BEAMFORMING", False))
SCENE_RADIUS            = float(cfg.get("SCENE_RADIUS", 100.0))

# ── Ionospheric Monte Carlo model ────────────────────────────────────────────
# Same semantics as run_elevation_sim.py: one (vtec, Ω_F) pair sampled per
# scene realisation (was "per batch" there; every "batch" here is one scene).
IONOSPHERE_ENABLED      = bool(cfg.get("IONOSPHERE_ENABLED", False))
VTEC_MEAN_TECU          = float(cfg.get("VTEC_MEAN_TECU",  10.0))
VTEC_STD_TECU           = float(cfg.get("VTEC_STD_TECU",   5.0))
B_L_TESLA               = float(cfg.get("B_L_TESLA",       2e-5))
VTEC_SIGMA_H            = float(cfg.get("VTEC_SIGMA_H",    0.0))

# ── Atmospheric Monte Carlo model ─────────────────────────────────────────────
ATMOSPHERE_ENABLED      = bool(cfg.get("ATMOSPHERE_ENABLED",   False))
ATM_T_K                 = float(cfg.get("ATM_T_K",             288.15))
ATM_P_HPA               = float(cfg.get("ATM_P_HPA",          1013.25))
ATM_RHO_G_M3            = float(cfg.get("ATM_RHO_G_M3",           7.5))
RAIN_R_MM_H             = float(cfg.get("RAIN_R_MM_H",            0.0))
ATM_H_STATION_KM        = float(cfg.get("ATM_H_STATION_KM",      0.0))
ATM_H_RAIN_KM           = float(cfg.get("ATM_H_RAIN_KM",         3.36))
ATM_POL_TILT_DEG        = float(cfg.get("ATM_POL_TILT_DEG",     45.0))
ATM_D_ANT_M             = float(cfg.get("ATM_D_ANT_M",           0.0))
ATM_SCINTILLATION       = bool(cfg.get("ATM_SCINTILLATION",      True))

# ── Doppler (SGP4 orbital mobility) ──────────────────────────────────────────
DOPPLER_ENABLED         = bool(cfg.get("DOPPLER_ENABLED", False)) and USE_WAYPOINT
DOPPLER_NUM_TIME_STEPS  = int(cfg.get("DOPPLER_NUM_TIME_STEPS", 128))
DOPPLER_OVERSAMPLE_FACTOR = float(cfg.get("DOPPLER_OVERSAMPLE_FACTOR", 10.0))
SCENE_EAST_HEADING_DEG  = float(cfg.get("SCENE_EAST_HEADING_DEG", 90.0))

from scipy import constants

# Compute TX position (and, for waypoint mode, velocity) for this run.
# Same per-run sizing as run_elevation_sim.py -- BEAM_HALF_ANGLE_RAD sized
# from the actual slant distance so the ground footprint's narrowest
# (cross-range) extent always reaches out to SCENE_RADIUS.
if USE_WAYPOINT:
    TX_POS, BEAM_HALF_ANGLE_RAD = tx_position_from_pass_sample(
        WAYPOINT, SCENE_CENTER, SCENE_RADIUS,
        scene_east_heading_deg=SCENE_EAST_HEADING_DEG,
    )
    TX_VEL = tx_velocity_scene_frame(
        WAYPOINT, scene_east_heading_deg=SCENE_EAST_HEADING_DEG,
    )
else:
    # SWEEP_ALTITUDE_KM, if set, replaces the fixed SLANT_DIST_M with the true
    # spherical-Earth range for this elevation.
    _alt = cfg.get("SWEEP_ALTITUDE_KM")
    _sweep_slant_m = float(slant_range_m(_alt, ELEVATION_DEG)) if _alt else SLANT_DIST_M
    TX_POS, BEAM_HALF_ANGLE_RAD = tx_position_from_elevation(
        ELEVATION_DEG, _sweep_slant_m, SCENE_CENTER, SCENE_RADIUS, TX_AZIMUTH_DEG
    )
    TX_VEL = None

SLANT_DIST_ACTUAL_M = float(WAYPOINT["range_m"]) if USE_WAYPOINT else _sweep_slant_m

if DOPPLER_ENABLED:
    DOPPLER_BOUND_HZ = (FREQUENCY / constants.c) * float(np.linalg.norm(TX_VEL))
    SAMPLING_FREQUENCY_HZ = DOPPLER_OVERSAMPLE_FACTOR * max(DOPPLER_BOUND_HZ, 1.0)
else:
    DOPPLER_BOUND_HZ = 0.0
    SAMPLING_FREQUENCY_HZ = 1.0

# Load scene ensemble manifest

with open(args.scenes_manifest, "rb") as f:
    manifest = pickle.load(f)
scene_entries = manifest["scenes"]
N_SCENES      = len(scene_entries)
TOTAL_RX      = N_SCENES * N_RX_PER_SCENE
# The manifest is the authoritative record of which materials the ensemble's
# scenes actually use -- override whatever MATERIAL_NAMES sim_config.json has
# so scattering-coefficient application in build_scene_batch always matches
# reality, even if the notebook's cfg cell and the manifest ever drift.
_roof_mat = manifest.get("roof_material", manifest["building_material"])
cfg["MATERIAL_NAMES"] = sorted({manifest["building_material"], _roof_mat,
                                 manifest["ground_material"]})
# Per-role a/b/c/d + scattering_coefficient (+ xpd_coefficient) -- additive,
# only present in manifests written by the updated
# scene_gen_ppp.generate_scene_ensemble(). Older manifests (pre-dating this
# feature) simply lack these keys, so cfg["MATERIAL_PARAMS"] is never set and
# build_scene_batch() falls back to its MATERIAL_NAMES/SCATTERING_COEFF path,
# identical to before this feature existed.
if all(k in manifest for k in ("wall_params", "roof_params", "ground_params")):
    cfg["MATERIAL_PARAMS"] = {
        manifest["building_material"]: manifest["wall_params"],
        _roof_mat:                     manifest["roof_params"],
        manifest["ground_material"]:   manifest["ground_params"],
    }

print(f"\n{'='*64}")
print(f"  run_stochastic_sim.py")
print(f"  Elevation : {ELEVATION_DEG:.0f}°")
print(f"  GPU       : {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
print(f"  TX pos    : [{TX_POS[0]/1e3:.2f} km, {TX_POS[1]/1e3:.2f} km, {TX_POS[2]/1e3:.2f} km]")
if USE_WAYPOINT:
    print(f"  Waypoint  : az={WAYPOINT['az_deg']:.2f}°  range={WAYPOINT['range_km']:.1f} km  "
          f"pass_t={WAYPOINT['t_s']:.1f} s  |v|={np.linalg.norm(WAYPOINT['enu_vel_kms']):.3f} km/s")
    print(f"  TX vel    : [{TX_VEL[0]:.1f}, {TX_VEL[1]:.1f}, {TX_VEL[2]:.1f}] m/s (scene frame)")
print(f"  Beam      : half-angle={np.degrees(BEAM_HALF_ANGLE_RAD):.4f}°  "
      f"(sized to SCENE_RADIUS={SCENE_RADIUS:.0f} m at slant={SLANT_DIST_ACTUAL_M/1e3:.2f} km, "
      f"elev={ELEVATION_DEG:.1f}°)")
print(f"  Pol       : TX={TX_POL_TYPE}  RX={RX_POL_TYPE}  base_pattern={POL_BASE_PATTERN}")
if DOPPLER_ENABLED:
    _doppler_window_s = DOPPLER_NUM_TIME_STEPS / SAMPLING_FREQUENCY_HZ
    print(f"  Doppler   : ENABLED  bound={DOPPLER_BOUND_HZ:.0f} Hz  "
          f"sampling={SAMPLING_FREQUENCY_HZ/1e3:.1f} kHz  "
          f"steps={DOPPLER_NUM_TIME_STEPS}  window={_doppler_window_s*1e6:.1f} us")
else:
    print(f"  Doppler   : DISABLED"
          + ("" if USE_WAYPOINT else " (requires --waypoints_file)"))
print(f"  max_depth : {MAX_DEPTH}")
print(f"  LoS={SIM_LOS}  Spec={SIM_SPECULAR}  Diff.refl={SIM_DIFFUSE}  "
      f"Diffr={SIM_DIFFRACTION}  Refr={SIM_REFRACTION}")
if IONOSPHERE_ENABLED:
    print(f"  Ionosphere: Monte Carlo  VTEC~LN(μ={VTEC_MEAN_TECU} TECU, "
          f"σ={VTEC_STD_TECU} TECU)  B_L={B_L_TESLA*1e6:.1f} μT")
    ionospheric_summary(VTEC_MEAN_TECU, VTEC_STD_TECU, FREQUENCY, ELEVATION_DEG, B_L_TESLA)
else:
    print(f"  Ionosphere: DISABLED")
if ATMOSPHERE_ENABLED:
    print(f"  Atmosphere: ρ={ATM_RHO_G_M3} g/m³  R={RAIN_R_MM_H} mm/h  "
          f"Scint={'ON' if ATM_SCINTILLATION else 'OFF'}")
    atmospheric_deterministic_summary(FREQUENCY, ELEVATION_DEG,
                                      rho_g_m3=ATM_RHO_G_M3, R_mm_h=RAIN_R_MM_H,
                                      T_K=ATM_T_K, P_hPa=ATM_P_HPA,
                                      h_station_km=ATM_H_STATION_KM,
                                      h_rain_km=ATM_H_RAIN_KM,
                                      pol_tilt_deg=ATM_POL_TILT_DEG)
else:
    print(f"  Atmosphere: DISABLED")
_actual_seeds = [e["seed"] for e in scene_entries]
print(f"  Scene ensemble : {N_SCENES} realisations  "
      f"(seeds {min(_actual_seeds)}..{max(_actual_seeds)}"
      + (f", {len(manifest['skipped_seeds'])} skipped for receiver collisions"
         if manifest.get("skipped_seeds") else "") + f")  "
      f"materials={cfg['MATERIAL_NAMES']}")
if RX_AT_ORIGIN:
    print(f"  Receivers : {N_RX_PER_SCENE}/scene pinned at the origin "
          f"(0, 0, {RX_HEIGHT} m)  ->  {TOTAL_RX} total")
else:
    print(f"  Receivers : {N_RX_PER_SCENE}/scene, uniform over free space "
          f"(clearance={RX_CLEARANCE} m, z={RX_HEIGHT} m)  ->  {TOTAL_RX} total")
if GROUND_RETRY_MAX:
    print(f"  Ground retry : up to {GROUND_RETRY_MAX} re-solve(s) on a dropped "
          f"ground path (jitter {GROUND_RETRY_JITTER_M} m)")
print(f"{'='*64}\n")

# Checkpoint resume or fresh start

os.makedirs(OUTPUT_DIR, exist_ok=True)

if RESUME_FROM_CHECKPOINT:
    ck = load_checkpoint(OUTPUT_DIR, ELEVATION_DEG)
else:
    ck = None

if ck is not None:
    completed_rx      = ck["completed_rx"]     # receivers done, NOT scenes
    # Every scene contributes exactly N_RX_PER_SCENE receivers (sample_rx_positions
    # always returns exactly n rows), so the scene cursor is an exact division --
    # same relationship as run_elevation_sim.py's start_batch = completed_rx // BATCH_SIZE.
    # NB: checkpoint_utils stores the config under "cfg_snapshot", not "cfg" --
    # reading the wrong key silently yields the default 1 and would reject every
    # legitimate N>1 resume.
    _ck_cfg  = ck.get("cfg_snapshot") or {}
    _ck_n_rx = int(_ck_cfg.get("N_RX_PER_SCENE", 1))
    if _ck_cfg and _ck_n_rx != N_RX_PER_SCENE:
        raise ValueError(
            f"checkpoint was written with N_RX_PER_SCENE={_ck_n_rx} but this run "
            f"has {N_RX_PER_SCENE}. Resuming would misalign the scene cursor and "
            f"every accumulator. Re-run with --no-resume, or restore the original "
            f"value."
        )
    start_scene        = completed_rx // N_RX_PER_SCENE
    all_rx_positions  = [list(x) for x in ck.get("nominal_rx_positions", [])]
    all_K_power       = list(ck["K_power_ratio"])
    all_K_moments     = list(ck["K_moments"])
    all_tau_rms       = list(ck["tau_rms_mean_s"])
    all_p_los         = list(ck["los_probability"])
    all_P_los         = list(ck.get("P_los",         [0.0] * completed_rx))
    all_P_specular    = list(ck.get("P_specular",    [0.0] * completed_rx))
    all_P_diffuse     = list(ck.get("P_diffuse",     [0.0] * completed_rx))
    all_P_diffraction = list(ck.get("P_diffraction", [0.0] * completed_rx))
    all_P_refraction  = list(ck.get("P_refraction",  [0.0] * completed_rx))
    all_P_nlos        = list(ck.get("P_nlos",        [0.0] * completed_rx))
    all_omega_F       = list(ck.get("omega_F",       [0.0] * completed_rx))
    all_vtec_sample   = list(ck.get("vtec_sample",   [0.0] * completed_rx))
    all_A_gas_db      = list(ck.get("A_gas_db",      [0.0] * completed_rx))
    all_R_sample      = list(ck.get("R_sample_mm_h", [0.0] * completed_rx))
    all_A_rain_db     = list(ck.get("A_rain_db",     [0.0] * completed_rx))
    all_scint_db      = list(ck.get("scint_db",      [0.0] * completed_rx))
    all_atm_total_db  = list(ck.get("atm_total_db",  [0.0] * completed_rx))
    all_doppler_mean_hz = list(ck.get("doppler_mean_hz", [0.0] * completed_rx))
    all_doppler_rms_hz  = list(ck.get("doppler_rms_hz",  [0.0] * completed_rx))
    all_doppler_mean_hz_td = list(ck.get("doppler_mean_hz_td", [0.0] * completed_rx))
    all_doppler_rms_hz_td  = list(ck.get("doppler_rms_hz_td",  [0.0] * completed_rx))
    # NTN angular spreads and scene provenance -- zero/empty-fill if the
    # checkpoint pre-dates this feature (same graceful-fallback pattern).
    all_ASD_deg       = list(ck.get("ASD_deg",       [0.0] * completed_rx))
    all_ASA_deg       = list(ck.get("ASA_deg",       [0.0] * completed_rx))
    all_ZSD_deg       = list(ck.get("ZSD_deg",       [0.0] * completed_rx))
    all_ZSA_deg       = list(ck.get("ZSA_deg",       [0.0] * completed_rx))
    all_scene_seeds   = list(ck.get("scene_seeds",   []))
    raw_results       = list(ck["raw_results"])
    print(f"  Resuming: {completed_rx}/{TOTAL_RX} receivers "
          f"= scene {start_scene}/{N_SCENES}")
else:
    completed_rx      = 0
    start_scene        = 0
    all_rx_positions  = []
    all_K_power       = []
    all_K_moments     = []
    all_tau_rms       = []
    all_p_los         = []
    all_P_los         = []
    all_P_specular    = []
    all_P_diffuse     = []
    all_P_diffraction = []
    all_P_refraction  = []
    all_P_nlos        = []
    all_omega_F       = []
    all_vtec_sample   = []
    all_A_gas_db      = []
    all_R_sample      = []
    all_A_rain_db     = []
    all_scint_db      = []
    all_atm_total_db  = []
    all_doppler_mean_hz = []
    all_doppler_rms_hz  = []
    all_doppler_mean_hz_td = []
    all_doppler_rms_hz_td  = []
    all_ASD_deg       = []
    all_ASA_deg       = []
    all_ZSD_deg       = []
    all_ZSA_deg       = []
    all_scene_seeds   = []
    raw_results       = []
    print(f"  Starting fresh run for elevation {ELEVATION_DEG}°")

# Main simulation loop -- one PPP scene realisation per iteration, with
# N_RX_PER_SCENE receivers sampled from that scene's free space.

rng         = np.random.default_rng(seed=RNG_SEED + int(ELEVATION_DEG * 7))
rng_iono    = np.random.default_rng(seed=RNG_SEED + int(ELEVATION_DEG * 13) + 99991)
rng_atm     = np.random.default_rng(seed=RNG_SEED + int(ELEVATION_DEG * 17) + 77773)
solver      = PathSolver()
total_start = time.perf_counter()
ground_retries = 0            # extra solves spent recovering dropped ground paths
ground_retry_exhausted = 0    # scenes where the ground path never reappeared

for scene_idx, entry in enumerate(scene_entries[start_scene:], start=start_scene):

    # N_RX_PER_SCENE UEs drawn uniformly from THIS scene's free space. The
    # footprint mask is analytic (no Mitsuba), so it costs ~0.1 s against a
    # ~100 s PathSolver call, and cannot leak the global Mitsuba variant.
    _spec       = sg.load_scene_spec(entry["spec_path"], entry["meta_path"])
    if RX_AT_ORIGIN:
        # Wall-ensemble (Chiu & Roy) mode: the terminal IS the PPP's reference
        # point, at (0, 0, h_g), and the scene is redrawn about it per
        # realisation. Sampling free space instead would be a different
        # experiment. The bias warning that motivates free-space sampling
        # applies to block scenes, whose origin gate inflates LoS; walls carry
        # no footprint area and the reference model has no exclusion zone.
        rx_batch = [[0.0, 0.0, RX_HEIGHT]] * N_RX_PER_SCENE
    else:
        _free_xy, _ = sg.free_space_grid(_spec, grid_res=RX_GRID_RES,
                                         margin=RX_MARGIN, clearance=RX_CLEARANCE)
        # Seeded by the SCENE SEED alone -- never by elevation or frequency -- so
        # every angle and both carriers see the IDENTICAL UE positions. That is what
        # makes per-UE statistics pairable across the sweep; seeding off the run RNG
        # would silently resample them per angle and destroy the pairing.
        _rx_rng  = np.random.default_rng(RNG_SEED + 7919 * int(entry["seed"]))
        rx_batch = sample_rx_positions(_free_xy, N_RX_PER_SCENE, RX_HEIGHT,
                                       rng=_rx_rng, grid_res=RX_GRID_RES).tolist()
    actual_bs   = len(rx_batch)             # == N_RX_PER_SCENE
    cfg["SCENE_XML"] = entry["xml_path"]    # build_scene_batch() re-reads this
                                             # fresh every call -- confirmed in
                                             # scene_utils.py, zero changes there.
    batch_start = time.perf_counter()

    # ── Memory cleanup ─────────────────────────────────────────────────────────
    for _name in ["paths", "amp_np", "tau_np", "inter_np", "scene"]:
        if _name in dir():
            del globals()[_name]
    gc.collect()
    try:
        drjit.flush_malloc_cache()
        drjit.eval()
    except Exception:
        pass

    # ── Sample atmospheric losses for this scene ───────────────────────────────
    if ATMOSPHERE_ENABLED:
        batch_atm = compute_atmospheric_effects(
            freq_hz               = FREQUENCY,
            elev_deg              = ELEVATION_DEG,
            rho_g_m3              = ATM_RHO_G_M3,
            R_mm_h                = RAIN_R_MM_H,
            rng                   = rng_atm,
            T_K                   = ATM_T_K,
            P_hPa                 = ATM_P_HPA,
            h_station_km          = ATM_H_STATION_KM,
            h_rain_km             = ATM_H_RAIN_KM,
            pol_tilt_deg          = ATM_POL_TILT_DEG,
            D_ant_m               = ATM_D_ANT_M,
            include_scintillation = ATM_SCINTILLATION,
        )
    else:
        batch_atm = {
            "A_gas_db": 0.0, "R_sample_mm_h": 0.0, "A_rain_db": 0.0,
            "scint_db": 0.0, "sigma_scint_db": 0.0, "total_loss_db": 0.0,
        }

    # ── Sample Faraday rotation angle for this scene ────────────────────────────
    if IONOSPHERE_ENABLED:
        batch_omega_F, batch_tec_slant = sample_faraday_angle(
            VTEC_MEAN_TECU, VTEC_STD_TECU,
            FREQUENCY, ELEVATION_DEG, B_L_TESLA,
            rng_iono,
            sigma_h=VTEC_SIGMA_H,
        )
    else:
        batch_omega_F   = 0.0
        batch_tec_slant = 0.0

    # ── Build this realisation's scene (TX polarisation pre-rotated by Ω_F) ────
    scene, _ = build_scene_batch(
        rx_batch, TX_POS, cfg, omega_F=batch_omega_F,
        tx_velocity=(TX_VEL if DOPPLER_ENABLED else None),
    )

    # ── Solve paths ────────────────────────────────────────────────────────────
    seed = drjit.cuda.ad.UInt(int(rng.integers(1, 1_000_000)))

    def _solve(sc):
        return solver(
            scene                 = sc,
            max_depth             = MAX_DEPTH,
            los                   = SIM_LOS,
            specular_reflection   = SIM_SPECULAR,
            diffuse_reflection    = SIM_DIFFUSE,
            refraction            = SIM_REFRACTION,
            diffraction           = SIM_DIFFRACTION,
            edge_diffraction      = SIM_EDGE_DIFFRACTION,
            synthetic_array       = False,
            beamforming           = BEAMFORMING,
            beam_angle            = BEAM_HALF_ANGLE_RAD,
            max_num_paths_per_src = BASE_MAX_PATHS,
            samples_per_src       = BASE_SAMPLES_PER_SRC,
            seed                  = seed,
        )

    paths = _solve(scene)

    # ── Ground-path dropout retry ─────────────────────────────────────────────
    # Sionna's specular refinement intermittently fails to return the
    # ground-reflected path (~6% of geometries): where it IS returned it is
    # exact to 0.00 dB against the image-method Fresnel value, and where it is
    # not, no amount of extra sampling recovers it -- but a sub-millimetre
    # change in RX position flips it. Since the ground bounce carries 69-92% of
    # the non-direct power in this model, silently dropping it would bias K
    # upward, systematically and per-elevation (the ground geometry is the same
    # in every scene, so an affected elevation would lose it in EVERY scene).
    # Re-solve with a small RX jitter until it reappears. A ground path that is
    # genuinely wall-blocked never reappears, so the attempts are bounded and
    # the exhausted count is reported.
    if GROUND_RETRY_MAX > 0:
        _jit_rng = np.random.default_rng(RNG_SEED + 104729 * int(entry["seed"]))
        for _attempt in range(GROUND_RETRY_MAX):
            if _has_los_and_ground(paths):
                break
            _off = _jit_rng.uniform(-GROUND_RETRY_JITTER_M, GROUND_RETRY_JITTER_M, 2)
            _rx_j = [[rx_batch[0][0] + _off[0], rx_batch[0][1] + _off[1],
                      rx_batch[0][2]]]
            scene, _ = build_scene_batch(
                _rx_j, TX_POS, cfg, omega_F=batch_omega_F,
                tx_velocity=(TX_VEL if DOPPLER_ENABLED else None),
            )
            paths = _solve(scene)
            ground_retries += 1
        else:
            if not _has_los_and_ground(paths):
                ground_retry_exhausted += 1

    amp_np, tau_np = paths.cir(normalize_delays=False, out_type="numpy")
    inter_np         = paths.interactions.numpy()
    doppler_np       = paths.doppler.numpy() if DOPPLER_ENABLED else None
    # Per-path departure/arrival angles [rad] -- same indexing convention as
    # tau_np/amp_np ([rx, rx_ant, tx, tx_ant, path]). Feeds ASD/ASA/ZSD/ZSA.
    theta_t_np = paths.theta_t.numpy()
    phi_t_np   = paths.phi_t.numpy()
    theta_r_np = paths.theta_r.numpy()
    phi_r_np   = paths.phi_r.numpy()
    # Interaction points [depth, rx, rx_ant, tx, tx_ant, path, 3] -- the only
    # way to tell a ground reflection from a wall reflection, since Sionna
    # labels both IT_SPECULAR. Optional: older Sionna builds lack it.
    try:
        vert_np = paths.vertices.numpy()
    except AttributeError:
        vert_np = None

    # ── Per-receiver extraction (always exactly 1 iteration: one scene, one RX) ─
    for local_rx in range(actual_bs):
        rx_nom    = rx_batch[local_rx]

        t           = tau_np[local_rx, 0, 0, 0, :]
        a_full      = amp_np[local_rx, 0, 0, 0, :, :]     # [n_paths, num_time_steps]
        inter_all   = inter_np[:, local_rx, 0, 0, 0, :]

        valid       = np.any(np.abs(a_full) > 0.0, axis=-1)
        t_v         = t[valid]
        a_full_v    = a_full[valid, :]             # [n_valid_paths, num_time_steps]
        inter_all_v = inter_all[:, valid]          # [max_depth, n_valid_paths]
        doppler_v   = doppler_np[local_rx, 0, 0, 0, :][valid] if DOPPLER_ENABLED else None
        # Angle slices -- corrections below modify t_v/a_full_v's delay/amplitude
        # values but never reorder or drop entries, so these stay index-aligned.
        theta_t_v   = theta_t_np[local_rx, 0, 0, 0, :][valid]
        phi_t_v     = phi_t_np[local_rx, 0, 0, 0, :][valid]
        theta_r_v   = theta_r_np[local_rx, 0, 0, 0, :][valid]
        phi_r_v     = phi_r_np[local_rx, 0, 0, 0, :][valid]
        vert_v      = vert_np[:, local_rx, 0, 0, 0, :, :][:, valid, :] \
            if vert_np is not None else None

        # ── Post-RT: ionospheric group delay + phase advance ───────────────────
        if IONOSPHERE_ENABLED and len(t_v) > 0:
            iono = apply_ionospheric_corrections(
                t_v, a_full_v,
                tec_slant_tecu = batch_tec_slant,
                freq_hz        = FREQUENCY,
            )
            t_v      = iono["taus_mod"]
            a_full_v = iono["amps_mod"]

        # ── Post-RT: atmospheric amplitude loss ────────────────────────────────
        if ATMOSPHERE_ENABLED and len(t_v) > 0:
            atm = apply_atmospheric_corrections(
                t_v, a_full_v,
                total_loss_db = batch_atm["total_loss_db"],
            )
            t_v      = atm["taus_mod"]
            a_full_v = atm["amps_mod"]

        a_v = a_full_v[:, 0] if a_full_v.size else a_full_v

        if DOPPLER_ENABLED and len(t_v) > 0:
            doppler_mean_hz, doppler_rms_hz = doppler_mean_rms_from_paths(doppler_v, a_v)
        else:
            doppler_mean_hz, doppler_rms_hz = 0.0, 0.0

        if DOPPLER_ENABLED and DOPPLER_NUM_TIME_STEPS > 1 and len(t_v) > 0:
            t_axis = np.arange(DOPPLER_NUM_TIME_STEPS) / SAMPLING_FREQUENCY_HZ
            a_t = a_v[:, None] * np.exp(1j * 2 * np.pi * doppler_v[:, None] * t_axis[None, :])
            h_t = np.sum(a_t, axis=0)
            doppler_mean_hz_td, doppler_rms_hz_td = rms_doppler_spread_hz(h_t, SAMPLING_FREQUENCY_HZ)
        else:
            doppler_mean_hz_td, doppler_rms_hz_td = 0.0, 0.0

        if len(t_v) > 0:
            pwr         = split_by_interaction_type(t_v, a_v, inter_all_v, FREQUENCY,
                                                     incoherent=False)
            pwr_inc     = split_by_interaction_type(t_v, a_v, inter_all_v, FREQUENCY,
                                                     incoherent=True)
            _, ds        = rms_delay_spread(t_v, a_v)
            h_total      = complex_channel(t_v, a_v, FREQUENCY)
            P_los        = pwr["P_los"]
            P_nlos       = pwr["P_nlos"]
            has_los_flag = pwr["has_los"]
            # NTN large-scale angular spreads (3GPP TR 38.901 Sec 7.5) --
            # ASD/ASA from departure/arrival azimuth, ZSD/ZSA from
            # departure/arrival zenith. Same a_v (post ionospheric/atmospheric
            # correction) that feeds K-factor/delay-spread above.
            _, ASD_rad = rms_angular_spread(phi_t_v, a_v)
            _, ASA_rad = rms_angular_spread(phi_r_v, a_v)
            _, ZSD_rad = rms_angular_spread(theta_t_v, a_v)
            _, ZSA_rad = rms_angular_spread(theta_r_v, a_v)
        else:
            pwr = pwr_inc = {
                "P_los": 0.0, "P_specular": 0.0, "P_diffuse": 0.0,
                "P_diffraction": 0.0, "P_refraction": 0.0, "P_nlos": 0.0,
                "has_los": False,
            }
            P_los = P_nlos = ds = 0.0
            h_total      = 0j
            has_los_flag = False
            ASD_rad = ASA_rad = ZSD_rad = ZSA_rad = 0.0

        K_pr  = P_los / (P_nlos + 1e-30)
        h_env = float(np.abs(h_total))
        ASD_deg, ASA_deg = np.degrees(ASD_rad), np.degrees(ASA_rad)
        ZSD_deg, ZSA_deg = np.degrees(ZSD_rad), np.degrees(ZSA_rad)

        all_K_power      .append(float(K_pr))
        all_K_moments    .append(float('nan'))
        all_p_los        .append(1.0 if has_los_flag else 0.0)
        all_tau_rms      .append(float(ds))
        all_P_los        .append(float(P_los))
        all_P_specular   .append(float(pwr_inc["P_specular"]))
        all_P_diffuse    .append(float(pwr_inc["P_diffuse"]))
        all_P_diffraction.append(float(pwr_inc["P_diffraction"]))
        all_P_refraction .append(float(pwr_inc["P_refraction"]))
        all_P_nlos       .append(float(P_nlos))
        all_omega_F      .append(float(batch_omega_F))
        all_vtec_sample  .append(float(batch_tec_slant))
        all_A_gas_db     .append(float(batch_atm["A_gas_db"]))
        all_R_sample     .append(float(batch_atm["R_sample_mm_h"]))
        all_A_rain_db    .append(float(batch_atm["A_rain_db"]))
        all_scint_db     .append(float(batch_atm["scint_db"]))
        all_atm_total_db .append(float(batch_atm["total_loss_db"]))
        all_doppler_mean_hz.append(float(doppler_mean_hz))
        all_doppler_rms_hz .append(float(doppler_rms_hz))
        all_doppler_mean_hz_td.append(float(doppler_mean_hz_td))
        all_doppler_rms_hz_td .append(float(doppler_rms_hz_td))
        all_ASD_deg      .append(float(ASD_deg))
        all_ASA_deg      .append(float(ASA_deg))
        all_ZSD_deg      .append(float(ZSD_deg))
        all_ZSA_deg      .append(float(ZSA_deg))
        all_scene_seeds  .append(int(entry["seed"]))
        all_rx_positions .append([float(v) for v in rx_nom])

        # Save raw results at regular intervals
        if (completed_rx + 1) % RAW_SAVE_INTERVAL == 0:
            raw_results.append({
                "seed"         : int(entry["seed"]),
                "rx_nominal"   : list(rx_nom),
                "taus"         : t_v.copy(),
                "amps"         : a_v.copy(),
                "inter_d0"     : inter_all_v[0].copy() if len(t_v) > 0 else np.array([], dtype=int),
                # Per-path geometry, index-aligned with taus/amps. Departure
                # angles give the residual Doppler in float64 (differencing
                # Sionna's own per-path Doppler against the ~40 kHz LoS shift
                # loses the ~0.1 Hz residual); arrival angles give the GT
                # antenna pattern weight G(theta) in post.
                "theta_t"      : theta_t_v.copy(),
                "phi_t"        : phi_t_v.copy(),
                "theta_r"      : theta_r_v.copy(),
                "phi_r"        : phi_r_v.copy(),
                "doppler_hz"   : (doppler_v.copy() if (DOPPLER_ENABLED and doppler_v is not None)
                                  else None),
                "vertices"     : (vert_v.copy() if (SAVE_VERTICES and vert_v is not None)
                                  else None),
                "P_los"        : float(P_los),
                "P_scat"       : float(P_nlos),       # kept for backward compat
                "P_specular"   : float(pwr["P_specular"]),
                "P_diffuse"    : float(pwr["P_diffuse"]),
                "P_diffraction": float(pwr["P_diffraction"]),
                "P_refraction" : float(pwr["P_refraction"]),
                "P_nlos"       : float(P_nlos),
                "tau_rms_s"    : float(ds),
                "h_env"        : float(h_env),
                "has_los"      : bool(has_los_flag),
                "ASD_deg"      : float(ASD_deg),
                "ASA_deg"      : float(ASA_deg),
                "ZSD_deg"      : float(ZSD_deg),
                "ZSA_deg"      : float(ZSA_deg),
                "omega_F_rad"  : float(batch_omega_F),
                "vtec_tecu"    : float(batch_tec_slant),
                "A_gas_db"     : float(batch_atm["A_gas_db"]),
                "R_sample_mm_h": float(batch_atm["R_sample_mm_h"]),
                "A_rain_db"    : float(batch_atm["A_rain_db"]),
                "scint_db"     : float(batch_atm["scint_db"]),
                "atm_total_db" : float(batch_atm["total_loss_db"]),
                "doppler_mean_hz": float(doppler_mean_hz),
                "doppler_rms_hz" : float(doppler_rms_hz),
                "doppler_mean_hz_td": float(doppler_mean_hz_td),
                "doppler_rms_hz_td" : float(doppler_rms_hz_td),
                "waypoint_az_deg": float(WAYPOINT["az_deg"]) if USE_WAYPOINT else None,
                "waypoint_range_km": float(WAYPOINT["range_km"]) if USE_WAYPOINT else None,
            })

        completed_rx += 1

    # ── Scene-batch summary (still "batch" of size 1, kept for log parity) ─────
    Plos_batch  = np.array(all_P_los[-actual_bs:])
    Pnlos_batch = np.array(all_P_nlos[-actual_bs:])
    Pdiffuse_batch  = np.array(all_P_diffuse[-actual_bs:])
    Pdiffraction_batch  = np.array(all_P_diffraction[-actual_bs:])
    Pspecular_batch  = np.array(all_P_specular[-actual_bs:])
    mean_Plos  = float(np.mean(Plos_batch))
    mean_Pnlos = float(np.mean(Pnlos_batch))
    K_mean_dB   = 10.0 * np.log10(mean_Plos / (mean_Pnlos + 1e-30) + 1e-30)

    elapsed    = time.perf_counter() - batch_start
    total_e    = time.perf_counter() - total_start

    _P_total_inc = Plos_batch + Pspecular_batch + Pdiffuse_batch + Pdiffraction_batch
    _mean_tot    = float(np.mean(_P_total_inc)) if float(np.mean(_P_total_inc)) > 0 else 1.0
    _frac_los    = float(np.mean(Plos_batch))   / _mean_tot * 100
    _frac_spec   = float(np.mean(Pspecular_batch))  / _mean_tot * 100
    _frac_diff   = float(np.mean(Pdiffuse_batch))  / _mean_tot * 100
    _frac_diffr  = float(np.mean(Pdiffraction_batch)) / _mean_tot * 100

    print()
    iono_str = f"  Ω_F={np.degrees(batch_omega_F):.1f}°  TEC={batch_tec_slant:.1f} TECU" \
               if IONOSPHERE_ENABLED else ""
    atm_str  = (f"  A_gas={batch_atm['A_gas_db']:.2f} dB"
                f"  A_rain={batch_atm['A_rain_db']:.2f} dB (R={batch_atm['R_sample_mm_h']:.1f} mm/h)"
                f"  scint={batch_atm['scint_db']:.2f} dB"
                f"  tot={batch_atm['total_loss_db']:.2f} dB") \
               if ATMOSPHERE_ENABLED else ""
    print(
        f"  [elev={ELEVATION_DEG:.0f}°]  "
        f"scene={scene_idx + 1}/{N_SCENES} (seed={entry['seed']})  "
        f"done={completed_rx}/{TOTAL_RX} rx  "
        f"K={K_mean_dB:.1f} dB  ASD={ASD_deg:.1f}° ASA={ASA_deg:.1f}° "
        f"ZSD={ZSD_deg:.1f}° ZSA={ZSA_deg:.1f}°  "
        f'P_los: {_frac_los  :6.2f} %'
        f'  P_specular: {_frac_spec :6.2f} %'
        f'  P_diffuse: {_frac_diff :6.2f} %'
        f'  P_diffraction: {_frac_diffr:6.2f} %'
        f"  scene_t={elapsed:.1f}s  total={total_e/3600:.2f}h"
        f"{iono_str}{atm_str}"
    )
    sys.stdout.flush()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if (scene_idx + 1) % CHECKPOINT_INTERVAL == 0 or completed_rx == TOTAL_RX:
        ck_path = save_checkpoint(
            output_dir     = OUTPUT_DIR,
            elevation_deg  = ELEVATION_DEG,
            completed_rx   = completed_rx,
            total_rx       = TOTAL_RX,
            batch_size     = N_RX_PER_SCENE,
            frequency      = FREQUENCY,
            tx_pos         = TX_POS,
            slant_dist_m   = SLANT_DIST_ACTUAL_M,
            cfg            = cfg,
            K_power_ratio  = all_K_power,
            K_moments      = all_K_moments,
            tau_rms_mean_s = all_tau_rms,
            los_probability= all_p_los,
            P_los          = all_P_los,
            P_specular     = all_P_specular,
            P_diffuse      = all_P_diffuse,
            P_diffraction  = all_P_diffraction,
            P_refraction   = all_P_refraction,
            P_nlos         = all_P_nlos,
            omega_F        = all_omega_F,
            vtec_sample    = all_vtec_sample,
            A_gas_db       = all_A_gas_db,
            R_sample_mm_h  = all_R_sample,
            A_rain_db      = all_A_rain_db,
            scint_db       = all_scint_db,
            atm_total_db   = all_atm_total_db,
            doppler_mean_hz = all_doppler_mean_hz,
            doppler_rms_hz  = all_doppler_rms_hz,
            doppler_mean_hz_td = all_doppler_mean_hz_td,
            doppler_rms_hz_td  = all_doppler_rms_hz_td,
            ASD_deg        = all_ASD_deg,
            ASA_deg        = all_ASA_deg,
            ZSD_deg        = all_ZSD_deg,
            ZSA_deg        = all_ZSA_deg,
            scene_seeds    = all_scene_seeds,
            nominal_rx_positions = all_rx_positions,
            raw_results    = raw_results,
        )
        print(f"  ✓ checkpoint → {ck_path}")
        sys.stdout.flush()

# Save final results

if GROUND_RETRY_MAX > 0:
    print(f"  Ground-path dropout: {ground_retries} extra solve(s), "
          f"{ground_retry_exhausted}/{len(all_P_los)} scene(s) still missing it "
          f"after {GROUND_RETRY_MAX} retries")

results_path = save_results(
    output_dir           = OUTPUT_DIR,
    elevation_deg        = ELEVATION_DEG,
    nominal_rx_positions = all_rx_positions,
    K_power_ratio        = all_K_power,
    K_moments            = all_K_moments,
    tau_rms_mean_s       = all_tau_rms,
    los_probability      = all_p_los,
    P_los                = all_P_los,
    P_specular           = all_P_specular,
    P_diffuse            = all_P_diffuse,
    P_diffraction        = all_P_diffraction,
    P_refraction         = all_P_refraction,
    P_nlos               = all_P_nlos,
    omega_F              = all_omega_F,
    vtec_sample          = all_vtec_sample,
    A_gas_db             = all_A_gas_db,
    R_sample_mm_h        = all_R_sample,
    A_rain_db            = all_A_rain_db,
    scint_db             = all_scint_db,
    atm_total_db         = all_atm_total_db,
    doppler_mean_hz      = all_doppler_mean_hz,
    doppler_rms_hz       = all_doppler_rms_hz,
    doppler_mean_hz_td   = all_doppler_mean_hz_td,
    doppler_rms_hz_td    = all_doppler_rms_hz_td,
    ASD_deg              = all_ASD_deg,
    ASA_deg              = all_ASA_deg,
    ZSD_deg              = all_ZSD_deg,
    ZSA_deg              = all_ZSA_deg,
    scene_seeds          = all_scene_seeds,
    raw_results          = raw_results,
    cfg                  = cfg,
    tx_pos               = TX_POS,
    slant_dist_m         = SLANT_DIST_ACTUAL_M,
)

total_elapsed = time.perf_counter() - total_start
print(f"\n{'='*64}")
print(f"  Elevation {ELEVATION_DEG:.0f}° — COMPLETE")
print(f"  Scenes    : {N_SCENES}   Receivers : {completed_rx}/{TOTAL_RX}")
print(f"  Elapsed   : {total_elapsed/3600:.2f} h  ({total_elapsed:.1f} s)")
print(f"  Results   : {results_path}")
print(f"{'='*64}\n")
