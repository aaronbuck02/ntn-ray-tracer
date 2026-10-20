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
import shutil

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
parser.add_argument("--config",    type=str,   default=None,
                    help="Path to sim_config.json")
parser.add_argument("--scenes_manifest", type=str, default=None,
                    help="Path to scenes_manifest.pkl (generated once by "
                         "scene_gen_ppp.generate_scene_ensemble, shared across "
                         "every elevation angle in this sweep)")
parser.add_argument("--output",    type=str,   default=None,
                    help="Output directory for checkpoints and results")
parser.add_argument("--group",     type=str,   default=None,
                    help="Solve-group JSON from sweep_utils: {members: [{name, dir, config, "
                         "manifest}]}. Members share each scene's ray trace; replaces "
                         "--config/--scenes_manifest/--output.")
parser.add_argument("--resume",    action="store_true", default=False,
                    help="Resume from checkpoint if one exists")
parser.add_argument("--no-resume", action="store_true", default=False,
                    dest="no_resume",
                    help="Force a fresh start, ignoring any existing checkpoint "
                         "(overrides --resume if both are passed)")
args = parser.parse_args()

if args.group:
    with open(args.group) as f:
        MEMBERS = json.load(f)["members"]
elif args.config and args.scenes_manifest and args.output:
    MEMBERS = [{"name": os.path.basename(os.path.normpath(args.output)), "dir": args.output,
                "config": args.config, "manifest": args.scenes_manifest}]
else:
    parser.error("give --group, or all of --config, --scenes_manifest and --output")

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
                              rms_doppler_spread_hz,
                              power_by_depth, object_roles, power_by_role,
                              rician_k_moment_fit)
# Patched fork of scene_utils (scattering pattern, UE pattern/orientation); see its docstring.
from Ray_Tracing.src.utils.scene_utils_sensitivity import (build_scene_batch, sample_rx_positions,
                                                            sample_rx_trajectories)
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

for _m in MEMBERS:
    with open(_m["config"], "r") as f:
        _m["cfg"] = json.load(f)
    with open(_m["manifest"], "rb") as f:
        _m["manifest"] = pickle.load(f)
# Members may differ only in UE-side keys; the launcher grouped them by this key.
from Ray_Tracing.src.utils.sweep_utils import solve_key
_keys = {solve_key(_m["cfg"], _m["manifest"]) for _m in MEMBERS}
if len(_keys) > 1:
    raise ValueError(f"group members do not share one solve: keys {sorted(_keys)}")
cfg = dict(MEMBERS[0]["cfg"])   # shared (non-UE) settings

# Verify the patched Sionna is loaded and record it in the results, so every
# run file carries its own provenance. scene_utils re-checks at solve time.
import Ray_Tracing.src.sionna_patch_check as sionna_patch_check
cfg["_SIONNA_PATCH"] = sionna_patch_check.require("run_stochastic_sim")
for _m in MEMBERS:
    _m["cfg"]["_SIONNA_PATCH"] = cfg["_SIONNA_PATCH"]
print(f"  {sionna_patch_check.summary_line(cfg['_SIONNA_PATCH'])}")

OUTPUT_DIR              = MEMBERS[0]["dir"]   # scratch space; results go to each member's dir

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
# Walking UEs: UE_TRAJ_STEPS > 1 solves each UE at t = 0, dt, ... along a straight
# walk (heading U[0, 360), speed U[UE_SPEED_RANGE_MPS]); the TX stays frozen at the
# elevation's orbit draw. Each snapshot is one receiver (one result row).
# Bank per-path interaction points in raw_results (ground-vs-wall reflection
# split). Off by default: it is the bulkiest thing in the raw payload.
SAVE_VERTICES           = bool(cfg.get("SAVE_PATH_VERTICES", False))
# Retries for the Sionna ground-path dropout (see the retry block below).
# 0 disables the workaround entirely.
GROUND_RETRY_MAX        = int(cfg.get("GROUND_RETRY_MAX", 0))
GROUND_RETRY_JITTER_M   = float(cfg.get("GROUND_RETRY_JITTER_M", 0.05))
GROUND_Z_TOL_M          = float(cfg.get("GROUND_Z_TOL_M", 0.05))
# Set -> solver seed is a pure function of (scene seed, elevation, offset): common
# random numbers across sweep variants and resume-safe. None keeps the legacy stream.
SOLVER_SEED_OFFSET      = cfg.get("SOLVER_SEED_OFFSET")
K_FIT_BW_HZ             = float(cfg.get("K_FIT_BW_HZ", 100e6))
K_FIT_N                 = int(cfg.get("K_FIT_N", 0))   # 0: skip the moment-fit K (NaN); it cost ~1.3 s/UE
# Keys a manifest entry may override per scene: only those read per scene below
# or inside build_scene_batch. Anything else is fixed for the whole run.
PER_SCENE_CFG_KEYS      = {"RX_HEIGHT", "SCATTERING_COEFF", "RX_ORIENTATION_DEG",
                           "RX_BASE_PATTERN", "POL_BASE_PATTERN",
                           "TX_POL_TYPE", "RX_POL_TYPE", "SCENE_EAST_HEADING_DEG",
                           "RX_PITCH_DEG", "RX_YAW_DEG", "VTEC_MEAN_TECU", "VTEC_STD_TECU",
                           "RAIN_R_MM_H", "ATM_RHO_G_M3"}
# Streamed scenes: verify the first scene of the run, plus every Nth if > 0.
PPP_VERIFY_EVERY        = int(cfg.get("PPP_VERIFY_EVERY", 0))


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
def _n_rx(c):
    """(UEs drawn, UEs kept) per scene at this elevation. N_RX_KEEP_BY_ELEV {"10": 25, "*": 50}
    keeps the first n of the usual draw, so the kept UEs are a subset of the full set's."""
    n = int(c.get("N_RX_PER_SCENE", 1))
    kb = c.get("N_RX_KEEP_BY_ELEV") or {}
    keep = kb.get(str(int(round(ELEVATION_DEG))), kb.get("*"))
    return n, min(n, int(keep)) if keep else n


def _traj(c):
    """(steps, dt [s], speed range [m/s]) of the member's UE walks; steps 1 = static UEs."""
    return (int(c.get("UE_TRAJ_STEPS", 1)), float(c.get("UE_TRAJ_DT_S", 1.0)),
            tuple(float(v) for v in c.get("UE_SPEED_RANGE_MPS", (1.0, 1.6))))


for _m in MEMBERS:
    _m["n_draw"], _m["n_ue"] = _n_rx(_m["cfg"])
    _m["traj"] = _traj(_m["cfg"])
    _m["n_rx"] = _m["n_ue"] * _m["traj"][0]   # result rows (receivers) per scene
    _m["rx_h"] = float(_m["cfg"].get("RX_HEIGHT", 1.5))
    if RX_AT_ORIGIN and _m["n_rx"] != 1:
        raise ValueError("PPP_RX_AT_ORIGIN pins one fixed position, so "
                         f"N_RX_PER_SCENE must be 1, got {_m['n_rx']} ({_m['name']}).")
    if _m["n_rx"] < 1:
        raise ValueError(f"N_RX_PER_SCENE must be >= 1, got {_m['n_rx']} ({_m['name']})")
if GROUND_RETRY_MAX > 0 and (len(MEMBERS) > 1 or MEMBERS[0]["n_rx"] > 1):
    raise ValueError("GROUND_RETRY_MAX re-solves a single receiver: it needs one "
                     "member with N_RX_PER_SCENE = 1")
# UEs per solve; 0 = all of a scene's receivers at once. An OOM halves it.
RX_CHUNK                = int(cfg.get("RX_CHUNK", 0))

# Polarization: one pair for the whole group keeps single-pol arrays (as before);
# several pairs put a dual-pol [V, H] element on the side(s) that vary and each
# member reads its pair out of the 2x2 coefficients (_pol_weights).
POL_BASE_PATTERN        = cfg.get("POL_BASE_PATTERN", "iso")
_POL_PAIRS = {(e.get("cfg_overrides", {}).get("TX_POL_TYPE", _m["cfg"].get("TX_POL_TYPE", "V")),
               e.get("cfg_overrides", {}).get("RX_POL_TYPE", _m["cfg"].get("RX_POL_TYPE", "V")))
              for _m in MEMBERS for e in _m["manifest"]["scenes"]}
TX_ARRAY_POL = next(iter({p[0] for p in _POL_PAIRS})) if len({p[0] for p in _POL_PAIRS}) == 1 else "VH"
RX_ARRAY_POL = next(iter({p[1] for p in _POL_PAIRS})) if len({p[1] for p in _POL_PAIRS}) == 1 else "VH"
cfg["TX_POL_TYPE"], cfg["RX_POL_TYPE"] = TX_ARRAY_POL, RX_ARRAY_POL   # what build_scene_batch builds

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
# Gated on a TX velocity: --waypoints_file, or per-scene orbits in the manifest (below).
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

manifest      = MEMBERS[0]["manifest"]
scene_entries = manifest["scenes"]
# Per-scene real-TLE orbits (orbit_sampling): each scene carries its own TX
# state at this elevation, so the per-run TX above is only the fallback.
USE_SCENE_ORBITS = all(ELEVATION_DEG in (e.get("orbit") or {}) for e in scene_entries)
if any(e.get("orbit") for e in scene_entries) and not USE_SCENE_ORBITS:
    raise ValueError(f"some manifest scenes lack an orbit waypoint for elevation {ELEVATION_DEG}")
if USE_SCENE_ORBITS and USE_WAYPOINT:
    raise ValueError("--waypoints_file and per-scene manifest orbits are mutually exclusive")
if USE_SCENE_ORBITS:
    DOPPLER_ENABLED = bool(cfg.get("DOPPLER_ENABLED", False))
STREAMING     = bool(manifest.get("streaming"))
SCRATCH_ROOT  = os.path.join(OUTPUT_DIR, f"scratch_elev{int(ELEVATION_DEG):03d}")
N_SCENES      = len(scene_entries)
for _m in MEMBERS:
    _m["total_rx"] = N_SCENES * _m["n_rx"]
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
for _m in MEMBERS:   # each member's cfg_snapshot records the same materials
    for _k in ("MATERIAL_NAMES", "MATERIAL_PARAMS"):
        if _k in cfg:
            _m["cfg"][_k] = cfg[_k]
    if len(MEMBERS) > 1 or "VH" in (TX_ARRAY_POL, RX_ARRAY_POL):
        _m["cfg"]["_SOLVE_GROUP"] = [m["name"] for m in MEMBERS]
        _m["cfg"]["_ARRAY_POL"] = [TX_ARRAY_POL, RX_ARRAY_POL]

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
print(f"  Arrays    : TX={TX_ARRAY_POL}  RX={RX_ARRAY_POL}  base_pattern={POL_BASE_PATTERN}"
      f"  rx_chunk={RX_CHUNK or 'all'}")
if DOPPLER_ENABLED and USE_SCENE_ORBITS:
    # Bound and sampling rate are set per scene from that scene's orbit velocity.
    print(f"  Doppler   : ENABLED  per-scene orbit velocity  steps={DOPPLER_NUM_TIME_STEPS}"
          + ("  (path-domain only)" if DOPPLER_NUM_TIME_STEPS <= 1 else ""))
elif DOPPLER_ENABLED:
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
for _m in MEMBERS:
    _where = "pinned at the origin" if RX_AT_ORIGIN else \
        f"uniform over free space (clearance={RX_CLEARANCE} m)"
    if _m["traj"][0] > 1:
        _where += (f", walking: {_m['traj'][0]} snapshots every {_m['traj'][1]:g} s at "
                   f"{_m['traj'][2][0]:g}-{_m['traj'][2][1]:g} m/s, TX frozen")
    print(f"  Receivers : {_m['name']}: {_m['n_ue']} UEs x {_m['traj'][0]} = {_m['n_rx']}/scene "
          f"{_where}, z={_m['rx_h']} m, "
          f"TX={_m['cfg'].get('TX_POL_TYPE', 'V')} RX={_m['cfg'].get('RX_POL_TYPE', 'V')}"
          f"  ->  {_m['total_rx']} total")
if GROUND_RETRY_MAX:
    print(f"  Ground retry : up to {GROUND_RETRY_MAX} re-solve(s) on a dropped "
          f"ground path (jitter {GROUND_RETRY_JITTER_M} m)")
print(f"{'='*64}\n")

# Checkpoint resume or fresh start

EXTRA_KEYS = ("PG_dB", "n_paths", "P_by_depth", "P_ground", "P_wall", "P_roof",
              "ue_id", "traj_t_s", "ue_speed_mps", "ue_heading_deg",
              "K_fit_dB", "rx_height", "itu_alpha", "itu_beta", "itu_gamma_fit",
              "heading_deg", "sat_az_deg", "slant_range_m", "orbit_peak_deg",
              "orbit_leg", "norad_id", "gs_lat", "gs_lon")
# Per-UE result lists, named as save_checkpoint / save_results take them.
ACC_KEYS = ("nominal_rx_positions", "K_power_ratio", "K_moments", "tau_rms_mean_s",
            "los_probability", "P_los", "P_specular", "P_diffuse", "P_diffraction",
            "P_refraction", "P_nlos", "omega_F", "vtec_sample", "A_gas_db",
            "R_sample_mm_h", "A_rain_db", "scint_db", "atm_total_db",
            "doppler_mean_hz", "doppler_rms_hz", "doppler_mean_hz_td",
            "doppler_rms_hz_td", "ASD_deg", "ASA_deg", "ZSD_deg", "ZSA_deg",
            "scene_seeds")
_REQUIRED_KEYS = {"K_power_ratio", "K_moments", "tau_rms_mean_s", "los_probability"}


def _scene_cfg(entry, base=None):
    """cfg for one scene: manifest per-scene material params and cfg_overrides applied."""
    c = dict(cfg if base is None else base)
    bad = set(entry.get("cfg_overrides", {})) - PER_SCENE_CFG_KEYS
    if bad:
        raise ValueError(f"scene {entry['seed']}: cfg_overrides {sorted(bad)} "
                         f"cannot vary per scene (allowed: {sorted(PER_SCENE_CFG_KEYS)})")
    c.update(entry.get("cfg_overrides", {}))
    if "RX_PITCH_DEG" in c or "RX_YAW_DEG" in c:
        # Boresight elevation p maps to Sionna pitch -p (pitch -90 points +x to zenith).
        c["RX_ORIENTATION_DEG"] = [float(c.get("RX_YAW_DEG", 0.0)),
                                   -float(c.get("RX_PITCH_DEG", 0.0)), 0.0]
    if any(k in entry for k in ("wall_params", "roof_params", "ground_params")):
        mp = dict(c.get("MATERIAL_PARAMS") or {})
        for role, name in (("wall", manifest["building_material"]),
                           ("roof", _roof_mat),
                           ("ground", manifest["ground_material"])):
            if f"{role}_params" in entry:
                mp[name] = entry[f"{role}_params"]
        c["MATERIAL_PARAMS"] = mp
    return c


def _scene_tx(entry, scene_cfg):
    """(tx_pos, beam_half_angle, tx_vel, waypoint, heading): the scene's orbit, else the per-run TX."""
    hd = float(scene_cfg.get("SCENE_EAST_HEADING_DEG", SCENE_EAST_HEADING_DEG))
    if not USE_SCENE_ORBITS:
        return TX_POS, BEAM_HALF_ANGLE_RAD, TX_VEL, WAYPOINT, hd
    wp = entry["orbit"][ELEVATION_DEG]
    pos, beam = tx_position_from_pass_sample(wp, SCENE_CENTER, SCENE_RADIUS,
                                             scene_east_heading_deg=hd)
    return pos, beam, tx_velocity_scene_frame(wp, scene_east_heading_deg=hd), wp, hd


def _scene_geometry(entry, scene_idx):
    """(xml_path, SceneSpec, scratch_dir|None): files from the manifest, or streamed into scratch."""
    if not STREAMING:
        return entry["xml_path"], sg.load_scene_spec(entry["spec_path"], entry["meta_path"]), None
    d = os.path.join(SCRATCH_ROOT, f"scene_{int(entry['seed']):07d}")
    shutil.rmtree(d, ignore_errors=True)
    verify = scene_idx == start_scene or (PPP_VERIFY_EVERY > 0 and scene_idx % PPP_VERIFY_EVERY == 0)
    xml, spec = sg.generate_scene_streaming(
        d, int(entry["seed"]), scene_name="ppp_scene",
        primitive=manifest.get("primitive", "block"),
        building_material=manifest["building_material"], roof_material=_roof_mat,
        ground_material=manifest["ground_material"],
        wall_params=manifest["wall_params"], roof_params=manifest["roof_params"],
        ground_params=manifest["ground_params"],
        one_ply_per_building=True, binary_ply=True, verify=verify,
        **{**manifest.get("sample_kwargs", {}), **entry.get("sample_kwargs", {})})
    return xml, spec, d


# (theta, phi) Jones vectors of the element polarizations; CP as in circular_polarization.py.
_JONES = {"V": (1.0, 0.0), "H": (0.0, 1.0),
          "RHCP": (2 ** -0.5, -1j * 2 ** -0.5), "LHCP": (2 ** -0.5, 1j * 2 ** -0.5)}
_CP_CONJ = {"RHCP": "LHCP", "LHCP": "RHCP"}   # RX handedness flip, as set_polarization


def _pol_weights(tx_pol, rx_pol):
    """(w_tx, w_rx) over the [V, H] patterns, or None on a single-pol side.

    TX coefficients are linear in the TX pattern; the RX side is u^H E, so
    conjugate-linear in the (handedness-flipped) RX pattern.
    """
    def _j(p):
        if p.upper() not in _JONES:
            raise ValueError(f"polarization {p!r} cannot be read out of a dual-pol solve")
        return np.asarray(_JONES[p.upper()], dtype=complex)
    wt = _j(tx_pol) if TX_ARRAY_POL == "VH" else None
    wr = np.conj(_j(_CP_CONJ.get(rx_pol.upper(), rx_pol))) if RX_ARRAY_POL == "VH" else None
    return wt, wr


def _new_state(m, ck):
    """Per-member accumulators, from a checkpoint if given."""
    if ck is None:
        m["done"] = 0
        m["acc"] = {k: [] for k in ACC_KEYS}
        m["extra"] = {k: [] for k in EXTRA_KEYS}
        m["raw"] = []
        return
    n = ck["completed_rx"]   # receivers done, NOT scenes
    # NB: checkpoint_utils stores the config under "cfg_snapshot", not "cfg".
    _ck_cfg = ck.get("cfg_snapshot") or {}
    _ck_n_rx = _n_rx(_ck_cfg)[1] * _traj(_ck_cfg)[0]
    if _ck_cfg and _ck_n_rx != m["n_rx"]:
        raise ValueError(
            f"{m['name']}: checkpoint was written with {_ck_n_rx} UEs per scene but this "
            f"run has {m['n_rx']}. Resuming would misalign the scene cursor and every "
            f"accumulator. Re-run with --no-resume, or restore the original value.")
    acc = {}
    for k in ACC_KEYS:
        if k in _REQUIRED_KEYS:
            acc[k] = list(ck[k])
        elif k == "nominal_rx_positions":
            acc[k] = [list(x) for x in ck.get(k, [])]
        elif k == "scene_seeds":
            acc[k] = list(ck.get(k, []))
        else:   # zero-fill checkpoints that pre-date a key
            acc[k] = list(ck.get(k, [0.0] * n))
    m["done"] = n
    m["acc"] = acc
    # NaN-pad checkpoints that pre-date these keys so old runs still resume.
    m["extra"] = {k: list(ck.get(k, [float("nan")] * n)) for k in EXTRA_KEYS}
    m["raw"] = list(ck["raw_results"])


for _m in MEMBERS:
    os.makedirs(_m["dir"], exist_ok=True)
    _new_state(_m, load_checkpoint(_m["dir"], ELEVATION_DEG) if RESUME_FROM_CHECKPOINT else None)

# Members advance in lockstep; one behind (e.g. newly added) pulls the rest back.
start_scene = min(_m["done"] // _m["n_rx"] for _m in MEMBERS)
_kept_seeds = {int(e["seed"]) for e in scene_entries[:start_scene]}
for _m in MEMBERS:
    _keep = start_scene * _m["n_rx"]
    if _m["done"] > _keep:
        print(f"  {_m['name']}: trimming checkpoint {_m['done']} -> {_keep} rx to match the group")
        _m["acc"] = {k: v[:_keep] for k, v in _m["acc"].items()}
        _m["extra"] = {k: v[:_keep] for k, v in _m["extra"].items()}
        _m["raw"] = [r for r in _m["raw"] if int(r["seed"]) in _kept_seeds]
        _m["done"] = _keep
if start_scene > 0:
    print(f"  Resuming at scene {start_scene}/{N_SCENES}")
else:
    print(f"  Starting fresh run for elevation {ELEVATION_DEG}°")


# Candidate-cap check: Sionna stores at most max_num_paths_per_src (ray, receiver)
# candidates per source, shared by every receiver in a solve, and silently drops
# the rest. Record the stored count against the cap from the buffer's shrink().
from sionna.rt.path_solvers.paths_buffer import PathsBuffer
_CAND = {}
_shrink = PathsBuffer.shrink


def _shrink_logged(self):
    _CAND["cap"] = int(self.buffer_size)
    _shrink(self)
    _CAND["n"] = int(self.buffer_size)


PathsBuffer.shrink = _shrink_logged


# Roles by material for merged shapes (MERGE_SHAPES): Sionna merges per radio
# material, so walls/roofs lose their names but keep distinct dummy materials.
_ROLE_BY_MATERIAL = {manifest["building_material"]: "wall", _roof_mat: "roof",
                     manifest["ground_material"]: "ground"}
if cfg.get("MERGE_SHAPES") and len(set(_ROLE_BY_MATERIAL)) < 3:
    raise ValueError("MERGE_SHAPES needs distinct wall/roof/ground materials to keep P_wall/P_roof")


def _scene_roles(scene):
    roles = object_roles(scene)
    for name, obj in scene.objects.items():
        r = _ROLE_BY_MATERIAL.get(obj.radio_material.name)
        if int(obj.object_id) not in roles and r:
            roles[int(obj.object_id)] = r
    return roles


def _scene_rng(entry, stream):
    """Per-scene draw stream: identical for every variant, carrier and resume (paired)."""
    return np.random.default_rng([RNG_SEED, int(entry["seed"]),
                                  int(round(ELEVATION_DEG * 100)), stream])


def _atm_draw(entry, c):
    """Atmospheric losses for one scene under cfg c (rain / water vapour may vary per member)."""
    if not ATMOSPHERE_ENABLED:
        return {"A_gas_db": 0.0, "R_sample_mm_h": 0.0, "A_rain_db": 0.0,
                "scint_db": 0.0, "sigma_scint_db": 0.0, "total_loss_db": 0.0}
    return compute_atmospheric_effects(
        freq_hz               = FREQUENCY,
        elev_deg              = ELEVATION_DEG,
        rho_g_m3              = float(c.get("ATM_RHO_G_M3", ATM_RHO_G_M3)),
        R_mm_h                = float(c.get("RAIN_R_MM_H", RAIN_R_MM_H)),
        rng                   = _scene_rng(entry, 2),
        T_K                   = ATM_T_K,
        P_hPa                 = ATM_P_HPA,
        h_station_km          = ATM_H_STATION_KM,
        h_rain_km             = ATM_H_RAIN_KM,
        pol_tilt_deg          = ATM_POL_TILT_DEG,
        D_ant_m               = ATM_D_ANT_M,
        include_scintillation = ATM_SCINTILLATION,
    )


def _free_gpu():
    gc.collect()
    try:
        drjit.flush_malloc_cache()
        drjit.eval()
    except Exception:
        pass


def _solve_chunk(rx_pos, rx_orient, rx_vel, scene_cfg, tx_pos, tx_vel, beam, omega_F, seed, scene_seed):
    """Build and solve one chunk of receivers; returns (per-receiver numpy slices, roles, retries)."""
    global ground_retries, ground_retry_exhausted

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
            beam_angle            = beam,
            max_num_paths_per_src = BASE_MAX_PATHS,
            samples_per_src       = BASE_SAMPLES_PER_SRC,
            seed                  = seed,
        )

    tx_v = tx_vel if DOPPLER_ENABLED else None
    rx_v = rx_vel if DOPPLER_ENABLED else None
    _t0 = time.perf_counter()
    scene, _ = build_scene_batch(rx_pos, tx_pos, scene_cfg, omega_F=omega_F,
                                 tx_velocity=tx_v, rx_orientations_deg=rx_orient,
                                 rx_velocities=rx_v)
    _t1 = time.perf_counter()
    paths = _solve(scene)
    drjit.eval()
    _t2 = time.perf_counter()

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
    # the exhausted count is reported. Single receiver only (checked at start).
    if GROUND_RETRY_MAX > 0:
        _jit_rng = np.random.default_rng(RNG_SEED + 104729 * int(scene_seed))
        for _attempt in range(GROUND_RETRY_MAX):
            if _has_los_and_ground(paths):
                break
            _off = _jit_rng.uniform(-GROUND_RETRY_JITTER_M, GROUND_RETRY_JITTER_M, 2)
            _rx_j = [[rx_pos[0][0] + _off[0], rx_pos[0][1] + _off[1], rx_pos[0][2]]]
            scene, _ = build_scene_batch(_rx_j, tx_pos, scene_cfg, omega_F=omega_F,
                                         tx_velocity=tx_v, rx_orientations_deg=rx_orient,
                                         rx_velocities=rx_v)
            paths = _solve(scene)
            ground_retries += 1
        else:
            if not _has_los_and_ground(paths):
                ground_retry_exhausted += 1

    # Index layout [rx, rx_ant, tx, tx_ant, path(, ...)]; rx_ant / tx_ant are
    # the [V, H] patterns on a dual-pol side. Geometry is the same for every
    # pattern pair, so it is read at pattern 0.
    amp, tau = paths.cir(normalize_delays=False, out_type="numpy")
    inter    = paths.interactions.numpy()
    dop      = paths.doppler.numpy() if DOPPLER_ENABLED else None
    th_t, ph_t = paths.theta_t.numpy(), paths.phi_t.numpy()
    th_r, ph_r = paths.theta_r.numpy(), paths.phi_r.numpy()
    # Interaction points [depth, rx, rx_ant, tx, tx_ant, path, 3] -- the only
    # way to tell a ground reflection from a wall reflection, since Sionna
    # labels both IT_SPECULAR. Optional: older Sionna builds lack it.
    try:
        vert = paths.vertices.numpy()
    except AttributeError:
        vert = None
    obj   = paths.objects.numpy()
    roles = _scene_roles(scene)
    cand  = dict(_CAND)
    del paths, scene
    cand["t"] = (_t1 - _t0, _t2 - _t1, time.perf_counter() - _t2)

    out = []
    for r in range(len(rx_pos)):
        a_r = amp[r][:, 0]                                     # [rx_pat, tx_pat, path, time]
        v = np.any(np.abs(a_r) > 0.0, axis=(0, 1, 3))         # path exists for any pattern pair
        out.append({
            "a": a_r[:, :, v, :], "t": tau[r, 0, 0, 0, :][v],
            "inter": inter[:, r, 0, 0, 0, :][:, v], "obj": obj[:, r, 0, 0, 0, :][:, v],
            "vert": vert[:, r, 0, 0, 0, :, :][:, v, :] if vert is not None else None,
            "dop": dop[r, 0, 0, 0, :][v] if dop is not None else None,
            "theta_t": th_t[r, 0, 0, 0, :][v], "phi_t": ph_t[r, 0, 0, 0, :][v],
            "theta_r": th_r[r, 0, 0, 0, :][v], "phi_r": ph_r[r, 0, 0, 0, :][v],
        })
    return out, roles, cand


def _extract_ue(rxd, w, sc, want_raw):
    """One UE's LSPs from its receiver's paths, read out with weights w=(w_tx, w_rx)."""
    wt, wr = w
    a = rxd["a"]
    single = wt is None and wr is None
    if single:
        a_full = a[0, 0]
    else:
        a_full = np.einsum("r,t,rtpk->pk",
                           wr if wr is not None else np.ones(a.shape[0]),
                           wt if wt is not None else np.ones(a.shape[1]), a)
    # Single-pol: drop paths that are zero for this antenna pair, exactly as
    # before. Dual-pol: a path counts if it exists for any pattern pair.
    valid = np.any(np.abs(a_full) > 0.0, axis=-1) if single \
        else np.ones(a_full.shape[0], dtype=bool)
    t_v         = rxd["t"][valid]
    a_full_v    = a_full[valid, :]             # [n_valid_paths, num_time_steps]
    inter_all_v = rxd["inter"][:, valid]       # [max_depth, n_valid_paths]
    doppler_v   = rxd["dop"][valid] if DOPPLER_ENABLED else None
    # Angle slices -- corrections below modify t_v/a_full_v's delay/amplitude
    # values but never reorder or drop entries, so these stay index-aligned.
    theta_t_v   = rxd["theta_t"][valid]
    phi_t_v     = rxd["phi_t"][valid]
    theta_r_v   = rxd["theta_r"][valid]
    phi_r_v     = rxd["phi_r"][valid]
    vert_v      = rxd["vert"][:, valid, :] if rxd["vert"] is not None else None
    obj_v       = rxd["obj"][:, valid]

    # ── Post-RT: ionospheric group delay + phase advance ───────────────────
    if IONOSPHERE_ENABLED and len(t_v) > 0:
        iono = apply_ionospheric_corrections(
            t_v, a_full_v,
            tec_slant_tecu = sc["tec_slant"],
            freq_hz        = FREQUENCY,
        )
        t_v      = iono["taus_mod"]
        a_full_v = iono["amps_mod"]

    # ── Post-RT: atmospheric amplitude loss ────────────────────────────────
    if ATMOSPHERE_ENABLED and len(t_v) > 0:
        atm = apply_atmospheric_corrections(
            t_v, a_full_v,
            total_loss_db = sc["atm"]["total_loss_db"],
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
    if len(t_v) > 0:
        _p_depth = power_by_depth(a_v, inter_all_v, MAX_DEPTH)
        _p_role  = power_by_role(a_v, obj_v, sc["roles"])
        _k_fit   = rician_k_moment_fit(t_v, a_v, K_FIT_BW_HZ, K_FIT_N) if K_FIT_N > 0 else float("nan")
    else:
        _p_depth = np.zeros(MAX_DEPTH + 1)
        _p_role  = {"P_ground": 0.0, "P_wall": 0.0, "P_roof": 0.0}
        _k_fit   = float("nan")
    wp, gs = sc["wp"], sc["gs"]
    extra = {
        "PG_dB": float(10 * np.log10(np.sum(np.abs(a_v) ** 2) + 1e-300)) if len(t_v) else float("nan"),
        "n_paths": int(len(t_v)), "P_by_depth": _p_depth.tolist(), **_p_role,
        "K_fit_dB": _k_fit, "rx_height": sc["rx_h"],
        **{k: float(sc["itu"][k]) for k in ("itu_alpha", "itu_beta", "itu_gamma_fit")},
        "heading_deg": sc["heading"],
        "sat_az_deg": float(wp["az_deg"]) if wp else float("nan"),
        "slant_range_m": float(np.linalg.norm(np.asarray(sc["tx_pos"])
                                              - np.array([*SCENE_CENTER, 0.0]))),
        "orbit_peak_deg": float(wp.get("peak_deg", np.nan)) if wp else float("nan"),
        "orbit_leg": wp.get("leg", "") if wp else "",
        "norad_id": int(wp.get("norad_id", -1)) if wp else -1,
        "gs_lat": float(gs[0]), "gs_lon": float(gs[1]),
    }
    h_env = float(np.abs(h_total))
    ASD_deg, ASA_deg = np.degrees(ASD_rad), np.degrees(ASA_rad)
    ZSD_deg, ZSA_deg = np.degrees(ZSD_rad), np.degrees(ZSA_rad)
    atm_s = sc["atm"]
    rec = {
        "nominal_rx_positions": [float(v) for v in sc["rx_nom"]],
        "K_power_ratio": float(K_pr), "K_moments": float("nan"),
        "tau_rms_mean_s": float(ds), "los_probability": 1.0 if has_los_flag else 0.0,
        "P_los": float(P_los), "P_specular": float(pwr_inc["P_specular"]),
        "P_diffuse": float(pwr_inc["P_diffuse"]),
        "P_diffraction": float(pwr_inc["P_diffraction"]),
        "P_refraction": float(pwr_inc["P_refraction"]), "P_nlos": float(P_nlos),
        "omega_F": float(sc["omega_F"]), "vtec_sample": float(sc["tec_slant"]),
        "A_gas_db": float(atm_s["A_gas_db"]), "R_sample_mm_h": float(atm_s["R_sample_mm_h"]),
        "A_rain_db": float(atm_s["A_rain_db"]), "scint_db": float(atm_s["scint_db"]),
        "atm_total_db": float(atm_s["total_loss_db"]),
        "doppler_mean_hz": float(doppler_mean_hz), "doppler_rms_hz": float(doppler_rms_hz),
        "doppler_mean_hz_td": float(doppler_mean_hz_td),
        "doppler_rms_hz_td": float(doppler_rms_hz_td),
        "ASD_deg": float(ASD_deg), "ASA_deg": float(ASA_deg),
        "ZSD_deg": float(ZSD_deg), "ZSA_deg": float(ZSA_deg),
        "scene_seeds": int(sc["seed"]),
    }
    raw = None
    if want_raw:
        raw = {
            "seed"         : int(sc["seed"]),
            "rx_nominal"   : list(sc["rx_nom"]),
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
            "omega_F_rad"  : float(sc["omega_F"]),
            "vtec_tecu"    : float(sc["tec_slant"]),
            "A_gas_db"     : float(atm_s["A_gas_db"]),
            "R_sample_mm_h": float(atm_s["R_sample_mm_h"]),
            "A_rain_db"    : float(atm_s["A_rain_db"]),
            "scint_db"     : float(atm_s["scint_db"]),
            "atm_total_db" : float(atm_s["total_loss_db"]),
            "doppler_mean_hz": float(doppler_mean_hz),
            "doppler_rms_hz" : float(doppler_rms_hz),
            "doppler_mean_hz_td": float(doppler_mean_hz_td),
            "doppler_rms_hz_td" : float(doppler_rms_hz_td),
            "waypoint_az_deg": float(WAYPOINT["az_deg"]) if USE_WAYPOINT else None,
            "waypoint_range_km": float(WAYPOINT["range_km"]) if USE_WAYPOINT else None,
        }
    return rec, extra, raw


def _save_member_checkpoint(m):
    return save_checkpoint(
        output_dir     = m["dir"],
        elevation_deg  = ELEVATION_DEG,
        completed_rx   = m["done"],
        total_rx       = m["total_rx"],
        batch_size     = m["n_rx"],
        frequency      = FREQUENCY,
        tx_pos         = TX_POS,
        slant_dist_m   = SLANT_DIST_ACTUAL_M,
        cfg            = m["cfg"],
        raw_results    = m["raw"],
        extra          = m["extra"],
        **m["acc"],
    )


# Main simulation loop -- one PPP scene realisation per iteration. Every
# member's UEs are placed in the scene and solved together, in chunks.

rng         = np.random.default_rng(seed=RNG_SEED + int(ELEVATION_DEG * 7))
solver      = PathSolver()
total_start = time.perf_counter()
ground_retries = 0            # extra solves spent recovering dropped ground paths
ground_retry_exhausted = 0    # scenes where the ground path never reappeared
chunk_ok    = RX_CHUNK or None   # largest chunk known to fit; lowered on OOM
n_cap_hits  = 0

for scene_idx, entry in enumerate(scene_entries[start_scene:], start=start_scene):

    scene_start = time.perf_counter()
    scene_cfg   = _scene_cfg(entry)
    scene_cfg["TX_POL_TYPE"], scene_cfg["RX_POL_TYPE"] = TX_ARRAY_POL, RX_ARRAY_POL
    _xml, _spec, _scratch = _scene_geometry(entry, scene_idx)
    _itu        = sg.itu_parameters(_spec)
    _tx_pos, _beam, _tx_vel, _wp, _heading = _scene_tx(entry, scene_cfg)
    if DOPPLER_ENABLED:
        DOPPLER_BOUND_HZ = (FREQUENCY / constants.c) * float(np.linalg.norm(_tx_vel))
        SAMPLING_FREQUENCY_HZ = DOPPLER_OVERSAMPLE_FACTOR * max(DOPPLER_BOUND_HZ, 1.0)
    scene_cfg["SCENE_XML"] = _xml   # build_scene_batch() re-reads this
    _free_gpu()

    # ── Receivers: every member's UEs, identical ones solved once ──────────────
    # The footprint mask is analytic (no Mitsuba), so it costs ~0.1 s against a
    # ~100 s PathSolver call, and cannot leak the global Mitsuba variant.
    if not RX_AT_ORIGIN:
        _free_xy, _ = sg.free_space_grid(_spec, grid_res=RX_GRID_RES,
                                         margin=RX_MARGIN, clearance=RX_CLEARANCE)
    rx_index, rx_pos, rx_orient, rx_vel = {}, [], [], []
    ues = []    # per member: [(receiver index, nominal position, UE row keys)], weights, height
    for m in MEMBERS:
        mcfg = _scene_cfg(m["manifest"]["scenes"][scene_idx], m["cfg"])
        _rx_h = float(mcfg.get("RX_HEIGHT", m["rx_h"]))
        if RX_AT_ORIGIN:
            # Wall-ensemble (Chiu & Roy) mode: the terminal IS the PPP's reference
            # point, at (0, 0, h_g), and the scene is redrawn about it per
            # realisation. Sampling free space instead would be a different
            # experiment. The bias warning that motivates free-space sampling
            # applies to block scenes, whose origin gate inflates LoS; walls carry
            # no footprint area and the reference model has no exclusion zone.
            pos = [[0.0, 0.0, _rx_h]] * m["n_rx"]
            vel = [[0.0, 0.0, 0.0]] * m["n_rx"]
            um = [dict(ue_id=k, traj_t_s=0.0, ue_speed_mps=0.0, ue_heading_deg=np.nan)
                  for k in range(m["n_rx"])]
        elif m["traj"][0] > 1:
            # Walks seeded by the scene alone, on their own stream (static draws untouched).
            _steps, _dt, _vr = m["traj"]
            _p, _v, _sp, _hd = sample_rx_trajectories(
                _spec, _free_xy, m["n_ue"], _rx_h, _steps, _dt, _vr,
                rng=np.random.default_rng([RNG_SEED, int(entry["seed"]), 7919, 1]),
                grid_res=RX_GRID_RES, margin=RX_MARGIN, clearance=RX_CLEARANCE)
            pos = _p.reshape(-1, 3).tolist()
            vel = np.repeat(_v, _steps, axis=0).tolist()
            um = [dict(ue_id=k, traj_t_s=float(s * _dt), ue_speed_mps=float(_sp[k]),
                       ue_heading_deg=float(_hd[k])) for k in range(m["n_ue"]) for s in range(_steps)]
        else:
            # Seeded by the SCENE SEED alone -- never by elevation or frequency -- so
            # every angle and both carriers see the IDENTICAL UE positions. That is what
            # makes per-UE statistics pairable across the sweep; seeding off the run RNG
            # would silently resample them per angle and destroy the pairing.
            _rx_rng = np.random.default_rng(RNG_SEED + 7919 * int(entry["seed"]))
            pos = sample_rx_positions(_free_xy, m["n_draw"], _rx_h,
                                      rng=_rx_rng, grid_res=RX_GRID_RES).tolist()[:m["n_ue"]]
            vel = [[0.0, 0.0, 0.0]] * len(pos)
            um = [dict(ue_id=k, traj_t_s=0.0, ue_speed_mps=0.0, ue_heading_deg=np.nan)
                  for k in range(len(pos))]
        orient = [float(v) for v in mcfg.get("RX_ORIENTATION_DEG", [0.0, 0.0, 0.0])]
        idx = []
        for p, v, u in zip(pos, vel, um):
            key = (*[float(x) for x in p], *orient, *[float(x) for x in v])
            if key not in rx_index:
                rx_index[key] = len(rx_pos)
                rx_pos.append([float(x) for x in p])
                rx_orient.append(orient)
                rx_vel.append([float(x) for x in v])
            idx.append((rx_index[key], p, u))
        w = _pol_weights(mcfg.get("TX_POL_TYPE", "V"), mcfg.get("RX_POL_TYPE", "V"))
        ues.append((idx, w, _rx_h, _atm_draw(entry, mcfg)))
    n_recv = len(rx_pos)

    # Atmospheric losses: per member (rain / water vapour are UE-side), see _atm_draw.

    # ── Sample Faraday rotation angle for this scene ────────────────────────────
    if IONOSPHERE_ENABLED:
        # VTEC is in the solve key: Faraday rotation rolls the TX before the trace.
        batch_omega_F, batch_tec_slant = sample_faraday_angle(
            float(scene_cfg.get("VTEC_MEAN_TECU", VTEC_MEAN_TECU)),
            float(scene_cfg.get("VTEC_STD_TECU", VTEC_STD_TECU)),
            FREQUENCY, ELEVATION_DEG, B_L_TESLA,
            _scene_rng(entry, 1),
            sigma_h=VTEC_SIGMA_H,
        )
    else:
        batch_omega_F   = 0.0
        batch_tec_slant = 0.0

    # ── Solver seed: one per scene, shared by every chunk (same rays) ──────────
    _legacy_seed = int(rng.integers(1, 1_000_000))   # always drawn: keeps the legacy stream intact
    if SOLVER_SEED_OFFSET is None:
        seed = drjit.cuda.ad.UInt(_legacy_seed)
    else:
        _ss = np.random.SeedSequence([RNG_SEED, int(entry["seed"]),
                                      int(round(ELEVATION_DEG * 100)), int(SOLVER_SEED_OFFSET)])
        seed = drjit.cuda.ad.UInt(int(_ss.generate_state(1)[0] % 999_999) + 1)

    # ── Solve in chunks (TX polarisation pre-rotated by Ω_F) ───────────────────
    per_rx, roles, chunk_log = [None] * n_recv, None, []
    chunk = min(n_recv, chunk_ok or n_recv)
    i = 0
    while i < n_recv:
        j = min(i + chunk, n_recv)
        oom = False
        try:
            res, roles, cand = _solve_chunk(rx_pos[i:j], rx_orient[i:j], rx_vel[i:j], scene_cfg, _tx_pos,
                                            _tx_vel, _beam, batch_omega_F, seed, entry["seed"])
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or j - i == 1:
                raise
            oom = True
        if oom:
            _free_gpu()
            chunk = (j - i + 1) // 2
            chunk_ok = chunk
            print(f"    OOM solving {j - i} receivers; retrying in chunks of {chunk}")
            sys.stdout.flush()
            continue
        per_rx[i:j] = res
        if cand.get("n", 0) >= cand.get("cap", 1):
            n_cap_hits += 1
            print(f"    WARNING: candidate cap hit ({cand['n']:,}/{cand['cap']:,}) for "
                  f"receivers {i}-{j - 1}: paths dropped; results now depend on the chunk size")
        chunk_log.append(f"{j - i}:{cand.get('n', 0) / max(cand.get('cap', 1), 1):.0%}"
                         f"[{'/'.join(f'{x:.1f}' for x in cand['t'])}s]")
        del res
        _free_gpu()
        i = j

    if _scratch:
        shutil.rmtree(_scratch, ignore_errors=True)   # streamed scene: never cached

    # ── Per-member extraction ──────────────────────────────────────────────────
    sc = {"seed": int(entry["seed"]), "omega_F": batch_omega_F, "tec_slant": batch_tec_slant,
          "roles": roles, "itu": _itu, "heading": _heading, "wp": _wp,
          "gs": entry.get("ground_station", (np.nan, np.nan)), "tx_pos": _tx_pos}
    print()
    for m, (idx, w, _rx_h, batch_atm) in zip(MEMBERS, ues):
        sc["rx_h"], sc["atm"] = _rx_h, batch_atm
        for r, p, u in idx:
            sc["rx_nom"] = p
            rec, extra, raw = _extract_ue(per_rx[r], w, sc,
                                          (m["done"] + 1) % RAW_SAVE_INTERVAL == 0)
            extra.update(u)
            for k, v in rec.items():
                m["acc"][k].append(v)
            for k, v in extra.items():
                m["extra"][k].append(v)
            if raw is not None:
                m["raw"].append(raw)
            m["done"] += 1

        # ── Scene summary for this member ───────────────────────────────────────
        elapsed = time.perf_counter() - scene_start     # whole scene: geometry .. extraction
        total_e = time.perf_counter() - total_start
        nb = m["n_rx"]
        Plos_batch         = np.array(m["acc"]["P_los"][-nb:])
        Pnlos_batch        = np.array(m["acc"]["P_nlos"][-nb:])
        Pdiffuse_batch     = np.array(m["acc"]["P_diffuse"][-nb:])
        Pdiffraction_batch = np.array(m["acc"]["P_diffraction"][-nb:])
        Pspecular_batch    = np.array(m["acc"]["P_specular"][-nb:])
        mean_Plos  = float(np.mean(Plos_batch))
        mean_Pnlos = float(np.mean(Pnlos_batch))
        K_mean_dB  = 10.0 * np.log10(mean_Plos / (mean_Pnlos + 1e-30) + 1e-30)
        _P_total_inc = Plos_batch + Pspecular_batch + Pdiffuse_batch + Pdiffraction_batch
        _mean_tot    = float(np.mean(_P_total_inc)) if float(np.mean(_P_total_inc)) > 0 else 1.0
        _frac_los    = float(np.mean(Plos_batch))   / _mean_tot * 100
        _frac_spec   = float(np.mean(Pspecular_batch))  / _mean_tot * 100
        _frac_diff   = float(np.mean(Pdiffuse_batch))  / _mean_tot * 100
        _frac_diffr  = float(np.mean(Pdiffraction_batch)) / _mean_tot * 100
        _last = {k: m["acc"][k][-1] for k in ("ASD_deg", "ASA_deg", "ZSD_deg", "ZSA_deg")}
        iono_str = f"  Ω_F={np.degrees(batch_omega_F):.1f}°  TEC={batch_tec_slant:.1f} TECU" \
                   if IONOSPHERE_ENABLED else ""
        atm_str  = (f"  A_gas={batch_atm['A_gas_db']:.2f} dB"
                    f"  A_rain={batch_atm['A_rain_db']:.2f} dB (R={batch_atm['R_sample_mm_h']:.1f} mm/h)"
                    f"  scint={batch_atm['scint_db']:.2f} dB"
                    f"  tot={batch_atm['total_loss_db']:.2f} dB") \
                   if ATMOSPHERE_ENABLED else ""
        tag = f"{m['name']}  " if len(MEMBERS) > 1 else ""
        print(
            f"  [elev={ELEVATION_DEG:.0f}°]  {tag}"
            f"scene={scene_idx + 1}/{N_SCENES} (seed={entry['seed']})  "
            f"done={m['done']}/{m['total_rx']} rx  "
            f"K={K_mean_dB:.1f} dB  ASD={_last['ASD_deg']:.1f}° ASA={_last['ASA_deg']:.1f}° "
            f"ZSD={_last['ZSD_deg']:.1f}° ZSA={_last['ZSA_deg']:.1f}°  "
            f'P_los: {_frac_los  :6.2f} %'
            f'  P_specular: {_frac_spec :6.2f} %'
            f'  P_diffuse: {_frac_diff :6.2f} %'
            f'  P_diffraction: {_frac_diffr:6.2f} %'
            f"  scene_t={elapsed:.1f}s  total={total_e/3600:.2f}h"
            f"{iono_str}{atm_str}"
        )
    print(f"    {n_recv} receivers in chunks [size:candidate-cap fill[build/solve/read s]] "
          f"{' '.join(chunk_log)}")
    sys.stdout.flush()
    del per_rx

    # ── Checkpoint (all members at the same scene) ────────────────────────────
    if (scene_idx + 1) % CHECKPOINT_INTERVAL == 0 or scene_idx + 1 == N_SCENES:
        for m in MEMBERS:
            ck_path = _save_member_checkpoint(m)
            print(f"  ✓ checkpoint → {ck_path}")
        sys.stdout.flush()

# Save final results

if GROUND_RETRY_MAX > 0:
    print(f"  Ground-path dropout: {ground_retries} extra solve(s), "
          f"{ground_retry_exhausted}/{N_SCENES} scene(s) still missing it "
          f"after {GROUND_RETRY_MAX} retries")
if n_cap_hits:
    print(f"  WARNING: candidate cap hit in {n_cap_hits} chunk(s); raise BASE_MAX_PATHS "
          f"or lower RX_CHUNK")

if STREAMING:
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)

total_elapsed = time.perf_counter() - total_start
for m in MEMBERS:
    results_path = save_results(
        output_dir   = m["dir"],
        elevation_deg = ELEVATION_DEG,
        raw_results  = m["raw"],
        extra        = m["extra"],
        cfg          = m["cfg"],
        tx_pos       = TX_POS,
        slant_dist_m = SLANT_DIST_ACTUAL_M,
        **m["acc"],
    )
    print(f"\n{'='*64}")
    print(f"  Elevation {ELEVATION_DEG:.0f}° — COMPLETE  ({m['name']})")
    print(f"  Scenes    : {N_SCENES}   Receivers : {m['done']}/{m['total_rx']}")
    print(f"  Elapsed   : {total_elapsed/3600:.2f} h  ({total_elapsed:.1f} s)")
    print(f"  Results   : {results_path}")
    print(f"{'='*64}\n")
