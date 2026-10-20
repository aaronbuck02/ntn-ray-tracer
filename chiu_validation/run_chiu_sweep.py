#!/usr/bin/env python3
"""Scene-outer sweep for the Chiu & Roy validation: generate one wall scene, solve
every elevation and polarization on it, discard it, repeat.

Inverted relative to run_stochastic_sim.py because at the paper's 2 km radius a
2000-scene ensemble is ~18 GB that would be re-parsed sixteen times. Workers
take contiguous scene chunks; chiu_setup.merge_chunks() reassembles them in
scene order. See docs/pipelines.md.
"""

import os
import sys
from pathlib import Path
import argparse
import json
import time
import shutil
import pickle

parser = argparse.ArgumentParser(description="Scene-outer Chiu & Roy sweep")
parser.add_argument("--scene-start", type=int, required=True,
                    help="Index of the first scene in this chunk (0-based)")
parser.add_argument("--scene-count", type=int, required=True,
                    help="Number of scenes in this chunk")
parser.add_argument("--config", type=str, required=True,
                    help="Sweep config JSON (chiu_setup.make_sweep_config)")
parser.add_argument("--output", type=str, required=True,
                    help="Directory for chunk result files")
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--scratch", type=str, default=None,
                    help="Where the one live scene is written (default: "
                         "<output>/scratch_gpu<id>). Point at fast local disk.")
parser.add_argument("--resume", action="store_true", default=False,
                    help="Skip this chunk if its result file already exists")
args = parser.parse_args()

# Lock GPU visibility before any drjit / mitsuba / sionna import.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
os.environ.setdefault("DRJIT_LIBLLVM_PATH",
                      "/usr/lib/x86_64-linux-gnu/libLLVM-20.so")

import numpy as np
from sionna.rt import PathSolver

# Import root is the directory containing Ray_Tracing/, so the
# Ray_Tracing.src.* package paths resolve however this is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Ray_Tracing.src.utils.channel_utils import (rms_delay_spread, rms_angular_spread,
                           complex_channel, split_by_interaction_type,
                           tx_position_from_elevation)
from Ray_Tracing.src.utils.scene_utils import build_scene_batch, set_tx_pose, set_polarization, \
                        set_rx_positions
from Ray_Tracing.src.utils.orbital_utils import slant_range_m
import Ray_Tracing.src.utils.scene_gen_ppp as sg
# Applied by scene_utils at first build_scene_batch(); imported here as well so
# the state can be asserted before any GPU time is spent and recorded with the
# results. Patched and unpatched runs are not comparable.
import Ray_Tracing.src.sionna_patch_check as sionna_patch_check

C0 = 299792458.0

with open(args.config, "r") as f:
    cfg = json.load(f)

SCENARIO      = cfg["SCENARIO"]
RADIUS        = float(cfg["R_CITY"])
WALL_THICK    = float(cfg["WALL_THICK"])
BASE_SEED     = int(cfg.get("PPP_BASE_SEED", 0))
ELEVATIONS    = [float(e) for e in cfg["ELEVATION_ANGLES"]]
# How the paper's polarization-AVERAGED Fresnel kernel is realised in RT.
#
#   "cp_total"         RHCP transmit into a complete dual-port ("VH") receive,
#                      powers summed over the two ports. Circular polarization
#                      is invariant under rotation about the propagation axis,
#                      so a CP wave splits exactly 50/50 into the LOCAL TE and
#                      TM modes at every interaction, whatever its plane of
#                      incidence; summing both orthogonal receive components
#                      then gives 0.5(|G_TE|^2+|G_TM|^2) identically. Verified
#                      against the reference model's own per-path amplitude on
#                      wall reflections at arbitrary orientation: 0.00 dB error.
#                      One solve instead of two.
#
#   "linear_copol_avg" the legacy two-run average 0.5(P_VV + P_HH). Exact only
#                      where the plane of incidence aligns with the V/H basis --
#                      true for the ground (vertical plane of incidence), false
#                      for a wall at arbitrary orientation, where it discards the
#                      cross-polar V->H conversion and under-counts by up to
#                      8.2 dB (measured mean -1.50 dB, worst -6.24 dB over ten
#                      reference wall reflections). Kept for cross-checking.
POLARIZATION_MODE = cfg.get("POLARIZATION_MODE", "cp_total")
if POLARIZATION_MODE == "cp_total":
    POL_SETTINGS = [("cp", "RHCP", "VH")]          # (label, tx_pol, rx_pol)
elif POLARIZATION_MODE == "linear_copol_avg":
    POL_SETTINGS = [(p, p, p) for p in cfg["POLARIZATIONS"]]
else:
    raise ValueError(f"unknown POLARIZATION_MODE {POLARIZATION_MODE!r}")
POLARIZATIONS = [lbl for lbl, _, _ in POL_SETTINGS]
FREQUENCY     = float(cfg["FREQUENCY"])
RX_HEIGHT     = float(cfg["RX_HEIGHT"])
ALT_KM        = float(cfg["SWEEP_ALTITUDE_KM"])
AZIMUTH       = float(cfg["TX_AZIMUTH_DEG"])
SCENE_CENTER  = cfg.get("SCENE_CENTER", [0.0, 0.0])
MAX_DEPTH     = int(cfg.get("MAX_DEPTH", 1))
SAMPLES       = int(cfg["BASE_SAMPLES_PER_SRC"])
MAX_PATHS     = int(cfg["BASE_MAX_PATHS"])
SIM_LOS       = bool(cfg.get("LOS", True))
SIM_SPEC      = bool(cfg.get("SPECULAR_REFLECTION", True))
SIM_DIFFUSE   = bool(cfg.get("DIFFUSE_REFLECTION", False))
SIM_DIFFR     = bool(cfg.get("DIFFRACTION", True))
SIM_EDGE      = bool(cfg.get("EDGE_DIFFRACTION", False))
SIM_REFR      = bool(cfg.get("REFRACTION", False))
BEAMFORMING   = bool(cfg.get("BEAMFORMING", True))
POL_BASE      = cfg.get("POL_BASE_PATTERN", "iso")
RNG_SEED      = int(cfg.get("RNG_SEED", 42))
GROUND_RETRY_MAX      = int(cfg.get("GROUND_RETRY_MAX", 4))
GROUND_RETRY_JITTER_M = float(cfg.get("GROUND_RETRY_JITTER_M", 0.05))
GROUND_Z_TOL_M        = float(cfg.get("GROUND_Z_TOL_M", 0.05))
VERIFY_EVERY  = int(cfg.get("VERIFY_EVERY", 0))

WALL_MAT   = cfg["WALL_MAT"]
ROOF_MAT   = cfg["ROOF_MAT"]
GROUND_MAT = cfg["GROUND_MAT"]
MATERIAL_PARAMS = cfg["MATERIAL_PARAMS"]

CHUNK_PATH = os.path.join(args.output,
                          f"chunk{args.scene_start:06d}.pkl")
if args.resume and os.path.exists(CHUNK_PATH):
    print(f"[chiu-sweep] {CHUNK_PATH} exists -- nothing to do (--resume)")
    sys.exit(0)

os.makedirs(args.output, exist_ok=True)
SCRATCH = args.scratch or os.path.join(args.output, f"scratch_gpu{args.gpu_id}")

print(f"\n{'='*66}")
print(f"  run_chiu_sweep.py  ({SCENARIO})")
print(f"  Scenes    : {args.scene_start} .. {args.scene_start + args.scene_count - 1}"
      f"  (seeds {BASE_SEED + args.scene_start} ..)")
print(f"  Scene     : wall PPP, R_city={RADIUS:.0f} m, slab {WALL_THICK} m, "
      f"UE at the origin, z={RX_HEIGHT} m")
print(f"  Inner loop: {len(ELEVATIONS)} elevations x {len(POL_SETTINGS)} "
      f"polarization setting(s) = {len(ELEVATIONS)*len(POL_SETTINGS)} solves/scene")
print(f"  Pol mode  : {POLARIZATION_MODE}  "
      + ", ".join(f"{l}: TX={t}/RX={r}" for l, t, r in POL_SETTINGS))
print(f"  Satellite : {ALT_KM:.0f} km, azimuth {AZIMUTH}deg, f_c={FREQUENCY/1e9:.2f} GHz")
print(f"  Propagate : max_depth={MAX_DEPTH} LoS={SIM_LOS} Spec={SIM_SPEC} "
      f"Diffuse={SIM_DIFFUSE} Diffr={SIM_DIFFR}")
print(f"  GPU       : {os.environ.get('CUDA_VISIBLE_DEVICES')}   scratch: {SCRATCH}")
print(f"{'='*66}\n")

# require() raises if either patch is missing -- the occlusion dead zone would
# silently corrupt the LoS state.
_PATCH = sionna_patch_check.require("run_chiu_sweep")
print(f"  {sionna_patch_check.summary_line(_PATCH)}")

solver = PathSolver()
GT = np.array([0.0, 0.0, RX_HEIGHT], dtype=np.float64)

# Per-elevation TX geometry is scene-independent -- compute it once.
TX_BY_ELEV = {}
for _e in ELEVATIONS:
    _slant = float(slant_range_m(ALT_KM, _e))
    _pos, _beam = tx_position_from_elevation(_e, _slant, SCENE_CENTER,
                                             RADIUS, AZIMUTH)
    TX_BY_ELEV[_e] = (_pos, _beam, _slant)


def _solve(scene, beam, seed):
    return solver(scene=scene, max_depth=MAX_DEPTH, los=SIM_LOS,
                  specular_reflection=SIM_SPEC, diffuse_reflection=SIM_DIFFUSE,
                  refraction=SIM_REFR, diffraction=SIM_DIFFR,
                  edge_diffraction=SIM_EDGE, synthetic_array=False,
                  beamforming=BEAMFORMING, beam_angle=beam,
                  max_num_paths_per_src=MAX_PATHS, samples_per_src=SAMPLES,
                  seed=seed)


def _unpack(paths):
    """Per-path arrays for the single receiver, valid paths only."""
    a, tau = paths.cir(normalize_delays=False, out_type="numpy")
    a, tau = np.asarray(a), np.asarray(tau)
    n_rx, n_ant = a.shape[0], a.shape[1]
    amps = a.reshape(n_rx, n_ant, -1)[0]                    # [ant, path]
    taus = tau.reshape(n_rx, n_ant, -1)[0, 0]               # [path]

    inter = paths.interactions.numpy()                      # [depth, rx, ant, ...]
    inter = inter.reshape(inter.shape[0], n_rx, n_ant, -1)[:, 0, 0, :]
    try:
        vert = paths.vertices.numpy()
        vert = vert.reshape(vert.shape[0], n_rx, n_ant, -1, 3)[:, 0, 0, :, :]
    except AttributeError:
        vert = None
    ang = [np.asarray(getattr(paths, n).numpy()).reshape(n_rx, n_ant, -1)[0, 0]
           for n in ("theta_t", "phi_t", "theta_r", "phi_r")]

    # A path counts as present if it carries power in ANY receive port.
    valid = (np.abs(amps) ** 2).sum(axis=0) > 0.0
    return dict(
        amps=amps[:, valid], taus=taus[valid], inter=inter[:, valid],
        vert=(vert[:, valid, :] if vert is not None else None),
        theta_t=ang[0][valid], phi_t=ang[1][valid],
        theta_r=ang[2][valid], phi_r=ang[3][valid],
    )


def _has_ground(d):
    """
    False only when the direct path is present but the ground reflection is
    not -- the Sionna specular-refinement dropout. A ground path is a
    first-order interaction whose interaction point lies on the ground plane;
    wall reflections and roof-edge diffractions sit above it.
    """
    it = d["inter"][0]
    if not np.any(it == 0):
        return True                      # NLoS state: nothing to repair
    if d["vert"] is None:
        return True                      # cannot tell; do not retry blindly
    vz = d["vert"][0][:, 2]
    return bool(np.any((it != 0) & (np.abs(vz) < GROUND_Z_TOL_M)))


def _path_power(d):
    """Per-path power, summed over receive ports.

    Orthogonal field components do not interfere in power, so the two "VH"
    ports add incoherently. This sum is what makes RHCP -> dual-port receive
    equal the paper's 0.5(|G_TE|^2+|G_TM|^2) kernel.
    """
    return (np.abs(d["amps"]) ** 2).sum(axis=0)


def _excess_delays(d, tx_pos):
    """Excess delay per path [s], in float64 from the interaction vertices."""
    S = np.asarray(tx_pos, dtype=np.float64)
    R_los = np.linalg.norm(S - GT)
    n = len(d["taus"])
    out = np.zeros(n, dtype=np.float64)
    if d["vert"] is None:
        return (d["taus"].astype(np.float64) - d["taus"].min()) if n else out
    it = d["inter"][0]
    for i in range(n):
        if it[i] == 0:
            continue                     # direct path: zero excess by definition
        q = d["vert"][0][i].astype(np.float64)
        out[i] = (np.linalg.norm(S - q) + np.linalg.norm(q - GT) - R_los) / C0
    return out


def _stats(d, tx_pos):
    """Per-(scene, elevation, polarization) record."""
    if len(d["taus"]) == 0:
        return dict(K_power_ratio=0.0, tau_rms_mean_s=0.0, los_probability=0.0,
                    P_los=0.0, P_specular=0.0, P_diffuse=0.0,
                    P_diffraction=0.0, P_refraction=0.0, P_nlos=0.0,
                    ASD_deg=0.0, ASA_deg=0.0, ZSD_deg=0.0, ZSA_deg=0.0,
                    h_env=0.0, n_paths=0, has_ground=False, n_ports=0,
                    taus=np.zeros(0), amps=np.zeros((0, 0), complex),
                    P_path=np.zeros(0),
                    dtau_s=np.zeros(0), inter_d0=np.zeros(0, int),
                    vert_z=np.zeros(0), theta_r=np.zeros(0), phi_r=np.zeros(0),
                    theta_t=np.zeros(0), phi_t=np.zeros(0))
    t_v, inter = d["taus"], d["inter"]
    amps = d["amps"]                     # [n_ports, n_paths]

    # Coherent WITHIN a receive port, incoherent ACROSS ports: paths interfere,
    # orthogonal field components do not. So the coherent LoS/NLoS split runs
    # per port and the resulting powers are summed.
    keys = ("P_los", "P_specular", "P_diffuse", "P_diffraction",
            "P_refraction", "P_nlos")
    pwr     = {k: 0.0 for k in keys}
    pwr_inc = {k: 0.0 for k in keys}
    has_los = False
    for port in range(amps.shape[0]):
        a_p = amps[port]
        c = split_by_interaction_type(t_v, a_p, inter, FREQUENCY, incoherent=False)
        i = split_by_interaction_type(t_v, a_p, inter, FREQUENCY, incoherent=True)
        for k in keys:
            pwr[k] += float(c[k]); pwr_inc[k] += float(i[k])
        has_los = has_los or bool(c["has_los"])

    # Power-weighted spreads use total per-path power. rms_delay_spread and
    # rms_angular_spread consume only |a|^2, so passing sqrt(P) as a real
    # amplitude is exact rather than an approximation.
    P_path = _path_power(d)
    a_eff  = np.sqrt(P_path)
    dtau = _excess_delays(d, tx_pos)
    # Delay spread on the float64 excess delays, not Sionna's float32 tau.
    _, ds = rms_delay_spread(dtau, a_eff)
    _, ASD = rms_angular_spread(d["phi_t"], a_eff)
    _, ASA = rms_angular_spread(d["phi_r"], a_eff)
    _, ZSD = rms_angular_spread(d["theta_t"], a_eff)
    _, ZSA = rms_angular_spread(d["theta_r"], a_eff)
    P_los, P_nlos = pwr["P_los"], pwr["P_nlos"]
    return dict(
        K_power_ratio=float(P_los / (P_nlos + 1e-30)),
        tau_rms_mean_s=float(ds),
        los_probability=1.0 if has_los else 0.0,
        P_los=float(P_los), P_nlos=float(P_nlos),
        P_specular=float(pwr_inc["P_specular"]),
        P_diffuse=float(pwr_inc["P_diffuse"]),
        P_diffraction=float(pwr_inc["P_diffraction"]),
        P_refraction=float(pwr_inc["P_refraction"]),
        ASD_deg=float(np.degrees(ASD)), ASA_deg=float(np.degrees(ASA)),
        ZSD_deg=float(np.degrees(ZSD)), ZSA_deg=float(np.degrees(ZSA)),
        h_env=float(np.sqrt(sum(
            np.abs(complex_channel(t_v, amps[k], FREQUENCY)) ** 2
            for k in range(amps.shape[0])))),
        n_paths=int(len(t_v)), has_ground=bool(_has_ground(d)),
        n_ports=int(amps.shape[0]),
        taus=t_v.copy(), amps=amps.copy(), P_path=P_path, dtau_s=dtau,
        inter_d0=inter[0].copy(),
        vert_z=(d["vert"][0][:, 2].copy() if d["vert"] is not None
                else np.zeros(len(t_v))),
        theta_r=d["theta_r"].copy(), phi_r=d["phi_r"].copy(),
        theta_t=d["theta_t"].copy(), phi_t=d["phi_t"].copy(),
    )


records = {(e, p): [] for e in ELEVATIONS for p in POLARIZATIONS}
scene_seeds = []
ground_retries = 0
ground_exhausted = 0
t_start = time.perf_counter()

for k in range(args.scene_start, args.scene_start + args.scene_count):
    seed = BASE_SEED + k
    scene_dir = os.path.join(SCRATCH, f"scene_{seed:07d}")
    shutil.rmtree(scene_dir, ignore_errors=True)
    t0 = time.perf_counter()

    xml_path, spec = sg.generate_scene_streaming(
        scene_dir, seed, scene_name="wall_scene", primitive="wall",
        building_material=WALL_MAT, roof_material=ROOF_MAT,
        ground_material=GROUND_MAT,
        wall_params=MATERIAL_PARAMS[WALL_MAT],
        roof_params=MATERIAL_PARAMS[ROOF_MAT],
        ground_params=MATERIAL_PARAMS[GROUND_MAT],
        verify=(VERIFY_EVERY > 0 and (k % VERIFY_EVERY == 0)),
        radius=RADIUS, scenario=SCENARIO, thickness=WALL_THICK, h_min=0.0,
    )
    t_gen = time.perf_counter() - t0

    # Build once at the first elevation, then re-aim in place for the rest.
    scene_cfg = dict(cfg)
    scene_cfg["SCENE_XML"] = xml_path
    scene_cfg["MERGE_SHAPES"] = True
    # POL_SETTINGS entries are (label, tx_pol, rx_pol); the label is a result
    # key, not an antenna type, so the initial build takes the real pair.
    scene_cfg["TX_POL_TYPE"] = POL_SETTINGS[0][1]
    scene_cfg["RX_POL_TYPE"] = POL_SETTINGS[0][2]
    t0 = time.perf_counter()
    tx0, beam0, _ = TX_BY_ELEV[ELEVATIONS[0]]
    scene, _ = build_scene_batch([[0.0, 0.0, RX_HEIGHT]], tx0, scene_cfg)
    t_load = time.perf_counter() - t0

    rng_jit = np.random.default_rng(RNG_SEED + 104729 * seed)
    t0 = time.perf_counter()
    for elev in ELEVATIONS:
        tx_pos, beam, _ = TX_BY_ELEV[elev]
        set_tx_pose(scene, tx_pos, SCENE_CENTER)
        for pol, tx_pol, rx_pol in POL_SETTINGS:
            set_polarization(scene, tx_pol, rx_pol, POL_BASE)
            paths = _solve(scene, beam, 1)
            d = _unpack(paths)

            # Ground-path dropout retry. Sionna's specular refinement
            # intermittently drops the ground reflection: exact where returned,
            # unrecoverable by extra sampling, but it flips with a sub-millimetre
            # RX move. That path carries most of the non-direct power here, so
            # losing it silently would bias K. Moving the receiver is enough --
            # no need to rebuild or reload the scene.
            for _ in range(GROUND_RETRY_MAX):
                if _has_ground(d):
                    break
                off = rng_jit.uniform(-GROUND_RETRY_JITTER_M,
                                      GROUND_RETRY_JITTER_M, 2)
                set_rx_positions(scene, [[float(off[0]), float(off[1]),
                                          RX_HEIGHT]])
                paths = _solve(scene, beam, 1)
                d = _unpack(paths)
                ground_retries += 1
            else:
                if not _has_ground(d):
                    ground_exhausted += 1
            set_rx_positions(scene, [[0.0, 0.0, RX_HEIGHT]])

            records[(elev, pol)].append(_stats(d, tx_pos))
    t_solve = time.perf_counter() - t0

    scene_seeds.append(seed)
    shutil.rmtree(scene_dir, ignore_errors=True)

    done = k - args.scene_start + 1
    if done == 1 or done % 10 == 0 or done == args.scene_count:
        el = time.perf_counter() - t_start
        print(f"  scene {done}/{args.scene_count} (seed={seed})  "
              f"walls={spec.n_buildings}  gen={t_gen:.1f}s load={t_load:.2f}s "
              f"solve={t_solve:.1f}s  |  {el/done:.1f} s/scene  "
              f"eta={(args.scene_count-done)*el/done/3600:.2f} h")
        sys.stdout.flush()

shutil.rmtree(SCRATCH, ignore_errors=True)

payload = {
    "scene_start": args.scene_start,
    "scene_count": args.scene_count,
    "scene_seeds": scene_seeds,
    "scenario": SCENARIO,
    "elevations": ELEVATIONS,
    "polarizations": POLARIZATIONS,
    "polarization_mode": POLARIZATION_MODE,
    "sionna_patch": _PATCH,
    "shadow_patch": _PATCH,   # deprecated alias; kept so older results compare

    "radius": RADIUS,
    "config": cfg,
    "ground_retries": ground_retries,
    "ground_exhausted": ground_exhausted,
    "records": {f"{e:g}|{p}": v for (e, p), v in records.items()},
}
tmp = CHUNK_PATH + ".tmp"
with open(tmp, "wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
os.replace(tmp, CHUNK_PATH)

el = time.perf_counter() - t_start
print(f"\n  Ground-path dropout: {ground_retries} extra solve(s), "
      f"{ground_exhausted} (elev,pol,scene) case(s) unrecovered")
print(f"  {args.scene_count} scenes in {el/3600:.2f} h "
      f"({el/max(args.scene_count,1):.1f} s/scene)  ->  {CHUNK_PATH}")
