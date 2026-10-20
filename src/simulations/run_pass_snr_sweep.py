#!/usr/bin/env python3
"""Fixed-UE Canary Wharf whole-pass SNR campaign using the site runner.

Prepare once, then run on a CUDA host. See docs/pass-snr.md.
"""

import argparse
from collections import deque
import csv
import json
import hashlib
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
from Ray_Tracing.src.utils.pass_snr import (
    make_real_target_passes, make_target_pass, band_gain, snr_from_band_gain,
    elevation_bin_statistics, tr38811_losses_db, dish_gain_db,
    TR38821_CASES, TR38821_ALTITUDE_KM, TR38821_MIN_ELEVATION_DEG)
from Ray_Tracing.src.utils.pickle_utils import load_pickle
from Ray_Tracing.src.utils.channel_utils import IT_DIFFUSE


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    from Ray_Tracing.src.utils.scene_utils import build_occupancy_grid, save_rx_positions

    out = args.output
    if (out / "campaign.json").exists():
        raise FileExistsError("Campaign already prepared; use run/analyze or a new output directory")
    cfg = json.loads(args.config.read_text())
    scene = (args.scene or REPO / "RT Scenarios/wharf_scene_v3/du/canarywharf3_du.xml").resolve()
    free, _ = build_occupancy_grid(str(scene), rx_height=cfg["RX_HEIGHT"],
                                   grid_res=2.0, variant="llvm_ad_rgb")
    if not len(free):
        raise ValueError("No outdoor UE cells found")
    preferred = np.asarray(args.ue_xy or cfg["SCENE_CENTER"])
    xy = free[np.argmin(np.sum((free - preferred) ** 2, axis=1))]
    if args.ue_xy and np.linalg.norm(xy - preferred) > 1.5:
        raise ValueError("Requested UE is not an eligible outdoor grid cell")
    ue = [float(xy[0]), float(xy[1]), float(cfg["RX_HEIGHT"])]
    # Convert the selected scene offset to the corresponding geographic UE.
    heading = np.radians(cfg.get("SCENE_EAST_HEADING_DEG", 90))
    x, y = xy - np.asarray(cfg["SCENE_CENTER"])
    east, north = x * np.sin(heading) - y * np.cos(heading), x * np.cos(heading) + y * np.sin(heading)
    lat = cfg["SIM_LAT"] + np.degrees(north / 6378137.0)
    lon = cfg["SIM_LON"] + np.degrees(east / (6378137.0 * np.cos(np.radians(cfg["SIM_LAT"]))))
    cfg.update(SCENE_XML=str(scene), SCENE_CENTER=ue[:2], N_RX_POSITIONS=1,
               BATCH_SIZE=1, PERTURB_HALF=0.0, RAW_SAVE_INTERVAL=1,
               N_CLUSTER_SAMPLES=0, DOPPLER_ENABLED=False,
               RESUME_FROM_CHECKPOINT=False, SIM_LAT=float(lat), SIM_LON=float(lon))
    # Include the whole scene within the ray cone after changing its centre.
    cfg["SCENE_RADIUS"] += float(np.hypot(x, y))
    catalog_text = args.tle_catalog.read_text()
    passes = make_real_target_passes(catalog_text, args.max_elevations, lat, lon,
                                    args.epoch, args.search_hours, args.min_elevation,
                                    args.step_s, args.peak_tolerance_deg, args.max_tle_age_days)
    out.mkdir(parents=True, exist_ok=True)
    save_rx_positions(np.asarray([ue]), str(out / "ue.pkl"), {"fixed_ue": True})
    (out / "simulation_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    manifest = dict(scene_xml=str(scene), ue_scene_xyz_m=ue, ue_latitude_deg=float(lat),
                    ue_longitude_deg=float(lon), bandwidth_hz=args.bandwidth_hz,
                    noise_figure_db=args.noise_figure_db, temperature_k=290.0,
                    snr_definition="band-mean |H(f)|^2 over bandwidth, H from CIR amps and delays",
                    tle_catalog_path=str(args.tle_catalog.resolve()),
                    tle_catalog_sha256=hashlib.sha256(catalog_text.encode()).hexdigest(),
                    search_hours=args.search_hours, peak_tolerance_deg=args.peak_tolerance_deg,
                    max_tle_age_days=args.max_tle_age_days,
                    passes=[])
    (out / "catalog_snapshot.tle").write_text(catalog_text)
    geometry_rows = []
    for index, data in enumerate(passes):
        name = f"pass_{index:02d}_max{args.max_elevations[index]:g}deg"
        folder = out / name
        folder.mkdir()
        with (folder / "waypoints.pkl").open("wb") as f:
            pickle.dump(data, f)
        manifest["passes"].append(dict(directory=name, **data["metadata"]))
        for i, wp in enumerate(data["waypoints"]):
            geometry_rows.append(dict(pass_id=name, sample_index=i,
                                      satellite_name=data["metadata"]["satellite_name"],
                                      norad_id=data["metadata"]["norad_id"],
                                      inclination_deg=data["metadata"]["inclination_deg"],
                                      **{k: wp[k] for k in ("t_s", "epoch_offset_s", "elev_deg", "az_deg", "range_m", "leg")}))
    (out / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_csv(out / "geometry.csv", geometry_rows)
    print(f"Prepared {len(passes)} passes, fixed UE={ue}; metadata: {out / 'campaign.json'}")


def sample_band_gains(raw, result, manifest, frequency_hz):
    """(band gain, carrier gain) of one fixed-UE result, all RX ports combined by MRC.

    A "dish" campaign weights each path by the TR 38.811 aperture pattern
    pointed at the satellite (ideal tracking). Diffuse paths add as power;
    results without the any-bounce flag fall back to the first interaction.
    """
    first_diffuse = (np.asarray(raw["inter_d0"]) & IT_DIFFUSE) > 0 if "inter_d0" in raw else None
    ports = [(raw["amps"], raw["taus"], raw.get("diffuse", first_diffuse))]
    ports += [(p["amps"], p["taus"], p["diffuse"]) for p in raw.get("rx_ports", [])]
    if manifest.get("rx_model") == "dish" and len(raw["amps"]):
        if len(ports) != 1:
            raise ValueError("Dish receiver expects one RX port")
        theta, phi = np.asarray(raw["theta_r"], float), np.asarray(raw["phi_r"], float)
        arrival = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
        to_sat = np.asarray(result["tx_pos"], float) - np.asarray(result["nominal_rx_positions"][0], float)
        off_axis = np.arccos(np.clip(arrival @ (to_sat / np.linalg.norm(to_sat)), -1, 1))
        if raw["has_los"] and off_axis.min() > np.radians(0.1):
            raise ValueError("LoS path does not arrive from the satellite; check the angle convention")
        # pattern applied after RT to the co-pol isotropic port; the CP basis
        # rotation of a pointed dish only changes phases of paths it already suppresses.
        gain = 10 ** (dish_gain_db(off_axis, frequency_hz, manifest["dish_diameter_m"],
                                   manifest["dish_gain_dbi"]) / 20)
        ports = [(raw["amps"] * gain, raw["taus"], ports[0][2])]
    scs = manifest.get("subcarrier_spacing_hz", 15e3)
    band = sum(band_gain(a, t, manifest["bandwidth_hz"], scs, incoherent=d) for a, t, d in ports)
    carrier = sum(band_gain(a, t, scs, scs, incoherent=d) for a, t, d in ports)
    return band, carrier


def collect(out, threshold_db, tx_power_dbm=None, extra_loss_db=0.0):
    """tx_power_dbm overrides the simulated TX power in the SNR arithmetic only.

    TR 38.821 campaigns also subtract the TR 38.811 atmospheric and scintillation
    losses; extra_loss_db adds e.g. the TR shadowing margin, which RT replaces.
    """
    manifest = json.loads((out / "campaign.json").read_text())
    cfg = json.loads((out / "simulation_config.json").read_text())
    tx = cfg["TX_POWER_DBM"] if tx_power_dbm is None else tx_power_dbm
    tr_case = manifest.get("tr38821_case")
    samples, bins = [], []
    for meta in manifest["passes"]:
        folder = out / meta["directory"]
        with (folder / "waypoints.pkl").open("rb") as f:
            data = load_pickle(f)
        pass_samples = []
        for i, wp in enumerate(data["waypoints"]):
            result_dir = folder / f"sample_{i:05d}"
            files = list(result_dir.glob("results_elev*deg.pkl"))
            if len(files) != 1:
                raise RuntimeError(f"Missing/ambiguous sample {result_dir}; complete run before analysis")
            with files[0].open("rb") as f:
                result = load_pickle(f)
            if any(result["cfg_snapshot"].get(k) != v for k, v in cfg.items()):
                raise ValueError(f"Result configuration differs from this campaign: {files[0]}")
            raw = result["raw_results"]
            if len(raw) != 1 or not np.allclose(result["nominal_rx_positions"][0], manifest["ue_scene_xyz_m"]):
                raise ValueError(f"Invalid fixed-UE result: {files[0]}")
            loss = extra_loss_db + (float(tr38811_losses_db(cfg["FREQUENCY"], wp["elev_deg"]))
                                    if tr_case else 0.0)
            link = (tx - loss, manifest["bandwidth_hz"], manifest["noise_figure_db"],
                    manifest["temperature_k"])
            band, carrier = sample_band_gains(raw[0], result, manifest, cfg.get("FREQUENCY"))
            snr, snr_eff = snr_from_band_gain(band, *link)
            snr_nb = snr_from_band_gain(carrier, *link)[0]
            pass_samples.append(dict(pass_id=meta["directory"], sample_index=i,
                                     target_max_elevation_deg=meta["target_max_elevation_deg"],
                                     actual_max_elevation_deg=meta["actual_max_elevation_deg"],
                                     inclination_deg=meta["inclination_deg"],
                                     satellite_name=meta.get("satellite_name", ""),
                                     norad_id=meta.get("norad_id", ""),
                                     ue_x_m=manifest["ue_scene_xyz_m"][0],
                                     ue_y_m=manifest["ue_scene_xyz_m"][1],
                                     ue_z_m=manifest["ue_scene_xyz_m"][2],
                                     time_s=wp["t_s"], elevation_deg=wp["elev_deg"],
                                     azimuth_deg=wp["az_deg"], range_m=wp["range_m"],
                                     leg=wp["leg"], tx_power_dbm=tx, loss_db=loss,
                                     has_los=bool(raw[0].get("has_los", False)), snr_db=snr,
                                     snr_effective_db=snr_eff, snr_narrowband_db=snr_nb,
                                     threshold_db=threshold_db))
        samples.extend(pass_samples)
        stats = elevation_bin_statistics([s["elevation_deg"] for s in pass_samples],
                                          [s["snr_db"] for s in pass_samples], threshold_db)
        bins.extend(dict(pass_id=meta["directory"], satellite_name=meta.get("satellite_name", ""),
                         norad_id=meta.get("norad_id", ""), inclination_deg=meta["inclination_deg"],
                         actual_max_elevation_deg=meta["actual_max_elevation_deg"],
                         threshold_db=threshold_db, **b) for b in stats)
    write_csv(out / "snr_samples.csv", samples)
    write_csv(out / "elevation_bins.csv", bins)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for meta in manifest["passes"]:
        name = meta["directory"]
        rows = [s for s in samples if s["pass_id"] == name]
        stats = [b for b in bins if b["pass_id"] == name]
        label = f"Peak {meta['actual_max_elevation_deg']:.1f}°"
        axes[0].plot([s["time_s"] for s in rows], [s["snr_db"] for s in rows], label=label)
        axes[1].plot([b["elevation_lo_deg"] + 5 for b in stats],
                     [b["percent_above_threshold"] for b in stats], "o-", label=label)
    axes[0].axhline(threshold_db, color="black", linestyle="--")
    axes[0].set(xlabel="Time since rise (s)", ylabel="RX SNR (dB)")
    axes[1].set(xlabel="Elevation bin centre (deg)", ylabel=f"SNR > {threshold_db:g} dB (%)", ylim=(-2, 102))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pass_snr.png", dpi=180)
    plt.close(fig)
    print(f"Saved SNR samples, elevation bins and plot in {out}")


def prepare_additional(args):
    """Prepare additional real passes using exactly an existing campaign's UE/config."""
    base = args.base_campaign.resolve()
    out = args.output
    if out.exists():
        raise FileExistsError("Additional campaign output exists; choose a new directory")
    manifest = json.loads((base / "campaign.json").read_text())
    if any(p.get("orbit_source") != "real catalog TLE propagated with SGP4" for p in manifest["passes"]):
        raise ValueError("Base campaign must contain real TLE passes")
    excluded = [p["norad_id"] for p in manifest["passes"]]
    catalog = (base / "catalog_snapshot.tle").read_text()
    if hashlib.sha256(catalog.encode()).hexdigest() != manifest["tle_catalog_sha256"]:
        raise ValueError("Base catalog snapshot hash differs from manifest")
    passes = make_real_target_passes(
        catalog, args.max_elevations, manifest["ue_latitude_deg"], manifest["ue_longitude_deg"],
        args.epoch or manifest["passes"][0]["epoch_utc"], args.search_hours,
        args.min_elevation, args.step_s, args.peak_tolerance_deg, args.max_tle_age_days,
        target_azimuths=args.peak_azimuths, azimuth_tolerance_deg=args.azimuth_tolerance_deg,
        exclude_norad_ids=excluded)
    out.mkdir(parents=True)
    for filename in ("simulation_config.json", "ue.pkl", "catalog_snapshot.tle"):
        shutil.copyfile(base / filename, out / filename)
    manifest.update(base_campaign=str(base), excluded_norad_ids=excluded, passes=[],
                    search_hours=args.search_hours, peak_tolerance_deg=args.peak_tolerance_deg,
                    azimuth_tolerance_deg=args.azimuth_tolerance_deg,
                    max_tle_age_days=args.max_tle_age_days)
    geometry_rows, summary = [], []
    for index, data in enumerate(passes):
        meta = data["metadata"]
        name = f"pass_{index:02d}_norad{meta['norad_id']}"
        folder = out / name
        folder.mkdir()
        with (folder / "waypoints.pkl").open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        manifest["passes"].append(dict(directory=name, **meta))
        summary.append(dict(pass_id=name, satellite_name=meta["satellite_name"],
                            norad_id=meta["norad_id"], peak_elevation_deg=meta["actual_max_elevation_deg"],
                            rise_azimuth_deg=meta["rise_azimuth_deg"], peak_azimuth_deg=meta["peak_azimuth_deg"],
                            set_azimuth_deg=meta["set_azimuth_deg"], rt_samples=len(data["waypoints"]),
                            sample_interval_s=meta["sample_interval_s"], peak_utc=meta["peak_utc"]))
        for i, wp in enumerate(data["waypoints"]):
            geometry_rows.append(dict(pass_id=name, sample_index=i, satellite_name=meta["satellite_name"],
                                      norad_id=meta["norad_id"], inclination_deg=meta["inclination_deg"],
                                      **{k: wp[k] for k in ("t_s", "epoch_offset_s", "elev_deg", "az_deg", "range_m", "leg")}))
    (out / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_csv(out / "geometry.csv", geometry_rows)
    write_csv(out / "pass_summary.csv", summary)
    print(f"Prepared {len(passes)} additional satellites, {len(geometry_rows)} RT samples in {out}")


def prepare_tr(args):
    """TR 38.821 SC6/SC9 on synthetic LEO-600 orbits, at an existing campaign's UE.

    The UE, scene and RT settings are copied from the base campaign; frequency,
    EIRP and receive antenna follow the TR case. Each inclination contributes
    northbound and southbound passes on both sides of the UE; passes that
    coincide (e.g. near a 53 deg orbit's apex) are kept once.
    """
    case = TR38821_CASES[args.case]
    base, out = args.base_campaign.resolve(), args.output
    if out.exists():
        raise FileExistsError("Output exists; choose a new directory")
    base_manifest = json.loads((base / "campaign.json").read_text())
    cfg = json.loads((base / "simulation_config.json").read_text())
    eirp_dbm = case["eirp_dbw_per_mhz"] + 30 + 10 * np.log10(case["bandwidth_hz"] / 1e6)
    cfg.update(FREQUENCY=case["frequency_hz"], TX_POWER_DBM=float(eirp_dbm),
               RX_POL_TYPE=case["rx_pol_type"], IONOSPHERE_ENABLED=False,
               ATMOSPHERE_ENABLED=False)
    lat, lon = base_manifest["ue_latitude_deg"], base_manifest["ue_longitude_deg"]
    passes, seen = [], set()
    for inclination in args.inclinations:
        for target in args.max_elevations:
            for northbound in (True, False):
                for side in ((1,) if target >= 90 else (1, -1)):
                    data = make_target_pass(target, lat, lon, TR38821_ALTITUDE_KM, inclination,
                                            TR38821_MIN_ELEVATION_DEG, args.step_s,
                                            northbound=northbound, side=side)
                    meta = data["metadata"]
                    key = tuple(round(meta[k], 1) for k in
                                ("inclination_deg", "rise_azimuth_deg", "set_azimuth_deg"))
                    if key not in seen:
                        seen.add(key)
                        passes.append(data)
    out.mkdir(parents=True)
    shutil.copyfile(base / "ue.pkl", out / "ue.pkl")
    (out / "simulation_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    manifest = dict({k: base_manifest[k] for k in ("scene_xml", "ue_scene_xyz_m",
                                                   "ue_latitude_deg", "ue_longitude_deg")},
                    base_campaign=str(base), tr38821_case=args.case, **case,
                    temperature_k=case["antenna_temperature_k"], eirp_dbm=float(eirp_dbm),
                    altitude_km=TR38821_ALTITUDE_KM, mask_elevation_deg=TR38821_MIN_ELEVATION_DEG,
                    snr_definition="band-mean MRC |H(f)|^2 over bandwidth minus TR 38.811 "
                                   "atmospheric and scintillation losses",
                    passes=[])
    geometry_rows = []
    for index, data in enumerate(passes):
        meta = data["metadata"]
        name = (f"pass_{index:02d}_i{meta['inclination_deg']:.0f}_max{meta['target_max_elevation_deg']:g}"
                f"_{'N' if meta['northbound'] else 'S'}_az{meta['peak_azimuth_deg']:.0f}")
        folder = out / name
        folder.mkdir()
        with (folder / "waypoints.pkl").open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        manifest["passes"].append(dict(directory=name, **meta))
        for i, wp in enumerate(data["waypoints"]):
            geometry_rows.append(dict(pass_id=name, sample_index=i, inclination_deg=meta["inclination_deg"],
                                      **{k: wp[k] for k in ("t_s", "epoch_offset_s", "elev_deg", "az_deg", "range_m", "leg")}))
    (out / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_csv(out / "geometry.csv", geometry_rows)
    print(f"Prepared TR 38.821 {args.case.upper()}: {len(passes)} synthetic orbits, "
          f"{len(geometry_rows)} RT samples in {out}")


def run_campaign(output, gpu_ids=(0, 1), python=None, threshold_db=0, poll_interval=0.5):
    """Dispatch prepared samples with at most one site runner per physical GPU.

    Used by the run notebook and CLI. Completed samples are retained, failed
    samples restart fresh, and interruption stops all active child processes.
    Geometry and simulation configuration are read unchanged from the campaign.
    """
    output = Path(output).resolve()
    python = str(python or sys.executable)
    gpu_ids = tuple(gpu_ids)
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(g < 0 for g in gpu_ids):
        raise ValueError("GPU IDs must be a nonempty list of distinct nonnegative integers")
    if poll_interval <= 0:
        raise ValueError("Polling interval must be positive")
    manifest = json.loads((output / "campaign.json").read_text())
    if manifest.get("tr38821_case"):
        if any(m.get("orbit_source") != "synthetic SGP4 reference orbit" for m in manifest["passes"]):
            raise ValueError("TR 38.821 campaign must contain only its synthetic LEO-600 orbits")
    elif any(m.get("orbit_source") != "real catalog TLE propagated with SGP4"
             or not m.get("tle_line1") or not m.get("tle_line2") for m in manifest["passes"]):
        raise ValueError("This runner requires a real-TLE campaign; prepare into a new output directory")
    for filename in ("simulation_config.json", "ue.pkl"):
        if not (output / filename).is_file():
            raise FileNotFoundError(output / filename)
    pending = deque()
    total = skipped = 0
    for meta in manifest["passes"]:
        folder = output / meta["directory"]
        with (folder / "waypoints.pkl").open("rb") as f:
            count = len(load_pickle(f)["waypoints"])
        for i in range(count):
            dest = folder / f"sample_{i:05d}"
            total += 1
            files = list(dest.glob("results_elev*deg.pkl"))
            if len(files) > 1:
                raise RuntimeError(f"Ambiguous sample results: {dest}")
            if files:
                skipped += 1
                continue
            pending.append((folder, i, count, dest))
    print(f"{total} samples: {skipped} complete, {len(pending)} pending; GPUs {gpu_ids}", flush=True)
    active = {}
    finished = skipped
    started = time.monotonic()
    try:
        while pending or active:
            for gpu in gpu_ids:
                if gpu in active:
                    process, log, dest = active[gpu]
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    del active[gpu]
                    if code:
                        raise RuntimeError(f"Site runner exited {code} on GPU {gpu}; inspect {dest / 'run.log'}")
                    if len(list(dest.glob("results_elev*deg.pkl"))) != 1:
                        raise RuntimeError(f"Site runner produced no unique result; inspect {dest / 'run.log'}")
                    finished += 1
                    print(f"Completed {finished}/{total} ({(time.monotonic() - started) / 60:.1f} min)", flush=True)
                if not pending:
                    continue
                folder, i, count, dest = pending.popleft()
                dest.mkdir(exist_ok=True)
                cmd = [python, str(REPO / "src/simulations/run_elevation_sim.py"),
                       "--waypoints_file", str(folder / "waypoints.pkl"), "--waypoint_index", str(i),
                       "--gpu_id", "0",  # Each subprocess sees only its assigned physical GPU.
                       "--config", str(output / "simulation_config.json"),
                       "--rx_file", str(output / "ue.pkl"), "--output", str(dest), "--no-resume"]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                log = (dest / "run.log").open("w")
                try:
                    process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                active[gpu] = (process, log, dest)
                print(f"GPU {gpu}, PID {process.pid}: {folder.name} sample {i + 1}/{count}", flush=True)
            if active:
                time.sleep(poll_interval)
    finally:
        # Stop both workers on a failure or notebook interrupt, retaining completed results.
        for process, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, log, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            finally:
                log.close()
    collect(output, threshold_db)


def run(args):
    gpu_ids = getattr(args, "gpu_ids", None)
    run_campaign(args.output, gpu_ids if gpu_ids is not None else [getattr(args, "gpu_id", 0)],
                 getattr(args, "python", None), getattr(args, "threshold_db", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "prepare-additional", "prepare-tr", "run", "analyze"))
    parser.add_argument("--case", choices=sorted(TR38821_CASES), help="prepare-tr: TR 38.821 link-budget case")
    parser.add_argument("--inclinations", type=float, nargs="+", default=[53, 97.8],
                        help="prepare-tr: synthetic orbit inclinations (deg)")
    parser.add_argument("--extra-loss-db", type=float, default=0,
                        help="analyze: further loss, e.g. the TR shadowing margin")
    parser.add_argument("--output", type=lambda p: Path(p).resolve(), default=REPO / "Results/wharf_real_pass_snr")
    parser.add_argument("--config", type=Path, default=REPO / "src/simulation_configs/sim_config_wharf_du_2GHz.json")
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--ue-xy", type=float, nargs=2)
    parser.add_argument("--max-elevations", type=float, nargs="+", default=[30, 60, 90])
    parser.add_argument("--base-campaign", type=Path, default=REPO / "Results/wharf_real_pass_snr")
    parser.add_argument("--peak-azimuths", type=float, nargs="+", help="Compass azimuth at each peak elevation")
    parser.add_argument("--azimuth-tolerance-deg", type=float, default=30)
    parser.add_argument("--tle-catalog", type=Path, default=REPO / "tle_catalog/starlink.tle")
    parser.add_argument("--epoch", help="UTC search start; defaults to median TLE epoch (historical catalog replay)")
    parser.add_argument("--search-hours", type=float, default=24)
    parser.add_argument("--peak-tolerance-deg", type=float, default=5)
    parser.add_argument("--max-tle-age-days", type=float, default=3)
    parser.add_argument("--min-elevation", type=float, default=0)
    parser.add_argument("--step-s", type=float, default=5)
    parser.add_argument("--bandwidth-hz", type=float, default=10e6)
    parser.add_argument("--noise-figure-db", type=float, default=7)
    parser.add_argument("--threshold-db", type=float, default=0)
    parser.add_argument("--tx-power-dbm", type=float, help="analyze: override simulated TX power/EIRP in the SNR")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-ids", type=int, nargs="+", help="Physical GPU IDs; one simultaneous sample per GPU")
    parser.add_argument("--python", default=sys.executable, help="Python executable with patched Sionna and CUDA")
    args = parser.parse_args()
    if args.bandwidth_hz <= 0 or not np.isfinite(args.bandwidth_hz):
        parser.error("Bandwidth must be finite and positive")
    if args.action == "prepare":
        prepare(args)
    elif args.action == "prepare-additional":
        prepare_additional(args)
    elif args.action == "prepare-tr":
        if not args.case:
            parser.error("prepare-tr requires --case")
        prepare_tr(args)
    elif args.action == "run":
        run(args)
    else:
        collect(args.output, args.threshold_db, args.tx_power_dbm, args.extra_loss_db)


if __name__ == "__main__":
    main()
