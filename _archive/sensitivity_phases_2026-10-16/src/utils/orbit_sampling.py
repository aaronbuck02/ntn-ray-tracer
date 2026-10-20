"""Per-scene real-TLE satellite passes for the sensitivity sweeps.

Each scene gets an area-uniform ground point; each (scene, elevation E) a peak
elevation ~ U[max(30, E + tol), 90] and a leg in {rising, setting}. The real
catalog pass nearest each peak (pass_snr.make_real_target_passes) is
SGP4-propagated to the instant el = E. Output is a small orbit table, built in
worker processes, never in the notebook kernel. See docs/sensitivity.md.
"""

import hashlib
import os
import pickle
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc

PEAK_MIN_DEG = 30.0
PEAK_MAX_DEG = 90.0
_J2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


def sample_ground_points(n, lat_max_deg=60.0, seed=0):
    """(lat, lon) [deg], uniform in area over |lat| <= lat_max (LHS in sin(lat), lon)."""
    U = qmc.LatinHypercube(d=2, seed=seed).random(n)
    s = np.sin(np.radians(lat_max_deg))
    return np.degrees(np.arcsin(-s + 2 * s * U[:, 0])), -180.0 + 360.0 * U[:, 1]


def sample_peaks_and_legs(n, elevations, tol_deg=1.0, seed=0):
    """peak[n, nE] ~ U[max(30, E + tol), 90] and leg[n, nE] in {rising, setting}, LHS per column."""
    nE = len(elevations)
    U = qmc.LatinHypercube(d=2 * nE, seed=seed).random(n)
    lo = np.array([max(PEAK_MIN_DEG, e + tol_deg) for e in elevations])
    peaks = lo + U[:, :nE] * (PEAK_MAX_DEG - lo)
    legs = np.where(U[:, nE:] < 0.5, "rising", "setting")
    return peaks, legs


def _jd_fr(epoch_utc):
    start = datetime.fromisoformat(epoch_utc)
    jd_full = 2451545.0 + (start - _J2000).total_seconds() / 86400
    jd = float(np.floor(jd_full))
    return jd, jd_full - jd


def waypoint_at_elevation(meta, lat, lon, elev_deg, leg):
    """SGP4 state of a selected pass at the instant el = elev_deg on `leg`, in waypoint schema."""
    from Ray_Tracing.src.utils.orbital_utils import load_tle_satrec, propagate_topocentric
    sat = load_tle_satrec(meta["tle_line1"], meta["tle_line2"])
    jd, fr = _jd_fr(meta["epoch_utc"])
    rise = meta["rise_epoch_offset_s"]
    t_peak = rise + meta["peak_time_s"]
    a, b = (rise, t_peak) if leg == "rising" else (t_peak, meta["set_epoch_offset_s"])

    def el(t):
        return propagate_topocentric(sat, jd, fr + t / 86400, lat, lon, 0)[1] - elev_deg

    t = brentq(el, a, b, xtol=1e-6)
    az, e, dist, pos, vel, rr = propagate_topocentric(sat, jd, fr + t / 86400, lat, lon, 0)
    return dict(t_s=float(t - rise), epoch_offset_s=float(t), elev_deg=float(e),
                matched_elev_deg=float(e), az_deg=float(az), range_km=float(dist),
                range_m=float(dist * 1e3), enu_pos_km=np.asarray(pos),
                enu_vel_kms=np.asarray(vel), range_rate_km_s=float(rr), leg=leg,
                norad_id=int(meta["norad_id"]), satellite_name=meta["satellite_name"].strip(),
                target_peak_deg=float(meta["target_max_elevation_deg"]),
                peak_deg=float(meta["actual_max_elevation_deg"]),
                inclination_deg=float(meta["inclination_deg"]),
                tle_line1=meta["tle_line1"], tle_line2=meta["tle_line2"],
                peak_utc=meta["peak_utc"])


def scene_passes(lat, lon, elevations, peaks, legs, catalog_text, tol_deg=1.0,
                 search_hours=(6.0, 24.0), redraw_seed=0, lat_max_deg=60.0, max_redraws=5):
    """{E: waypoint} for one scene; widens the search window, then redraws the location."""
    import contextlib
    import io
    from Ray_Tracing.src.utils.pass_snr import make_real_target_passes
    rng = np.random.default_rng(redraw_seed)
    s = np.sin(np.radians(lat_max_deg))
    for attempt in range(max_redraws + 1):
        for hours in search_hours:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    sel = make_real_target_passes(catalog_text, [float(p) for p in peaks],
                                                  float(lat), float(lon), search_hours=hours,
                                                  tolerance_deg=tol_deg)
                wps = {float(E): waypoint_at_elevation(x["metadata"], lat, lon, float(E), leg)
                       for E, leg, x in zip(elevations, legs, sel)}
                return dict(lat=float(lat), lon=float(lon), redraws=attempt,
                            search_hours=hours, waypoints=wps)
            except (RuntimeError, ValueError):
                continue
        lat = float(np.degrees(np.arcsin(rng.uniform(-s, s))))
        lon = float(rng.uniform(-180.0, 180.0))
    raise RuntimeError(f"no passes within {tol_deg} deg after {max_redraws} redraws")


def _worker(args):
    seed, lat, lon, elevations, peaks, legs, catalog_path, tol, lat_max = args
    with open(catalog_path) as f:
        txt = f.read()
    return seed, scene_passes(lat, lon, elevations, peaks, legs, txt, tol_deg=tol,
                              redraw_seed=10_007 * seed + 1, lat_max_deg=lat_max)


def build_orbit_table(seeds, elevations, out_path, catalog_path, procs=8, seed=0,
                      lat_max_deg=60.0, tol_deg=1.0):
    """Orbit table {seed: scene_passes(...)} plus provenance; pickled to out_path."""
    from multiprocessing import Pool
    seeds = [int(s) for s in seeds]
    elevations = [float(e) for e in elevations]
    lat, lon = sample_ground_points(len(seeds), lat_max_deg, seed)
    peaks, legs = sample_peaks_and_legs(len(seeds), elevations, tol_deg, seed + 1)
    jobs = [(s, lat[i], lon[i], elevations, peaks[i], legs[i], catalog_path, tol_deg,
             lat_max_deg) for i, s in enumerate(seeds)]
    scenes = {}
    with Pool(procs) as pool:
        for k, (s, r) in enumerate(pool.imap_unordered(_worker, jobs), 1):
            scenes[s] = r
            if k % max(1, len(jobs) // 20) == 0 or k == len(jobs):
                print(f"[orbit_sampling] {k}/{len(jobs)} scenes", flush=True)
    with open(catalog_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    table = dict(scenes=scenes, elevations=elevations, lat_max_deg=lat_max_deg,
                 tol_deg=tol_deg, design_seed=seed, catalog_path=os.path.abspath(catalog_path),
                 catalog_sha256=sha, source="real catalog TLE, SGP4 (pass_snr.make_real_target_passes)")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
    return table


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    ap = argparse.ArgumentParser(description="Build the per-scene real-TLE orbit table.")
    ap.add_argument("--seeds", required=True, help="first:count, e.g. 0:100")
    ap.add_argument("--elevations", required=True, type=float, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lat_max", type=float, default=60.0)
    ap.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()
    first, count = (int(x) for x in a.seeds.split(":"))
    build_orbit_table(range(first, first + count), a.elevations, a.out, a.catalog,
                      procs=a.procs, seed=a.seed, lat_max_deg=a.lat_max, tol_deg=a.tol)
    print(f"[orbit_sampling] wrote {a.out}")
