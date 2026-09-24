"""SGP4 orbit propagation and topocentric geometry for the NTN RT pipeline.

Synthetic circular orbit (default: TR 38.821 reference LEO, 600 km) or a real
TLE, propagated TEME -> ECEF -> ground-station ENU to az/el/range/range-rate.

Earth is a sphere and TEME -> ECEF ignores polar motion and precession: fine at
channel accuracy, not for orbit determination. See docs/physics.md.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from sgp4.api import Satrec, WGS72, jday

# ─── Constants ─────────────────────────────────────────────────────────────

MU_EARTH_KM3_S2 = 398600.4418          # Earth gravitational parameter [km^3/s^2]
R_EARTH_KM      = 6378.137             # Mean equatorial radius [km]
OMEGA_EARTH     = 7.2921150e-5         # Earth rotation rate [rad/s]

# 3GPP TR 38.821 reference LEO altitudes [km]
REFERENCE_LEO_ALTITUDES_KM = {"leo600": 600.0, "leo1200": 1200.0}


def slant_range_m(altitude_km, elevation_deg):
    """Spherical-Earth slant range [m] to a satellite at `altitude_km`, seen at `elevation_deg`."""
    se = np.sin(np.radians(elevation_deg))
    rs = R_EARTH_KM + altitude_km
    return (np.sqrt(rs**2 - R_EARTH_KM**2 * (1.0 - se**2)) - R_EARTH_KM * se) * 1e3


# ─── Synthetic circular-orbit TLE construction ─────────────────────────────

def make_reference_satrec(altitude_km=600.0, inclination_deg=53.0,
                           raan_deg=0.0, arg_lat_deg=0.0,
                           epoch_year=2026, epoch_month=1, epoch_day=1,
                           epoch_hour=0, epoch_min=0, epoch_sec=0.0,
                           satnum=90000):
    """Build an `sgp4.api.Satrec` for an idealised circular orbit at the given
altitude/inclination, without needing a real published TLE.
    """
    a_km = R_EARTH_KM + altitude_km
    n_rad_s = np.sqrt(MU_EARTH_KM3_S2 / a_km ** 3)     # mean motion [rad/s]
    no_kozai = n_rad_s * 60.0                          # SGP4 wants rad/min

    jd, fr = jday(epoch_year, epoch_month, epoch_day,
                  epoch_hour, epoch_min, epoch_sec)
    epoch_days = (jd + fr) - 2433281.5   # SGP4 epoch: days since 1949-12-31 00:00 UT

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,              # gravity model
        'i',                # 'i' = improved (afspc) mode
        satnum,
        epoch_days,
        0.0,                                    # bstar (no drag for idealised orbit)
        0.0, 0.0,                               # ndot, nddot (unused by SGP4/SDP4)
        1.0e-4,                                 # ecco (~0, avoid exact-zero singularities)
        np.radians(0.0),                        # argpo (irrelevant, e~0)
        np.radians(inclination_deg),             # inclo
        np.radians(arg_lat_deg),                 # mo (mean anomaly at epoch)
        no_kozai,                                # no_kozai [rad/min]
        np.radians(raan_deg),                    # nodeo
    )
    return satrec


def orbital_period_s(altitude_km):
    """Keplerian orbital period [s] for a circular orbit at the given altitude."""
    a_km = R_EARTH_KM + altitude_km
    return 2.0 * np.pi * np.sqrt(a_km ** 3 / MU_EARTH_KM3_S2)


# ─── Real-satellite TLE loading ─────────────────────────────────────────────

def load_tle_satrec(line1, line2):
    """Build a Satrec from a real published TLE; raises ValueError if malformed.

    The real-satellite counterpart to `make_reference_satrec`: unlike the
    synthetic circular builders this carries actual eccentricity, drag (bstar)
    and perturbation terms, so the geometry reflects the true orbit.
    """
    line1 = line1.strip()
    line2 = line2.strip()
    satrec = Satrec.twoline2rv(line1, line2)
    # A basic sanity check: twoline2rv doesn't raise on malformed lines by
    # itself, but a completely bogus mean-motion (no_kozai) will show up as
    # zero/negative, which downstream propagation code can't handle safely.
    if satrec.no_kozai <= 0:
        raise ValueError(
            "Parsed TLE has non-positive mean motion — check that line1/line2 "
            "are correctly formatted, unmodified TLE lines (including column "
            "alignment and checksums)."
        )
    return satrec


def parse_tle_text(tle_text):
    """Parse a block of TLE text (2 or 3 lines — an optional name line followed
by the two numbered element lines) into (name, line1, line2).
    """
    lines = [ln for ln in tle_text.strip().splitlines() if ln.strip()]
    if len(lines) == 3:
        name, line1, line2 = lines
    elif len(lines) == 2:
        name, line1, line2 = None, lines[0], lines[1]
    else:
        raise ValueError(f"Expected 2 or 3 non-empty lines of TLE text, got {len(lines)}")
    return name, line1, line2


# ─── Real-satellite catalog search ──────────────────────────────────────────
#
# `find_pass_with_geometry_class` above searches over *invented* RAAN/arg_lat
# for one fabricated circular orbit -- useful for a representative synthetic
# scenario, but it never involves a real tracked object. The functions below
# are the real-satellite counterpart: given a bulk TLE catalog (e.g. all
# active Starlink satellites from celestrak), they scan each satellite's own,
# fixed, real orbital elements and report which ones actually produce a pass
# over the ground station -- nothing about the orbit is searched or invented,
# only which of the (many) real objects happens to qualify.

def parse_tle_catalog(text):
    """Parse a bulk multi-satellite TLE text blob -- the "3-line" format used by
celestrak's bulk group files (e.g. gp.php?GROUP=starlink&FORMAT=TLE):
repeating blocks of [name line, line1, line2] for many real satellites in
a row, with no blank-line separators required.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) % 3 != 0:
        raise ValueError(
            f"Expected a multiple of 3 non-empty lines (name + line1 + line2 "
            f"per satellite), got {len(lines)}. Is this a bulk 3-line-format "
            f"TLE file (celestrak FORMAT=TLE), not a 2-line-only file?"
        )
    catalog = []
    for i in range(0, len(lines), 3):
        name, line1, line2 = lines[i].strip(), lines[i + 1], lines[i + 2]
        catalog.append((name, line1, line2))
    return catalog


def _altitude_km_from_mean_motion(satrec):
    """
    Approximate circular-orbit altitude [km] from a satrec's mean motion
    (Kepler's third law; ignores eccentricity, which is ~0 for Starlink-class
    LEO). Used only for the coarse geometric pre-filter below -- full
    propagation still uses real SGP4 dynamics.
    """
    n_rad_s = satrec.no_kozai / 60.0   # SGP4 stores mean motion in rad/min
    a_km = (MU_EARTH_KM3_S2 / n_rad_s ** 2) ** (1.0 / 3.0)
    return a_km - R_EARTH_KM


def can_reach_latitude(satrec, gs_lat_deg, min_elev_deg=10.0):
    """Cheap pre-filter: could this orbit ever clear `min_elev_deg` from this latitude?"""
    inclination_deg = np.degrees(satrec.inclo)
    max_gt_lat_deg = inclination_deg if inclination_deg <= 90.0 else 180.0 - inclination_deg

    altitude_km = _altitude_km_from_mean_motion(satrec)
    eps = np.radians(min_elev_deg)
    coverage_angle_deg = np.degrees(
        np.arccos((R_EARTH_KM / (R_EARTH_KM + altitude_km)) * np.cos(eps)) - eps
    )
    return abs(gs_lat_deg) <= max_gt_lat_deg + coverage_angle_deg


def search_tle_catalog_for_pass(
    catalog, gs_lat_deg, gs_lon_deg, gs_alt_km,
    jd_start, fr_start, search_duration_s, min_elev_deg=10.0, step_s=5.0,
    geometry_class=None, max_satellites=None,
):
    """Scan a catalog of REAL satellites (from `parse_tle_catalog`) for whichever
ones actually pass over the ground station within the search window.
    """
    entries = catalog[:max_satellites] if max_satellites else catalog
    matches = []
    for name, line1, line2 in entries:
        try:
            satrec = load_tle_satrec(line1, line2)
        except Exception:
            continue   # skip malformed/unparseable entries rather than aborting the whole scan
        if not can_reach_latitude(satrec, gs_lat_deg, min_elev_deg):
            continue   # orbit can never reach gs_lat_deg -- skip without propagating
        try:
            pass_dict = find_pass(
                satrec, gs_lat_deg, gs_lon_deg, gs_alt_km,
                jd_start, fr_start, search_duration_s,
                min_elev_deg=min_elev_deg, step_s=step_s,
            )
        except RuntimeError:
            # SGP4 propagation error (e.g. code 6 "decayed") -- happens when a
            # satellite's TLE is propagated well outside its valid epoch window,
            # or the object has genuinely decayed. Skip it rather than aborting
            # the whole catalog scan over one bad entry.
            continue
        if pass_dict is None:
            continue
        max_elev = pass_dict["max_elev_deg"]
        cls = classify_pass_geometry(max_elev, min_elev_deg)
        if geometry_class not in (None, "any") and cls != geometry_class:
            continue
        matches.append({
            "name": name, "satnum": satrec.satnum, "satrec": satrec,
            "pass": pass_dict, "max_elev_deg": max_elev, "geometry_class": cls,
        })

    matches.sort(key=lambda m: m["pass"]["rows"][0]["t_s"])
    return matches


def best_pass_in_catalog(matches, select="soonest"):
    """Pick a single match out of `search_tle_catalog_for_pass`'s results."""
    if not matches:
        return None
    if select == "highest_elev":
        return max(matches, key=lambda m: m["max_elev_deg"])
    if select == "soonest":
        return matches[0]   # already sorted by soonest rise time
    raise ValueError(f"Unknown select='{select}', choose 'soonest' or 'highest_elev'")


# ─── Propagation & frame conversions ────────────────────────────────────────

def propagate_teme(satrec, jd, fr):
    """Propagate to a single Julian date/fraction."""
    e, r, v = satrec.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 propagation error code {e} at jd={jd}+{fr}")
    return np.array(r, dtype=float), np.array(v, dtype=float)


def propagate_teme_array(satrec, jd_arr, fr_arr):
    """Vectorized propagation over many (jd, fr) pairs in one call, using
`Satrec.sgp4_array` instead of looping `propagate_teme` in Python.
    """
    jd_arr = np.asarray(jd_arr, dtype=float)
    fr_arr = np.asarray(fr_arr, dtype=float)
    err, r, v = satrec.sgp4_array(jd_arr, fr_arr)
    return np.asarray(r, dtype=float), np.asarray(v, dtype=float), np.asarray(err)


def gmst_rad(jd, fr):
    """
    Greenwich Mean Sidereal Time [rad], IAU-1982 formula (Vallado).
    `jd + fr` must be UT1 (UTC is used as an approximation here, adequate
    at the sub-second-of-arc-per-day level irrelevant to this pipeline).
    """
    jd_ut1 = jd + fr
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * T
                + 0.093104 * T ** 2
                - 6.2e-6 * T ** 3)
    gmst_deg = (gmst_sec % 86400.0) / 240.0    # 86400 s <-> 360 deg => 240 s/deg
    return np.radians(gmst_deg % 360.0)


def teme_to_ecef(r_teme_km, v_teme_kms, jd, fr):
    """
    Rotate TEME position/velocity into ECEF using the Earth rotation angle
    (GMST) about the Z axis, including the Earth-rotation velocity term.
    """
    theta = gmst_rad(jd, fr)
    ct, st = np.cos(theta), np.sin(theta)
    Rz = np.array([[ct, st, 0.0],
                   [-st, ct, 0.0],
                   [0.0, 0.0, 1.0]])
    r_ecef = Rz @ r_teme_km
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH])
    v_ecef = Rz @ v_teme_kms - np.cross(omega_vec, r_ecef)
    return r_ecef, v_ecef


def gmst_rad_array(jd_arr, fr_arr):
    """Vectorized `gmst_rad` -- same IAU-1982 formula, over [N] arrays."""
    jd_ut1 = jd_arr + fr_arr
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * T
                + 0.093104 * T ** 2
                - 6.2e-6 * T ** 3)
    gmst_deg = (gmst_sec % 86400.0) / 240.0
    return np.radians(gmst_deg % 360.0)


def teme_to_ecef_array(r_teme_km, v_teme_kms, jd_arr, fr_arr):
    """
    Vectorized `teme_to_ecef` over [N, 3] position/velocity arrays (one GMST
    angle per row, from `jd_arr`/`fr_arr`). Same rotation as the scalar
    version, applied per-row without building N separate 3x3 matrices.
    """
    theta = gmst_rad_array(jd_arr, fr_arr)   # [N]
    ct, st = np.cos(theta), np.sin(theta)

    x, y, z = r_teme_km[:, 0], r_teme_km[:, 1], r_teme_km[:, 2]
    r_ecef = np.stack([ct * x + st * y, -st * x + ct * y, z], axis=1)

    vx, vy, vz = v_teme_kms[:, 0], v_teme_kms[:, 1], v_teme_kms[:, 2]
    v_rot = np.stack([ct * vx + st * vy, -st * vx + ct * vy, vz], axis=1)
    # cross([0,0,OMEGA_EARTH], r_ecef) = [-OMEGA*ry, OMEGA*rx, 0]
    omega_cross_r = np.stack([-OMEGA_EARTH * r_ecef[:, 1],
                               OMEGA_EARTH * r_ecef[:, 0],
                               np.zeros_like(theta)], axis=1)
    v_ecef = v_rot - omega_cross_r
    return r_ecef, v_ecef


def geodetic_to_ecef_spherical(lat_deg, lon_deg, alt_km=0.0):
    """Ground-station geodetic (spherical-Earth) -> ECEF [km]."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    r = R_EARTH_KM + alt_km
    return np.array([r * np.cos(lat) * np.cos(lon),
                      r * np.cos(lat) * np.sin(lon),
                      r * np.sin(lat)])


def ecef_to_enu_matrix(lat_deg, lon_deg):
    """3x3 rotation matrix mapping ECEF vectors to local ENU at (lat, lon)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    east  = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up    = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return np.vstack([east, north, up])   # rows: E, N, U


def ecef_to_topocentric(sat_ecef_km, sat_vel_ecef_kms, gs_lat_deg, gs_lon_deg, gs_alt_km=0.0):
    """Convert satellite ECEF position/velocity to ground-station-relative
ENU position and velocity (ground station assumed fixed in ECEF).
    """
    gs_ecef = geodetic_to_ecef_spherical(gs_lat_deg, gs_lon_deg, gs_alt_km)
    rho_ecef = sat_ecef_km - gs_ecef
    M = ecef_to_enu_matrix(gs_lat_deg, gs_lon_deg)
    enu_pos = M @ rho_ecef
    enu_vel = M @ sat_vel_ecef_kms       # ground station velocity in ECEF = 0
    return enu_pos, enu_vel


def ecef_to_topocentric_array(sat_ecef_km, sat_vel_ecef_kms, gs_lat_deg, gs_lon_deg, gs_alt_km=0.0):
    """Vectorized `ecef_to_topocentric` over [N, 3] arrays (fixed ground station)."""
    gs_ecef = geodetic_to_ecef_spherical(gs_lat_deg, gs_lon_deg, gs_alt_km)   # [3]
    rho_ecef = sat_ecef_km - gs_ecef[None, :]                                 # [N, 3]
    M = ecef_to_enu_matrix(gs_lat_deg, gs_lon_deg)                            # [3, 3], rows E,N,U
    enu_pos = rho_ecef @ M.T
    enu_vel = sat_vel_ecef_kms @ M.T
    return enu_pos, enu_vel


def enu_to_az_el_range(enu_pos_km):
    """(E, N, U) [km] -> (azimuth_deg [0,360) from North, CW; elevation_deg; range_km)."""
    e, n, u = enu_pos_km
    rng = float(np.linalg.norm(enu_pos_km))
    elev_deg = float(np.degrees(np.arcsin(np.clip(u / rng, -1.0, 1.0)))) if rng > 0 else 90.0
    az_deg = float(np.degrees(np.arctan2(e, n))) % 360.0
    return az_deg, elev_deg, rng


def enu_to_az_el_range_array(enu_pos_km):
    """Vectorized `enu_to_az_el_range` over an [N, 3] (E, N, U) array."""
    e, n, u = enu_pos_km[:, 0], enu_pos_km[:, 1], enu_pos_km[:, 2]
    rng = np.linalg.norm(enu_pos_km, axis=1)
    safe_rng = np.where(rng > 0, rng, 1.0)   # avoid 0-division; overwritten below
    elev_deg = np.where(rng > 0,
                         np.degrees(np.arcsin(np.clip(u / safe_rng, -1.0, 1.0))),
                         90.0)
    az_deg = np.degrees(np.arctan2(e, n)) % 360.0
    return az_deg, elev_deg, rng


def range_rate_km_s(enu_pos_km, enu_vel_kms):
    """Line-of-sight range-rate [km/s]: rho_dot = (r . v) / |r|."""
    rng = np.linalg.norm(enu_pos_km)
    if rng == 0:
        return 0.0
    return float(np.dot(enu_pos_km, enu_vel_kms) / rng)


def range_rate_km_s_array(enu_pos_km, enu_vel_kms):
    """Vectorized `range_rate_km_s` over [N, 3] arrays."""
    rng = np.linalg.norm(enu_pos_km, axis=1)
    safe_rng = np.where(rng > 0, rng, 1.0)
    dot = np.einsum('ij,ij->i', enu_pos_km, enu_vel_kms)
    return np.where(rng > 0, dot / safe_rng, 0.0)


def propagate_topocentric(satrec, jd, fr, gs_lat_deg, gs_lon_deg, gs_alt_km=0.0):
    """One exact SGP4 propagation + full frame conversion chain, at a single
(jd, fr) instant -- the scalar building block used by `generate_waypoints`
to solve for exact elevation-crossing times, instead of snapping to the
nearest row of a `sample_pass_table` grid.
    """
    r_teme, v_teme = propagate_teme(satrec, jd, fr)
    r_ecef, v_ecef = teme_to_ecef(r_teme, v_teme, jd, fr)
    enu_pos, enu_vel = ecef_to_topocentric(r_ecef, v_ecef, gs_lat_deg, gs_lon_deg, gs_alt_km)
    az_deg, elev_deg, rng_km = enu_to_az_el_range(enu_pos)
    rr = range_rate_km_s(enu_pos, enu_vel)
    return az_deg, elev_deg, rng_km, enu_pos, enu_vel, rr


# ─── Pass geometry classes (synthetic orbits only) ──────────────────────────
#
# For a synthetic circular orbit we're free to choose RAAN and argument of
# latitude at epoch, which — for a fixed ground station — control how close
# the ground track passes to zenith. This lets us search for a pass whose
# character matches a requested class, instead of taking whatever pass a
# single fixed (RAAN, arg_lat) happens to produce.
#
#   "overhead": near-zenith pass, satellite nearly directly above
#   "near"    : close but not overhead — a strong, high-elevation pass
#   "far"     : still above the mask elevation (link-connected) but low on
#               the horizon — long slant range, grazing geometry
#
# Bounds are on the pass's max_elev_deg. "far"'s lower bound is filled in
# with the caller's min_elev_deg (the link's own connectivity mask) at
# call time, since that's the true floor of "still connected".
PASS_GEOMETRY_BANDS = {
    "overhead": (80.0, 90.0),
    "near":     (40.0, 80.0),
    "far":      (None, 40.0),
}


def classify_pass_geometry(max_elev_deg, min_elev_deg=10.0):
    """Return which PASS_GEOMETRY_BANDS key a given pass max-elevation falls into."""
    for name, (lo, hi) in PASS_GEOMETRY_BANDS.items():
        lo_eff = min_elev_deg if lo is None else lo
        if lo_eff <= max_elev_deg <= hi:
            return name
    return None


def find_pass_with_geometry_class(
    make_satrec_fn, gs_lat_deg, gs_lon_deg, gs_alt_km,
    jd_start, fr_start, min_elev_deg, geometry_class,
    step_s=5.0, raan_grid=8, arg_lat_grid=8, search_periods=1.1,
):
    """Search over a grid of (RAAN, argument-of-latitude) values for a synthetic
circular orbit until a pass is found whose max elevation falls inside
the requested geometry class band (see PASS_GEOMETRY_BANDS). This is the
synthetic-orbit counterpart to a plain `find_pass` call — instead of a
single fixed geometry, it actively looks for one matching the requested
"overhead" / "near" / "far" character.
    """
    if geometry_class not in PASS_GEOMETRY_BANDS:
        raise ValueError(
            f"Unknown geometry_class '{geometry_class}', "
            f"choose from {list(PASS_GEOMETRY_BANDS)}"
        )
    lo, hi = PASS_GEOMETRY_BANDS[geometry_class]
    lo = min_elev_deg if lo is None else lo

    raan_values = np.linspace(0.0, 360.0, raan_grid, endpoint=False)
    arg_lat_values = np.linspace(0.0, 360.0, arg_lat_grid, endpoint=False)

    best = None   # (distance_to_band, satrec, pass_dict, raan, arg_lat)
    for raan in raan_values:
        for arg_lat in arg_lat_values:
            satrec = make_satrec_fn(float(raan), float(arg_lat))
            period_s = 2.0 * np.pi / (satrec.no_kozai / 60.0)   # no_kozai: rad/min
            pass_dict = find_pass(
                satrec, gs_lat_deg, gs_lon_deg, gs_alt_km,
                jd_start, fr_start, search_duration_s=period_s * search_periods,
                min_elev_deg=min_elev_deg, step_s=step_s,
            )
            if pass_dict is None:
                continue
            me = pass_dict["max_elev_deg"]
            if lo <= me <= hi:
                return {
                    "satrec": satrec, "pass": pass_dict,
                    "raan_deg": float(raan), "arg_lat_deg": float(arg_lat),
                    "matched_band": True,
                    "geometry_class": geometry_class, "band": (lo, hi),
                }
            dist = min(abs(me - lo), abs(me - hi))
            if best is None or dist < best[0]:
                best = (dist, satrec, pass_dict, float(raan), float(arg_lat))

    if best is None:
        return None
    _, satrec, pass_dict, raan, arg_lat = best
    return {
        "satrec": satrec, "pass": pass_dict,
        "raan_deg": raan, "arg_lat_deg": arg_lat,
        "matched_band": False,
        "geometry_class": geometry_class, "band": (lo, hi),
    }


# ─── Pass finding & waypoint generation ─────────────────────────────────────

def sample_pass_table(satrec, gs_lat_deg, gs_lon_deg, gs_alt_km,
                       jd_start, fr_start, duration_s, step_s=5.0):
    """Densely sample satellite geometry over [t_start, t_start+duration_s]."""
    n_steps = int(np.floor(duration_s / step_s)) + 1
    t_s_arr = np.arange(n_steps) * step_s
    fr_arr = fr_start + (t_s_arr / 86400.0)
    jd_arr = np.full(n_steps, jd_start, dtype=float)

    r_teme, v_teme, err = propagate_teme_array(satrec, jd_arr, fr_arr)
    ok = (err == 0)
    if not np.all(ok):
        t_s_arr, jd_arr, fr_arr = t_s_arr[ok], jd_arr[ok], fr_arr[ok]
        r_teme, v_teme = r_teme[ok], v_teme[ok]
    if len(t_s_arr) == 0:
        return []

    r_ecef, v_ecef = teme_to_ecef_array(r_teme, v_teme, jd_arr, fr_arr)
    enu_pos, enu_vel = ecef_to_topocentric_array(r_ecef, v_ecef, gs_lat_deg, gs_lon_deg, gs_alt_km)
    az, el, rng = enu_to_az_el_range_array(enu_pos)
    rr = range_rate_km_s_array(enu_pos, enu_vel)

    return [{
        "t_s": float(t_s_arr[i]), "jd": float(jd_arr[i]), "fr": float(fr_arr[i]),
        "az_deg": float(az[i]), "elev_deg": float(el[i]), "range_km": float(rng[i]),
        "enu_pos_km": enu_pos[i], "enu_vel_kms": enu_vel[i],
        "range_rate_km_s": float(rr[i]),
    } for i in range(len(t_s_arr))]


def find_pass(satrec, gs_lat_deg, gs_lon_deg, gs_alt_km,
              jd_start, fr_start, search_duration_s, min_elev_deg=10.0,
              step_s=5.0, coarse_step_s=None, refine_margin_s=600.0):
    """Scan a time window and return the first contiguous pass whose elevation
stays >= `min_elev_deg`, as a dict:
    {rows: [...], max_elev_deg, max_elev_idx, rise_idx, set_idx}
`rows` is the full contiguous sample list for that pass (from rise to set,
inclusive), each row matching the schema from `sample_pass_table`.
    """
    if coarse_step_s is None:
        coarse_step_s = step_s * 12.0

    if coarse_step_s > step_s and search_duration_s > 4 * coarse_step_s:
        coarse_table = sample_pass_table(satrec, gs_lat_deg, gs_lon_deg, gs_alt_km,
                                          jd_start, fr_start, search_duration_s, coarse_step_s)
        candidate_idx = next((i for i, row in enumerate(coarse_table)
                               if row["elev_deg"] >= min_elev_deg), None)
        if candidate_idx is None:
            return None   # no coarse sample ever reaches min_elev_deg -> no pass here

        t_center = coarse_table[candidate_idx]["t_s"]
        t_lo = max(0.0, t_center - coarse_step_s - refine_margin_s)
        t_hi = min(search_duration_s, t_center + coarse_step_s + refine_margin_s)
        refine_fr_start = fr_start + (t_lo / 86400.0)
        full_table = sample_pass_table(satrec, gs_lat_deg, gs_lon_deg, gs_alt_km,
                                        jd_start, refine_fr_start, t_hi - t_lo, step_s)
        for row in full_table:
            row["t_s"] += t_lo   # rebase back onto the original jd_start/fr_start clock
    else:
        full_table = sample_pass_table(satrec, gs_lat_deg, gs_lon_deg, gs_alt_km,
                                        jd_start, fr_start, search_duration_s, step_s)

    above = [row["elev_deg"] >= min_elev_deg for row in full_table]

    rise_idx = None
    for i, ok in enumerate(above):
        if ok:
            rise_idx = i
            break
    if rise_idx is None:
        return None

    set_idx = rise_idx
    for i in range(rise_idx, len(above)):
        if above[i]:
            set_idx = i
        else:
            break

    pass_rows = full_table[rise_idx:set_idx + 1]
    max_elev_idx_local = int(np.argmax([row["elev_deg"] for row in pass_rows]))
    return {
        "rows": pass_rows,
        "max_elev_deg": pass_rows[max_elev_idx_local]["elev_deg"],
        "max_elev_idx": max_elev_idx_local,
        "rise_idx": 0,
        "set_idx": len(pass_rows) - 1,
    }


def generate_waypoints(pass_dict, satrec, gs_lat_deg, gs_lon_deg, gs_alt_km,
                        elev_step_deg=10.0, mask_elev_deg=10.0, leg="rising",
                        elev_tol_deg=1e-5):
    """Generate elevation waypoints in `elev_step_deg` increments from
`mask_elev_deg` up to the pass's actual maximum elevation, matching one
leg of the pass (rising or setting).
    """
    rows = pass_dict["rows"]
    max_idx = pass_dict["max_elev_idx"]

    if leg == "rising":
        leg_rows = rows[: max_idx + 1]
    elif leg == "setting":
        leg_rows = rows[max_idx:]
    else:
        raise ValueError("leg must be 'rising' or 'setting'")

    if not leg_rows:
        return []

    # jd is constant across all of a pass_dict's rows and fr(t_s) is affine
    # in t_s (see `sample_pass_table`) -- recover that mapping once so any
    # t_s (not just sampled ones) can be propagated to directly.
    jd_ref = leg_rows[0]["jd"]
    fr_ref = leg_rows[0]["fr"] - leg_rows[0]["t_s"] / 86400.0

    def _elev_at_t(t_s):
        return propagate_topocentric(
            satrec, jd_ref, fr_ref + t_s / 86400.0, gs_lat_deg, gs_lon_deg, gs_alt_km,
        )[1]

    # Refine the true continuous-time elevation peak. The discrete argmax in
    # `rows` is only accurate to +/- step_s/2; elevation is smooth near the
    # peak, so a bounded 1-D maximization over its immediate neighbours (in
    # the full, not per-leg, row list -- the peak sits between rows regardless
    # of which leg we're generating waypoints for) recovers the true max.
    i_lo, i_hi = max(0, max_idx - 1), min(len(rows) - 1, max_idx + 1)
    t_lo_peak, t_hi_peak = rows[i_lo]["t_s"], rows[i_hi]["t_s"]
    if t_hi_peak > t_lo_peak:
        peak = minimize_scalar(lambda t: -_elev_at_t(t),
                                bounds=(t_lo_peak, t_hi_peak),
                                method="bounded", options={"xatol": 1e-6})
        t_peak, max_elev_exact = float(peak.x), float(-peak.fun)
    else:
        t_peak, max_elev_exact = rows[max_idx]["t_s"], pass_dict["max_elev_deg"]

    targets = list(np.arange(mask_elev_deg, max_elev_exact, elev_step_deg))
    if not targets or abs(targets[-1] - max_elev_exact) > elev_tol_deg:
        targets.append(max_elev_exact)

    leg_elevs = np.array([r["elev_deg"] for r in leg_rows])
    leg_t = np.array([r["t_s"] for r in leg_rows])

    waypoints = []
    for target in targets:
        if abs(target - max_elev_exact) <= elev_tol_deg:
            # Top waypoint: already solved above as a turning point, not a
            # simple crossing (elev(t) - target has no sign change there).
            t_star = t_peak
        else:
            # Bracket the target between the two adjacent leg samples whose
            # elevations straddle it (a plain sign-change scan, so it works
            # regardless of the leg's monotonic direction), then solve
            # elev(t) - target = 0 exactly.
            t_lo = t_hi = None
            for i in range(len(leg_rows) - 1):
                e0, e1 = leg_elevs[i], leg_elevs[i + 1]
                if e0 == target:
                    t_lo = t_hi = leg_t[i]
                    break
                if (e0 - target) * (e1 - target) <= 0:
                    t_lo, t_hi = leg_t[i], leg_t[i + 1]
                    break
            if t_lo is None:
                # Target sits below this leg's lowest *sampled* elevation --
                # this is the mask_elev_deg target itself: `find_pass` only
                # keeps rows already >= mask_elev_deg, so the true crossing
                # time lies just outside the sampled window (backward in
                # time for the rising leg, forward for the setting leg).
                # Step outward from that boundary sample via direct SGP4
                # propagation (exponentially growing step) until the target
                # is bracketed, then solve exactly -- same precision as the
                # interior case, just without a pre-existing bracket.
                direction = -1.0 if leg == "rising" else 1.0
                t_edge = leg_t[0] if leg == "rising" else leg_t[-1]
                e_edge = leg_elevs[0] if leg == "rising" else leg_elevs[-1]
                dt = abs(leg_t[1] - leg_t[0]) if len(leg_t) > 1 else 5.0
                dt = dt or 5.0
                t_probe, e_probe, n_expand = t_edge, e_edge, 0
                while e_probe > target and n_expand < 40:
                    t_probe = t_edge + direction * dt
                    e_probe = _elev_at_t(t_probe)
                    dt *= 1.5
                    n_expand += 1
                if e_probe <= target:
                    lo, hi = (t_probe, t_edge) if direction < 0 else (t_edge, t_probe)
                    t_star = brentq(lambda t: _elev_at_t(t) - target, lo, hi, xtol=1e-6)
                else:
                    # Expansion cap reached without bracketing -- fall back
                    # to the nearest sample rather than failing the run.
                    t_star = float(leg_t[int(np.argmin(np.abs(leg_elevs - target)))])
            elif t_lo == t_hi:
                t_star = float(t_lo)
            else:
                t_star = brentq(lambda t: _elev_at_t(t) - target, t_lo, t_hi, xtol=1e-6)

        az_deg, elev_exact, range_km, enu_pos_km, enu_vel_kms, rr = propagate_topocentric(
            satrec, jd_ref, fr_ref + t_star / 86400.0, gs_lat_deg, gs_lon_deg, gs_alt_km,
        )
        waypoints.append({
            "elev_deg": float(target),
            "matched_elev_deg": float(elev_exact),
            "az_deg": float(az_deg),
            "range_km": float(range_km),
            "range_m": float(range_km * 1e3),
            "enu_pos_km": enu_pos_km,
            "enu_vel_kms": enu_vel_kms,
            "range_rate_km_s": float(rr),
            "t_s": float(t_star),
        })
    return waypoints


# ─── Window-validity check (quasi-static geometry assumption) ──────────────

def doppler_window_validity(pass_dict, waypoint, window_s, max_drift_deg=0.05):
    """Check whether holding TX position/velocity fixed for `window_s` seconds
at the given waypoint keeps elevation drift under `max_drift_deg`.

Estimates the local elevation angular rate via finite differencing the
pass table around the waypoint's sample time, then checks
|rate| * window_s <= max_drift_deg.
    """
    rows = pass_dict["rows"]
    t_target = waypoint["t_s"]
    ts = np.array([r["t_s"] for r in rows])
    elevs = np.array([r["elev_deg"] for r in rows])
    i = int(np.argmin(np.abs(ts - t_target)))
    i0, i1 = max(i - 1, 0), min(i + 1, len(rows) - 1)
    if i1 == i0:
        rate = 0.0
    else:
        rate = (elevs[i1] - elevs[i0]) / (ts[i1] - ts[i0])
    est_drift = abs(rate) * window_s
    return {
        "valid": est_drift <= max_drift_deg,
        "elev_rate_deg_s": float(rate),
        "est_drift_deg": float(est_drift),
    }


# ─── Compass azimuth / ENU velocity -> scene local frame ────────────────────

def compass_to_scene_azimuth(az_compass_deg, scene_east_heading_deg=90.0):
    """Convert a real (compass) azimuth [deg, 0=North, clockwise] into the
scene's local math-angle azimuth convention used by
`tx_position_from_elevation` / `tx_position_from_pass_sample`
(degrees, counter-clockwise from the scene's +X axis).
    """
    return (scene_east_heading_deg - az_compass_deg) % 360.0


def rotate_enu_to_scene(vec_enu, scene_east_heading_deg=90.0):
    """Rotate an ENU vector (E, N, U) — e.g. satellite velocity — into the
scene's local Cartesian frame (X, Y, Z), using the same
rotation-only transform implied by `compass_to_scene_azimuth`
(no translation — correct for velocity vectors, and equivalent to the
position transform's ground-vector rotation).
    """
    e, n, u = vec_enu
    phi0 = np.radians(scene_east_heading_deg)
    vx = e * np.sin(phi0) + n * np.cos(phi0)
    vy = -e * np.cos(phi0) + n * np.sin(phi0)
    vz = u
    return np.array([vx, vy, vz])
