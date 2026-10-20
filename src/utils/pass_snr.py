"""Whole-pass geometry and elevation-bin SNR statistics (no GPU imports)."""

import numpy as np


def make_real_target_passes(catalog_text, targets, latitude, longitude,
                            epoch_utc=None, search_hours=24.0, mask_deg=0.0,
                            step_s=5.0, tolerance_deg=5.0, max_tle_age_days=3.0,
                            target_azimuths=None, azimuth_tolerance_deg=30.0,
                            exclude_norad_ids=()):
    """Choose real TLE passes nearest the requested peaks without changing elements.

    Search the first visible pass of each catalog satellite in the window.
    With no explicit start, use the median catalog TLE epoch for a reproducible
    historical campaign. Reject elements too far from the search window.
    """
    from datetime import datetime, timedelta, timezone
    from scipy.optimize import minimize_scalar, brentq
    from Ray_Tracing.src.utils.orbital_utils import (
        parse_tle_catalog, load_tle_satrec, search_tle_catalog_for_pass,
        propagate_topocentric)

    if not targets or any(not mask_deg < t <= 90 for t in targets):
        raise ValueError("Require mask < each requested peak <= 90")
    if search_hours <= 0 or step_s <= 0 or tolerance_deg <= 0 or max_tle_age_days <= 0:
        raise ValueError("Search duration, sampling step, tolerance and TLE age must be positive")
    if target_azimuths is not None:
        if len(target_azimuths) != len(targets) or any(not 0 <= a < 360 for a in target_azimuths):
            raise ValueError("Require one compass azimuth in [0, 360) per peak target")
        if not 0 < azimuth_tolerance_deg <= 180:
            raise ValueError("Azimuth tolerance must be in (0, 180]")
    excluded = set(exclude_norad_ids)
    catalog = parse_tle_catalog(catalog_text)
    parsed = [(name, l1, l2, load_tle_satrec(l1, l2)) for name, l1, l2 in catalog]
    epoch_jds = [s.jdsatepoch + s.jdsatepochF for _, _, _, s in parsed]
    origin = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    if epoch_utc is None:
        start_jd = float(np.median(epoch_jds))
        start = origin + timedelta(days=start_jd - 2451545.0)
    else:
        start = datetime.fromisoformat(epoch_utc.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        start = start.astimezone(timezone.utc)
        start_jd = 2451545.0 + (start - origin).total_seconds() / 86400
    jd = float(np.floor(start_jd))
    fr = start_jd - jd
    eligible = [(n, l1, l2) for n, l1, l2, s in parsed
                if s.satnum not in excluded and max(abs(start_jd - s.jdsatepoch - s.jdsatepochF),
                       abs(start_jd + search_hours / 24 - s.jdsatepoch - s.jdsatepochF)) <= max_tle_age_days]
    print(f"Searching {len(eligible)}/{len(catalog)} real satellites from {start.isoformat()}", flush=True)
    matches = search_tle_catalog_for_pass(eligible, latitude, longitude, 0,
                                          jd, fr, search_hours * 3600,
                                          min_elev_deg=mask_deg, step_s=10)
    if not matches:
        raise RuntimeError("No qualifying real passes; check TLE age, epoch and search duration")
    elements = {load_tle_satrec(l1, l2).satnum: (l1, l2) for _, l1, l2 in eligible}

    def geometry(sat, t):
        return propagate_topocentric(sat, jd, fr + t / 86400, latitude, longitude, 0)

    def crossing(sat, t_peak, direction):
        span = 30.0
        while geometry(sat, t_peak + direction * span)[1] > mask_deg:
            span *= 1.5
            if span > 3600:
                raise RuntimeError("Could not bracket a complete real pass")
        a, b = sorted((t_peak, t_peak + direction * span))
        return brentq(lambda t: geometry(sat, t)[1] - mask_deg, a, b, xtol=1e-7)

    # Refine nearby candidates and all high passes; avoid coarse sampling
    # choosing a lower peak simply because a zenith crossing fell between samples.
    candidates = []
    for m in matches:
        rough = m["max_elev_deg"]
        if not any((rough >= 75 if t >= 80 else abs(rough - t) <= max(2, tolerance_deg)) for t in targets):
            continue
        rows = m["pass"]["rows"]
        t0 = rows[m["pass"]["max_elev_idx"]]["t_s"]
        fit = minimize_scalar(lambda t: -geometry(m["satrec"], t)[1],
                              bounds=(t0 - 15, t0 + 15), method="bounded",
                              options={"xatol": 1e-6})
        if -fit.fun <= mask_deg:
            continue
        rise = crossing(m["satrec"], float(fit.x), -1)
        setting = crossing(m["satrec"], float(fit.x), 1)
        if rise < 0 or setting > search_hours * 3600:
            continue  # Select only complete passes inside the search window.
        candidates.append(dict(match=m, t_peak=float(fit.x), peak=float(-fit.fun),
                               rise=rise, setting=setting,
                               peak_azimuth=float(geometry(m["satrec"], float(fit.x))[0])))
    selected, used = [], set()
    for index, target in enumerate(targets):
        target_azimuth = None if target_azimuths is None else target_azimuths[index]
        choices = [c for c in candidates if c["match"]["satnum"] not in used]
        if target_azimuth is not None:
            az_error = lambda c: abs((c["peak_azimuth"] - target_azimuth + 180) % 360 - 180)
            choices = [c for c in choices if abs(c["peak"] - target) <= tolerance_deg
                       and az_error(c) <= azimuth_tolerance_deg]
        if not choices:
            raise RuntimeError(f"No distinct real satellite near peak {target}°, peak azimuth {target_azimuth}")
        chosen = min(choices, key=lambda c: abs(c["peak"] - target) if target_azimuth is None
                     else (abs(c["peak"] - target) / tolerance_deg) ** 2
                     + (az_error(c) / azimuth_tolerance_deg) ** 2)
        if abs(chosen["peak"] - target) > tolerance_deg:
            raise RuntimeError(f"Nearest real peak to {target}° is {chosen['peak']:.3f}°; expand search/tolerance")
        m, t_peak, peak = chosen["match"], chosen["t_peak"], chosen["peak"]
        sat = m["satrec"]
        used.add(sat.satnum)

        rise, setting = chosen["rise"], chosen["setting"]
        count = int(np.ceil((setting - rise) / step_s))
        dt = (setting - rise) / count
        waypoints = []
        for t in rise + (np.arange(count) + 0.5) * dt:
            az, el, distance, position, velocity, rr = geometry(sat, float(t))
            waypoints.append(dict(t_s=float(t - rise), epoch_offset_s=float(t),
                                  elev_deg=float(el), matched_elev_deg=float(el),
                                  az_deg=float(az), range_km=float(distance),
                                  range_m=float(distance * 1e3), enu_pos_km=position,
                                  enu_vel_kms=velocity, range_rate_km_s=float(rr),
                                  leg="rising" if t < t_peak else "setting"))
        l1, l2 = elements[sat.satnum]
        tle_jd = sat.jdsatepoch + sat.jdsatepochF
        utc_at = lambda t: (start + timedelta(seconds=t)).isoformat()
        metadata = dict(target_max_elevation_deg=target, actual_max_elevation_deg=peak,
                        sampled_max_elevation_deg=max(w["elev_deg"] for w in waypoints),
                        inclination_deg=float(np.degrees(sat.inclo)),
                        raan_deg=float(np.degrees(sat.nodeo)),
                        mean_anomaly_deg=float(np.degrees(sat.mo)),
                        eccentricity=float(sat.ecco), satellite_name=m["name"],
                        norad_id=int(sat.satnum), tle_line1=l1, tle_line2=l2,
                        tle_epoch_utc=(origin + timedelta(days=tle_jd - 2451545)).isoformat(),
                        tle_age_at_peak_days=start_jd + t_peak / 86400 - tle_jd,
                        epoch_utc=start.isoformat(), rise_utc=utc_at(rise),
                        peak_utc=utc_at(t_peak), set_utc=utc_at(setting),
                        rise_epoch_offset_s=rise, set_epoch_offset_s=setting,
                        duration_s=setting - rise, peak_time_s=t_peak - rise,
                        sample_interval_s=dt, mask_elevation_deg=mask_deg,
                        ground_latitude_deg=latitude, ground_longitude_deg=longitude,
                        peak_azimuth_deg=chosen["peak_azimuth"],
                        rise_azimuth_deg=float(geometry(sat, rise)[0]),
                        set_azimuth_deg=float(geometry(sat, setting)[0]),
                        orbit_source="real catalog TLE propagated with SGP4")
        if target_azimuth is not None:
            metadata["target_peak_azimuth_deg"] = target_azimuth
        selected.append(dict(waypoints=waypoints, metadata=metadata))
        print(f"{m['name'].strip()} (NORAD {sat.satnum}): peak {peak:.4f}°, inclination {metadata['inclination_deg']:.4f}°", flush=True)
    return selected


# TR 38.821 Table 6.1.3.3-1 link-budget cases (LEO-600, Set 1, frequency reuse 1).
# Satellite EIRP density from Table 6.1.1.1-1, terminals from Table 6.1.1.1-3.
# Service-link minimum elevation 10 deg (Table 4.1-1).
TR38821_ALTITUDE_KM, TR38821_MIN_ELEVATION_DEG = 600.0, 10.0
TR38821_CASES = {
    "sc9": dict(terminal="handheld", frequency_hz=2e9, eirp_dbw_per_mhz=34.0,
                bandwidth_hz=30e6, noise_figure_db=7.0, antenna_temperature_k=290.0,
                rx_pol_type="CROSS", rx_model="mrc", subcarrier_spacing_hz=15e3,
                shadow_margin_db=3.0, tr_cnr_db=6.6),
    "sc6": dict(terminal="VSAT", frequency_hz=20e9, eirp_dbw_per_mhz=4.0,
                bandwidth_hz=400e6, noise_figure_db=1.2, antenna_temperature_k=150.0,
                rx_pol_type="RHCP", rx_model="dish", dish_diameter_m=0.6, dish_gain_dbi=39.7,
                subcarrier_spacing_hz=120e3, shadow_margin_db=0.0, tr_cnr_db=8.5),
}
# TR 38.811 Sec 6.6.4 zenith attenuation (ITU-R P.676 Fig. 6, values as in ns-3).
_ZENITH_ATM_DB = {2e9: 0.035, 20e9: 0.25}
# TR 38.811 Table 6.6.6.2.1-1, tropospheric scintillation at 20 GHz (99%).
_TROPO_ELEV = [10, 20, 30, 40, 50, 60, 70, 80, 90]
_TROPO_DB = [1.08, 0.48, 0.30, 0.22, 0.17, 0.13, 0.12, 0.12, 0.12]


def tr38811_losses_db(frequency_hz, elevation_deg):
    """Atmospheric absorption plus scintillation, as TR 38.821 Table 6.1.3.2-1 specifies.

    Absorption A_zenith/sin(el) (38.811 eq. 6.6-8); ionospheric scintillation
    6.22/f^1.5 below 6 GHz (P3, 1% time), tropospheric table otherwise.
    """
    el = np.asarray(elevation_deg, dtype=float)
    atm = _ZENITH_ATM_DB[frequency_hz] / np.sin(np.radians(el))
    if frequency_hz < 6e9:
        return atm + 6.22 / (frequency_hz / 1e9) ** 1.5
    # table clamped below 10 deg, which is the TR minimum elevation anyway
    return atm + np.interp(el, _TROPO_ELEV, _TROPO_DB)


def dish_gain_db(off_axis_rad, frequency_hz, diameter_m, peak_gain_dbi):
    """TR 38.811 Sec 6.4.1 aperture pattern, 4|J1(ka sin t)/(ka sin t)|^2 scaled to peak.

    Nothing is received from behind the aperture (off-axis > 90 deg).
    """
    from scipy.special import j1
    t = np.asarray(off_axis_rad, dtype=float)
    x = np.pi * diameter_m * frequency_hz / 299792458.0 * np.sin(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(np.abs(x) < 1e-9, 1.0, (2 * j1(x) / x) ** 2)
    rel = np.where(t > np.pi / 2, 0.0, rel)
    with np.errstate(divide="ignore"):
        return peak_gain_dbi + 10 * np.log10(rel)


def _noise_dbm(bandwidth_hz, noise_figure_db, temperature_k):
    """k (T_ant + 290 (F - 1)) B; with T_ant = 290 K this is k 290 F B."""
    if bandwidth_hz <= 0 or temperature_k <= 0:
        raise ValueError("Bandwidth and temperature must be positive")
    t_sys = temperature_k + 290.0 * (10 ** (noise_figure_db / 10) - 1)
    return 10 * np.log10(1.380649e-23 * t_sys * bandwidth_hz * 1e3)


def snr_db_from_cir(amps, tx_power_dbm, bandwidth_hz, noise_figure_db,
                    temperature_k=290.0):
    """Narrowband coherent SNR at the carrier frequency only, |sum(amps)|^2.

    CIR amplitudes already contain delay phase. TX power is applied once;
    noise is k*T*B times the linear receiver noise factor. No paths = -inf.
    """
    noise_dbm = _noise_dbm(bandwidth_hz, noise_figure_db, temperature_k)
    gain = float(abs(np.sum(amps)) ** 2)
    return float(tx_power_dbm + 10 * np.log10(gain) - noise_dbm) if gain else -np.inf


def wideband_snr_db(amps, taus, tx_power_dbm, bandwidth_hz, noise_figure_db,
                    temperature_k=290.0, subcarrier_spacing_hz=15e3, chunk=4096):
    """(band-mean SNR, capacity-effective SNR) in dB over the bandwidth.

    H(f) = sum a_i exp(-j2pi f tau_i) on a subcarrier grid centred on the
    carrier (amps already carry carrier phase). Band-mean power equals that of
    a flat-spectrum signal convolved with the CIR. No paths = -inf.
    """
    p = band_gain(amps, taus, bandwidth_hz, subcarrier_spacing_hz, chunk)
    return snr_from_band_gain(p, tx_power_dbm, bandwidth_hz, noise_figure_db, temperature_k)


def band_gain(amps, taus, bandwidth_hz, subcarrier_spacing_hz=15e3, chunk=4096,
              incoherent=None):
    """Expected |H(f)|^2 on the subcarrier grid; zeros when there are no paths.

    Paths flagged `incoherent` (diffuse scattering) have random phase, so they
    add their power, flat across the band. Receive branches with equal noise
    combine by MRC as the sum of their band gains.
    """
    amps = np.asarray(amps, dtype=np.complex128).ravel()
    taus = np.asarray(taus, dtype=np.float64).ravel()
    if amps.shape != taus.shape:
        raise ValueError("amps and taus must match")
    n = max(1, int(round(bandwidth_hz / subcarrier_spacing_hz)))
    if not amps.size:
        return np.zeros(n)
    floor = 0.0
    if incoherent is not None:
        incoherent = np.asarray(incoherent, dtype=bool).ravel()
        floor = float(np.sum(np.abs(amps[incoherent]) ** 2))
        amps, taus = amps[~incoherent], taus[~incoherent]
    f = (np.arange(n) - (n - 1) / 2) * subcarrier_spacing_hz
    tau = taus - taus.min() if taus.size else taus   # common delay is a common phase
    h = np.zeros(n, dtype=np.complex128)
    for s in range(0, amps.size, chunk):
        h += np.exp(-2j * np.pi * np.outer(f, tau[s:s + chunk])) @ amps[s:s + chunk]
    return np.abs(h) ** 2 + floor


def snr_from_band_gain(p, tx_power_dbm, bandwidth_hz, noise_figure_db, temperature_k=290.0):
    """(band-mean SNR, capacity-effective SNR) in dB from band_gain output."""
    noise_dbm = _noise_dbm(bandwidth_hz, noise_figure_db, temperature_k)
    if not p.any():
        return -np.inf, -np.inf
    mean_db = float(tx_power_dbm + 10 * np.log10(p.mean()) - noise_dbm)
    gamma = p * 10 ** ((tx_power_dbm - noise_dbm) / 10)
    eff = 2 ** np.mean(np.log2(1 + gamma)) - 1
    return mean_db, float(10 * np.log10(eff)) if eff > 0 else -np.inf


def elevation_bin_statistics(elevations, snrs, threshold_db=0.0):
    """Count strictly SNR > threshold in [lo, hi), final bin includes 90°.

    Uniform time samples from both legs are pooled. Empty bins have NaN
    percentages; -inf SNR is an outage and remains in the denominator.
    Missing (NaN) SNR is an error, never silently removed.
    """
    elevations, snrs = np.asarray(elevations), np.asarray(snrs)
    if elevations.shape != snrs.shape or elevations.ndim != 1:
        raise ValueError("Elevation and SNR must be matching 1-D arrays")
    if np.any(~np.isfinite(elevations)) or np.any((elevations < 0) | (elevations > 90)):
        raise ValueError("Elevation must be finite and in [0, 90]")
    if np.any(np.isnan(snrs)) or not np.isfinite(threshold_db):
        raise ValueError("Missing SNR or invalid threshold")
    bins = []
    for lo in range(0, 90, 10):
        mask = (elevations >= lo) & ((elevations <= 90) if lo == 80 else (elevations < lo + 10))
        n = int(mask.sum())
        above = int(np.count_nonzero(snrs[mask] > threshold_db))
        bins.append(dict(elevation_lo_deg=lo, elevation_hi_deg=lo + 10,
                         sample_count=n, above_threshold_count=above,
                         percent_above_threshold=100 * above / n if n else float("nan")))
    return bins


def make_target_pass(target_deg, latitude, longitude, altitude_km=600.0,
                     inclination_deg=53.0, mask_deg=0.0, step_s=5.0,
                     northbound=None, side=1):
    """Fit synthetic SGP4 geometry to a peak elevation, then sample rise to set.

    Fix epoch, altitude and inclination; change RAAN and argument of latitude.
    `northbound` picks the ascending or descending crossing of the UE latitude
    (None: whichever fits best); `side` = +1/-1 shifts RAAN east/west, putting
    the track on either side of the UE. Midpoint samples represent equal time
    intervals covering the entire pass. The continuous peak is recorded
    separately from the sampled maximum.
    """
    from scipy.optimize import least_squares, minimize_scalar, brentq
    from sgp4.api import jday
    from Ray_Tracing.src.utils.orbital_utils import make_reference_satrec, propagate_topocentric

    if not 0 <= mask_deg < target_deg <= 90 or step_s <= 0:
        raise ValueError("Require 0 <= mask < target <= 90 and step > 0")
    jd, fr = jday(2026, 1, 1, 0, 0, 0)

    def satellite(raan, arg):
        return make_reference_satrec(altitude_km, inclination_deg, raan, arg)

    def geometry(sat, t):
        return propagate_topocentric(sat, jd, fr + t / 86400, latitude, longitude, 0.0)

    # Solve the overhead position at epoch; try both ascending/descending seeds.
    best = None
    for raan0, arg in ((r, a) for r in (100, 190, 280, 10) for a in (60, 120, 240, 300)):
        fit = least_squares(lambda x: geometry(satellite(*x), 0)[3][:2] / altitude_km,
                            [longitude + raan0, arg], max_nfev=150,
                            xtol=1e-11, ftol=1e-11, gtol=1e-11)
        # Northbound means the argument of latitude is in the ascending half.
        # The antipode also has zero east/north offset; keep only overhead fits.
        if (northbound is not None and (np.cos(np.radians(fit.x[1])) > 0) != northbound
                or geometry(satellite(*fit.x), 0)[3][2] < 0):
            continue
        if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
            best = fit
    if best is None or np.linalg.norm(best.fun) > 1e-6:
        raise ValueError("Inclination cannot produce an overhead pass at this latitude")
    base_raan, arg = best.x

    def peak(raan):
        sat = satellite(raan, arg)
        fit = minimize_scalar(lambda t: -geometry(sat, t)[1], bounds=(-1200, 1200),
                              method="bounded", options={"xatol": 1e-7})
        return float(fit.x), float(-fit.fun)

    raan = base_raan
    if target_deg < 90:
        offsets = side * np.arange(1.0, 91.0)
        bracket = next((d for d in offsets if peak(base_raan + d)[1] < target_deg), None)
        if bracket is None:
            raise RuntimeError("Could not bracket requested peak elevation")
        raan = brentq(lambda r: peak(r)[1] - target_deg,
                      *sorted((base_raan, base_raan + bracket)), xtol=1e-9)
    sat = satellite(raan, arg)
    t_peak, actual_peak = peak(raan)

    def crossing(direction):
        span = 30.0
        while geometry(sat, t_peak + direction * span)[1] > mask_deg:
            span *= 1.5
            if span > 2400:
                raise RuntimeError("Unable to locate full pass boundary")
        a, b = sorted((t_peak, t_peak + direction * span))
        return brentq(lambda t: geometry(sat, t)[1] - mask_deg, a, b, xtol=1e-7)

    rise, setting = crossing(-1), crossing(1)
    count = int(np.ceil((setting - rise) / step_s))
    dt = (setting - rise) / count
    waypoints = []
    for t in rise + (np.arange(count) + 0.5) * dt:
        az, el, distance, position, velocity, rr = geometry(sat, float(t))
        waypoints.append(dict(t_s=float(t - rise), epoch_offset_s=float(t),
                              elev_deg=float(el), matched_elev_deg=float(el),
                              az_deg=float(az), range_km=float(distance),
                              range_m=float(distance * 1e3), enu_pos_km=position,
                              enu_vel_kms=velocity, range_rate_km_s=float(rr),
                              leg="rising" if t < t_peak else "setting"))
    return dict(waypoints=waypoints, metadata=dict(
        target_max_elevation_deg=target_deg, actual_max_elevation_deg=actual_peak,
        sampled_max_elevation_deg=max(w["elev_deg"] for w in waypoints),
        inclination_deg=float(np.degrees(sat.inclo)), altitude_km=altitude_km,
        raan_deg=float(raan % 360), arg_lat_deg=float(arg % 360),
        northbound=bool(np.cos(np.radians(arg)) > 0), side=side,
        peak_azimuth_deg=float(geometry(sat, t_peak)[0]),
        rise_azimuth_deg=float(geometry(sat, rise)[0]),
        set_azimuth_deg=float(geometry(sat, setting)[0]),
        epoch_utc="2026-01-01T00:00:00Z", rise_epoch_offset_s=rise,
        set_epoch_offset_s=setting, duration_s=setting - rise,
        peak_time_s=t_peak - rise, sample_interval_s=dt, mask_elevation_deg=mask_deg,
        ground_latitude_deg=latitude, ground_longitude_deg=longitude,
        orbit_source="synthetic SGP4 reference orbit"))
