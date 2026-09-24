"""Signal-processing helpers for Sionna RT ray-tracing post-processing.

Covers CIR statistics (delay/angular/Doppler spreads), the mechanism-wise
power split that defines K = P_los/P_nlos, and TX placement from either a
prescribed elevation or a real SGP4 pass sample (see orbital_utils.py).

See docs/physics.md.
"""

import numpy as np
from Ray_Tracing.src.utils.orbital_utils import compass_to_scene_azimuth, rotate_enu_to_scene


# ─── RMS Delay Spread ─────────────────────────────────────────────────────────

def rms_delay_spread(taus, amps):
    """Power-weighted mean excess delay and RMS delay spread [s] for one CIR."""
    t = np.asarray(taus, dtype=float)
    a = np.asarray(amps)
    if len(t) == 0:
        return 0.0, 0.0
    P   = np.abs(a) ** 2
    den = np.sum(P)
    if den == 0:
        return 0.0, 0.0
    tau0 = t[0]
    dt   = t - tau0
    mu   = np.dot(dt, P) / den
    var  = np.dot((dt - mu) ** 2, P) / den
    return float(mu), float(np.sqrt(var))


# ─── RMS Angular Spread ────────────────────────────────────────────────────────

def rms_angular_spread(angles_rad, amps):
    """Power-weighted circular mean and RMS angular spread [rad].

    Serves all four 3GPP angular parameters. TR 38.901 Sec 7.5 applies this
    circular form to zenith as well, although zenith does not wrap at
    +/-180 deg -- an assumption inherited from the spec, not derived here.
    """
    phi = np.asarray(angles_rad, dtype=float)
    a   = np.asarray(amps)
    if len(phi) == 0:
        return 0.0, 0.0
    P   = np.abs(a) ** 2
    den = np.sum(P)
    if den == 0:
        return 0.0, 0.0
    z = np.sum(P * np.exp(1j * phi)) / den
    R = np.clip(np.abs(z), 0.0, 1.0)   # fp guard: |z| <= 1 in exact math
    return float(np.angle(z)), float(np.sqrt(-2.0 * np.log(max(R, 1e-300))))


# ─── Complex Channel & Power Split ────────────────────────────────────────────

def complex_channel(taus, amps, freq):
    """Narrowband complex channel gain h = Σ a_k · exp(−j 2π f τ_k)."""
    a = np.asarray(amps, dtype=complex)
    t = np.asarray(taus, dtype=float)
    return np.sum(a * np.exp(-1j * 2 * np.pi * freq * t))


# Sionna's InteractionType constants are BIT FLAGS, not an ordinal sequence:
# NONE=0, SPECULAR=1, DIFFUSE=2, REFRACTION=4, DIFFRACTION=8.
IT_NONE        = 0
IT_SPECULAR    = 1
IT_DIFFUSE     = 2
IT_DIFFRACTION = 8
IT_REFRACTION  = 4


def split_by_interaction_type(taus, amps, inter_all_depths, freq,
                              incoherent=False):
    """Split the CIR into LoS and per-mechanism NLoS powers.

    Each path takes its *first* non-zero interaction as its primary type, so a
    path that diffracts then reflects counts as diffraction.

    ``incoherent`` drops the cross-terms, making the per-mechanism powers
    additive so fractions sum to 100 % -- use it for power breakdowns, not for
    narrowband power. P_los and P_nlos are always incoherent so that
    K = P_los/P_nlos keeps its Rician meaning.
    """
    a    = np.asarray(amps, dtype=complex)
    t    = np.asarray(taus, dtype=float)
    ph   = np.exp(-1j * 2 * np.pi * freq * t)
    iall = np.asarray(inter_all_depths, dtype=int)   # [max_depth, n_paths]

    n_paths = len(a)

    # Vectorised primary-type detection:
    #   nonzero_mask[d, p] = True  if depth d of path p has an interaction
    #   first_nonzero_depth = index of first True across depth axis (0 if none)
    nonzero_mask = iall != 0                             # [max_depth, n_paths]
    any_nonzero  = np.any(nonzero_mask, axis=0)          # [n_paths]

    primary_type = np.zeros(n_paths, dtype=int)          # default = 0 (LoS)
    if np.any(any_nonzero):
        idx = np.where(any_nonzero)[0]
        first_depth = np.argmax(nonzero_mask[:, idx], axis=0)   # [n_active]
        primary_type[idx] = iall[first_depth, idx]

    def _h_sq_coherent(mask):
        """Return |Σ a_k e^{-j2πfτ_k}|² for a path subset (coherent)."""
        if not np.any(mask):
            return 0.0
        return float(np.abs(np.sum(a[mask] * ph[mask])) ** 2)

    def _power_incoherent(mask):
        """Return (Σ |a_k|²) for a path subset (sum of path powers, additive)."""
        if not np.any(mask):
            return 0.0
        return float(np.sum(np.abs(a[mask])**2))

    # P_los and P_nlos always use incoherent sums — these define K = P_los/P_nlos
    los_mask  = primary_type == IT_NONE
    nlos_mask = ~los_mask
    P_los  = _power_incoherent(los_mask)
    P_nlos = _power_incoherent(nlos_mask)

    # Per-mechanism breakdown: use incoherent sums when requested so that
    # P_spec + P_diff + P_diffr + P_refr = P_nlos_incoherent (additive).
    _mech_fn = _power_incoherent if incoherent else _h_sq_coherent

    return {
        "P_los"        : P_los,
        "P_specular"   : _mech_fn(primary_type == IT_SPECULAR),
        "P_diffuse"    : _mech_fn(primary_type == IT_DIFFUSE),
        "P_diffraction": _mech_fn(primary_type == IT_DIFFRACTION),
        "P_refraction" : _mech_fn(primary_type == IT_REFRACTION),
        "P_nlos"       : P_nlos,
        "has_los"      : bool(np.any(los_mask)),
        "primary_type" : primary_type,
    }


def tx_position_from_elevation(elev_deg, slant_dist_m, scene_center_xy, radius, azimuth_deg=180.0):
    """TX position [x, y, z] and beam half-angle for one elevation/slant range.

    ``azimuth_deg`` is degrees from +X, counter-clockwise (scene convention,
    not compass).
    """
    elev_rad = np.radians(elev_deg)
    azim_rad = np.radians(azimuth_deg)

    ground_dist = slant_dist_m * np.cos(elev_rad)
    height      = slant_dist_m * np.sin(elev_rad)

    dx = ground_dist * np.cos(azim_rad)
    dy = ground_dist * np.sin(azim_rad)

    cx, cy = float(scene_center_xy[0]), float(scene_center_xy[1])

    # Half-angle whose ground footprint CONTAINS the scene circle, rather than
    # merely matching its area. The footprint is an ellipse with cross-range
    # semi-axis d*tan(theta) and along-range d*tan(theta)/sin(elev); the latter is
    # always the larger, so cross-range binds and solving d*tan(theta) = radius
    # covers both. An equal-area ellipse would instead fall short in cross-range
    # at every elevation, trading it away against the along-range stretch.
    beam_half_angle = np.arctan(radius / slant_dist_m)

    # Cast to Python float — mitsuba.Point3f rejects numpy float64
    return [float(cx + dx), float(cy + dy), float(height)], float(beam_half_angle)


def elevation_summary(elev_deg, slant_dist_m):
    """
    Print a short summary of TX geometry for a given elevation angle.
    """
    elev_rad    = np.radians(elev_deg)
    ground_dist = slant_dist_m * np.cos(elev_rad)
    height      = slant_dist_m * np.sin(elev_rad)
    print(
        f"  elev={elev_deg:4.0f}°  slant={slant_dist_m/1e3:.1f} km  "
        f"ground_dist={ground_dist/1e3:.2f} km  height={height/1e3:.2f} km"
    )


# ─── TX Position & Velocity from a Real SGP4 Pass Sample ──────────────────────

def tx_position_from_pass_sample(waypoint, scene_center_xy, radius,
                                  scene_east_heading_deg=90.0):
    """TX position for one waypoint of an SGP4-propagated pass.

    Replaces the fixed-azimuth assumption of tx_position_from_elevation: the
    waypoint's compass azimuth is rotated into the scene's math-angle
    convention via compass_to_scene_azimuth, and its true slant range used.
    """
    azimuth_scene_deg = compass_to_scene_azimuth(waypoint["az_deg"], scene_east_heading_deg)
    return tx_position_from_elevation(
        waypoint["elev_deg"], waypoint["range_m"], scene_center_xy, radius,
        azimuth_deg=azimuth_scene_deg,
    )


def tx_velocity_scene_frame(waypoint, scene_east_heading_deg=90.0):
    """Rotate a waypoint's ENU satellite velocity into the scene frame [m/s].

    Rotation only -- velocity needs no translation. ``scene_east_heading_deg``
    must match the value given to tx_position_from_pass_sample.
    """
    v_enu_m_s = np.asarray(waypoint["enu_vel_kms"], dtype=float) * 1e3
    return rotate_enu_to_scene(v_enu_m_s, scene_east_heading_deg)


# ─── Doppler-Modulated Time Series (post apply_doppler) ────────────────────────

def complex_channel_time_series(taus, amps_t, freq):
    """Narrowband complex channel gain h(t) per time step.

    ``taus`` is assumed quasi-static over the Doppler window; ``amps_t`` is
    [num_paths, num_time_steps] as returned by Paths.apply_doppler.
    """
    a = np.asarray(amps_t, dtype=complex)
    t = np.asarray(taus, dtype=float)
    phase = np.exp(-1j * 2 * np.pi * freq * t)          # [num_paths]
    return np.sum(a * phase[:, None], axis=0)           # -> [num_time_steps]


def doppler_psd(h_t, sampling_frequency):
    """Doppler PSD of a channel time series, via FFT; bins are zero-centred [Hz].

    ``sampling_frequency`` must be the one passed to apply_doppler.
    """
    h = np.asarray(h_t, dtype=complex)
    n = len(h)
    H = np.fft.fftshift(np.fft.fft(h))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sampling_frequency))
    psd = np.abs(H) ** 2
    return freqs, psd


def rms_doppler_spread_hz(h_t, sampling_frequency):
    """Power-weighted mean and RMS Doppler [Hz] from a PSD.
    """
    freqs, psd = doppler_psd(h_t, sampling_frequency)
    total = np.sum(psd)
    if total <= 0:
        return 0.0, 0.0
    f_mean = np.sum(freqs * psd) / total
    f_rms = np.sqrt(np.sum((freqs - f_mean) ** 2 * psd) / total)
    return float(f_mean), float(f_rms)


def doppler_mean_rms_from_paths(doppler_hz, amps):
    """Power-weighted mean and RMS Doppler [Hz] from per-path shifts.

    Closed-form equivalent of rms_doppler_spread_hz, taking Paths.doppler
    directly: no time-domain sampling, so no Nyquist choice to get wrong.
    Returns (0.0, 0.0) when there is no path power.
    """
    doppler_hz = np.asarray(doppler_hz, dtype=float)
    p = np.abs(np.asarray(amps)) ** 2
    total = np.sum(p)
    if total <= 0:
        return 0.0, 0.0
    f_mean = np.sum(doppler_hz * p) / total
    f_rms = np.sqrt(np.sum((doppler_hz - f_mean) ** 2 * p) / total)
    return float(f_mean), float(f_rms)
