"""Atmospheric loss models for NTN ray tracing (ITU-R P.676/P.838/P.618).

OFF BY DEFAULT (ATMOSPHERE_ENABLED=false in every shipped config).

Gaseous, rain and scintillation losses, applied post-RT as a field-amplitude
scaling amps * 10^(-A_total/20). Uniform across paths, so it cancels in the
K-factor. See docs/physics.md.
"""

import numpy as np

# ─── Physical constants ────────────────────────────────────────────────────────
C = 2.99792458e8   # speed of light [m/s]  (CODATA 2018)

# Equivalent zenith scale heights for gaseous absorption (P.676 Annex 2)
H_O2_KM   = 6.0   # oxygen equivalent height [km]
H_H2O_KM  = 2.1   # water-vapour equivalent height [km] (standard conditions)

# Minimum elevation angle for cosecant mapping validity [deg]
_MIN_ELEV_DEG = 5.0


# ─── Scenario presets ─────────────────────────────────────────────────────────

# Surface weather scenarios:
#   (T_K, P_hPa, rho_mean_g_m3, rho_std_g_m3, h_station_km)
# rho is water-vapour density; log-normal spread models diurnal / seasonal variation.
# std/mean ≈ 0.3 gives σ_ln ≈ 0.29 — mild day-to-day variability.
WEATHER_PRESETS = {
    "cold_dry"         : (263.15,  900.0,  2.0,  0.6, 0.0),   # winter, continental
    "temperate_moderate": (288.15, 1013.25,  7.5,  2.2, 0.0),  # mid-lat reference
    "warm_humid"       : (303.15, 1010.0, 15.0,  4.5, 0.0),   # summer, coastal
    "tropical"         : (303.15, 1010.0, 20.0,  6.0, 0.0),   # tropical/equatorial
    "high_altitude"    : (278.15,  750.0,  3.0,  0.9, 2.0),   # mountain station
}

# Rain climate zones:
#   (R_mean_mm_h, R_std_mm_h, p_rain)
# R_mean / R_std are the conditional lognormal parameters (given rain occurs).
# p_rain is the fraction of time precipitation is present.
# These loosely correspond to ITU-R P.838 rain climate zones.
RAIN_PRESETS = {
    "arid"              : ( 2.0,   3.0, 0.01),  # desert / arid regions
    "temperate_maritime": (10.0,  15.0, 0.06),  # NW Europe, Pacific NW US
    "temperate_continental": (8.0, 12.0, 0.04), # central Europe / continental US
    "subtropical"       : (18.0,  25.0, 0.08),  # SE US, Mediterranean summer
    "tropical"          : (35.0,  50.0, 0.12),  # tropics / equatorial
    "monsoon"           : (50.0,  70.0, 0.15),  # South / SE Asia monsoon belt
}


# ─── ITU-R P.676-12 Annex 2 — Gaseous Absorption ─────────────────────────────

def _phi(rp: float, theta: float, a: float, b: float, c: float, d: float) -> float:
    """Auxiliary function φ used in the P.676-12 Annex 2 oxygen expression.

    φ(ρₚ, θ, a, b, c, d) = ρₚᵃ exp[b(1−ρₚ)] θᶜ exp[d(1−θ)]
    """
    return (rp ** a) * np.exp(b * (1.0 - rp)) * (theta ** c) * np.exp(d * (1.0 - theta))


def gaseous_specific_attenuation(f_ghz: float,
                                  T_K: float = 288.15,
                                  P_hPa: float = 1013.25,
                                  rho_g_m3: float = 7.5) -> tuple[float, float]:
    """Oxygen and water-vapour specific attenuation [dB/km], P.676-12 Annex 2.

    Valid 1-350 GHz, within 1 dB/km of the line-by-line calculation under
    typical atmospheric conditions.
    """
    rp    = P_hPa / 1013.25          # normalised pressure
    theta = 300.0 / T_K              # normalised inverse temperature
    f     = float(f_ghz)

    # ── Oxygen ────────────────────────────────────────────────────────────────
    # Table 2 coefficients (P.676-12 Annex 2, equations 22a–22f)
    xi1 = _phi(rp, theta,  0.0717, -1.8132,  0.0156, -1.6515)
    xi2 = _phi(rp, theta,  0.5146, -4.6368, -0.1921, -5.7416)
    xi3 = _phi(rp, theta,  0.3414, -6.5851,  0.2130, -8.5854)

    gamma_o = (
        7.2 / (f**2 + 0.34)
        + 0.62 * xi3 / ((54.0 - f)**2 + 1.16 * xi1)
    ) * f**2 * rp**2 * theta**2 * 1e-3

    # ── Water vapour ──────────────────────────────────────────────────────────
    # Resonance line contributions (P.676-12 Annex 2, equations 23a–23j)
    # The dominant lines for satellite-link frequencies:
    #   22.235 GHz  (critical for L/S/C/Ku/Ka)
    #  183.310 GHz  (millimetre-wave)
    #  321.226 GHz, 325.153 GHz, 380 GHz, 448 GHz, 557 GHz, 752 GHz  (mm/sub-mm)
    rho = float(rho_g_m3)
    eta1 = 0.955 * rp * theta**0.68 + 0.006 * rho
    eta2 = 0.735 * rp * theta**0.50 + 0.0353 * theta**4 * rho

    def _g(fi: float) -> float:
        """Shape correction factor g(f, fi) = 1 + ((f−fi)/(f+fi))²"""
        return 1.0 + ((f - fi) / (f + fi))**2

    # Individual resonance terms
    denom_22  = (f - 22.235)**2  + 9.42  * eta1**2
    denom_183 = (f - 183.310)**2 + 11.14 * eta1**2
    denom_321 = (f - 321.226)**2 +  6.29 * eta1**2
    denom_325 = (f - 325.153)**2 +  9.22 * eta1**2
    denom_380 = (f - 380.000)**2
    denom_448 = (f - 448.000)**2
    denom_557 = (f - 557.000)**2
    denom_752 = (f - 752.000)**2
    denom_1780= (f - 1780.00)**2

    # Guard against exact resonance (f == fi would give division by zero)
    _eps = 1e-10

    t1  = 3.98   * eta1 * np.exp( 2.23 * (1 - theta)) / (denom_22  + _eps) * _g(22.235)
    t2  = 11.96  * eta1 * np.exp( 0.70 * (1 - theta)) / (denom_183 + _eps)
    t3  = 0.081  * eta1 * np.exp( 6.44 * (1 - theta)) / (denom_321 + _eps)
    t4  = 3.66   * eta1 * np.exp( 1.60 * (1 - theta)) / (denom_325 + _eps)
    t5  = 25.37  * eta1 * np.exp( 1.09 * (1 - theta)) / (denom_380 + _eps)
    t6  = 17.40  * eta1 * np.exp( 1.46 * (1 - theta)) / (denom_448 + _eps)
    t7  = 844.6  * eta1 * np.exp( 0.17 * (1 - theta)) / (denom_557 + _eps) * _g(557.0)
    t8  = 290.0  * eta1 * np.exp( 0.41 * (1 - theta)) / (denom_752 + _eps) * _g(752.0)
    t9  = 83328. * eta2 * np.exp( 0.99 * (1 - theta)) / (denom_1780+ _eps) * _g(1780.0)

    gamma_w = (t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9) \
              * f**2 * rho * theta**1.5 * np.exp(2.143 * (1.0 - theta)) * 1e-4

    return float(max(0.0, gamma_o)), float(max(0.0, gamma_w))


def gaseous_attenuation_db(freq_hz: float,
                            elev_deg: float,
                            T_K: float = 288.15,
                            P_hPa: float = 1013.25,
                            rho_g_m3: float = 7.5,
                            h_station_km: float = 0.0) -> float:
    """Slant-path gaseous attenuation [dB] for a ground-satellite link.

    Cosecant mapping A_slant = A_zenith/sin(theta), floored at 5 deg elevation
    to avoid divergence at grazing angles, with the zenith term built from
    equivalent scale heights A_zenith = gamma_o*h_o + gamma_w*h_w
    (h_o = 6 km, h_w = 2.1 km; station height reduces both proportionally).
    """
    f_ghz = freq_hz / 1e9
    gamma_o, gamma_w = gaseous_specific_attenuation(f_ghz, T_K, P_hPa, rho_g_m3)

    # Effective scale heights above the station
    h_o_eff = max(0.0, H_O2_KM  - h_station_km)
    h_w_eff = max(0.0, H_H2O_KM - h_station_km)

    A_zenith = gamma_o * h_o_eff + gamma_w * h_w_eff   # [dB]

    # Cosecant mapping (clamp elevation to avoid numerical explosion)
    sin_el   = np.sin(np.radians(max(elev_deg, _MIN_ELEV_DEG)))
    return float(A_zenith / sin_el)


# ─── ITU-R P.838-3 — Rain Specific Attenuation ────────────────────────────────

# Tabulated (k_H, α_H, k_V, α_V) from ITU-R P.838-3 Table 1.
# Log-log interpolation is used rather than the polynomial-fit formulas so that
# the reference values are reproduced exactly at the tabulated frequencies.
# Frequency axis in GHz (log10 for interpolation).
_P838_F_GHZ = np.array([
    1,   2,   4,   6,   7,   8,  10,  12,  15,  20,
   25,  30,  35,  40,  45,  50,  60,  70,  80,  90, 100
], dtype=float)

_P838_KH = np.array([
    3.87e-5, 1.54e-4, 6.50e-4, 1.75e-3, 3.01e-3, 4.54e-3, 1.217e-2,
    2.386e-2, 4.481e-2, 9.164e-2, 0.1571, 0.2403, 0.3374, 0.4495,
    0.5765, 0.7272, 1.119, 1.702, 2.397, 3.144, 3.950
], dtype=float)

_P838_AH = np.array([
    0.912, 0.963, 1.121, 1.308, 1.332, 1.327, 1.264,
    1.185, 1.128, 1.065, 1.030, 1.000, 0.979, 0.967,
    0.955, 0.942, 0.923, 0.908, 0.893, 0.876, 0.869
], dtype=float)

_P838_KV = np.array([
    3.52e-5, 1.38e-4, 5.91e-4, 1.55e-3, 2.65e-3, 3.95e-3, 1.001e-2,
    1.963e-2, 3.545e-2, 7.158e-2, 0.1218, 0.1866, 0.2634, 0.3505,
    0.4495, 0.5765, 0.8553, 1.251, 1.728, 2.228, 2.785
], dtype=float)

_P838_AV = np.array([
    0.880, 0.923, 1.075, 1.265, 1.312, 1.310, 1.260,
    1.177, 1.129, 1.065, 1.030, 1.000, 0.979, 0.967,
    0.955, 0.942, 0.923, 0.908, 0.893, 0.876, 0.869
], dtype=float)

_P838_LOG10_F = np.log10(_P838_F_GHZ)
_P838_LOG10_KH = np.log10(_P838_KH)
_P838_LOG10_KV = np.log10(_P838_KV)


def rain_coefficients_p838(freq_hz: float,
                            pol_tilt_deg: float = 45.0) -> tuple[float, float]:
    """Rain coefficients (k, alpha) for gamma_R = k*R^alpha, ITU-R P.838-3."""
    lg10_f = float(np.log10(freq_hz / 1e9))
    lg10_f = np.clip(lg10_f, _P838_LOG10_F[0], _P838_LOG10_F[-1])

    # Log-log interpolation for k (linear for α)
    kH = 10.0 ** float(np.interp(lg10_f, _P838_LOG10_F, _P838_LOG10_KH))
    kV = 10.0 ** float(np.interp(lg10_f, _P838_LOG10_F, _P838_LOG10_KV))
    aH = float(np.interp(lg10_f, _P838_LOG10_F, _P838_AH))
    aV = float(np.interp(lg10_f, _P838_LOG10_F, _P838_AV))

    tau_rad = np.radians(pol_tilt_deg)
    cos2tau = np.cos(2.0 * tau_rad)

    k     = (kH + kV + (kH - kV) * cos2tau) / 2.0
    alpha = (kH * aH + kV * aV + (kH * aH - kV * aV) * cos2tau) / (2.0 * k)

    return float(k), float(alpha)


def rain_slant_attenuation_db(freq_hz: float,
                               R_mm_h: float,
                               elev_deg: float,
                               h_station_km: float = 0.0,
                               h_rain_km: float = 3.36,
                               pol_tilt_deg: float = 45.0) -> float:
    """Slant-path rain attenuation [dB], simplified P.618-13 Sec 2.2.1 model."""
    if R_mm_h <= 0.0 or h_rain_km <= h_station_km:
        return 0.0

    k, alpha  = rain_coefficients_p838(freq_hz, pol_tilt_deg)
    gamma_R   = k * (R_mm_h ** alpha)                              # [dB/km]

    sin_el    = np.sin(np.radians(max(elev_deg, _MIN_ELEV_DEG)))
    L_S       = (h_rain_km - h_station_km) / sin_el               # [km]

    return float(gamma_R * L_S)


# ─── ITU-R P.618-13 §2.4 — Tropospheric Scintillation ─────────────────────────

def _wet_refractivity(T_K: float, rho_g_m3: float) -> float:
    """Surface wet component of radio refractivity N_wet."""
    P_w = rho_g_m3 * T_K / 216.7     # [hPa]
    return 72.0 * P_w / T_K + 3.75e5 * P_w / T_K**2


def scintillation_sigma_db(freq_hz: float,
                            elev_deg: float,
                            T_K: float = 288.15,
                            rho_g_m3: float = 7.5,
                            D_ant_m: float = 0.0) -> float:
    """Tropospheric scintillation fade standard deviation [dB], P.618-13 Sec 2.4.1."""
    _K_S = 0.00978    # calibration constant [dB · N-unit^(-0.4588) · GHz^(-7/12)]
    _H_TUR_M = 1000.0  # turbulent layer height [m]

    f_ghz = freq_hz / 1e9
    N_wet = _wet_refractivity(T_K, rho_g_m3)
    N_wet = max(N_wet, 1e-3)   # guard

    # Reference sigma (point antenna, zenith)
    sigma_ref = _K_S * N_wet**0.4588 * f_ghz**(7.0 / 12.0)

    # Aperture averaging (only matters for large dish antennas)
    if D_ant_m > 0.0:
        lam    = C / freq_hz                              # wavelength [m]
        D_F    = np.sqrt(lam * _H_TUR_M)                 # Fresnel zone radius [m]
        if D_ant_m < D_F:
            eta = np.sqrt(D_ant_m / D_F)
            sigma_ref *= (1.0 / np.sqrt(eta)) if eta > 0 else 1.0

    # Elevation scaling
    sin_el      = np.sin(np.radians(max(elev_deg, _MIN_ELEV_DEG)))
    sigma_scint = sigma_ref / sin_el**1.2

    return float(max(0.0, sigma_scint))


def sample_scintillation_db(freq_hz: float,
                             elev_deg: float,
                             rng: np.random.Generator,
                             T_K: float = 288.15,
                             rho_g_m3: float = 7.5,
                             D_ant_m: float = 0.0) -> tuple[float, float]:
    """Sample one scintillation fade from N(0, sigma^2); returns (fade, sigma) [dB].

    Positive is enhancement, negative is fade.
    """
    sigma_scint = scintillation_sigma_db(freq_hz, elev_deg, T_K, rho_g_m3, D_ant_m)
    scint_db    = float(rng.normal(0.0, sigma_scint))
    return scint_db, sigma_scint


def compute_atmospheric_effects(freq_hz: float,
                                elev_deg: float,
                                rho_g_m3: float,
                                R_mm_h: float,
                                rng: np.random.Generator,
                                T_K: float = 288.15,
                                P_hPa: float = 1013.25,
                                h_station_km: float = 0.0,
                                h_rain_km: float = 3.36,
                                pol_tilt_deg: float = 45.0,
                                D_ant_m: float = 0.0,
                                include_scintillation: bool = True) -> dict:
    """One realisation of all three atmospheric effects."""
    # ── 1. Gaseous absorption (deterministic at fixed ρ) ─────────────────────
    A_gas_db = gaseous_attenuation_db(
        freq_hz, elev_deg, T_K, P_hPa, rho_g_m3, h_station_km
    )

    # ── 2. Rain attenuation (deterministic at fixed R; 0 → no rain) ──────────
    A_rain_db = rain_slant_attenuation_db(
        freq_hz, R_mm_h, elev_deg, h_station_km, h_rain_km, pol_tilt_deg
    )

    # ── 3. Tropospheric scintillation (stochastic — fast per-batch fade) ──────
    if include_scintillation:
        scint_db, sigma_scint = sample_scintillation_db(
            freq_hz, elev_deg, rng, T_K, rho_g_m3, D_ant_m
        )
    else:
        scint_db    = 0.0
        sigma_scint = 0.0

    total_loss_db = A_gas_db + A_rain_db + (-scint_db)

    return {
        "rho_sample"    : float(rho_g_m3),
        "A_gas_db"      : float(A_gas_db),
        "R_sample_mm_h" : float(R_mm_h),
        "A_rain_db"     : float(A_rain_db),
        "scint_db"      : float(scint_db),
        "sigma_scint_db": float(sigma_scint),
        "total_loss_db" : float(total_loss_db),
    }


def apply_atmospheric_corrections(taus: np.ndarray,
                                   amps: np.ndarray,
                                   total_loss_db: float) -> dict:
    """Apply one total atmospheric loss uniformly to every CIR path."""
    amplitude_scale = float(10.0 ** (-total_loss_db / 20.0))
    amps_mod        = np.asarray(amps, dtype=complex) * amplitude_scale
    return {
        "taus_mod"       : np.asarray(taus),
        "amps_mod"       : amps_mod,
        "amplitude_scale": amplitude_scale,
    }


def atmospheric_deterministic_summary(freq_hz: float,
                                      elev_deg: float,
                                      rho_g_m3: float,
                                      R_mm_h: float,
                                      T_K: float = 288.15,
                                      P_hPa: float = 1013.25,
                                      h_station_km: float = 0.0,
                                      h_rain_km: float = 3.36,
                                      pol_tilt_deg: float = 45.0) -> None:
    """
    Print a summary of the atmospheric model with deterministic (fixed) inputs.

    Shows the exact loss values that will be applied for the given fixed ρ and R,
    plus the scintillation sigma (the only stochastic component remaining).
    """
    f_ghz = freq_hz / 1e9
    k, alpha = rain_coefficients_p838(freq_hz, pol_tilt_deg)
    sin_el   = np.sin(np.radians(max(elev_deg, _MIN_ELEV_DEG)))
    L_S      = max(0.0, (h_rain_km - h_station_km)) / sin_el

    A_gas  = gaseous_attenuation_db(freq_hz, elev_deg, T_K, P_hPa, rho_g_m3, h_station_km)
    A_rain = rain_slant_attenuation_db(freq_hz, R_mm_h, elev_deg, h_station_km,
                                       h_rain_km, pol_tilt_deg)
    sigma_sc = scintillation_sigma_db(freq_hz, elev_deg, T_K, rho_g_m3)
    g0_o, g0_w = gaseous_specific_attenuation(f_ghz, T_K, P_hPa, rho_g_m3)

    print(f"\n{'─'*64}")
    print(f"  Atmospheric propagation model summary  [deterministic inputs]")
    print(f"  Frequency   : {f_ghz:.3f} GHz")
    print(f"  Elevation   : {elev_deg:.1f}°")
    print(f"{'─'*64}")

    print(f"\n  1) Gaseous absorption (ITU-R P.676-12)  "
          f"T={T_K:.0f} K  P={P_hPa:.0f} hPa")
    print(f"     ρ_water_vapour = {rho_g_m3:.2f} g/m³  (fixed)")
    print(f"     γ_O₂={g0_o:.4f} dB/km   γ_H₂O={g0_w:.4f} dB/km")
    print(f"     A_gas = {A_gas:.3f} dB")

    print(f"\n  2) Rain attenuation (ITU-R P.838-3 + P.618-13)")
    print(f"     R = {R_mm_h:.2f} mm/h  (fixed{'  →  clear sky' if R_mm_h == 0 else ''})")
    print(f"     k={k:.5f}  α={alpha:.4f}  L_slant_rain={L_S:.2f} km  pol_tilt={pol_tilt_deg:.0f}°")
    print(f"     A_rain = {A_rain:.3f} dB")

    print(f"\n  3) Tropospheric scintillation (ITU-R P.618-13)  [stochastic]")
    print(f"     σ_scint = {sigma_sc:.4f} dB")
    print(f"     ±1σ fades span ±{sigma_sc:.3f} dB  "
          f"|  99% interval ≈ ±{2.576*sigma_sc:.3f} dB")
    print(f"\n  Fixed total deterministic loss (excl. scintillation): "
          f"{A_gas + A_rain:.3f} dB")
    print(f"{'─'*64}\n")
