"""Ionospheric corrections for NTN ray tracing (ITU-R P.531-15).

OFF BY DEFAULT (IONOSPHERE_ENABLED=false in every shipped config).

Faraday rotation is applied as a TX antenna roll BEFORE tracing, not to the CIR
after, so Sionna projects the rotated field onto its own TE/TM basis and the
Brewster null disappears. See docs/physics.md.
"""

import numpy as np
from scipy import constants

# ─── Physical constants ────────────────────────────────────────────────────────
C      = constants.c          # speed of light [m/s]
R_EARTH = 6371e3              # mean Earth radius [m]
H_SHELL = 350e3               # ionospheric thin-shell height (F2 peak) [m]

# Ionospheric dispersion coefficient [m³/s²]
#   Group delay:    τ_iono = K_IONO × TEC [el/m²] / (c × f²)
#   Phase advance:  Δφ     = 2π × K_IONO × TEC [el/m²] / (c × f)
K_IONO = 40.308

# Faraday rotation coefficient [rad · Hz² / (T · el/m²)]
#   Ω_F = K_FARADAY × B_L [T] × TEC_slant [el/m²] / f² [Hz²]
K_FARADAY = 2.36e4


# ─── Representative TEC scenarios (VTEC_MEAN_TECU, VTEC_STD_TECU) ─────────────
TEC_PRESETS = {
    # (vtec_mean_tecu, vtec_std_tecu)
    # Coefficient of variation (std/mean) held at ~0.5, giving σ_ln ≈ 0.47
    "quiet_night_midlat"  : (3.0,   1.5),   # night, solar minimum, mid-lat
    "quiet_day_midlat"    : (10.0,  5.0),   # day, solar minimum, mid-lat
    "active_day_midlat"   : (30.0, 15.0),   # day, solar maximum, mid-lat
    "quiet_day_equatorial": (20.0, 10.0),   # equatorial — inherently higher TEC
    "storm_midlat"        : (60.0, 30.0),   # geomagnetic storm
}


# ─── TEC / Faraday sampling ───────────────────────────────────────────────────

def tec_lognormal_params(vtec_mean_tecu: float,
                          vtec_std_tecu: float,
                          elev_deg: float = 90.0,
                          sigma_h: float = 0.0,
                          Re: float = R_EARTH,
                          hs: float = H_SHELL) -> tuple[float, float]:
    """Log-normal (mu_ln, sigma_ln) of slant TEC at one elevation."""
    # Base vertical TEC log-normal params
    cv            = vtec_std_tecu / vtec_mean_tecu
    sigma_ln_vtec = float(np.sqrt(np.log(1.0 + cv**2)))
    mu_ln_vtec    = float(np.log(vtec_mean_tecu) - 0.5 * sigma_ln_vtec**2)

    # Spherical mapping function at this elevation
    M_sph, _ = mapping_function(elev_deg, Re=Re, hs=hs)
    ln_M      = float(np.log(M_sph))          # zero at zenith, positive below

    # Slant TEC distribution parameters
    mu_ln    = mu_ln_vtec + ln_M              # mean scales with M(θ)
    sigma_ln = sigma_ln_vtec + sigma_h * ln_M # std gains elevation-dependent spread

    return mu_ln, float(max(sigma_ln, 1e-6))  # guard against numerical zero


def sample_faraday_angle(vtec_mean_tecu: float,
                          vtec_std_tecu: float,
                          freq_hz: float,
                          elev_deg: float,
                          B_L_tesla: float,
                          rng: np.random.Generator,
                          sigma_h: float = 0.0) -> tuple[float, float]:
    """Draw one (omega_F [rad], slant TEC [TECU]) pair for this elevation.

    Sampling happens directly in slant-TEC space, so the mapping function is
    already absorbed into the distribution and needs no post-hoc scaling.
    """
    mu_ln, sigma_ln = tec_lognormal_params(
        vtec_mean_tecu, vtec_std_tecu,
        elev_deg=elev_deg, sigma_h=sigma_h,
    )
    # Draw directly from the slant TEC distribution (already elevation-adjusted)
    tec_slant_sample = float(rng.lognormal(mean=mu_ln, sigma=sigma_ln))  # TECU
    tec_slant_si     = tec_slant_sample * 1e16                           # el/m²
    omega_F = K_FARADAY * B_L_tesla * tec_slant_si / freq_hz**2
    return omega_F, tec_slant_sample


# ─── TEC mapping ──────────────────────────────────────────────────────────────

def slant_tec(vtec_tecu: float, elev_deg: float,
              Re: float = R_EARTH, hs: float = H_SHELL) -> float:
    """Vertical TEC [TECU] -> slant TEC [el/m^2], spherical thin-shell mapping."""
    MAX_SIN  = 0.9999                          # prevents cos(z_IPP) → 0
    sin_zipp = (Re / (Re + hs)) * np.cos(np.radians(elev_deg))
    sin_zipp = np.clip(sin_zipp, 0.0, MAX_SIN)
    M        = 1.0 / np.sqrt(1.0 - sin_zipp**2)   # spherical mapping function
    return vtec_tecu * 1e16 * M


# ─── Ionospheric effect magnitudes ────────────────────────────────────────────

def faraday_rotation_angle(vtec_tecu: float, freq_hz: float, elev_deg: float,
                            B_L_tesla: float = 2e-5) -> float:
    """Faraday rotation angle Ω_F [rad]."""
    tec_si = slant_tec(vtec_tecu, elev_deg)
    return K_FARADAY * B_L_tesla * tec_si / freq_hz**2


def ionospheric_group_delay(vtec_tecu: float, freq_hz: float,
                             elev_deg: float) -> float:
    """Extra propagation group delay introduced by the ionosphere [s]."""
    tec_si = slant_tec(vtec_tecu, elev_deg)
    return K_IONO * tec_si / (C * freq_hz**2)


# ─── Post-RT CIR correction (Stage 1) ────────────────────────────────────────

def apply_ionospheric_corrections(taus: np.ndarray, amps: np.ndarray,
                                   tec_slant_tecu: float, freq_hz: float) -> dict:
    """Apply ionospheric group delay and phase advance to a CIR post-RT."""
    # Convert slant TEC from TECU to SI [el/m²] and apply dispersion formulas
    tec_si    = tec_slant_tecu * 1e16                         # el/m²
    tau_iono  = K_IONO * tec_si / (C * freq_hz**2)           # group delay [s]
    delta_phi = 2.0 * np.pi * K_IONO * tec_si / (C * freq_hz)  # phase advance [rad]

    taus_mod = taus + tau_iono
    amps_mod = amps * np.exp(1j * delta_phi)

    return {
        "taus_mod" : taus_mod,
        "amps_mod" : amps_mod,
        "tau_iono" : tau_iono,
        "delta_phi": delta_phi,
    }


# ─── Convenience: parameter and scenario summary ──────────────────────────────

def mapping_function(elev_deg: float,
                     Re: float = R_EARTH, hs: float = H_SHELL) -> tuple[float, float]:
    """Return the spherical and flat-Earth mapping functions for a given elevation.

Useful for diagnostics — shows how much the flat approximation overestimates
slant TEC at low elevation angles.
    """
    sin_zipp   = np.clip((Re / (Re + hs)) * np.cos(np.radians(elev_deg)), 0.0, 0.9999)
    M_spherical = 1.0 / np.sqrt(1.0 - sin_zipp**2)
    M_flat      = 1.0 / np.clip(np.sin(np.radians(elev_deg)), 0.05, 1.0)
    return float(M_spherical), float(M_flat)


def ionospheric_summary(vtec_mean_tecu: float, vtec_std_tecu: float,
                         freq_hz: float, elev_deg: float,
                         B_L_tesla: float = 2e-5):
    """
    Print a summary of the ionospheric model setup and expected Ω_F statistics.
    Useful for sanity-checking before a simulation run.
    """
    mu_ln, sigma_ln = tec_lognormal_params(vtec_mean_tecu, vtec_std_tecu)

    vtec_p16 = np.exp(mu_ln - sigma_ln)   # ~16th percentile
    vtec_p50 = np.exp(mu_ln)              # median
    vtec_p84 = np.exp(mu_ln + sigma_ln)   # ~84th percentile

    omf = lambda v: np.degrees(faraday_rotation_angle(v, freq_hz, elev_deg, B_L_tesla))
    tau = lambda v: ionospheric_group_delay(v, freq_hz, elev_deg) * 1e9   # ns
    M_sph, M_flat = mapping_function(elev_deg)

    print(f"\n{'─'*60}")
    print(f"  Ionospheric model summary (Monte Carlo)")
    print(f"  VTEC distribution : LN(μ={vtec_mean_tecu} TECU, σ={vtec_std_tecu} TECU)")
    print(f"  Log-normal params : μ_ln={mu_ln:.3f}  σ_ln={sigma_ln:.3f}")
    print(f"  Frequency         : {freq_hz/1e9:.3f} GHz")
    print(f"  Elevation         : {elev_deg:.1f}°")
    print(f"  B_L               : {B_L_tesla*1e6:.1f} μT")
    print(f"  Mapping function  : M_spherical={M_sph:.3f}  "
          f"M_flat={M_flat:.3f}  "
          f"(flat overestimates by {(M_flat/M_sph - 1)*100:.0f}%)")
    print(f"{'─'*60}")
    print(f"  {'Percentile':>14}  {'VTEC [TECU]':>12}  {'Ω_F [°]':>10}  {'τ_iono [ns]':>12}")
    for pct, vtec in [("16th", vtec_p16), ("50th (median)", vtec_p50), ("84th", vtec_p84)]:
        print(f"  {pct:>14}  {vtec:>12.2f}  {omf(vtec):>10.2f}  {tau(vtec):>12.3f}")
    print(f"{'─'*60}\n")
