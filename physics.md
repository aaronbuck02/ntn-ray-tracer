# Propagation and geometry models

Reference for the physics modules: `orbital_utils.py`, `ionospheric_utils.py`,
`atmospheric_utils.py`, `circular_polarization.py`. These carry the content that
used to live in their module docstrings.

---

## Orbital geometry — `orbital_utils.py`

SGP4 orbit propagation and topocentric geometry. Builds either a synthetic
reproducible circular orbit (default: 3GPP TR 38.821 reference LEO at 600 km) or
a real satellite from a published TLE, propagates it through
TEME → ECEF → ground-station ENU, and extracts azimuth, elevation, slant range
and range rate. `find_pass` locates one pass above a mask elevation;
`generate_waypoints` then solves exact elevation crossings along one leg of it.

### Two deliberate simplifications

Both are well inside channel-level accuracy and **not** suitable for precision
orbit determination:

- Earth is a sphere of radius `R_EARTH_KM`, matching the spherical thin-shell
  ionosphere used elsewhere in the pipeline.
- TEME → ECEF uses the IAU-1982 GMST formula and ignores polar motion and
  precession/nutation, which keeps the dependencies to `sgp4` + `numpy`.

### Frame conventions

Compass azimuth is degrees from North, clockwise. Scene azimuth is degrees from
+X, counter-clockwise. `compass_to_scene_azimuth` converts between them using
`SCENE_EAST_HEADING_DEG`, the compass heading of the scene's +X axis (default
90° = East). `rotate_enu_to_scene` applies the same rotation to a vector — used
for satellite velocity, which needs rotation but no translation.

Getting this wrong rotates the whole scene relative to the orbit, which shows up
as an azimuth-dependent bias in the angular spreads rather than an obvious error.

### Slant range

For a circular orbit at altitude *h*, range varies with elevation as

```
d(theta) = sqrt((Re + h)^2 - Re^2 cos^2(theta)) - Re sin(theta)
```

implemented as `slant_range_m`. At the TR 38.821 reference LEO (600 km) this
spans 1932 km at 10° down to 600 km at zenith — a 10.2 dB swing in free-space
loss that a fixed slant range omits entirely. Setting `SWEEP_ALTITUDE_KM` makes
a sweep use the true per-elevation range instead of `SLANT_DIST_M`.

---

## Ionosphere — `ionospheric_utils.py`

ITU-R P.531-15. **Off by default** — the shipped configs set
`IONOSPHERE_ENABLED=false`, so nothing here runs unless that is turned on. It is
disabled for the TR 38.811 cross-validation because the standard carries
ionospheric loss as a separate additive link-budget term; enabling it inside the
ray tracer and then comparing against the tables would double-count.

TEC is time-varying, so Faraday rotation Ω_F is a random variable. Per batch:

1. `sample_faraday_angle()` draws one (VTEC, Ω_F) pair from a log-normal TEC
   distribution.
2. `omega_F` goes to `scene_utils.build_scene_batch()`, which applies it as a
   roll offset on the TX antenna **before** tracing, so Sionna propagates the
   rotated field and projects it onto the TM/TE Fresnel basis itself.
3. `apply_ionospheric_corrections()` then adds group delay and phase advance to
   the CIR.

### Why rotate before the trace

It removes Sionna's Brewster null. Pure V-pol (TM) Fresnel coefficients vanish
at the Brewster angle, but after rotation the field arrives as
`cos(Ω_F)·V̂ + sin(Ω_F)·Ĥ`, and the TE part is non-zero at every angle. Applying
the rotation post-hoc to the CIR cannot recover the power that the null removed.

Ω_F ∝ TEC/f², so it is ~10–60° at L-band and negligible (<0.2°) at Ka.

---

## Troposphere — `atmospheric_utils.py`

**Off by default**, same reasoning and same config gate as the ionosphere.

Three effects, all from ITU-R closed forms:

| Effect | Model | Notes |
|---|---|---|
| Gaseous absorption | P.676-12 Annex 2 | O₂ + H₂O, valid 1–350 GHz; slant path by cosecant mapping `A = A_zenith / sin(theta)` |
| Rain | P.838-3 | specific attenuation `k·R^alpha`, path length from the P.839-4 mean rain height |
| Scintillation | P.618-13 Sec 2.4 | σ scales as `f^(7/12) / sin(theta)^1.2`; fade drawn from `N(0, sigma^2)` in dB |

All three live above the urban scene, so they apply equally to every path and
are handled post-RT as a **field-amplitude** scaling, not a power scaling:

```
amps_mod = amps * 10^(-A_total/20),    A_total = A_gas + A_rain + A_scint
```

Per batch: `compute_atmospheric_effects()` for the three losses, then
`apply_atmospheric_corrections()` on the solver output. Delays are untouched —
the refractometric group delay is <1 ns here, well under Sionna's CIR
resolution. `WEATHER_PRESETS` and `RAIN_PRESETS` carry the named scenarios.

Because the loss is uniform across paths, it cancels in the K-factor ratio and
leaves the PDP shape unchanged; the observable effect is on total received power
only.

---

## Polarization — `circular_polarization.py`

Registers RHCP/LHCP antenna patterns with Sionna. Two useful configurations:
circular polarization at both ends (`TX_POL_TYPE`/`RX_POL_TYPE` = `RHCP`), and
circular transmit into a complete dual-port receive with the port powers summed
(`cp_total`).

`cp_total` exists because a circularly polarized wave divides equally between
the local TE and TM modes at any interaction, independently of the orientation of
its plane of incidence. Summing the two port powers therefore evaluates the
polarization-averaged Fresnel and UTD kernels `½(|Γ_TE|² + |Γ_TM|²)` directly,
in one solve — which is what analytical models tabulate.

The obvious alternative, averaging two co-polar linear runs, is exact **only**
where the plane of incidence aligns with the V/H basis. That holds for a flat
ground but not for an arbitrarily oriented wall, where it discards the
cross-polar V→H conversion. Measured on ten arbitrarily oriented wall
reflections: `cp_total` reproduces the reference amplitudes to 0.00 dB mean
error, the linear average to −1.50 dB mean and −6.24 dB worst case.

`src/diagnostics/test_cp_smoke.py` is the executable check for this module.
