"""Freeze reference values from the Python ITU functions for the MATLAB self-test.

The MATLAB side cannot call Python, and MATLAB itself may not be launchable when
the port is written. Storing inputs and expected outputs here turns numeric
parity into a check tools/selftest.m can run on its own, at any later date.

Usage: python gen_golden.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

# Import root is the directory containing Ray_Tracing/, so the
# Ray_Tracing.src.* package paths resolve however this is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import Ray_Tracing.src.utils.atmospheric_utils as A  # noqa: E402
import Ray_Tracing.src.utils.ionospheric_utils as I  # noqa: E402
import Ray_Tracing.src.utils.orbital_utils as O  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "golden_values.mat"

FREQS = [1.9e9, 2e9, 6e9, 11e9, 11.7e9, 20e9, 30e9]
ELEVS = [5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 90.0]
TECS = [3.0, 10.0, 30.0, 60.0]
RAINS = [0.0, 5.0, 25.0, 50.0]
RHOS = [2.0, 7.5, 15.0, 20.0]


def grid(*axes):
    return [x.ravel() for x in np.meshgrid(*axes, indexing="ij")]


def main():
    g = {}

    # --- ionosphere -------------------------------------------------------
    v, e = grid(np.array(TECS), np.array(ELEVS))
    g["iono_in_vtec"] = v.reshape(-1, 1)
    g["iono_in_elev"] = e.reshape(-1, 1)
    g["iono_mapping"] = np.array([I.mapping_function(x)[0] for x in e]).reshape(-1, 1)
    g["iono_slant_tec"] = np.array([I.slant_tec(a, b) for a, b in zip(v, e)]).reshape(-1, 1)

    f, v2, e2 = grid(np.array(FREQS), np.array(TECS), np.array(ELEVS))
    g["iono_in_freq3"] = f.reshape(-1, 1)
    g["iono_in_vtec3"] = v2.reshape(-1, 1)
    g["iono_in_elev3"] = e2.reshape(-1, 1)
    g["iono_faraday"] = np.array([I.faraday_rotation_angle(a, b, c)
                                  for a, b, c in zip(v2, f, e2)]).reshape(-1, 1)
    g["iono_group_delay"] = np.array([I.ionospheric_group_delay(a, b, c)
                                      for a, b, c in zip(v2, f, e2)]).reshape(-1, 1)

    # --- atmosphere -------------------------------------------------------
    fg, rho = grid(np.array(FREQS) / 1e9, np.array(RHOS))
    go, gw = zip(*[A.gaseous_specific_attenuation(a, 288.15, 1013.25, b)
                   for a, b in zip(fg, rho)])
    g["atm_in_fghz"] = fg.reshape(-1, 1)
    g["atm_in_rho"] = rho.reshape(-1, 1)
    g["atm_gamma_o"] = np.array(go).reshape(-1, 1)
    g["atm_gamma_w"] = np.array(gw).reshape(-1, 1)

    f2, e3, r2 = grid(np.array(FREQS), np.array(ELEVS), np.array(RHOS))
    g["atm_in_freq3"] = f2.reshape(-1, 1)
    g["atm_in_elev3"] = e3.reshape(-1, 1)
    g["atm_in_rho3"] = r2.reshape(-1, 1)
    g["atm_gas_db"] = np.array([A.gaseous_attenuation_db(a, b, 288.15, 1013.25, c)
                                for a, b, c in zip(f2, e3, r2)]).reshape(-1, 1)
    g["atm_scint_sigma"] = np.array([A.scintillation_sigma_db(a, b, 288.15, c)
                                     for a, b, c in zip(f2, e3, r2)]).reshape(-1, 1)

    f3, r3, e4 = grid(np.array(FREQS), np.array(RAINS), np.array(ELEVS))
    g["rain_in_freq"] = f3.reshape(-1, 1)
    g["rain_in_rate"] = r3.reshape(-1, 1)
    g["rain_in_elev"] = e4.reshape(-1, 1)
    g["rain_db"] = np.array([A.rain_slant_attenuation_db(a, b, c)
                             for a, b, c in zip(f3, r3, e4)]).reshape(-1, 1)
    kk, aa = zip(*[A.rain_coefficients_p838(a) for a in f3])
    g["rain_k"] = np.array(kk).reshape(-1, 1)
    g["rain_alpha"] = np.array(aa).reshape(-1, 1)

    g["wet_refractivity"] = np.array([A._wet_refractivity(288.15, x)
                                      for x in RHOS]).reshape(-1, 1)
    g["wet_in_rho"] = np.array(RHOS).reshape(-1, 1)

    # --- orbit ------------------------------------------------------------
    alt, e5 = grid(np.array([600.0, 1200.0]), np.array(ELEVS))
    g["orb_in_alt"] = alt.reshape(-1, 1)
    g["orb_in_elev"] = e5.reshape(-1, 1)
    g["orb_slant_range"] = np.array([O.slant_range_m(a, b)
                                     for a, b in zip(alt, e5)]).reshape(-1, 1)

    azc = np.arange(0.0, 360.0, 15.0)
    g["orb_in_az_compass"] = azc.reshape(-1, 1)
    g["orb_scene_azimuth"] = np.array([O.compass_to_scene_azimuth(a)
                                       for a in azc]).reshape(-1, 1)
    enu = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [3.0, -4.0, 5.0]])
    g["orb_in_enu"] = enu
    g["orb_enu_scene"] = np.array([O.rotate_enu_to_scene(v_) for v_ in enu])

    savemat(OUT, g, do_compression=True)
    n = sum(v.size for k, v in g.items() if k.startswith(("iono_", "atm_", "rain_", "orb_", "wet_"))
            and not k.startswith(("iono_in", "atm_in", "rain_in", "orb_in", "wet_in")))
    print(f"wrote {OUT}")
    print(f"  {len(g)} arrays, {n} reference values")


if __name__ == "__main__":
    main()
