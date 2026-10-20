"""RHCP/LHCP antenna patterns and their registration with Sionna.

test_cp_smoke.py is the executable check for this module. See docs/physics.md.
"""

import math
import numpy as np
import drjit as dr
import mitsuba as mi

from sionna.rt import AntennaPattern, register_antenna_pattern
from sionna.rt.antenna_pattern import (
    v_iso_pattern,
    v_dipole_pattern,
    v_hw_dipole_pattern,
    v_tr38901_pattern,
)


# Single circularly polarized pattern

class CircularlyPolarizedPattern(AntennaPattern):
    """Single circularly polarized pattern wrapping a v-pol amplitude function.

    `v_pattern_fn` is any of v_iso_pattern, v_dipole_pattern, v_hw_dipole_pattern
    or v_tr38901_pattern; `handedness` is "RHCP" (default) or "LHCP". Assign the
    instance to an array's `.antenna_pattern`.
    """

    def __init__(self, v_pattern_fn, handedness: str = "RHCP"):
        super().__init__()
        if handedness not in ("RHCP", "LHCP"):
            raise ValueError(f"handedness must be 'RHCP' or 'LHCP', got {handedness!r}")
        self._v_fn      = v_pattern_fn
        self._handedness = handedness
        # sign = −1 → RHCP (C_φ = −jA),  sign = +1 → LHCP (C_φ = +jA)
        self._sign = -1.0 if handedness == "RHCP" else 1.0

    @property
    def patterns(self):
        v_fn = self._v_fn
        sign = self._sign
        s    = 1.0 / math.sqrt(2.0)   # normalization scalar (Python float)

        def _cp_pattern(theta, phi):
            # v_fn returns Complex2f(amplitude, 0) for the standard v_* patterns
            A = v_fn(theta, phi)
            a, b = dr.real(A), dr.imag(A)
            c_theta = mi.Complex2f(a * s, b * s)
            # sign*j*(a + jb) = Complex2f(-sign*b, sign*a)
            c_phi = mi.Complex2f(-sign * b * s, sign * a * s)

            return c_theta, c_phi

        return [_cp_pattern]

    def __repr__(self):
        return f"CircularlyPolarizedPattern(handedness={self._handedness!r})"


# Registration helpers

def register_cp_patterns():
    """Register {rhcp,lhcp}_{iso,dipole,hw_dipole,tr38901} as PlanarArray strings."""
    _entries = [
        ("rhcp_iso",       v_iso_pattern,      "RHCP"),
        ("lhcp_iso",       v_iso_pattern,      "LHCP"),
        ("rhcp_dipole",    v_dipole_pattern,   "RHCP"),
        ("lhcp_dipole",    v_dipole_pattern,   "LHCP"),
        ("rhcp_hw_dipole", v_hw_dipole_pattern,"RHCP"),
        ("lhcp_hw_dipole", v_hw_dipole_pattern,"LHCP"),
        ("rhcp_tr38901",   v_tr38901_pattern,  "RHCP"),
        ("lhcp_tr38901",   v_tr38901_pattern,  "LHCP"),
    ]
    for name, fn, hand in _entries:
        _fn, _hand = fn, hand   # capture loop vars
        register_antenna_pattern(name, lambda f=_fn, h=_hand, **kw: CircularlyPolarizedPattern(f, h))
    print("Registered CP patterns:", [e[0] for e in _entries])


# Validation

def validate_cp_pattern(pattern, theta_deg: float = 45.0, phi_deg: float = 0.0):
    """Check CP at one direction: equal amplitudes, 90 deg phase, AR ~ 0 dB.

    Defaults to theta=45 deg to stay off the pole singularities.
    Returns dicts with amp_ratio, phase_diff_deg, axial_ratio_dB.
    """
    theta = mi.Float(np.radians(theta_deg))
    phi   = mi.Float(np.radians(phi_deg))

    results = []
    for i, fn in enumerate(pattern.patterns):
        c_theta, c_phi = fn(theta, phi)

        # Convert to Python complex
        ct = complex(float(dr.real(c_theta)), float(dr.imag(c_theta)))
        cp = complex(float(dr.real(c_phi)),   float(dr.imag(c_phi)))

        amp_ratio     = abs(cp) / abs(ct) if abs(ct) > 1e-12 else float("inf")
        phase_diff    = np.degrees(np.angle(cp / ct)) if abs(ct) > 1e-12 else 0.0
        # Axial ratio formula valid for general elliptical polarization
        num = abs(ct + 1j * cp) + abs(ct - 1j * cp)
        den = abs(abs(ct + 1j * cp) - abs(ct - 1j * cp))
        ar_dB = 0.0 if den < 1e-12 else 20 * np.log10(num / den)

        label = f"Pattern[{i}]"
        print(f"\n{label}  (θ={theta_deg}°, φ={phi_deg}°)")
        print(f"  C_θ           = {ct:.6f}")
        print(f"  C_φ           = {cp:.6f}")
        print(f"  |C_φ|/|C_θ|  = {amp_ratio:.6f}  (ideal: 1.0)")
        print(f"  ∠C_φ − ∠C_θ  = {phase_diff:.2f}°  (ideal: ±90°)")
        print(f"  Axial Ratio   = {ar_dB:.4f} dB  (ideal: 0 dB)")

        results.append(dict(amp_ratio=amp_ratio, phase_diff_deg=phase_diff,
                            axial_ratio_dB=ar_dB))
    return results


# Entry point

if __name__ == "__main__":
    # 1. Validate math
    print("=" * 60)
    print("RHCP PATTERN VALIDATION (isotropic amplitude)")
    print("=" * 60)
    rhcp = CircularlyPolarizedPattern(v_iso_pattern, "RHCP")
    validate_cp_pattern(rhcp, theta_deg=45.0, phi_deg=0.0)

    print("\n" + "=" * 60)
    print("LHCP PATTERN VALIDATION (isotropic amplitude)")
    print("=" * 60)
    lhcp = CircularlyPolarizedPattern(v_iso_pattern, "LHCP")
    validate_cp_pattern(lhcp, theta_deg=45.0, phi_deg=0.0)

