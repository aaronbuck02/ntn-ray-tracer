"""Widen ITU material frequency ranges past their measured bands.

Sionna raises ValueError from itu_material() for any material used outside the
(fmin, fmax) band ITU-R P.2040 measured it over. The three ground materials are
capped at 1-10 GHz, which blocks the 11 GHz and 20 GHz campaigns. Widening the
cap extrapolates the a*f^b / c*f^d laws past their validated range -- state that
wherever results above 10 GHz are reported.
"""

from sionna.rt.radio_materials.itu import ITU_MATERIALS_PROPERTIES

GROUND = ("very_dry_ground", "medium_dry_ground", "wet_ground")   # measured 1-10 GHz only


def widen(fmax_ghz=100.0, names=GROUND):
    """Extend `names` up to `fmax_ghz`. Idempotent; returns the names changed."""
    changed = []
    for name in names:
        props = ITU_MATERIALS_PROPERTIES[name]        # mutate in place: Sionna aliases this dict
        (lo, hi), coeffs = next(iter(props.items()))
        if hi >= fmax_ghz:
            continue
        props.clear()
        props[(lo, fmax_ghz)] = coeffs
        changed.append(name)
    return changed
