"""Gate checks of the Chiu & Roy analytical model against the RT wall ensemble.

Pure geometry: LoS probability from the closed form versus a ray test over the
generated scenes, plus the scene-radius convergence sweep. No Sionna, no GPU.
See docs/pipelines.md.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.special import erfc

import Ray_Tracing.src.utils.scene_gen_ppp as sg

C0 = 299792458.0
FC = 2e9
H_G = 1.5              # terminal height [m]
H_SAT_KM = 550.0

DEFAULT_REF_DIR = os.environ.get("CHIU_REF_DIR", "/path/to/chiu_roy_ref")


# ══════════════════════════════════════════════════════════════════════════════
#  Closed forms
# ══════════════════════════════════════════════════════════════════════════════

def scenario_params(scenario: str) -> tuple:
    """(gamma, lambda_b, l_f, lambda_1D) for 'urban'/'rural'."""
    p = sg.WALL_SCENARIOS[scenario]
    lam_1d = 2.0 * p["lam_b"] * p["l_f"] / np.pi
    return p["gamma"], p["lam_b"], p["l_f"], lam_1d


def plos_analytic(beta_deg, scenario: str, radius: float | None = None,
                  h_g: float = H_G) -> np.ndarray:
    """
    E[1_0], the LoS-state probability.

    ``radius=None`` gives eq. (27)/(57), the R_city -> inf limit the paper
    plots. A finite ``radius`` gives eq. (56), which retains the upper endpoint
    -- that is the form to compare against any finite simulated scene.
    """
    gamma, lam_b, l_f, _ = scenario_params(scenario)
    beta = np.radians(np.asarray(beta_deg, dtype=float))
    pref = np.sqrt(2.0 / np.pi) * lam_b * l_f * gamma / np.tan(beta)
    inner = erfc(h_g / (np.sqrt(2.0) * gamma))
    if radius is not None:
        z_top = h_g + radius * np.tan(beta)
        inner = inner - erfc(z_top / (np.sqrt(2.0) * gamma))
    return np.exp(-pref * inner)


def slant_range_m(beta_deg, alt_km: float = H_SAT_KM) -> np.ndarray:
    """Spherical-Earth slant range [m] (mirrors orbital_utils.slant_range_m)."""
    from Ray_Tracing.src.utils.orbital_utils import slant_range_m as _s
    return _s(alt_km, beta_deg)


# ══════════════════════════════════════════════════════════════════════════════
#  Geometry gate: ray-traced LoS over the wall ensemble
# ══════════════════════════════════════════════════════════════════════════════

def _los_clear(spec: sg.SceneSpec, beta_deg: float, alpha_deg: float = 0.0,
               h_g: float = H_G) -> bool:
    """
    True if the terminal->satellite ray clears every wall of ``spec``.

    Segment-crossing test in the horizontal plane, then a height test at the
    crossing -- the same rule as the reference model's Model.clear(), written
    independently against our SceneSpec. The ray is extended well past the
    scene so the far endpoint never truncates it.
    """
    n = spec.n_buildings
    if n == 0:
        return True

    beta, alpha = np.radians(beta_deg), np.radians(alpha_deg)
    ray = np.array([np.cos(beta) * np.cos(alpha),
                    np.cos(beta) * np.sin(alpha),
                    np.sin(beta)]) * (4.0 * spec.radius + 1.0)

    e = np.array([np.cos(spec.orientations), np.sin(spec.orientations)])
    centers = spec.centers.T                                  # [2, n]
    # 2-D cross products: solve ray(t) == center + s*e for t, s.
    den = ray[0] * e[1] - ray[1] * e[0]
    cx = centers[0] * e[1] - centers[1] * e[0]
    ca = centers[0] * ray[1] - centers[1] * ray[0]
    ok = np.abs(den) > 1e-12
    t = np.divide(cx, den, out=np.full(n, np.inf), where=ok)
    s = np.divide(ca, den, out=np.full(n, np.inf), where=ok)
    z = h_g + t * ray[2]
    hit = (t > 1e-9) & (t < 1.0 - 1e-9) & (np.abs(s) <= spec.sides / 2.0) \
        & (z >= 0.0) & (spec.heights > z)
    return not bool(np.any(hit))


def gate(scenarios=("urban", "rural"),
         elevations=(10, 20, 30, 40, 50, 60, 70, 80, 90),
         radii=(2000.0,),
         n_scenes: int = 4000, base_seed: int = 0, h_g: float = H_G,
         alpha_deg: float = 0.0):
    """
    Monte-Carlo P_LoS over ray-tested wall scenes vs the closed form.

    Returns a dict keyed by (scenario, radius) -> dict(elev, mc, se, eq56, eq57).
    Cheap because scene_gen_ppp is pure numpy: one scene is redrawn per seed and
    reused across every elevation, exactly as the reference evaluators do.
    """
    out = {}
    elevations = np.asarray(elevations, dtype=float)
    for scenario in scenarios:
        for radius in radii:
            clear = np.zeros((n_scenes, len(elevations)), dtype=bool)
            for k in range(n_scenes):
                spec = sg.sample_ppp_walls(radius=radius, scenario=scenario,
                                           seed=base_seed + k)
                for j, b in enumerate(elevations):
                    clear[k, j] = _los_clear(spec, b, alpha_deg, h_g)
            mc = clear.mean(axis=0)
            out[(scenario, radius)] = dict(
                elev=elevations, mc=mc,
                se=np.sqrt(np.maximum(mc * (1.0 - mc), 0.0) / n_scenes),
                eq56=plos_analytic(elevations, scenario, radius, h_g),
                eq57=plos_analytic(elevations, scenario, None, h_g),
                n_scenes=n_scenes,
            )
    return out


def print_gate(res: dict) -> None:
    for (scenario, radius), r in res.items():
        print(f"\n{scenario}  R_city={radius:.0f} m  ({r['n_scenes']} scenes)")
        print(f"  {'elev':>5} {'MC':>8} {'+/-':>7} {'eq56':>8} {'dev/se':>7} "
              f"{'eq57':>8}")
        for j, b in enumerate(r["elev"]):
            se = max(r["se"][j], 1e-12)
            dev = (r["mc"][j] - r["eq56"][j]) / se
            print(f"  {b:5.0f} {r['mc'][j]:8.4f} {r['se'][j]:7.4f} "
                  f"{r['eq56'][j]:8.4f} {dev:7.2f} {r['eq57'][j]:8.4f}")


def _cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    g = sub.add_parser("gate", help="pure-geometry LoS-probability gate")
    g.add_argument("--n-scenes", type=int, default=4000)
    g.add_argument("--base-seed", type=int, default=0)
    # The paper's own R_city. Pass several to re-run the convergence sweep
    # (P_LoS is converged by 250 m, and eq. 56's truncation term is numerically
    # zero from there up, so eq. 56 and eq. 27 coincide at 2 km).
    g.add_argument("--radii", type=float, nargs="+", default=[2000.0])
    g.add_argument("--scenarios", nargs="+", default=["urban", "rural"])
    g.add_argument("--elevations", type=float, nargs="+",
                   default=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    g.add_argument("--out", type=str, default=None, help="save results .npz")

    a = p.parse_args()
    if a.stage == "gate":
        res = gate(a.scenarios, a.elevations, a.radii, a.n_scenes, a.base_seed)
        print_gate(res)
        if a.out:
            flat = {f"{s}_{r:.0f}_{k}": v for (s, r), d in res.items()
                    for k, v in d.items() if k != "n_scenes"}
            np.savez_compressed(a.out, **flat)
            print(f"\nwrote {a.out}")


if __name__ == "__main__":
    _cli()
