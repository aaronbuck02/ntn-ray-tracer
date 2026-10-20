"""Scene manifests for the sensitivity sweeps: streaming ensembles and their variants.

A streaming manifest lists scenes as (seed, sampler kwargs, per-scene params,
orbit); run_stochastic_sim builds each scene in scratch and deletes it, so no
geometry is ever cached. Variants layer LHS draws or global material overrides
on the same entries, so they stay paired with the baseline.
See docs/sensitivity.md and docs/pipelines.md (Sensitivity sweeps).
"""

import copy
import os
import pickle

import numpy as np
from scipy.stats import qmc

from Ray_Tracing.src.utils import scene_gen_ppp as sg

ROLES = ("wall", "roof", "ground")
# Absolute keys copied straight into a role's params dict.
_ABS_KEYS = {"scattering_coefficient", "xpd_coefficient", "thickness",
             "scattering_pattern", "alpha_r", "alpha_i", "lambda_"}


def load_manifest(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_manifest(m, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def virtual_manifest(n_scenes, sample_kwargs, base_seed=0, primitive="block",
                     building_material="itu_concrete", roof_material="itu_brick",
                     ground_material="itu_plasterboard",
                     wall_params=None, roof_params=None, ground_params=None):
    """File-free manifest of n_scenes seeds; the runner streams each scene."""
    return dict(streaming=True, primitive=primitive, base_seed=base_seed,
                n_scenes=n_scenes, sample_kwargs=dict(sample_kwargs), skipped_seeds=[],
                building_material=building_material, roof_material=roof_material,
                ground_material=ground_material,
                wall_params=apply_material_override(wall_params or sg.DEFAULT_WALL_PARAMS, {}),
                roof_params=apply_material_override(roof_params or sg.DEFAULT_ROOF_PARAMS, {}),
                ground_params=apply_material_override(ground_params or sg.DEFAULT_GROUND_PARAMS, {}),
                scenes=[{"seed": base_seed + i} for i in range(n_scenes)])


def attach_orbits(m, table):
    """Copy of m with each entry's per-elevation waypoints from an orbit_sampling table."""
    m = copy.deepcopy(m)
    for e in m["scenes"]:
        sc = table["scenes"].get(e["seed"])
        if sc is None:
            raise KeyError(f"orbit table has no scene seed {e['seed']}")
        e["orbit"] = sc["waypoints"]
        e["ground_station"] = (sc["lat"], sc["lon"])
    m["orbit_table"] = {k: v for k, v in table.items() if k != "scenes"}
    return m


def first_scenes(m, n):
    """Copy of m restricted to its first n scenes (design rows included)."""
    m = copy.deepcopy(m)
    m["scenes"] = m["scenes"][:n]
    if "n_scenes" in m:
        m["n_scenes"] = len(m["scenes"])
    d = m.get("design")
    if d is not None:
        d["X"], d["seeds"] = np.asarray(d["X"])[:n], list(d["seeds"])[:n]
    return m


def apply_material_override(params, ov):
    """Role params with ov applied: eps_scale/sigma_scale scale a/c, other keys set absolutely."""
    p = dict(params)
    p.setdefault("thickness", sg.DEFAULT_THICKNESS_M)
    for k, v in ov.items():
        if k == "eps_scale":
            p["a"] = float(p["a"]) * float(v)
        elif k == "sigma_scale":
            p["c"] = float(p["c"]) * float(v)
        elif k in _ABS_KEYS:
            p[k] = v
        else:
            raise KeyError(f"unknown material override {k!r}")
    return p


def _expand_roles(material_overrides):
    """{'all'|role: ov} -> {role: merged ov}; role-specific keys win over 'all'."""
    out = {r: dict(material_overrides.get("all", {})) for r in ROLES}
    for r in ROLES:
        out[r].update(material_overrides.get(r, {}))
    return out


def variant_manifest(base, material_overrides=None, out_path=None, name=None):
    """Copy of base with global per-role overrides, applied to manifest and per-scene params."""
    m = copy.deepcopy(base)
    for r, ov in _expand_roles(material_overrides or {}).items():
        m[f"{r}_params"] = apply_material_override(m[f"{r}_params"], ov)
        for e in m["scenes"]:
            if f"{r}_params" in e:
                e[f"{r}_params"] = apply_material_override(e[f"{r}_params"], ov)
    m["variant"] = {"name": name, "material_overrides": material_overrides or {}}
    if out_path:
        save_manifest(m, out_path)
    return m


def _scale(u, lo, hi, kind):
    if kind == "int":   # discrete-uniform on lo..hi: equal mass per integer
        return float(min(np.floor(lo + u * (hi - lo + 1)), hi))
    if kind == "log":
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return float(lo + u * (hi - lo))


def lhs_design(n, factors, seed=0):
    """(names, X[n, d]) Latin hypercube; factors = {name: (lo, hi[, 'lin'|'int'|'log'])}."""
    names = list(factors)
    U = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    X = np.empty_like(U)
    for j, nm in enumerate(names):
        lo, hi, *kind = factors[nm]
        X[:, j] = [_scale(u, lo, hi, kind[0] if kind else "lin") for u in U[:, j]]
    return names, X


def _route(name, value, mat_ov, cfg_ov, geo_kw):
    """Route 'role.key' / 'all.key' / 'cfg.KEY' / 'geo.kwarg' factor names."""
    tgt, key = name.split(".", 1)
    if tgt == "cfg":
        cfg_ov[key] = value
    elif tgt == "geo":
        geo_kw[key] = value
    elif tgt in ROLES or tgt == "all":
        if key == "alpha_r":
            value = int(value)
        mat_ov.setdefault(tgt, {})[key] = value
    else:
        raise KeyError(f"factor {name!r}: target must be a role, 'all', 'cfg' or 'geo'")


def _geo_kwargs(base_kw, geo):
    """Sampler kwargs; geo 'beta' [buildings/km^2] becomes cell_res for the 1-occupied-cell-per-building grid."""
    kw = {**base_kw, **{k: v for k, v in geo.items() if k != "beta"}}
    if "beta" in geo:
        occ = 1.0 - np.exp(-float(kw["points_per_cell"]))
        kw["cell_res"] = float(np.sqrt(occ * 1e6 / float(geo["beta"])))
    return kw


def lhs_manifest(base, factors, seed=0, out_path=None):
    """Copy of base with one LHS draw per scene, layered on any draws the entries already carry."""
    m = copy.deepcopy(base)
    for r in ROLES:   # inject default thickness into pre-thickness manifests
        m[f"{r}_params"] = apply_material_override(m[f"{r}_params"], {})
    names, X = lhs_design(len(m["scenes"]), factors, seed)
    for e, x in zip(m["scenes"], X):
        mat_ov, cfg_ov, geo = {}, {}, {}
        for nm, v in zip(names, x):
            _route(nm, v, mat_ov, cfg_ov, geo)
        if geo:
            if not m.get("streaming"):
                raise ValueError("geo.* factors need a streaming manifest (virtual_manifest)")
            e["sample_kwargs"] = _geo_kwargs({**m["sample_kwargs"], **e.get("sample_kwargs", {})}, geo)
        for r, ov in _expand_roles(mat_ov).items():
            if ov:
                e[f"{r}_params"] = apply_material_override(e.get(f"{r}_params", m[f"{r}_params"]), ov)
        if cfg_ov:
            e["cfg_overrides"] = {**e.get("cfg_overrides", {}), **cfg_ov}
    prev = m.get("design")
    if prev is not None:
        assert list(prev["seeds"]) == [e["seed"] for e in m["scenes"]]
        names, X = list(prev["factors"]) + names, np.hstack([prev["X"], X])
        factors = {**prev["spec"], **factors}
    m["design"] = {"factors": names, "X": X, "seeds": [e["seed"] for e in m["scenes"]],
                   "spec": dict(factors), "lhs_seed": seed}
    if out_path:
        save_manifest(m, out_path)
    return m


def oat_uniform_manifest(base, factor, spec, seed=0, out_path=None):
    """One-at-a-time uniform variant: only `factor` is drawn per scene from spec=(lo, hi[, kind])."""
    return lhs_manifest(base, {factor: tuple(spec)}, seed=seed, out_path=out_path)


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Write a material-override variant manifest.")
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--overrides", required=True,
                    help='JSON, e.g. \'{"ground": {"eps_scale": 1.5}, "all": {"thickness": 2.0}}\'')
    a = ap.parse_args()
    variant_manifest(load_manifest(a.base), json.loads(a.overrides), a.out)
    print(f"wrote {a.out}")
