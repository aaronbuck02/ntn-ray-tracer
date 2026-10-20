"""Scene-classification dataset from Results/scene_class/<sweep>/<freq>__<class>/ runs.

links: one row per (scene, UE, elevation, carrier[, walk snapshot]). scenes: one row per (scene,
elevation, carrier), the LSPs distilled over the scene's UEs. Every column has a
role (COLUMN_ROLES); only 'feature' and 'context' columns may feed a classifier.
See docs/scene_classification.md.
"""

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

from Ray_Tracing.src.utils.sensitivity import load_results_frame

C0 = 299_792_458.0

# feature: distilled LSP; context: known to the terminal; nuisance: drawn, unobservable;
# label: target; oracle: simulator truth, never a feature; id: grouping keys.
LINK_FEATURES = ("is_los", "K_dB", "lgDS", "lgASA", "lgZSA", "lgASD", "lgZSD",
                 "excess_loss_dB", "lg_doppler_rms_hz")
CONTEXT = ("elevation", "frequency", "pol_pair")
LABELS = ("scene_class", "geo_alpha", "geo_beta", "geo_gamma")
IDS = ("sweep", "scene_seed", "ue", "traj_t_s")
ORACLE_PREFIX = ("P_", "_P_", "itu_", "n_paths", "SF_dB", "loss_db", "DS_s", "PG_dB", "K_fit_dB",
                 "ASD_deg", "ASA_deg", "ZSD_deg", "ZSA_deg", "doppler_")
MLP_LSPS = ("is_los", "lgDS")   # classifier input: LoS state, delay spread (K dropped 2026-10-20)
SCENE_LSPS = ("K_dB", "lgDS", "lgASA", "lgZSA", "lgASD", "lgZSD", "excess_loss_dB",
              "lg_doppler_rms_hz")
QUANTILES = (0.1, 0.5, 0.9)


def _role(col):
    if col in IDS:
        return "id"
    if col in LABELS:
        return "label"
    if col in CONTEXT:
        return "context"
    if col in LINK_FEATURES or col.startswith(("f_", "P_LoS")) or \
            col.split("@")[0].split("~")[0] in LINK_FEATURES:   # '<lsp>[~stat]@<elev>[@t<k>]' 
        return "feature"
    if col.startswith(ORACLE_PREFIX):
        return "oracle"
    return "nuisance"


def column_roles(df):
    return {c: _role(c) for c in df.columns}


def scene_meta(manifest):
    """Per-scene label / nuisance table from a class manifest's entries and LHS design."""
    rows = []
    for e in manifest["scenes"]:
        co = e.get("cfg_overrides", {})
        r = {"scene_seed": int(e["seed"]), "scene_class": e.get("scene_class", manifest.get("scene_class")),
             "pol_pair": f"{co.get('TX_POL_TYPE', '?')}/{co.get('RX_POL_TYPE', '?')}"}
        for role in ("wall", "roof", "ground"):
            p = e.get(f"{role}_params", manifest.get(f"{role}_params", {}))
            r[f"{role}_anchor"] = p.get("base", "default")
            r[f"{role}_scattering"] = float(p.get("scattering_coefficient", np.nan))
        rows.append(r)
    meta = pd.DataFrame(rows)
    d = manifest.get("design")
    if d is not None:   # continuous draws; categorical ones are already named above
        X = pd.DataFrame(d["X"], columns=[f.replace(".", "_") for f in d["factors"]])
        X = X.drop(columns=[c for c in X if c.endswith("_base") or c == "cfg_POL_PAIR"])
        X["scene_seed"] = [int(s) for s in d["seeds"]]
        meta = meta.merge(X, on="scene_seed", how="left")
    return meta


def _fspl_db(d_m, f_hz):
    return 20 * np.log10(4 * np.pi * np.asarray(d_m, float) * np.asarray(f_hz, float) / C0)


def load_links(sweep_dir, sweep=""):
    """Per-UE frame over every <freq>__<class> variant of one sweep, with scene metadata."""
    frames = []
    for mp in sorted(glob.glob(os.path.join(sweep_dir, "*", "scenes_manifest.pkl"))):
        d = os.path.dirname(mp)
        with open(mp, "rb") as f:
            meta = scene_meta(pickle.load(f))
        for rp in sorted(glob.glob(os.path.join(d, "results_elev*deg.pkl"))):
            with open(rp, "rb") as f:
                res = pickle.load(f)
            df = load_results_frame(res, os.path.basename(d))
            n = len(df)
            for k in ("ue_id", "traj_t_s", "ue_speed_mps", "ue_heading_deg"):
                v = res.get(k)
                df[k] = np.asarray(v, float) if v is not None and len(v) == n else np.nan
            # walking runs store the UE id; static runs predate it (one row per UE)
            df["ue"] = df.ue_id.astype(int) if df.ue_id.notna().all() else df.groupby("scene_seed").cumcount()
            df["traj_t_s"] = df.traj_t_s.fillna(0.0)
            df = df.drop(columns="ue_id")
            frames.append(df.merge(meta, on="scene_seed", how="left", validate="many_to_one"))
    if not frames:
        raise FileNotFoundError(f"no results under {sweep_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["sweep"] = sweep
    df["K_dB"] = df.K_dB.where(np.isfinite(df.K_dB))          # NLoS: -inf -> NaN
    df["excess_loss_dB"] = -df.PG_dB - _fspl_db(df.slant_range_m, df.frequency)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lg_doppler_rms_hz"] = np.log10(df.doppler_rms_hz.where(df.doppler_rms_hz > 0))
    return df.drop(columns=["variant"])


def _stats(x, prefix):
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"{prefix}_{s}": np.nan for s in ("mean", "std", *(f"q{int(q * 100)}" for q in QUANTILES))}
    out = {f"{prefix}_mean": x.mean(), f"{prefix}_std": x.std()}
    out.update({f"{prefix}_q{int(q * 100)}": np.quantile(x, q) for q in QUANTILES})
    return out


SCENE_KEYS = ["sweep", "scene_seed", "elevation", "frequency"]


def scene_features(links, n_ue=None, seed=0):
    """Distilled per-(scene, elevation, carrier) features; n_ue subsamples UEs per scene."""
    if n_ue is not None:   # subsample UEs, keeping each one's whole walk
        u = links[SCENE_KEYS + ["ue"]].drop_duplicates()
        if u.groupby(SCENE_KEYS).size().min() < n_ue:
            raise ValueError(f"some scenes have fewer than {n_ue} UEs")
        u = u.groupby(SCENE_KEYS).sample(n=n_ue, random_state=seed)
        links = links.merge(u, on=SCENE_KEYS + ["ue"])
    grp = links.groupby(SCENE_KEYS)
    # carry only columns constant within a scene row (labels, draws, orbit, realised P.1410)
    nun = grp.nunique(dropna=False).max()
    keep = SCENE_KEYS + [c for c in links if c not in SCENE_KEYS and c != "ue"
                         and _role(c) != "feature" and nun.get(c, 2) <= 1]
    rows = []
    for _, g in grp:
        r = g.iloc[0][keep].to_dict()
        los = g.is_los.to_numpy(bool)
        r["n_ue"] = g.ue.nunique()
        r["P_LoS"] = los.mean()
        Pl, Pn = g._P_los.to_numpy()[los].sum(), g._P_nlos.to_numpy()[los].sum()
        r["f_K_rom_dB"] = 10 * np.log10(Pl / Pn) if Pn > 0 and Pl > 0 else np.nan
        for lsp in SCENE_LSPS:
            v = g[lsp].to_numpy(float)
            for st, m in (("all", np.ones_like(los)), ("los", los), ("nlos", ~los)):
                if lsp == "K_dB" and st != "los":
                    continue
                r.update(_stats(v[m], f"f_{lsp}_{st}"))
        rows.append(r)
    return pd.DataFrame(rows)


CURVE_KEYS = ["sweep", "scene_seed", "ue", "frequency"]
CURVE_CARRY = ("scene_class", "pol_pair", "geo_alpha", "geo_beta", "geo_gamma")


def curve_col(lsp, elev, step=None):
    return f"{lsp}@{elev:g}" + ("" if step is None else f"@t{step}")


def elevation_curves(links, lsps=MLP_LSPS, elevations=None, traj="full"):
    """One row per (scene, UE, carrier); columns '<lsp>@<elev>' hold the UE's LSP at each elevation.

    UEs are seeded by the scene alone, so ue i is the same terminal at every elevation. A
    missing (scene, elevation) leaves NaNs. elevations: subset to keep (default: all present).
    Walking UEs (several traj_t_s): traj='full' gives '<lsp>@<elev>@t<k>' per snapshot k,
    'summary' gives '<lsp>~mean@<elev>' and '<lsp>~std@<elev>' over the walk, 't0' keeps
    the first snapshot only (a static UE).
    """
    lsps = [c for c in lsps if c in links]
    els = sorted(links.elevation.unique()) if elevations is None else sorted(elevations)
    l = links[links.elevation.isin(els)]
    ts = sorted(l.traj_t_s.unique()) if "traj_t_s" in l else [0.0]
    if traj == "t0" or len(ts) == 1:
        l, steps = l[l.traj_t_s == ts[0]] if "traj_t_s" in l else l, None
    elif traj == "full":
        l, steps = l.assign(_step=l.traj_t_s.map({t: k for k, t in enumerate(ts)})), range(len(ts))
    elif traj == "summary":
        g = l.assign(**{c: l[c].astype(float) for c in lsps}).groupby(CURVE_KEYS + ["elevation"])
        agg = pd.concat({f"{c}~{st}": g[c].agg(st) for c in lsps for st in ("mean", "std")}, axis=1)
        l = g[[c for c in CURVE_CARRY if c in l]].first().join(agg).reset_index()
        lsps, steps = list(agg.columns), None
    else:
        raise ValueError("traj must be 'full', 'summary' or 't0'")
    n = l.groupby(["sweep", "scene_seed", "frequency", "elevation"]).ue.nunique()
    assert (n.groupby(level=[0, 1, 2]).nunique() == 1).all(), "UE count differs across elevations of a scene"
    if steps is None:
        wide = l.set_index(CURVE_KEYS + ["elevation"])[lsps].astype(float).unstack("elevation")
        wide.columns = [curve_col(c, e) for c, e in wide.columns]
        order = [curve_col(c, e) for c in lsps for e in els]
    else:
        wide = l.set_index(CURVE_KEYS + ["elevation", "_step"])[lsps].astype(float).unstack(["elevation", "_step"])
        wide.columns = [curve_col(c, e, k) for c, e, k in wide.columns]
        order = [curve_col(c, e, k) for c in lsps for e in els for k in steps]
    wide = wide[[c for c in order if c in wide]]
    carry = [c for c in CURVE_CARRY if c in l]
    meta = l.groupby(CURVE_KEYS)[carry].first()
    return meta.join(wide).reset_index()


def feature_columns(df, context=True):
    roles = column_roles(df)
    return [c for c, r in roles.items() if r == "feature" or (context and r == "context")]


def build_dataset(root, sweeps=None, out_dir=None):
    """Write <out_dir>/<sweep>/{links,scenes}.pkl + columns.json for each sweep under root."""
    out_dir = out_dir or os.path.join(root, "dataset")
    sweeps = sweeps or sorted(s for s in os.listdir(root)
                              if os.path.isdir(os.path.join(root, s)) and s != "dataset"
                              and glob.glob(os.path.join(root, s, "*", "scenes_manifest.pkl")))
    out = {}
    for s in sweeps:
        links = load_links(os.path.join(root, s), s)
        scenes = scene_features(links)
        d = os.path.join(out_dir, s)
        os.makedirs(d, exist_ok=True)
        links.to_pickle(os.path.join(d, "links.pkl"))
        scenes.to_pickle(os.path.join(d, "scenes.pkl"))
        with open(os.path.join(d, "columns.json"), "w") as f:
            json.dump({"links": column_roles(links), "scenes": column_roles(scenes)}, f, indent=1)
        out[s] = (links, scenes)
        print(f"[scene_dataset] {s}: {len(links)} links, {len(scenes)} scene rows -> {d}")
    return out
