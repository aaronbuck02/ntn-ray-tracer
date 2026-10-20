"""Results loaders and LSP statistics shared by the stochastic sweeps: per-RX frames,
scene-bootstrap CIs, K estimators, P.1410 binning and distance to TR 38.811.

Bootstraps resample scenes, not UEs: UEs in one scene share geometry, so UE
resampling understates the CI. See docs/analysis.md.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
from scipy import stats

from Ray_Tracing.src.utils.lsp_stats import per_rx_lsp_frame
from Ray_Tracing.src.utils.ns3_reference import band_for_frequency, reference_frame

_ELEV_RE = re.compile(r"results_elev(\d{3})deg\.pkl$")


# ─── Loading ──────────────────────────────────────────────────────────────────

def _flat_factors(v):
    """variant.json -> {factor_name: value} for its global overrides (lists as JSON strings)."""
    out = {f"cfg.{k}": val for k, val in v.get("cfg_overrides", {}).items()}
    for role, ov in (v.get("material_overrides") or {}).items():
        out.update({f"{role}.{k}": val for k, val in ov.items()})
    return {k: json.dumps(x) if isinstance(x, (list, tuple, dict)) else x
            for k, x in out.items()}


def load_results_frame(res, variant="", factors=None, design=None):
    """Per-RX frame of one results dict, with sensitivity columns and factor values."""
    df = per_rx_lsp_frame(res)
    n = len(df)

    def col(k, fill=np.nan):
        v = res.get(k)
        return np.asarray(v, dtype=float) if v is not None and len(v) == n else np.full(n, fill)

    df["variant"] = variant
    df["elevation"] = float(res["elevation_deg"])
    df["frequency"] = float(res["frequency"])
    df["scene_seed"] = col("scene_seeds", -1).astype(int)
    for k in ("K_fit_dB", "PG_dB", "omega_F", "vtec_sample", "atm_total_db",
              "n_paths", "P_ground", "P_wall", "P_roof",
              "rx_height", "doppler_rms_hz", "doppler_rms_hz_td",
              "itu_alpha", "itu_beta", "itu_gamma_fit", "heading_deg", "sat_az_deg",
              "slant_range_m", "orbit_peak_deg", "norad_id", "gs_lat", "gs_lon"):
        df[k] = col(k)
    leg = res.get("orbit_leg")
    df["orbit_leg"] = list(leg) if leg is not None and len(leg) == n else ""
    pd_ = res.get("P_by_depth")
    if pd_ is not None and len(pd_) == n and np.ndim(pd_[0]):
        D = np.asarray(pd_, dtype=float)
        for d in range(D.shape[1]):
            df[f"P_d{d}"] = D[:, d]
    for k, val in (factors or {}).items():
        df[k] = val
    if design is not None:
        X = pd.DataFrame(design["X"], columns=design["factors"])
        X["scene_seed"] = design["seeds"]
        df = df.merge(X, on="scene_seed", how="left")
    return df


def load_sweep(sweep_dir, variants=None, strip_freq_tag=True):
    """Long per-RX frame over every variant/elevation results file under sweep_dir.

    strip_freq_tag drops the notebook's '<f>GHz__' name prefix: frequency is its own
    column, so baselines then pair across carriers and across phase directories.
    """
    frames = []
    for vj in sorted(glob.glob(os.path.join(sweep_dir, "*", "variant.json"))):
        d = os.path.dirname(vj)
        with open(vj) as f:
            v = json.load(f)
        if variants and v["name"] not in variants and \
                re.sub(r"^[\d.]+GHz__", "", v["name"]) not in variants:
            continue
        design = None
        mp = os.path.join(d, "scenes_manifest.pkl")
        if os.path.exists(mp):
            with open(mp, "rb") as f:
                design = pickle.load(f).get("design")
        for rp in sorted(glob.glob(os.path.join(d, "results_elev*deg.pkl"))):
            if not _ELEV_RE.search(rp):
                continue
            with open(rp, "rb") as f:
                res = pickle.load(f)
            frames.append(load_results_frame(res, v["name"], _flat_factors(v), design))
    if not frames:
        raise FileNotFoundError(f"no results under {sweep_dir}")
    df = pd.concat(frames, ignore_index=True)
    if strip_freq_tag:
        df["variant"] = df.variant.str.replace(r"^[\d.]+GHz__", "", regex=True)
    return df


def load_dirs(dirs, elevations=None):
    """Per-RX frame from plain results dirs {label: dir}; label doubles as variant.

    Fixed-site runs have no scene_seeds, so each site gets one negative
    scene id and enters variance_components as a single scene.
    """
    frames = []
    for i, (label, d) in enumerate(dirs.items()):
        for rp in sorted(glob.glob(os.path.join(d, "results_elev*deg.pkl"))):
            m = _ELEV_RE.search(rp)
            if not m or (elevations and int(m.group(1)) not in elevations):
                continue
            with open(rp, "rb") as f:
                res = pickle.load(f)
            df = load_results_frame(res, label)
            if (df.scene_seed < 0).all():
                df["scene_seed"] = -(i + 1)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ─── Summary statistics ───────────────────────────────────────────────────────

def _db(x):
    return 10 * np.log10(x) if x > 0 else np.nan


def _ms(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (x.mean(), x.std()) if len(x) else (np.nan, np.nan)


def lsp_summary(df):
    """The scalar μ/σ statistics every sweep compares (K stats over LoS UEs)."""
    los = df[df.is_los]
    out = {"P_LoS": float(df.is_los.mean()), "n_rx": len(df)}
    out["mu_K_dB"], out["sigma_K_dB"] = _ms(los.K_dB)
    out["K_rom_dB"] = _db(los._P_los.sum() / los._P_nlos.sum()) if len(los) else np.nan
    out["K_median_dB"] = float(np.nanmedian(los.K_dB)) if len(los) else np.nan
    out["mu_K_fit_dB"], out["sigma_K_fit_dB"] = _ms(los.K_fit_dB)
    for st, sub in (("LOS", los), ("NLOS", df[~df.is_los])):
        for c in ("lgDS", "lgASA", "lgZSA"):
            out[f"mu_{c}_{st}"], out[f"sigma_{c}_{st}"] = _ms(sub[c])
    tot = (df._P_los + df._P_nlos).sum()
    if tot > 0:
        for k in ("P_ground", "P_wall", "P_roof") + tuple(c for c in df if c.startswith("P_d")):
            if k in df:
                out[f"share_{k}"] = float(df[k].sum() / tot)
    out["mu_n_paths"] = float(df.n_paths.mean()) if "n_paths" in df else np.nan
    return out


# ─── Scene-cluster bootstrap ──────────────────────────────────────────────────

def _group_index(df, group):
    return {g: np.asarray(ix) for g, ix in df.groupby(group).indices.items()}


def scene_bootstrap(df, stat=lsp_summary, B=500, group="scene_seed", ci=0.95, seed=0):
    """Point estimate and percentile CI of each stat() key, resampling whole scenes."""
    gi = _group_index(df, group)
    keys = list(gi)
    rng = np.random.default_rng(seed)
    point = stat(df)
    boots = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys))
        boots.append(stat(df.iloc[np.concatenate([gi[keys[i]] for i in pick])]))
    bt = pd.DataFrame(boots)
    a = (1 - ci) / 2
    return pd.DataFrame({"est": pd.Series(point), "lo": bt.quantile(a), "hi": bt.quantile(1 - a),
                         "se": bt.std()}).loc[list(point)]


def summary_table(df, by=("variant", "frequency", "elevation"), B=500, **kw):
    """scene_bootstrap per group, long format."""
    rows = []
    for key, sub in df.groupby(list(by)):
        t = scene_bootstrap(sub, B=B, **kw).reset_index(names="stat")
        for k, v in zip(by, key if isinstance(key, tuple) else (key,)):
            t[k] = v
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


# ─── P.1410 class binning ─────────────────────────────────────────────────────

# ITU-R P.1410 recommended (alpha, beta [/km^2], gamma [m]) and the 38.811 table each maps to.
P1410_CLASSES = {"Suburban": (0.1, 750.0, 8.0), "Urban": (0.3, 500.0, 15.0),
                 "DenseUrban": (0.5, 300.0, 20.0)}


def p1410_class(df):
    """Nearest P.1410 class per row from realised itu_alpha/beta/gamma_fit (range-normalised)."""
    P = np.array(list(P1410_CLASSES.values()))
    span = P.max(0) - P.min(0)
    X = df[["itu_alpha", "itu_beta", "itu_gamma_fit"]].to_numpy(float)
    dist = (((X[:, None, :] - P[None]) / span) ** 2).sum(-1)
    return pd.Series(np.array(list(P1410_CLASSES))[np.argmin(dist, 1)], index=df.index)


# ─── Distance to the TR 38.811 distributions ──────────────────────────────────

_REF_COLS = {"K_dB": ("uK", "sigK"), "lgDS": ("uLgDS", "sigLgDS"),
             "lgASA": ("uLgASA", "sigLgASA"), "lgZSA": ("uLgZSA", "sigLgZSA")}


def _w1_normal(x, mu, sig):
    x = np.sort(x)
    q = stats.norm.ppf((np.arange(1, len(x) + 1) - 0.5) / len(x), mu, sig)
    return float(np.mean(np.abs(x - q)))


def dist_metrics(df, scenario="Urban", lsps=("K_dB", "lgDS", "lgASA", "lgZSA"),
                 by=("variant", "frequency", "elevation")):
    """KS statistic/p and Wasserstein-1 of each per-UE LSP vs the 38.811 normal (log domain).

    scenario="by_class" bins scenes with p1410_class and scores each bin against its own table.
    """
    if scenario == "by_class":
        df = df.assign(p1410=p1410_class(df))
        parts = [dist_metrics(g, cls, lsps, by).assign(p1410=cls)
                 for cls, g in df.groupby("p1410")]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    ref = reference_frame(scenario)
    rows = []
    for key, sub in df.groupby(list(by)):
        g = dict(zip(by, key))
        band = band_for_frequency(g["frequency"], warn=False)
        for state, s in (("LOS", sub[sub.is_los]), ("NLOS", sub[~sub.is_los])):
            r = ref[(ref.state == state) & (ref.band == band)]
            if r.empty:
                continue
            r = r.iloc[(r.elevation_deg - g["elevation"]).abs().argmin()]
            for lsp in lsps:
                if lsp == "K_dB" and state == "NLOS":
                    continue
                mu, sig = r[_REF_COLS[lsp][0]], r[_REF_COLS[lsp][1]]
                x = s[lsp].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                if len(x) < 10 or not np.isfinite(mu) or not sig > 0:
                    continue
                ks = stats.kstest(x, "norm", args=(mu, sig))
                rows.append({**g, "state": state, "lsp": lsp, "n": len(x),
                             "ref_elev": r.elevation_deg, "ref_mu": mu, "ref_sigma": sig,
                             "rt_mu": x.mean(), "rt_sigma": x.std(),
                             "ks": ks.statistic, "ks_p": ks.pvalue,
                             "w1": _w1_normal(x, mu, sig)})
    return pd.DataFrame(rows)


# ─── K estimators ─────────────────────────────────────────────────────────────

def estimator_table(df, by=("variant", "frequency", "elevation")):
    """K by estimator: mean of dB, median of dB, ratio of means, CFR moment fit."""
    rows = []
    for key, sub in df.groupby(list(by)):
        for state, s in (("LOS", sub[sub.is_los]), ("NLOS", sub[~sub.is_los])):
            if s.empty:
                continue
            k = s.K_dB.replace([np.inf, -np.inf], np.nan)
            kf = s.K_fit_dB.replace([np.inf, -np.inf], np.nan)
            rows.append({**dict(zip(by, key)), "state": state, "n": len(s),
                         "K_mean_dB": k.mean(), "K_median_dB": k.median(),
                         "K_rom_dB": _db(s._P_los.sum() / s._P_nlos.sum()) if s._P_nlos.sum() > 0 else np.nan,
                         "K_fit_mean_dB": kf.mean(), "K_fit_median_dB": kf.median(),
                         "K_fit_inf_frac": float(np.isinf(s.K_fit_dB).mean())})
    return pd.DataFrame(rows)
