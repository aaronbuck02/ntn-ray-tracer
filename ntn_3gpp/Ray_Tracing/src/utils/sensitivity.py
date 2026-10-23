"""Sensitivity-sweep analysis: load, bootstrap CIs, variance split, OAT, Sobol, 38.811 distance.

Bootstraps resample scenes, not UEs: UEs in one scene share geometry, so UE
resampling understates the CI. Sobol uses a Legendre PCE fitted on scene-level
outputs, whose indices are analytic in the coefficients; the unexplained
(1 - R^2_cv) share is the scene/solver randomness. See docs/analysis.md.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from numpy.polynomial import legendre
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


# ─── Within- vs between-scene variance ────────────────────────────────────────

def variance_components(df, cols=("K_dB", "lgDS", "lgASA", "lgZSA"), group="scene_seed"):
    """One-way random-effects ANOVA (unbalanced): between/within variance and share."""
    rows = []
    for c in cols:
        d = df[[group, c]].replace([np.inf, -np.inf], np.nan).dropna()
        g = d.groupby(group)[c]
        n_i, m_i = g.size().values, g.mean().values
        N, k = n_i.sum(), len(n_i)
        if k < 2 or N <= k:
            continue
        grand = d[c].mean()
        msb = np.sum(n_i * (m_i - grand) ** 2) / (k - 1)
        msw = np.sum((d[c] - g.transform("mean")) ** 2) / (N - k)
        n0 = (N - np.sum(n_i ** 2) / N) / (k - 1)
        sb = max((msb - msw) / n0, 0.0)
        rows.append({"lsp": c, "var_between": sb, "var_within": msw,
                     "share_between": sb / (sb + msw) if sb + msw > 0 else np.nan,
                     "n_scenes": k, "n_rx": int(N)})
    return pd.DataFrame(rows)


# ─── One-at-a-time effects ────────────────────────────────────────────────────

def paired_effect(df_v, df_b, stat=lsp_summary, B=500, group="scene_seed", ci=0.95, seed=0):
    """Δ = stat(variant) - stat(baseline), CI from resampling the shared scenes in both."""
    gv, gb = _group_index(df_v, group), _group_index(df_b, group)
    keys = sorted(set(gv) & set(gb))
    if not keys:
        raise ValueError("variant and baseline share no scenes; effects are not paired")
    sv = df_v[df_v[group].isin(keys)]
    sb = df_b[df_b[group].isin(keys)]
    pv, pb = stat(sv), stat(sb)
    point = {k: pv[k] - pb[k] for k in pv}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        pick = [keys[i] for i in rng.choice(len(keys), len(keys))]
        a = stat(df_v.iloc[np.concatenate([gv[s] for s in pick])])
        b = stat(df_b.iloc[np.concatenate([gb[s] for s in pick])])
        boots.append({k: a[k] - b[k] for k in a})
    bt = pd.DataFrame(boots)
    q = (1 - ci) / 2
    return pd.DataFrame({"delta": pd.Series(point), "lo": bt.quantile(q),
                         "hi": bt.quantile(1 - q)}).loc[list(point)]


def noise_floor(df, replicates, stat=lsp_summary, by=("frequency", "elevation")):
    """Std of each stat across solver-seed replicate variants (same scenes, same params)."""
    rows = []
    for key, sub in df[df.variant.isin(replicates)].groupby(list(by)):
        s = pd.DataFrame([stat(r) for _, r in sub.groupby("variant")]).std()
        rows.append(s.rename(key))
    out = pd.DataFrame(rows)
    out.index = pd.MultiIndex.from_tuples(out.index, names=list(by))
    return out


def oat_effects(df, baseline, stat=lsp_summary, floor=None, by=("frequency", "elevation"),
                B=500, k_floor=2.0):
    """paired_effect of every variant vs baseline per group; flags effects clear of noise."""
    rows = []
    for key, sub in df.groupby(list(by)):
        base = sub[sub.variant == baseline]
        if base.empty:
            continue
        for v, dv in sub[sub.variant != baseline].groupby("variant"):
            t = paired_effect(dv, base, stat=stat, B=B).reset_index(names="stat")
            t["variant"] = v
            for k, val in zip(by, key):
                t[k] = val
            if floor is not None and key in floor.index:
                t["noise"] = t.stat.map(floor.loc[key])
            rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    excl0 = (out.lo > 0) | (out.hi < 0)
    out["significant"] = excl0 & (out.delta.abs() > k_floor * out["noise"].fillna(0)
                                  if "noise" in out else excl0)
    return out


# ─── Global sensitivity: Legendre PCE → Sobol ─────────────────────────────────

def _multi_indices(d, degree):
    return [a for a in itertools.product(range(degree + 1), repeat=d)
            if 0 < sum(a) <= degree]


def _legendre01(u, n):
    """Orthonormal shifted Legendre on U[0,1]."""
    c = np.zeros(n + 1)
    c[n] = 1
    return np.sqrt(2 * n + 1) * legendre.legval(2 * u - 1, c)


def _pce_matrix(U, alphas):
    cache = {}
    cols = [np.ones(len(U))]
    for a in alphas:
        v = np.ones(len(U))
        for j, n in enumerate(a):
            if n:
                if (j, n) not in cache:
                    cache[(j, n)] = _legendre01(U[:, j], n)
                v = v * cache[(j, n)]
        cols.append(v)
    return np.column_stack(cols)


def _pce_fit(U, y, alphas, ridge):
    A = _pce_matrix(U, alphas)
    R = ridge * np.eye(A.shape[1])
    R[0, 0] = 0
    return np.linalg.solve(A.T @ A + R, A.T @ y)


def _pce_sobol(c, alphas, d):
    c2 = np.asarray(c[1:]) ** 2
    V = c2.sum()
    A = np.asarray(alphas)
    first = np.array([c2[(A[:, j] > 0) & (A.sum(1) == A[:, j])].sum() for j in range(d)]) / V
    total = np.array([c2[A[:, j] > 0].sum() for j in range(d)]) / V
    return first, total


def to_unit(X, spec):
    """Design values -> U[0,1] using the factor spec (lo, hi[, 'lin'|'log'|'int'])."""
    U = np.empty_like(X, dtype=float)
    for j, (lo, hi, *kind) in enumerate(spec):
        k = kind[0] if kind else "lin"
        if k == "log":
            U[:, j] = (np.log(X[:, j]) - np.log(lo)) / (np.log(hi) - np.log(lo))
        elif k == "int":   # discrete-uniform lo..hi: integer i sits at its stratum centre
            U[:, j] = (X[:, j] - lo + 0.5) / (hi - lo + 1)
        else:
            U[:, j] = (X[:, j] - lo) / (hi - lo)
    return np.clip(U, 0, 1)


def sobol_pce(U, y, names, degree=2, ridge=1e-6, folds=5, B=200, seed=0):
    """First/total Sobol indices of a degree-p Legendre PCE on U[0,1]^d, with bootstrap CI and CV R²."""
    U, y = np.asarray(U, float), np.asarray(y, float)
    ok = np.isfinite(y) & np.all(np.isfinite(U), 1)
    U, y = U[ok], y[ok]
    d = U.shape[1]
    alphas = _multi_indices(d, degree)
    if len(y) < 2 * (len(alphas) + 1):
        raise ValueError(f"{len(y)} samples for {len(alphas) + 1} PCE terms; lower degree")
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % folds
    pred = np.empty_like(y)
    for f in range(folds):
        tr = fold != f
        pred[~tr] = _pce_matrix(U[~tr], alphas) @ _pce_fit(U[tr], y[tr], alphas, ridge)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    S1, ST = _pce_sobol(_pce_fit(U, y, alphas, ridge), alphas, d)
    bs1, bst = [], []
    for _ in range(B):
        i = rng.choice(len(y), len(y))
        s1, st = _pce_sobol(_pce_fit(U[i], y[i], alphas, ridge), alphas, d)
        bs1.append(s1)
        bst.append(st)
    bs1, bst = np.array(bs1), np.array(bst)
    return pd.DataFrame({"factor": names, "S1": S1, "S1_lo": np.quantile(bs1, .025, 0),
                         "S1_hi": np.quantile(bs1, .975, 0), "ST": ST,
                         "ST_lo": np.quantile(bst, .025, 0), "ST_hi": np.quantile(bst, .975, 0),
                         "r2_cv": r2, "n": len(y)})


def scene_table(df, factors, outputs=("K_dB", "lgDS", "is_los", "K_fit_dB")):
    """Scene-level means of outputs (K over LoS UEs only) with each scene's factor values."""
    def agg(s):
        los = s[s.is_los]
        r = {o: (np.nanmean(los[o].replace([np.inf, -np.inf], np.nan)) if o.startswith("K")
                 else np.nanmean(s[o].astype(float))) for o in outputs}
        r.update({f: s[f].iloc[0] for f in factors})
        return pd.Series(r)
    return df.groupby("scene_seed").apply(agg, include_groups=False).reset_index()


def sobol_from_design(df, design_spec, outputs=("K_dB", "lgDS", "is_los"), **kw):
    """sobol_pce per output on the scene table; design_spec = {factor: (lo, hi[, kind])}."""
    names = list(design_spec)
    st = scene_table(df, names, outputs)
    U = to_unit(st[names].to_numpy(float), [design_spec[n] for n in names])
    out = []
    for o in outputs:
        try:
            t = sobol_pce(U, st[o].to_numpy(float), names, **kw)
        except ValueError as e:   # e.g. too few scenes with a LoS UE for K
            warnings.warn(f"sobol skipped for {o}: {e}")
            continue
        t["output"] = o
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ─── Uniform one-at-a-time response ────────────────────────────────────────────

SCENE_OUTPUTS = ("K_dB", "lgDS", "is_los", "PG_dB")   # K_fit_dB is NaN unless K_FIT_N > 0


def noise_floor_scene(df, replicates, outputs=SCENE_OUTPUTS, by=("frequency", "elevation")):
    """Std across solver-seed replicates of the ensemble mean of each scene-level output."""
    rows = []
    for key, sub in df[df.variant.isin(replicates)].groupby(list(by)):
        means = [scene_table(g, [], outputs)[list(outputs)].mean() for _, g in sub.groupby("variant")]
        rows.append(pd.DataFrame(means).std().rename(key))
    out = pd.DataFrame(rows)
    out.index = pd.MultiIndex.from_tuples(out.index, names=list(by))
    return out


def _quad_through_x0(x, d, x0):
    """Least-squares Δ = b1 (x - x0) + b2 (x - x0)^2 (Δ(x0) = 0 by construction)."""
    z = x - x0
    A = np.column_stack([z, z * z])
    return np.linalg.lstsq(A, d, rcond=None)[0]


def _same_ues(b, v):
    """Baseline rows at the variant's UE positions (x, y per scene), so a variant run on a
    UE subset (N_RX_KEEP_BY_ELEV) or a scene subset is compared like for like."""
    if b.empty or v.empty:
        return b
    key = lambda d: list(zip(d.scene_seed, d.x.round(3), d.y.round(3)))
    keep = pd.Series(key(b), index=b.index).isin(set(key(v)))
    return b[keep] if keep.any() else b


def oat_response(df, baseline, factor, spec, x0, outputs=SCENE_OUTPUTS, floor=None,
                 by=("frequency", "elevation"), B=500, k_floor=2.0, seed=0):
    """Paired per-scene Δ(x) for a uniform one-at-a-time variant, fitted through Δ(x0) = 0.

    swing = max - min of the fit over [lo, hi] (Δ(x0) = 0 included): the change the
    factor's full range produces. CIs resample scenes. A swing is >= 0 by
    construction, so significance uses the fitted Δ at the range ends instead:
    either end's CI must exclude 0, and swing > k_floor x the scene noise floor.
    """
    lo, hi = spec[0], spec[1]
    grid = np.linspace(lo, hi, 101)
    rng = np.random.default_rng(seed)
    rows = []
    for key, sub in df.groupby(list(by)):
        v = sub[sub[factor].notna() & (sub.variant != baseline)]
        b = _same_ues(sub[sub.variant == baseline], v)
        if v.empty or b.empty:
            continue
        tv = scene_table(v, [factor], outputs).set_index("scene_seed")
        tb = scene_table(b, [], outputs).set_index("scene_seed")
        common = tv.index.intersection(tb.index)
        x = tv.loc[common, factor].to_numpy(float)
        for o in outputs:
            d = (tv.loc[common, o] - tb.loc[common, o]).to_numpy(float)
            ok = np.isfinite(d) & np.isfinite(x)
            if ok.sum() < 5:
                continue
            xs, ds = x[ok], d[ok]

            def fit(ix):
                c = _quad_through_x0(xs[ix], ds[ix], x0)
                f = c[0] * (grid - x0) + c[1] * (grid - x0) ** 2
                return np.ptp(np.r_[f, 0.0]), c[0], f[0], f[-1]

            sw, sl, dlo, dhi = fit(np.arange(len(xs)))
            bs = np.array([fit(rng.choice(len(xs), len(xs))) for _ in range(B)])
            q = lambda j: (np.quantile(bs[:, j], .025), np.quantile(bs[:, j], .975))
            row = {**dict(zip(by, key)), "factor": factor, "output": o, "n_scenes": int(ok.sum()),
                   "swing": sw, "swing_lo": q(0)[0], "swing_hi": q(0)[1],
                   "slope_at_x0": sl, "slope_lo": q(1)[0], "slope_hi": q(1)[1],
                   "delta_at_lo": dlo, "delta_at_lo_ci": q(2), "delta_at_hi": dhi,
                   "delta_at_hi_ci": q(3)}
            nf = floor.loc[key, o] if floor is not None and key in floor.index else 0.0
            ends_clear = any(lo_ > 0 or hi_ < 0 for lo_, hi_ in (q(2), q(3)))
            row["noise"] = nf
            row["significant"] = bool(ends_clear and sw > k_floor * nf)
            rows.append(row)
    return pd.DataFrame(rows)


# ─── Geometry variance split (Phase 3) ────────────────────────────────────────

def geometry_variance(df, factors, cols=("K_dB", "lgDS", "lgASA", "lgZSA"), folds=5, seed=0):
    """Total = within-scene + between-scene explained by `factors` (quadratic, CV R²) + residual."""
    rows = []
    for c in cols:
        d = df[["scene_seed", c, *factors]].replace([np.inf, -np.inf], np.nan).dropna()
        if d.scene_seed.nunique() < 10:
            continue
        vc = variance_components(d, cols=(c,)).iloc[0]
        g = d.groupby("scene_seed").agg({c: "mean", **{f: "first" for f in factors}})
        X = g[list(factors)].to_numpy(float)
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
        A = np.column_stack([np.ones(len(X)), X, *[X[:, i] * X[:, j] for i in range(X.shape[1])
                                                     for j in range(i, X.shape[1])]])
        y = g[c].to_numpy(float)
        fold = np.random.default_rng(seed).permutation(len(y)) % folds
        pred = np.empty_like(y)
        for f in range(folds):
            tr = fold != f
            pred[~tr] = A[~tr] @ np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        r2 = max(0.0, 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        tot = vc.var_between + vc.var_within
        rows.append({"lsp": c, "var_total": tot, "share_within": vc.var_within / tot,
                     "share_between_explained": r2 * vc.var_between / tot,
                     "share_between_residual": (1 - r2) * vc.var_between / tot,
                     "r2_cv_scene_means": r2, "n_scenes": int(vc.n_scenes)})
    return pd.DataFrame(rows, columns=["lsp", "var_total", "share_within",
                                       "share_between_explained", "share_between_residual",
                                       "r2_cv_scene_means", "n_scenes"])


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
