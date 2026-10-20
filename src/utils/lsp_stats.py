"""Distil per-receiver RT output into TR 38.811-style large-scale parameters.

Produces mu/sigma of log10(DS) and log10(spread), per-link K in dB, shadow
fading as a zero-mean residual, the LSP cross-correlation matrix in ns-3's LSP
orders, and per-LSP decorrelation distances.

Three K estimators coexist and are NOT interchangeable -- per-link dB, ratio of
means over the state subset, and ratio over all receivers. They diverge by ~5 dB
at 10 deg where the LoS population is bimodal. See docs/analysis.md.
"""

from __future__ import annotations

import numpy as np

from Ray_Tracing.src.utils.checkpoint_utils import load_all_results
from Ray_Tracing.src.ns3_integration.elevation_stats_to_ns3 import (free_space_path_loss_db,  # shared convention
                                    loss_db_from_power)

# LSP column orders used by ns-3's Cholesky matrices (see ns3_reference).
LSP_ORDER_LOS = ("SF", "K", "DS", "ASD", "ASA", "ZSD", "ZSA")
LSP_ORDER_NLOS = ("SF", "DS", "ASD", "ASA", "ZSD", "ZSA")

# Fraction of non-positive samples above which a spread column is declared
# degenerate (satellite departure spreads) rather than log-transformed.
DEGENERATE_FRAC = 0.5

NOTE8 = ("degenerate: point-source TX, departure spread ~0 -- matches "
         "TR 38.811 NOTE 8 / ns-3 three-gpp-channel-model.cc satellite override")

# Raw (pre-log) column backing each spread LSP, for the degeneracy test.
RAW_SPREAD_COL = {"DS": "DS_s", "ASD": "ASD_deg", "ASA": "ASA_deg",
                  "ZSD": "ZSD_deg", "ZSA": "ZSA_deg"}


def degenerate_spreads(df, frac: float = DEGENERATE_FRAC) -> set:
    """Spread LSPs that are ~0 by geometry, so log10 of them is meaningless.

    Decided over whichever receiver set is passed. ``elevation_lsp_rows`` passes
    both states pooled: the per-state fractions straddle the threshold (ASD is
    0.67 non-positive in LOS but 0.49 in NLOS at 2 GHz / 30 deg), which would
    otherwise class the same physical quantity both ways in one table.
    """
    out = set()
    for lsp, col in RAW_SPREAD_COL.items():
        raw = df.get(col)
        if raw is None or not len(raw):
            continue
        if float(np.mean(~(np.asarray(raw, dtype=float) > 0))) > frac:
            out.add(lsp)
    return out


# ─── Per-receiver frame ───────────────────────────────────────────────────────

def per_rx_lsp_frame(res: dict, sf_smooth_radius_m: float = 0.0):
    """One row per receiver, in the domains 3GPP tabulates."""
    import pandas as pd

    P_los = np.asarray(res["P_los"], dtype=float)
    P_nlos = np.asarray(res["P_nlos"], dtype=float)
    is_los = np.asarray(res["los_probability"], dtype=float) >= 0.5

    # Identical to elevation_stats_to_ns3.py so the two tables stay consistent.
    # RX with no received power are NaN, not floored -- see loss_db_from_power.
    loss_db = loss_db_from_power(P_los + P_nlos)

    with np.errstate(divide="ignore", invalid="ignore"):
        K_dB = 10.0 * np.log10(np.where(P_nlos > 0, P_los / np.maximum(P_nlos, 1e-300), np.nan))

    rx = np.asarray(res.get("nominal_rx_positions", []), dtype=float)
    n = len(loss_db)
    if rx.ndim != 2 or len(rx) != n:
        rx = np.full((n, 3), np.nan)

    df = pd.DataFrame({
        "x": rx[:, 0], "y": rx[:, 1], "z": rx[:, 2] if rx.shape[1] > 2 else np.nan,
        "is_los": is_los,
        "loss_db": loss_db,
        "K_dB": K_dB,
        "DS_s": np.asarray(res["tau_rms_mean_s"], dtype=float),
        "ASD_deg": np.asarray(res.get("ASD_deg", [0.0] * n), dtype=float),
        "ASA_deg": np.asarray(res.get("ASA_deg", [0.0] * n), dtype=float),
        "ZSD_deg": np.asarray(res.get("ZSD_deg", [0.0] * n), dtype=float),
        "ZSA_deg": np.asarray(res.get("ZSA_deg", [0.0] * n), dtype=float),
        # Underscored: consumed by lsp_marginals for the ratio-of-means K, not an LSP column.
        "_P_los": P_los,
        "_P_nlos": P_nlos,
    })

    # SF: zero-mean residual *within each state* (see module docstring).
    df["SF_dB"] = np.nan
    for state_mask in (df.is_los.values, ~df.is_los.values):
        if state_mask.any():
            v = df.loc[state_mask, "loss_db"]
            df.loc[state_mask, "SF_dB"] = v - v.mean()

    if sf_smooth_radius_m and sf_smooth_radius_m > 0 and np.isfinite(rx[:, 0]).any():
        r = float(sf_smooth_radius_m)
        keys = list(zip(np.floor(df.x.values / r).astype("int64"),
                        np.floor(df.y.values / r).astype("int64")))
        df["_blk"] = keys
        df["SF_dB_smooth"] = np.nan
        for state_mask in (df.is_los.values, ~df.is_los.values):
            if state_mask.any():
                sub = df.loc[state_mask]
                med = sub.groupby("_blk")["loss_db"].transform("median")
                df.loc[state_mask, "SF_dB_smooth"] = sub["loss_db"] - med
        df = df.drop(columns="_blk")

    # 3GPP log domains. log10 of a non-positive spread is undefined; those are
    # left NaN here and handled explicitly (and counted) in lsp_marginals.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["lgDS"] = np.log10(np.where(df.DS_s > 0, df.DS_s, np.nan))
        for a in ("ASD", "ASA", "ZSD", "ZSA"):
            col = df[f"{a}_deg"].values
            df[f"lg{a}"] = np.log10(np.where(col > 0, col, np.nan))
    return df


# ─── Marginals ────────────────────────────────────────────────────────────────

def lsp_marginals(df, state: str = "LOS", degenerate: set = None) -> dict:
    """
    ``mu``/``sigma`` per LSP for one state, in 3GPP's own domain.

    Degenerate spread columns (satellite ASD/ZSD) report ``mu=-inf, sigma=0``
    plus a ``*_note`` entry, instead of a spurious finite number. Pass
    ``degenerate`` from ``degenerate_spreads`` over both states pooled;
    ``None`` falls back to deciding it per state.
    """
    sub = df[df.is_los] if state.upper() == "LOS" else df[~df.is_los]
    out = {"state": state.upper(), "n": int(len(sub))}
    if len(sub) == 0:
        return out
    if degenerate is None:
        degenerate = degenerate_spreads(sub)

    def _mu_sig(vals, name, lsp=None):
        v = np.asarray(vals, dtype=float)
        finite = v[np.isfinite(v)]
        if lsp is not None and lsp in degenerate:
            out[f"mu_{name}"] = -np.inf
            out[f"sigma_{name}"] = 0.0
            out[f"n_{name}"] = 0
            out[f"note_{name}"] = NOTE8
            return
        out[f"mu_{name}"] = float(np.mean(finite)) if len(finite) else np.nan
        out[f"sigma_{name}"] = float(np.std(finite)) if len(finite) else np.nan
        out[f"n_{name}"] = int(len(finite))
        out[f"n_dropped_{name}"] = int(len(v) - len(finite))

    _mu_sig(sub.lgDS, "lgDS", "DS")
    for a in ("ASD", "ASA", "ZSD", "ZSA"):
        _mu_sig(sub[f"lg{a}"], f"lg{a}", a)

    # K is a property of the LOS state only.
    if state.upper() == "LOS":
        k = sub.K_dB.values
        k = k[np.isfinite(k)]
        out["mu_K_dB"] = float(np.mean(k)) if len(k) else np.nan
        out["sigma_K_dB"] = float(np.std(k)) if len(k) else np.nan
        out["n_K"] = int(len(k))
        # Historical ratio-of-means estimator, kept for continuity with
        # elevation_stats_to_ns3.py / plot_utils. Differs from mu_K_dB by a
        # Jensen gap; both are reported so the comparison is unambiguous.
        pl, pn = sub.get("_P_los"), sub.get("_P_nlos")
        if pl is not None and pn is not None and pn.sum() > 0:
            out["K_dB_ratio_of_means"] = float(10 * np.log10(pl.mean() / pn.mean()))

    sf = sub.SF_dB.values
    sf = sf[np.isfinite(sf)]
    out["mu_SF_dB"] = float(np.mean(sf)) if len(sf) else np.nan     # ~0 by construction
    out["sigma_SF_dB"] = float(np.std(sf)) if len(sf) else np.nan
    # RX the solver found no paths for: excluded from every loss statistic above.
    out["n_no_power"] = int((~np.isfinite(sub.loss_db.values)).sum())
    if "SF_dB_smooth" in sub:
        s2 = sub.SF_dB_smooth.values
        s2 = s2[np.isfinite(s2)]
        out["sigma_SF_dB_smooth"] = float(np.std(s2)) if len(s2) else np.nan
    out["mean_loss_db"] = float(np.nanmean(sub.loss_db))
    return out


# ─── Cross-correlation ────────────────────────────────────────────────────────

def lsp_correlation(df, state: str = "LOS", degenerate: set = None):
    """LSP cross-correlation matrix for one state, in ns-3's own column order."""
    sub = df[df.is_los] if state.upper() == "LOS" else df[~df.is_los]
    order = LSP_ORDER_LOS if state.upper() == "LOS" else LSP_ORDER_NLOS
    colmap = {"SF": "SF_dB", "K": "K_dB", "DS": "lgDS",
              "ASD": "lgASD", "ASA": "lgASA", "ZSD": "lgZSD", "ZSA": "lgZSA"}
    if degenerate is None:
        degenerate = degenerate_spreads(sub)

    n = len(order)
    C = np.full((n, n), np.nan)
    idx = [i for i, k in enumerate(order) if k not in degenerate]
    if len(idx) < 2 or len(sub) == 0:
        return C, list(order), 0

    M = np.column_stack([sub[colmap[order[i]]].values.astype(float) for i in idx])
    good = np.all(np.isfinite(M), axis=1)
    n_used = int(good.sum())
    if n_used >= 3:
        X = M[good]
        keep = X.std(axis=0) > 0
        if keep.sum() >= 2:
            Csub = np.corrcoef(X[:, keep], rowvar=False)
            kept = [idx[j] for j in np.where(keep)[0]]
            for a, ia in enumerate(kept):
                for b, ib in enumerate(kept):
                    C[ia, ib] = Csub[a, b]
    return C, list(order), n_used


# ─── Spatial decorrelation distance ───────────────────────────────────────────

def decorrelation_distance(values, xy, max_lag_m: float = 150.0,
                           bin_m: float = 2.0, max_points: int = 4000,
                           rho_floor: float = 0.05,
                           rng: np.random.Generator = None) -> dict:
    """Fit ``rho(d) = exp(-d / d_cor)`` to the empirical spatial autocorrelation."""
    from scipy.optimize import curve_fit

    v = np.asarray(values, dtype=float)
    p = np.asarray(xy, dtype=float)[:, :2]
    ok = np.isfinite(v) & np.all(np.isfinite(p), axis=1)
    v, p = v[ok], p[ok]
    if len(v) < 50 or np.std(v) == 0:
        return {"d_cor_m": np.nan, "d_cor_1e_m": np.nan, "n": int(len(v)),
                "d": np.array([]), "rho": np.array([]), "n_pairs": np.array([])}

    if len(v) > max_points:
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(len(v), size=max_points, replace=False)
        v, p = v[sel], p[sel]

    z = (v - v.mean()) / v.std()
    iu = np.triu_indices(len(z), k=1)
    d = np.hypot(p[iu[0], 0] - p[iu[1], 0], p[iu[0], 1] - p[iu[1], 1])
    prod = z[iu[0]] * z[iu[1]]

    m = d <= max_lag_m
    d, prod = d[m], prod[m]
    if len(d) < 100:
        return {"d_cor_m": np.nan, "d_cor_1e_m": np.nan, "n": int(len(z)),
                "d": np.array([]), "rho": np.array([]), "n_pairs": np.array([])}

    nb = max(int(np.ceil(max_lag_m / bin_m)), 2)
    which = np.clip((d / bin_m).astype(int), 0, nb - 1)
    cnt = np.bincount(which, minlength=nb).astype(float)
    ssum = np.bincount(which, weights=prod, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = ssum / cnt
    centres = (np.arange(nb) + 0.5) * bin_m

    valid = (cnt >= 20) & np.isfinite(rho)
    # Adaptive truncation: keep only lags up to the first bin that falls below
    # rho_floor, so the long zero-correlation tail cannot dominate the fit.
    below_floor = np.where(valid & (rho < rho_floor))[0]
    if len(below_floor):
        valid = valid & (np.arange(nb) <= below_floor[0])

    d_cor = np.nan
    if valid.sum() >= 3:
        try:
            popt, _ = curve_fit(lambda dd, dc: np.exp(-dd / dc),
                                centres[valid], rho[valid],
                                p0=[30.0], bounds=(1.0, 500.0), maxfev=10000)
            d_cor = float(popt[0])
        except Exception:
            d_cor = np.nan

    d_1e = np.nan
    below = np.where(valid & (rho < np.exp(-1.0)))[0]
    if len(below):
        d_1e = float(centres[below[0]])

    return {"d_cor_m": d_cor, "d_cor_1e_m": d_1e, "n": int(len(z)),
            "d": centres[valid], "rho": rho[valid], "n_pairs": cnt[valid]}


# ─── Per-elevation assembly ───────────────────────────────────────────────────

LSP_FOR_DCOR = {"SF": "SF_dB", "K": "K_dB", "DS": "lgDS",
                "ASA": "lgASA", "ZSA": "lgZSA"}


def elevation_lsp_rows(elev_deg: float, res: dict, label: str = "",
                       sf_smooth_radius_m: float = 5.0,
                       with_decorrelation: bool = True) -> list:
    """Both state rows (LOS and NLOS) for one elevation angle."""
    df = per_rx_lsp_frame(res, sf_smooth_radius_m=sf_smooth_radius_m)
    degenerate = degenerate_spreads(df)   # once, over both states
    rows = []
    for state in ("LOS", "NLOS"):
        row = lsp_marginals(df, state, degenerate=degenerate)
        row["elevation_deg"] = float(elev_deg)
        row["config"] = label
        row["frequency_hz"] = float(res.get("frequency", np.nan))
        row["los_probability"] = float(np.mean(df.is_los))
        row["n_rx_total"] = int(len(df))

        # Areal K over the whole scene, not the state subset: the quantity
        # run_elevation_sim.py logs as K_mean. A whole-scene diagnostic, not
        # what the report tabulates (see K_dB_ratio_of_means in lsp_marginals).
        # Computed here rather than in lsp_marginals because that function only
        # ever sees one state's receivers.
        _pl, _pn = df._P_los, df._P_nlos
        row["K_dB_ratio_all"] = (
            float(10 * np.log10(_pl.mean() / _pn.mean()))
            if _pn.mean() > 0 else np.nan)

        C, labels, n_used = lsp_correlation(df, state, degenerate=degenerate)
        row["_corr"] = C
        row["_corr_labels"] = labels
        row["corr_n"] = n_used

        if with_decorrelation:
            sub = df[df.is_los] if state == "LOS" else df[~df.is_los]
            for name, col in LSP_FOR_DCOR.items():
                if col not in sub or len(sub) < 50:
                    continue
                dd = decorrelation_distance(sub[col].values, sub[["x", "y"]].values)
                row[f"dcor_{name}_m"] = dd["d_cor_m"]
                row[f"dcor1e_{name}_m"] = dd["d_cor_1e_m"]
        rows.append(row)
    return rows


def build_lsp_table(results_dirs, elevation_angles, labels=None,
                    sf_smooth_radius_m: float = 5.0,
                    with_decorrelation: bool = True):
    """Full RT LSP table across elevations (and optionally several result dirs)."""
    import pandas as pd

    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    labels = labels or [str(d) for d in results_dirs]

    rows = []
    for d, lab in zip(results_dirs, labels):
        allres = load_all_results(d, elevation_angles)
        for elev, res in sorted(allres.items()):
            rows.extend(elevation_lsp_rows(
                elev, res, label=lab,
                sf_smooth_radius_m=sf_smooth_radius_m,
                with_decorrelation=with_decorrelation))
    df = pd.DataFrame(rows)
    lead = ["config", "frequency_hz", "elevation_deg", "state", "n",
            "los_probability"]
    return df[lead + [c for c in df.columns if c not in lead]]


