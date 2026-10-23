"""Join the RT-derived LSP table against the TR 38.811 reference and score deltas.

Reference side comes from ns3_reference; RT side from lsp_stats. Cells that are
vacuous under NOTE 8 are reported as matches by construction, not scored.
See docs/analysis.md and validation/METHODS.md.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

import Ray_Tracing.src.utils.ns3_reference as ref
from Ray_Tracing.src.utils.lsp_stats import build_lsp_table

# lsp -> (rt_mu, rt_sigma, ref_mu, ref_sigma); None where a side has no counterpart.
MARGINALS = {
    "DS":  ("mu_lgDS",  "sigma_lgDS",  "uLgDS",  "sigLgDS"),
    "ASD": ("mu_lgASD", "sigma_lgASD", "uLgASD", "sigLgASD"),
    "ASA": ("mu_lgASA", "sigma_lgASA", "uLgASA", "sigLgASA"),
    "ZSD": ("mu_lgZSD", "sigma_lgZSD", "uLgZSD", "sigLgZSD"),
    "ZSA": ("mu_lgZSA", "sigma_lgZSA", "uLgZSA", "sigLgZSA"),
    "K":   ("mu_K_dB",  "sigma_K_dB",  "uK",     "sigK"),
    "SF":  (None,       "sigma_SF_dB", None,     "sigma_SF_dB"),   # 3GPP SF mean is 0 by definition
}

# NOTE 8: point-source TX, so both sides force these to -inf. Agreement is
# true by construction and is not evidence about the ray tracer.
DEGENERATE = ("ASD", "ZSD")

# K is defined for the LOS state only (ns-3's LSP_ORDER_NLOS omits it, and the
# reference table carries a placeholder 0 there).
LOS_ONLY = ("K",)

# (name, rt column in cluster_table, reference column, forced status or None).
# cASA/cZSA are flagged: the tabulated 11 deg / 7 deg are TR 38.901's terrestrial
# per-cluster constants, and they exceed the tabulated TOTAL ASA/ZSA at every
# elevation in both bands -- an intra-cluster spread cannot exceed the total, so
# they are not usable targets.
CLUSTER_FIELDS = (
    ("numOfCluster",           "mu_n_clusters",              "numOfCluster",           None),
    ("rTau",                   "mu_rTau",                    "rTau",                   None),
    ("cDS",                    "mu_cDS_ns",                  "cDS",                    None),
    ("perClusterShadowingStd", "mu_perClusterShadowingStd",  "perClusterShadowingStd", None),
    ("cASA",                   "mu_cASA_deg",                "cASA",                   "inconsistent reference"),
    ("cZSA",                   "mu_cZSA_deg",                "cZSA",                   "inconsistent reference"),
    ("cASD",                   None,                         "cASD",                   "vacuous (NOTE 8)"),
    ("raysPerCluster",         None,                         "raysPerCluster",         "not an observable"),
)

# Scenarios for which ns3_reference wires up SFCL / LOS-probability.
# Per-LSP decorrelation literals are a separate matter -- see compare_decorrelation.
_FULL_SCENARIOS = ("DenseUrban", "Urban", "Suburban", "Rural")

# Below this many receivers a (state, elevation) cell is noise, not a measurement:
# the 90 deg NLOS cell holds 48 UEs and reports sigma_SF = 14.8 dB.
MIN_N = 200

# Cluster tables sample a few hundred receivers per elevation, not all 10 000, so
# they need their own floor: NLOS at 90 deg lands 4 receivers.
MIN_N_CLUSTER = 30

# SF is reported, never scored. TR 38.811's sigma_SF is a variance over UE
# positions AND environment realisations; the RT residual is taken within one
# fixed building layout, so by the law of total variance it estimates only the
# within-scene term and sits below the tabulated value by construction (measured
# LOS 0.03-0.74 dB against a tabulated 0.72-1.79 dB). `rt_alt` carries
# sigma_SF_dB_smooth so the share of the residual below the 5 m smoothing radius
# -- which 3GPP removes by local averaging and this estimator does not -- is
# visible next to the number.
SCENE_LIMITED = "scene-limited"


def _band(freq_hz):
    """(band, off_band) for a carrier; off_band is True in the 3-13 GHz Ku gap."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        band = ref.band_for_frequency(freq_hz)
    return band, bool(caught)


def _delta(rt, rf):
    """rt - rf, or NaN when either side is non-finite (e.g. both -inf under NOTE 8)."""
    if not (np.isfinite(rt) and np.isfinite(rf)):
        return np.nan
    return float(rt - rf)


def rt_table(results_dirs, elevations, labels=None, **kw):
    """RT LSP table. Thin wrapper so callers can cache it across the comparisons."""
    return build_lsp_table(list(results_dirs), list(elevations), labels=labels, **kw)


def compare_marginals(rt, scenario="Suburban"):
    """Long-form RT vs reference marginals."""
    import pandas as pd

    rf = ref.reference_frame(scenario)
    zen = {s: ref.zenith_anomaly_check(scenario, s) for s in ("LOS", "NLOS")}

    rows = []
    for _, r in rt.iterrows():
        band, off_band = _band(float(r["frequency_hz"]))
        state, elev = str(r["state"]), float(r["elevation_deg"])
        m = rf[(rf.state == state) & (rf.band == band)
               & (rf.elevation_deg == elev)]
        rref = m.iloc[0] if len(m) else None

        suspect = bool(zen[state].get(int(elev), {}).get("s_band_suspect", False)) \
            if band == "S" else False
        n_rx = int(r.get("n", 0))

        for lsp, (rt_mu, rt_sig, rf_mu, rf_sig) in MARGINALS.items():
            if lsp in LOS_ONLY and state != "LOS":
                continue
            for stat, rk, fk in (("mu", rt_mu, rf_mu), ("sigma", rt_sig, rf_sig)):
                if rk is None:
                    continue
                rt_v = float(r.get(rk, np.nan))
                rf_v = float(rref[fk]) if rref is not None else np.nan

                if rref is None:
                    status = "no reference"
                elif lsp in DEGENERATE:
                    status = "vacuous (NOTE 8)"
                elif off_band:
                    status = "off-band"
                elif n_rx < MIN_N or not np.isfinite(rt_v):
                    status = "insufficient sample"
                elif lsp == "SF":
                    status = SCENE_LIMITED
                else:
                    status = "ok"

                if lsp == "K" and stat == "mu":
                    alt = float(r.get("K_dB_ratio_of_means", np.nan))
                elif lsp == "SF" and stat == "sigma":
                    alt = float(r.get("sigma_SF_dB_smooth", np.nan))
                else:
                    alt = np.nan

                rows.append({
                    "config": r["config"], "elevation_deg": elev, "state": state,
                    "band": band, "lsp": lsp, "stat": stat,
                    "rt": rt_v, "ref": rf_v, "delta": _delta(rt_v, rf_v),
                    "rt_alt": alt, "n": n_rx,
                    "status": status,
                    "zenith_suspect": suspect and lsp in ("ZSA", "ZSD"),
                })
    return pd.DataFrame(rows)


def compare_clusters(rtc, scenario="Suburban"):
    """RT cluster statistics vs the TR 38.811 cluster columns.

    `rtc` is a cluster_stats.cluster_table(...) frame. Same long-form columns and
    same status vocabulary as compare_marginals.
    """
    import pandas as pd

    rf = ref.reference_frame(scenario)
    rows = []
    for _, r in rtc.iterrows():
        band, off_band = _band(float(r["frequency_hz"]))
        state, elev = str(r["state"]), float(r["elevation_deg"])
        m = rf[(rf.state == state) & (rf.band == band) & (rf.elevation_deg == elev)]
        rref = m.iloc[0] if len(m) else None

        n_rx = int(r["n_rx"])
        for name, rt_col, rf_col, note in CLUSTER_FIELDS:
            rt_v = float(r.get(rt_col, np.nan)) if rt_col else np.nan
            rf_v = float(rref[rf_col]) if rref is not None else np.nan
            if rref is None:
                status = "no reference"
            elif note:
                status = note
            elif off_band:
                status = "off-band"
            elif n_rx < MIN_N_CLUSTER or not np.isfinite(rt_v):
                status = "insufficient sample"
            else:
                status = "ok"
            rows.append({
                "config": r["config"], "elevation_deg": elev, "state": state,
                "band": band, "param": name, "rt": rt_v, "ref": rf_v,
                "delta": _delta(rt_v, rf_v), "n_rx": n_rx,
                "method": str(r.get("method", "")), "status": status,
            })
    return pd.DataFrame(rows)


def compare_correlation(rt, scenario="Suburban", state="LOS"):
    """Per-(config, elevation) RT vs reference LSP cross-correlation."""
    import pandas as pd

    C_ref = ref.correlation_frame(scenario, state)
    rows, matrices = [], {}

    for _, r in rt[rt.state == state].iterrows():
        C = r.get("_corr")
        labels = r.get("_corr_labels")
        if C is None or labels is None or np.size(C) == 0:
            continue
        labels = list(labels)
        R = C_ref.loc[labels, labels].values
        C = np.asarray(C, dtype=float)
        D = C - R
        # ASD/ZSD rows and columns are degenerate under NOTE 8; exclude them
        # from the norms so a vacuous block cannot flatter the agreement.
        keep = [i for i, l in enumerate(labels) if l not in DEGENERATE]
        Dk = D[np.ix_(keep, keep)]

        key = (r["config"], float(r["elevation_deg"]))
        matrices[key] = (pd.DataFrame(C, index=labels, columns=labels),
                         pd.DataFrame(R, index=labels, columns=labels),
                         pd.DataFrame(D, index=labels, columns=labels))
        rows.append({
            "config": r["config"], "elevation_deg": float(r["elevation_deg"]),
            "state": state, "n_lsp": len(labels),
            "n_rx": int(r.get("corr_n", 0)),
            "frobenius": float(np.linalg.norm(Dk)),
            "max_abs_delta": float(np.abs(Dk).max()),
            "mean_abs_delta": float(np.abs(Dk).mean()),
        })
    return pd.DataFrame(rows), matrices


def compare_decorrelation(rt, scenario="Suburban"):
    """RT decorrelation distances vs the TR 38.811 literals."""
    import pandas as pd

    if scenario not in _FULL_SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}, "
                         f"expected one of {_FULL_SCENARIOS}")

    tbl = ref.TR38811_CORR_DIST_M.get(scenario) or {}
    if not tbl:
        raise ValueError(
            f"per-LSP decorrelation literals not transcribed for scenario "
            f"{scenario!r}. ns-3 encodes none of them; fill in "
            f"ns3_reference.TR38811_CORR_DIST_M[{scenario!r}] from TR 38.811 "
            f"Table 6.7.2-1a/2a to enable this comparison.")
    sf_parsed = ref.parse_shadowing_corr_distance(scenario)

    rows = []
    for _, r in rt.iterrows():
        state = str(r["state"])
        for lsp, d_ref in tbl.get(state, {}).items():
            col = f"dcor_{lsp}_m"
            if col not in r:
                continue
            d_rt = float(r[col])
            rows.append({
                "config": r["config"], "elevation_deg": float(r["elevation_deg"]),
                "state": state, "lsp": lsp,
                "rt_m": d_rt, "ref_m": float(d_ref),
                "rt_1e_m": float(r.get(f"dcor1e_{lsp}_m", np.nan)),
                "ratio": d_rt / d_ref if d_ref else np.nan,
                # SF is the one entry ns-3 also states in source; use it to
                # check the hand-transcribed literals above.
                "ref_parsed_m": sf_parsed[f"{state}_m"] if lsp == "SF" else np.nan,
            })
    return pd.DataFrame(rows)


def compare_clutter_loss(rt, scenario="Suburban"):
    """RT NLOS excess loss vs the tabulated clutter loss."""
    import pandas as pd

    rf = ref.reference_frame(scenario)
    if "clutter_loss_dB" not in rf.columns:
        raise ValueError(f"no clutter-loss table wired for scenario {scenario!r}")

    rows = []
    for (cfg, elev), g in rt.groupby(["config", "elevation_deg"]):
        los = g[g.state == "LOS"]
        nlos = g[g.state == "NLOS"]
        if not len(los) or not len(nlos):
            continue
        l, n = los.iloc[0], nlos.iloc[0]
        band, off_band = _band(float(l["frequency_hz"]))
        m = rf[(rf.state == "NLOS") & (rf.band == band)
               & (rf.elevation_deg == float(elev))]
        if not len(m):
            continue

        rt_v = float(n.get("mean_loss_db", np.nan)) - float(l.get("mean_loss_db", np.nan))
        rf_v = float(m.iloc[0]["clutter_loss_dB"])
        n_los, n_nlos = int(l.get("n", 0)), int(n.get("n", 0))

        if off_band:
            status = "off-band"
        elif min(n_los, n_nlos) < MIN_N or not np.isfinite(rt_v):
            status = "insufficient sample"
        else:
            status = "ok"

        rows.append({
            "config": cfg, "elevation_deg": float(elev), "band": band,
            "rt": rt_v, "ref": rf_v, "delta": _delta(rt_v, rf_v),
            "n_los": n_los, "n_nlos": n_nlos, "status": status,
        })
    return pd.DataFrame(rows)


def compare_los_probability(rt, scenario="Suburban"):
    """RT geometric LoS fraction vs the tabulated LoS probability."""
    import pandas as pd

    rf = ref.reference_frame(scenario)
    if "los_prob" not in rf.columns:
        raise ValueError(f"no LOS-probability table wired for scenario {scenario!r}")

    rows = []
    for _, r in rt[rt.state == "LOS"].iterrows():
        band, off_band = _band(float(r["frequency_hz"]))
        elev = float(r["elevation_deg"])
        m = rf[(rf.state == "LOS") & (rf.band == band) & (rf.elevation_deg == elev)]
        if not len(m):
            continue
        p_rt, p_ref = float(r["los_probability"]), float(m.iloc[0]["los_prob"])
        rows.append({
            "config": r["config"], "elevation_deg": elev, "band": band,
            "rt": p_rt, "ref": p_ref, "delta": p_rt - p_ref,
            "status": "off-band" if off_band else "ok",
        })
    return pd.DataFrame(rows)


def run(results_dirs, elevations, out_dir, scenario="Suburban", labels=None,
        rt=None, rtc=None):
    """Run every comparison, write CSVs to `out_dir`, print a summary."""
    os.makedirs(out_dir, exist_ok=True)
    elevations = list(elevations)

    if rt is None:
        rt = rt_table(results_dirs, elevations, labels=labels)

    marg = compare_marginals(rt, scenario)
    losp = compare_los_probability(rt, scenario)
    corr = {s: compare_correlation(rt, scenario, s)[0] for s in ("LOS", "NLOS")}

    out = {"marginals": marg, "los_probability": losp,
           "clutter_loss": compare_clutter_loss(rt, scenario)}

    # Decorrelation is the one comparison that can be legitimately unavailable
    # (no transcribed literals for this scenario). Skip it loudly rather than
    # losing every other CSV to the exception.
    skipped = None
    try:
        out["decorrelation"] = compare_decorrelation(rt, scenario)
    except ValueError as e:
        skipped = str(e)
        warnings.warn(f"skipping decorrelation comparison: {e}", stacklevel=2)

    if rtc is not None and len(rtc):
        out["clusters"] = compare_clusters(rtc, scenario)
    for s, c in corr.items():
        out[f"correlation_{s}"] = c
    for name, df in out.items():
        p = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(p, index=False)
        print(f"[validate_3gpp] wrote {p}  ({len(df)} rows)")

    scored = marg[marg.status == "ok"]
    print(f"\nscenario={scenario}  elevations={len(elevations)}  "
          f"rows={len(marg)}  scored={len(scored)}")
    if skipped:
        print(f"\nNOT SCORED -- decorrelation: {skipped}")
    print("status counts:", marg.status.value_counts().to_dict())
    if len(scored):
        print("\nmean |delta| of scored marginals, by LSP:")
        print(scored.assign(a=scored.delta.abs())
                    .groupby(["lsp", "stat"])["a"].mean()
                    .round(3).to_string())
    if marg.zenith_suspect.any():
        n = int(marg.zenith_suspect.sum())
        print(f"\n{n} ZSA/ZSD rows flagged: {ref.ZENITH_ANOMALY_NOTE}")
    return out
