"""Delay-cluster and PDP peak extraction in the TR 38.811 parameter domains.

See docs/analysis.md and validation/METHODS.md.
"""

from __future__ import annotations

import numpy as np

from Ray_Tracing.src.utils.channel_utils import rms_angular_spread
from Ray_Tracing.src.utils.checkpoint_utils import load_all_results

BIN_NS = 1.0            # PDP bin width; float32 dtau_s supports ~0.2 ns
PROMINENCE_DB = 3.0     # peak prominence for a delay cluster
MIN_SEP_NS = 5.0        # minimum cluster separation
DYN_RANGE_DB = 30.0     # PDP bins below this (rel. peak) are not cluster candidates
MAX_NS = 2000.0        # observed excess-delay spans reach ~1850 ns


def _pdp(dtau_ns, power, bin_ns=BIN_NS, max_ns=MAX_NS):
    """Binned power delay profile. Returns (bin_centres_ns, power_per_bin).

    Weights are cast to float64: np.histogram accumulates in the weight dtype, and
    with float32 the weak long-delay tail (~1e-30 into bins holding ~1e-16) rounds
    away entirely -- which halves the delay spread, since DS weights by tau^2.
    """
    edges = np.arange(0.0, max_ns + bin_ns, bin_ns)
    h, _ = np.histogram(dtau_ns, bins=edges, weights=np.asarray(power, dtype=np.float64))
    return edges[:-1] + bin_ns / 2.0, h


def _normalize(rec):
    """Common view over a raw_results or cluster_samples record, delay-sorted.

    Both stores are unsorted in delay (cluster_samples is power-ordered), so this
    sorts. Angles are None for raw_results, which does not carry them.
    """
    if "dtau_s" in rec:                      # cluster_samples
        t = np.asarray(rec["dtau_s"], dtype=float)
        amp, prim = np.asarray(rec["amp"]), np.asarray(rec["prim_type"], dtype=int)
        ang = (np.asarray(rec["phi_r"], dtype=float),
               np.asarray(rec["theta_r"], dtype=float))
        src = "cluster_samples"
    else:                                    # raw_results: absolute delays, no angles
        t = np.asarray(rec["taus"], dtype=float)
        t = t - t.min()
        amp, prim = np.asarray(rec["amps"]), np.asarray(rec["inter_d0"], dtype=int)
        ang, src = None, "raw_results"
    o = np.argsort(t)
    # float64 power: the stored complex64 gives float32 magnitudes, whose dynamic
    # range (1e-16 down to 1e-30) does not survive summation in float32.
    return (t[o] * 1e9, (np.abs(amp[o]) ** 2).astype(np.float64), prim[o],
            None if ang is None else (ang[0][o], ang[1][o]), src)


def cluster_rx(rec, bin_ns=BIN_NS, prominence_db=PROMINENCE_DB,
               min_sep_ns=MIN_SEP_NS, dyn_range_db=DYN_RANGE_DB, max_ns=MAX_NS):
    """Delay clusters for one raw_results or cluster_samples record.

    Returns (clusters, meta). Each cluster carries power, mean delay and
    intra-cluster delay spread (cDS); arrival spreads (cASA, cZSA) only when the
    record carries angles.
    """
    from scipy.signal import find_peaks

    dtau_ns, power, prim, ang, src = _normalize(rec)
    phi_r, theta_r = ang if ang is not None else (None, None)

    # Exclude the specular LoS ray: TR 38.901 Sec 7.5 adds it separately, scaled by
    # the K-factor, outside the cluster structure. Leaving it in would put ~98% of
    # the power in one bin and drive every intra-cluster spread to zero.
    keep = (dtau_ns <= max_ns) & (prim != 0)
    p_los = float(power[(prim == 0)].sum())
    dtau_ns, power = dtau_ns[keep], power[keep]
    if phi_r is not None:
        phi_r, theta_r = phi_r[keep], theta_r[keep]
    total = power.sum()
    if total <= 0 or len(power) < 2:
        return [], {"rx_index": rec.get("rx_index"), "n_clusters": 0, "source": src}

    ctr, h = _pdp(dtau_ns, power, bin_ns, max_ns)
    hn = h / h.max()
    h_db = np.where(hn > 0, 10.0 * np.log10(np.maximum(hn, 1e-300)), -np.inf)
    floor = -abs(dyn_range_db)
    # Pad so a peak in the first or last bin is detectable -- find_peaks skips edges,
    # which would otherwise hide the earliest (strongest) scattered cluster.
    padded = np.concatenate(([floor - 1.0],
                             np.where(np.isfinite(h_db), h_db, floor - 1.0),
                             [floor - 1.0]))
    peaks, _ = find_peaks(padded, prominence=prominence_db,
                          distance=max(1, int(round(min_sep_ns / bin_ns))))
    peaks = peaks - 1
    peaks = peaks[(peaks >= 0) & (peaks < len(h_db))]
    peaks = peaks[h_db[peaks] > floor]
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(h))])

    # Assign every path to the nearest peak in delay.
    centres = ctr[peaks]
    assign = np.argmin(np.abs(dtau_ns[:, None] - centres[None, :]), axis=1)

    clusters = []
    for c in range(len(centres)):
        m = assign == c
        p = power[m]
        if p.sum() <= 0:
            continue
        t = dtau_ns[m]
        tbar = float(np.sum(p * t) / p.sum())
        cds = float(np.sqrt(max(np.sum(p * t**2) / p.sum() - tbar**2, 0.0)))
        if phi_r is not None:
            # rms_angular_spread returns radians; these columns are named _deg.
            _, casa = rms_angular_spread(phi_r[m], np.sqrt(p))
            _, czsa = rms_angular_spread(theta_r[m], np.sqrt(p))
            casa, czsa = np.degrees(casa), np.degrees(czsa)
        else:
            casa = czsa = np.nan
        clusters.append({
            "peak_ns": float(centres[c]), "mean_delay_ns": tbar,
            "power": float(p.sum()), "power_frac": float(p.sum() / total),
            "cDS_ns": cds, "cASA_deg": float(casa), "cZSA_deg": float(czsa),
            "n_paths": int(m.sum()),
        })
    clusters.sort(key=lambda c: c["mean_delay_ns"])

    meta = {"rx_index": rec.get("rx_index"), "has_los": bool(rec["has_los"]),
            "n_clusters": len(clusters), "n_paths_stored": int(len(power)),
            "n_paths_total": int(rec.get("n_paths_total", len(power))),
            "power_scattered": float(total), "power_los": p_los,
            "power_stored": float(total + p_los), "source": src}
    return clusters, meta


def _r_tau(clusters, ds_ns):
    """Moment estimator for 3GPP's delay scaling rTau from cluster excess delays."""
    if len(clusters) < 2 or not np.isfinite(ds_ns) or ds_ns <= 0:
        return np.nan
    t = np.array([c["mean_delay_ns"] for c in clusters])
    return float((t - t.min()).mean() / ds_ns)


def _per_cluster_shadowing_db(clusters, ds_ns, r_tau):
    """Std [dB] of cluster power about 3GPP's exponential power-delay law."""
    if len(clusters) < 3 or not np.isfinite(r_tau) or r_tau <= 1 or ds_ns <= 0:
        return np.nan
    t = np.array([c["mean_delay_ns"] for c in clusters])
    p = np.array([c["power"] for c in clusters])
    expected_db = -10.0 * np.log10(np.e) * (t - t.min()) * (r_tau - 1.0) / (r_tau * ds_ns)
    got_db = 10.0 * np.log10(np.maximum(p, 1e-300) / p.max())
    resid = got_db - (expected_db - expected_db.max())
    return float(np.std(resid))


def cluster_elevation(res, elev_deg, label="", store="raw_results", **kw):
    """Per-state cluster aggregates for one elevation's results dict.

    `store` defaults to raw_results: cluster_samples' top-N pruning destroys the
    delay statistics (see module docstring), so it is only valid for cASA/cZSA.
    """
    rows = []
    for state in ("LOS", "NLOS"):
        want = state == "LOS"
        per_rx = []
        for rec in res.get(store, []):
            if bool(rec["has_los"]) != want:
                continue
            cl, meta = cluster_rx(rec, **kw)
            if not cl:
                continue
            # DS over the same path set the clusters came from, so rTau is self-consistent.
            dt = np.array([c["mean_delay_ns"] for c in cl])
            w = np.array([c["power"] for c in cl])
            tb = float(np.sum(w * dt) / w.sum())
            ds = float(np.sqrt(max(np.sum(w * dt**2) / w.sum() - tb**2, 0.0)))
            rtau = _r_tau(cl, ds)
            per_rx.append({
                "n_clusters": len(cl),
                "cDS_ns": float(np.average([c["cDS_ns"] for c in cl], weights=w)),
                "cASA_deg": float(np.average([c["cASA_deg"] for c in cl], weights=w)),
                "cZSA_deg": float(np.average([c["cZSA_deg"] for c in cl], weights=w)),
                "rTau": rtau,
                "perClusterShadowingStd": _per_cluster_shadowing_db(cl, ds, rtau),
            })
        row = {"config": label, "frequency_hz": float(res.get("frequency", np.nan)),
               "elevation_deg": float(elev_deg), "state": state, "n_rx": len(per_rx),
               "store": store}
        for k in ("n_clusters", "cDS_ns", "cASA_deg", "cZSA_deg", "rTau",
                  "perClusterShadowingStd"):
            v = np.array([r[k] for r in per_rx], dtype=float) if per_rx else np.array([np.nan])
            v = v[np.isfinite(v)]
            row[f"mu_{k}"] = float(np.mean(v)) if len(v) else np.nan
            row[f"sigma_{k}"] = float(np.std(v)) if len(v) else np.nan
        rows.append(row)
    return rows


def cluster_table(results_dirs, elevations, labels=None, **kw):
    """Cluster aggregates across elevations and result dirs."""
    import pandas as pd
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    labels = labels or [str(d) for d in results_dirs]
    rows = []
    for d, lab in zip(results_dirs, labels):
        for elev, res in sorted(load_all_results(d, list(elevations)).items()):
            rows.extend(cluster_elevation(res, elev, label=lab, **kw))
    return pd.DataFrame(rows)


def cluster_sensitivity(res, prominences=(2.0, 3.0, 5.0, 8.0), state="LOS",
                        store="raw_results"):
    """numOfCluster vs peak-prominence threshold -- the method dependence, quantified."""
    want = state == "LOS"
    out = {}
    for pr in prominences:
        n = [cluster_rx(r, prominence_db=pr)[1]["n_clusters"]
             for r in res.get(store, []) if bool(r["has_los"]) == want]
        n = [x for x in n if x]
        out[pr] = (float(np.mean(n)), float(np.std(n)), len(n)) if n else (np.nan, np.nan, 0)
    return out


def ensemble_pdp(res, state="LOS", bin_ns=BIN_NS, max_ns=MAX_NS, normalize=True,
                 store="raw_results", include_los=True):
    """Power-averaged PDP over all captured UEs in one state.

    Averaged in power: cross-UE phase is meaningless (tau0_s differs by hundreds
    of carrier cycles), so amplitudes are never combined coherently across UEs.
    Each UE is normalised to unit power first, so no single strong UE dominates.
    """
    want = state == "LOS"
    acc, n = None, 0
    for rec in res.get(store, []):
        if bool(rec["has_los"]) != want:
            continue
        dtau_ns, p, prim, _, _ = _normalize(rec)
        if not include_los:
            m = prim != 0
            dtau_ns, p = dtau_ns[m], p[m]
        if p.sum() <= 0:
            continue
        ctr, h = _pdp(dtau_ns, p / p.sum(), bin_ns, max_ns)
        acc = h if acc is None else acc + h
        n += 1
    if not n:
        return np.array([]), np.array([]), 0
    acc /= n
    if normalize and acc.max() > 0:
        acc = acc / acc.max()
    return ctr, acc, n


def pdp_delay_spread_ns(ctr_ns, pdp):
    """Second moment of a PDP -- cross-check against the stored per-UE tau_rms."""
    p = np.asarray(pdp, dtype=float)
    if p.sum() <= 0:
        return np.nan
    t = np.asarray(ctr_ns, dtype=float)
    tbar = np.sum(p * t) / p.sum()
    return float(np.sqrt(max(np.sum(p * t**2) / p.sum() - tbar**2, 0.0)))
