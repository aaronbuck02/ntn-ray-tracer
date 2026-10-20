"""Joint delay-angle multipath clustering in MCD space.

TR 38.811 tabulates cluster parameters that TR 38.901 Sec 7.5 *generates*: the
spec defines no way to identify a cluster in data. `cluster_stats.py` picks peaks
of the binned PDP, which is delay-only and counts prominences of a diffuse
continuum -- it returns 6-12 clusters against a tabulated 3-4. This module uses
the estimator family the measurement literature actually used to produce those
tables: power-weighted KPowerMeans over the Multipath Component Distance
(Steinbauer; Czink), with the cluster count chosen by a validity index and the
spec's own -25 dB pruning applied afterwards.

Consumes `cluster_samples` (the only store carrying per-path angles). Emits the
column names `validate_3gpp.compare_clusters` reads, so it is a drop-in for
`cluster_stats.cluster_table`.

Departure angles are dropped: measured spread within one receiver is 0.0006 deg
in zenith and 0.002 deg in azimuth, i.e. TR 38.811 NOTE 8 as data.
"""

from __future__ import annotations

import numpy as np

from Ray_Tracing.src.utils.channel_utils import rms_angular_spread
from Ray_Tracing.src.utils.checkpoint_utils import load_results
from Ray_Tracing.src.utils.cluster_stats import _normalize

ZETA = 1.0                  # delay/angle weighting in MCD
DELAY_NORM = "spread"       # "spread" (whiten by sigma_tau) | "czink" (span^2)
PATH_DYN_RANGE_DB = 40.0    # per-path floor; near-no-op, see module notes
CLUSTER_PRUNE_DB = 25.0     # TR 38.901 Sec 7.5 Step 6
K_MAX = 12
N_INIT = 4
MAX_ITER = 50
MAX_NS = 2000.0
VALIDITY = "xb"
METHOD = "kpowermeans"
XB_FLATNESS_THRESH = 0.15
# Parsimony tie-break: take the smallest K scoring within this fraction of the
# best. Ray fans are elongated in the embedding, so splitting one true cluster
# often wins the index by a hair; without the tolerance a near-tie at large K
# silently over-counts.
K_TOL = 0.10

# TR 38.901 Table 7.5-3 ray offsets, and Table 7.5-2 C_phi (NLOS).
RAY_OFFSETS = np.array([0.0447, 0.1413, 0.2492, 0.3715, 0.5129,
                        0.6797, 0.8844, 1.1481, 1.5195, 2.1551])
RAY_OFFSETS = np.concatenate([RAY_OFFSETS, -RAY_OFFSETS])
_C_PHI = {4: 0.779, 5: 0.860, 8: 1.018, 10: 1.090, 11: 1.123, 12: 1.146,
          14: 1.190, 15: 1.211, 16: 1.226, 19: 1.273, 20: 1.289}


# ─── Metric ───────────────────────────────────────────────────────────────────

def arrival_unit_vectors(theta_r, phi_r):
    """Unit vectors on the arrival sphere, shape (N, 3)."""
    th, ph = np.asarray(theta_r, float), np.asarray(phi_r, float)
    st = np.sin(th)
    return np.column_stack([st * np.cos(ph), st * np.sin(ph), np.cos(th)])


def mcd_embed(dtau_ns, power, theta_r, phi_r, zeta=ZETA, delay_norm=DELAY_NORM):
    """Embed paths in R^4 so Euclidean distance IS the MCD. Returns (X, scales).

    Angular half of the MCD is Steinbauer's chord distance
    ||Omega_i - Omega_j||/2 = sin(dpsi/2), which needs no wrap convention -- the
    reason it is correct for zenith where the circular estimator is not.

    Delay half is whitened by the power-weighted delay spread. Czink's published
    sigma_tau/span^2 form is selectable but wrong here: span ~470 ns against
    sigma_tau ~28 ns compresses the delay axis by (span/sigma_tau)^2 ~ 280 and the
    clustering degenerates to angle-only.
    """
    Om = arrival_unit_vectors(theta_r, phi_r)
    p = np.asarray(power, dtype=np.float64)
    w = p / p.sum()
    Om_bar = w @ Om
    sig_om = 0.5 * float(np.sqrt(np.sum(w * np.sum((Om - Om_bar) ** 2, axis=1))))
    t = np.asarray(dtau_ns, dtype=np.float64)
    tbar = float(w @ t)
    sig_tau = float(np.sqrt(max(w @ (t - tbar) ** 2, 0.0)))

    # sig_om == 0 means no angular information (the delay-only form used by
    # truncation_bias); fall back to plain whitened delay so the embedding does
    # not collapse to a point.
    ref_om = sig_om if sig_om > 0 else 1.0
    if delay_norm == "czink":
        span = float(t.max() - t.min())
        s = zeta * sig_tau / span ** 2 if span > 0 else 0.0
    else:
        s = zeta * ref_om / sig_tau if sig_tau > 0 else 0.0

    X = np.column_stack([0.5 * Om, s * (t - tbar)])
    return X, {"sigma_omega": sig_om, "sigma_tau_ns": sig_tau, "delay_scale": s}


def mcd_matrix(X):
    """Pairwise MCD via the Gram trick."""
    g = X @ X.T
    d = np.diag(g)
    return np.sqrt(np.maximum(d[:, None] + d[None, :] - 2.0 * g, 0.0))


# ─── KPowerMeans ──────────────────────────────────────────────────────────────

def _sqdist(X, x2, C):
    """(N, K) squared distances, one BLAS call, no N*K*d temporary."""
    return x2[:, None] - 2.0 * (X @ C.T) + np.einsum("kj,kj->k", C, C)[None, :]


def _kmeanspp_power(X, p, K, rng):
    """Power-weighted k-means++ seeding."""
    n = len(X)
    idx = [int(rng.choice(n, p=p / p.sum()))]
    d2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
    for _ in range(1, K):
        w = p * d2
        s = w.sum()
        j = int(rng.choice(n, p=w / s)) if s > 0 else int(rng.integers(n))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((X - X[j]) ** 2, axis=1))
    return np.array(idx)


def kpowermeans(X, p, K, rng, n_init=N_INIT, max_iter=MAX_ITER):
    """Power-weighted Lloyd in the R^4 MCD embedding. Returns (J, labels, C)."""
    x2 = np.einsum("ij,ij->i", X, X)
    best = None
    for _ in range(n_init):
        C = X[_kmeanspp_power(X, p, K, rng)].copy()
        prev = None
        for _ in range(max_iter):
            D = _sqdist(X, x2, C)
            lab = D.argmin(1)
            if prev is not None and np.array_equal(lab, prev):
                break
            prev = lab
            dmin = D.min(1)
            for k in range(K):
                m = lab == k
                if m.any():
                    C[k] = (p[m, None] * X[m]).sum(0) / p[m].sum()
                else:
                    C[k] = X[int(np.argmax(p * dmin))]
        D = _sqdist(X, x2, C)
        lab = D.argmin(1)
        J = float((p * np.maximum(D.min(1), 0.0)).sum())
        if best is None or J < best[0]:
            best = (J, lab.copy(), C.copy())
    return best


def xie_beni(J, C, p):
    """J / (P_tot * min inter-centroid squared distance)."""
    if len(C) < 2:
        return np.inf
    d2 = mcd_matrix(C) ** 2
    np.fill_diagonal(d2, np.inf)
    dmin = float(d2.min())
    return np.inf if dmin <= 0 else float(J / (p.sum() * dmin))


def calinski_harabasz(X, p, labels, C):
    """Power-weighted CH. Monotone in K on a continuum -- diagnostic only."""
    K = len(C)
    n = len(X)
    if K < 2 or n <= K:
        return np.nan
    gbar = (p[:, None] * X).sum(0) / p.sum()
    B = sum(p[labels == k].sum() * np.sum((C[k] - gbar) ** 2) for k in range(K))
    W = float((p * np.sum((X - C[labels]) ** 2, axis=1)).sum())
    return np.nan if W <= 0 else float((B / (K - 1)) / (W / (n - K)))


def davies_bouldin(X, p, labels, C):
    """Power-weighted DB. Averages the worst pair per cluster, so unlike XB one
    close pair among many does not dominate the score."""
    K = len(C)
    if K < 2:
        return np.inf
    S = np.zeros(K)
    for k in range(K):
        m = labels == k
        if m.any():
            S[k] = float((p[m] * np.linalg.norm(X[m] - C[k], axis=1)).sum() / p[m].sum())
    D = mcd_matrix(C)
    np.fill_diagonal(D, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (S[:, None] + S[None, :]) / D
    return float(np.nanmean(np.nanmax(np.where(np.isfinite(R), R, np.nan), axis=1)))


def kim_parks(curves):
    """Czink's CombinedValidate: min-max normalised under+over-partition sum."""
    ks = sorted(curves)
    vu = np.array([curves[k]["vu"] for k in ks])
    vo = np.array([curves[k]["vo"] for k in ks])

    def _nrm(v):
        lo, hi = np.nanmin(v), np.nanmax(v)
        return np.zeros_like(v) if hi <= lo else (v - lo) / (hi - lo)

    tot = _nrm(vu) + _nrm(vo)
    return {k: float(t) for k, t in zip(ks, tot)}


def select_k(X, p, rng, k_max=K_MAX, validity=VALIDITY, n_init=N_INIT,
             k_tol=K_TOL):
    """Sweep K, score by a validity index, return the winning partition + curves."""
    k_max = int(min(k_max, len(X) - 1))
    fits, curves = {}, {}
    for K in range(2, max(k_max, 2) + 1):
        J, lab, C = kpowermeans(X, p, K, rng, n_init=n_init)
        d2 = mcd_matrix(C) ** 2
        np.fill_diagonal(d2, np.inf)
        dmin = float(np.sqrt(d2.min()))
        fits[K] = (J, lab, C)
        curves[K] = {
            "J": J,
            "xb": xie_beni(J, C, p),
            "ch": calinski_harabasz(X, p, lab, C),
            "db": davies_bouldin(X, p, lab, C),
            "vu": float((p * np.linalg.norm(X - C[lab], axis=1)).sum() / p.sum()),
            "vo": float(1.0 / dmin) if dmin > 0 else np.inf,
        }
    if not fits:
        return {"k_hat": 1, "labels": np.zeros(len(X), int),
                "C": X.mean(0)[None, :], "curves": {}, "xb_flatness": 0.0}

    score = kim_parks(curves) if validity == "kimparks" \
        else {k: curves[k][validity] for k in curves}
    ks = sorted(score)
    v = np.array([score[k] if np.isfinite(score[k]) else np.nan for k in ks])
    if np.all(np.isnan(v)):
        k_hat = ks[0]
    elif validity == "ch":                    # CH is maximised, not minimised
        best = np.nanmax(v)
        k_hat = ks[int(np.argmax(v >= best * (1.0 - k_tol)))]
    else:
        best = np.nanmin(v)
        thr = best * (1.0 + k_tol) if best > 0 else best + k_tol
        k_hat = ks[int(np.argmax(np.nan_to_num(v, nan=np.inf) <= thr))]

    xb = np.array([curves[k]["xb"] for k in sorted(curves)])
    xb = xb[np.isfinite(xb)]
    flat = float((xb.max() - xb.min()) / xb.min()) if len(xb) and xb.min() > 0 else 0.0

    J, lab, C = fits[k_hat]
    return {"k_hat": int(k_hat), "labels": lab, "C": C, "curves": curves,
            "xb_flatness": flat}


# ─── Kernel power density (K-free cross-check) ────────────────────────────────

def kernel_power_density(X, p, m_frac=0.02, k_max=K_MAX):
    """Density-peaks clustering on the MCD matrix. Returns (labels, diag)."""
    D = mcd_matrix(X)
    n = len(X)
    m = int(min(max(1, np.ceil(m_frac * n)), n - 1))
    h = float(np.median(np.partition(D, m, axis=1)[:, m]))
    if h <= 0:
        pos = D[D > 0]
        h = float(pos.min()) if pos.size else 1.0

    f = (p[None, :] * np.exp(-(D ** 2) / (2.0 * h * h))).sum(1)
    order = np.argsort(-f)
    delta = np.zeros(n)
    nn = np.full(n, -1, dtype=int)
    for r in range(1, n):
        i = order[r]
        higher = order[:r]
        d = D[i, higher]
        j = int(np.argmin(d))
        delta[i] = d[j]
        nn[i] = higher[j]
    delta[order[0]] = delta.max() if n > 1 else 0.0

    gamma = f * delta
    g = np.sort(gamma)[::-1]
    kk = int(min(len(g) - 1, k_max))
    if kk < 1:
        K = 1
    else:
        ratios = g[:kk] / np.maximum(g[1:kk + 1], 1e-300)
        K = int(np.argmax(ratios)) + 1

    lab = np.full(n, -1, dtype=int)
    for k, i in enumerate(np.argsort(-gamma)[:K]):
        lab[i] = k
    for i in order:                       # descending density: parent already set
        if lab[i] < 0:
            lab[i] = lab[nn[i]]
    return lab, {"h": h, "k_hat": K}


# ─── Spec-matched postprocessing ──────────────────────────────────────────────

def prune_clusters_25db(labels, power, prune_db=CLUSTER_PRUNE_DB):
    """TR 38.901 Sec 7.5 Step 6. Returns (kept ids, cluster powers).

    Referenced to the strongest SCATTERED cluster, LoS excluded. TR 38.811
    tabulates numOfCluster as a state-dependent constant, so pruning against a
    K-inflated cluster 1 would make the count a function of the K-factor, which
    the reference table demonstrably is not.
    """
    K = int(labels.max()) + 1 if len(labels) else 0
    P = np.array([power[labels == k].sum() for k in range(K)], dtype=np.float64)
    if not len(P) or P.max() <= 0:
        return np.array([], dtype=int), P
    keep = np.where(P >= P.max() * 10.0 ** (-abs(prune_db) / 10.0))[0]
    return keep, P


def _r_tau_corrected(clusters, ds_ns):
    """rTau from cluster excess delays, debiased for the sorted-minimum shift.

    Step 5 makes tau'_n ~ Exp(m) with m = rTau*DS, and tau_n = sort(tau') - min.
    E[min] = m/N, so E[mean(tau_n)] = m(1 - 1/N). Omitting that factor -- as
    cluster_stats._r_tau does -- biases rTau low by 33% at N=3, 50% at N=2.
    """
    n = len(clusters)
    if n < 2 or not np.isfinite(ds_ns) or ds_ns <= 0:
        return np.nan
    t = np.array([c["mean_delay_ns"] for c in clusters])
    return float((t - t.min()).mean() / ((1.0 - 1.0 / n) * ds_ns))


def _per_cluster_shadowing_db(clusters, ds_ns, r_tau):
    """Std [dB] of cluster power about Step 6's exponential power-delay law."""
    if len(clusters) < 3 or not np.isfinite(r_tau) or r_tau <= 1 or ds_ns <= 0:
        return np.nan
    t = np.array([c["mean_delay_ns"] for c in clusters])
    p = np.array([c["power"] for c in clusters])
    expected = -10.0 * np.log10(np.e) * (t - t.min()) * (r_tau - 1.0) / (r_tau * ds_ns)
    got = 10.0 * np.log10(np.maximum(p, 1e-300) / p.max())
    return float(np.std(got - (expected - expected.max())))


def pooled_shadowing_db(pairs):
    """perClusterShadowingStd from (delay_ns, power, ds_ns, rTau) pooled over UEs.

    The per-receiver estimator needs N>=3, and the corrected N is 2-3, so most
    receivers return NaN. Pooling the residuals keeps the parameter estimable.
    """
    resid = []
    for t, p, ds_ns, r_tau in pairs:
        if len(t) < 2 or not np.isfinite(r_tau) or r_tau <= 1 or ds_ns <= 0:
            continue
        t, p = np.asarray(t, float), np.asarray(p, float)
        expected = -10.0 * np.log10(np.e) * (t - t.min()) * (r_tau - 1.0) / (r_tau * ds_ns)
        got = 10.0 * np.log10(np.maximum(p, 1e-300) / p.max())
        resid.append(got - (expected - expected.max()))
    if not resid:
        return np.nan
    r = np.concatenate(resid)
    return float(np.std(r[np.isfinite(r)]))


# ─── Per-receiver ─────────────────────────────────────────────────────────────

def cluster_id_rx(rec, *, zeta=ZETA, delay_norm=DELAY_NORM, method=METHOD,
                  validity=VALIDITY, k_max=K_MAX, n_init=N_INIT, k_tol=K_TOL,
                  path_dyn_range_db=PATH_DYN_RANGE_DB,
                  prune_db=CLUSTER_PRUNE_DB, max_ns=MAX_NS,
                  ds_full_ns=None, ds_source="self", seed=0,
                  angles_required=True):
    """Joint delay-angle clusters for one cluster_samples record.

    Returns (clusters, meta). Set angles_required=False to run the delay-only
    form on an angle-less record (used by truncation_bias).
    """
    dtau_ns, power, prim, ang, src = _normalize(rec)
    if ang is None:
        if angles_required:
            raise ValueError("record carries no angles; cluster_samples required")
        phi_r = np.zeros_like(dtau_ns)
        theta_r = np.full_like(dtau_ns, np.pi / 2)
    else:
        phi_r, theta_r = ang

    # LoS ray is added outside the cluster structure, scaled by K (TR 38.901
    # Sec 7.5); leaving it in puts ~98% of the power in one point.
    keep = (dtau_ns <= max_ns) & (prim != 0)
    p_los = float(power[prim == 0].sum())
    dtau_ns, power = dtau_ns[keep], power[keep]
    phi_r, theta_r = phi_r[keep], theta_r[keep]

    meta = {"rx_index": rec.get("rx_index"), "has_los": bool(rec["has_los"]),
            "source": src, "method": method, "power_los": p_los,
            "n_paths_total": int(rec.get("n_paths_total", len(power)))}
    if len(power) < 3 or power.sum() <= 0:
        meta.update(n_clusters=0, k_hat=0, xb_flatness=np.nan, trunc_frac=np.nan)
        return [], meta

    floor = power.max() * 10.0 ** (-abs(path_dyn_range_db) / 10.0)
    m = power >= floor
    if m.sum() >= 3:
        dtau_ns, power, phi_r, theta_r = dtau_ns[m], power[m], phi_r[m], theta_r[m]

    X, scales = mcd_embed(dtau_ns, power, theta_r, phi_r, zeta, delay_norm)
    rng = np.random.default_rng(seed)
    if method == "kpd":
        labels, diag = kernel_power_density(X, power, k_max=k_max)
        k_hat, flat = diag["k_hat"], np.nan
    else:
        sel = select_k(X, power, rng, k_max=k_max, validity=validity,
                       n_init=n_init, k_tol=k_tol)
        labels, k_hat, flat = sel["labels"], sel["k_hat"], sel["xb_flatness"]

    kept, P = prune_clusters_25db(labels, power, prune_db)

    # Delay spread the rTau denominator is taken from.
    w = power / power.sum()
    tbar = float(w @ dtau_ns)
    ds_self = float(np.sqrt(max(w @ (dtau_ns - tbar) ** 2, 0.0)))
    ds_ns = ds_self if (ds_source == "self" or ds_full_ns is None) else float(ds_full_ns)

    clusters = []
    for k in kept:
        sel_k = labels == k
        p, t = power[sel_k], dtau_ns[sel_k]
        if p.sum() <= 0:
            continue
        tb = float(np.sum(p * t) / p.sum())
        cds = float(np.sqrt(max(np.sum(p * t ** 2) / p.sum() - tb ** 2, 0.0)))
        th, ph = theta_r[sel_k], phi_r[sel_k]
        thb = float(np.sum(p * th) / p.sum())
        # Azimuth wraps, so use the circular estimator; zenith does not, so use
        # the linear one and carry the circular value to size the difference.
        casa = np.degrees(rms_angular_spread(ph, np.sqrt(p))[1])
        czsa = np.degrees(np.sqrt(max(np.sum(p * (th - thb) ** 2) / p.sum(), 0.0)))
        czsa_c = np.degrees(rms_angular_spread(th, np.sqrt(p))[1])
        clusters.append({
            "mean_delay_ns": tb, "power": float(p.sum()),
            "power_frac": float(p.sum() / power.sum()),
            "cDS_ns": cds, "cASA_deg": float(casa), "cZSA_deg": float(czsa),
            "cZSA_circ_deg": float(czsa_c), "n_paths": int(sel_k.sum()),
        })
    clusters.sort(key=lambda c: c["mean_delay_ns"])

    n_tot = meta["n_paths_total"]
    meta.update(n_clusters=len(clusters), k_hat=int(k_hat), xb_flatness=flat,
                ds_ns=ds_ns, ds_self_ns=ds_self, n_paths_used=int(len(power)),
                trunc_frac=float(1.0 - len(power) / n_tot) if n_tot else np.nan,
                **scales)
    return clusters, meta


# ─── Aggregation ──────────────────────────────────────────────────────────────

_AGG = ("n_clusters", "cDS_ns", "cASA_deg", "cZSA_deg", "cZSA_circ_deg",
        "rTau", "rTau_fullds", "perClusterShadowingStd", "k_hat", "trunc_frac")


def cluster_id_elevation(res, elev_deg, label="", store="cluster_samples", **kw):
    """Per-state cluster aggregates for one elevation's results dict."""
    ds_full = np.asarray(res.get("tau_rms_mean_s", []), dtype=float) * 1e9
    rows = []
    for state in ("LOS", "NLOS"):
        want = state == "LOS"
        per_rx, pooled = [], []
        for rec in res.get(store, []):
            if bool(rec["has_los"]) != want:
                continue
            i = rec.get("rx_index")
            ds_f = float(ds_full[i]) if (i is not None and i < len(ds_full)) else None
            cl, meta = cluster_id_rx(rec, ds_full_ns=ds_f, **kw)
            if not cl:
                continue
            t = np.array([c["mean_delay_ns"] for c in cl])
            p = np.array([c["power"] for c in cl])
            rtau = _r_tau_corrected(cl, meta["ds_ns"])
            rtau_f = _r_tau_corrected(cl, ds_f) if ds_f else np.nan
            pooled.append((t, p, meta["ds_ns"], rtau))
            per_rx.append({
                "n_clusters": len(cl),
                "cDS_ns": float(np.average([c["cDS_ns"] for c in cl], weights=p)),
                "cASA_deg": float(np.average([c["cASA_deg"] for c in cl], weights=p)),
                "cZSA_deg": float(np.average([c["cZSA_deg"] for c in cl], weights=p)),
                "cZSA_circ_deg": float(np.average([c["cZSA_circ_deg"] for c in cl],
                                                  weights=p)),
                "rTau": rtau, "rTau_fullds": rtau_f,
                "perClusterShadowingStd": _per_cluster_shadowing_db(
                    cl, meta["ds_ns"], rtau),
                "k_hat": meta["k_hat"], "trunc_frac": meta["trunc_frac"],
            })

        row = {"config": label, "frequency_hz": float(res.get("frequency", np.nan)),
               "elevation_deg": float(elev_deg), "state": state,
               "n_rx": len(per_rx), "store": store,
               "method": kw.get("method", METHOD), "zeta": kw.get("zeta", ZETA)}
        for k in _AGG:
            v = np.array([r[k] for r in per_rx], dtype=float) if per_rx \
                else np.array([np.nan])
            v = v[np.isfinite(v)]
            row[f"mu_{k}"] = float(np.mean(v)) if len(v) else np.nan
            row[f"sigma_{k}"] = float(np.std(v)) if len(v) else np.nan
        row["mu_perClusterShadowingStd_pooled"] = pooled_shadowing_db(pooled)
        rows.append(row)
    return rows


def cluster_id_table(results_dirs, elevations, labels=None, **kw):
    """Cluster aggregates across elevations and result dirs."""
    import pandas as pd
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    labels = labels or [str(d) for d in results_dirs]
    rows = []
    for d, lab in zip(results_dirs, labels):
        for elev in elevations:                 # one elevation in memory at a time
            res = load_results(d, elev)
            if res is None:
                continue
            rows.extend(cluster_id_elevation(res, elev, label=lab, **kw))
    return pd.DataFrame(rows)


# ─── Ground truth ─────────────────────────────────────────────────────────────

def synth_38901_channel(n_clusters=4, r_tau=2.3, ds_ns=30.0, asa_deg=40.0,
                        zsa_deg=20.0, c_ds_ns=1.6, c_asa_deg=11.0,
                        c_zsa_deg=7.0, xi_db=3.0, m_rays=20,
                        diffuse_floor=0, floor_db=-30.0, seed=0):
    """A TR 38.901 Sec 7.5 channel with known cluster structure, as a record.

    Packaged in the cluster_samples schema so cluster_id_rx consumes it unchanged.
    `diffuse_floor` adds isotropic weak paths to mimic the RT continuum.
    """
    rng = np.random.default_rng(seed)
    N = int(n_clusters)

    tau = -r_tau * ds_ns * np.log(rng.uniform(size=N))
    tau = np.sort(tau) - tau.min()
    Z = rng.normal(0.0, xi_db, size=N)
    P = np.exp(-tau * (r_tau - 1.0) / (r_tau * ds_ns)) * 10.0 ** (-Z / 10.0)
    P /= P.sum()

    c_phi = _C_PHI.get(N, 0.779)
    lp = np.sqrt(-np.log(np.maximum(P / P.max(), 1e-300)))
    phi = (2.0 * (asa_deg / 1.4) * lp / c_phi) * rng.choice([-1.0, 1.0], size=N) \
        + rng.normal(0.0, asa_deg / 7.0, size=N)
    theta = -zsa_deg * np.log(np.maximum(P / P.max(), 1e-300)) / c_phi \
        * rng.choice([-1.0, 1.0], size=N) + rng.normal(0.0, zsa_deg / 7.0, size=N) + 90.0

    off = RAY_OFFSETS[:m_rays] if m_rays <= len(RAY_OFFSETS) else RAY_OFFSETS
    t_l, p_l, ph_l, th_l = [], [], [], []
    for n in range(N):
        t_l.append(tau[n] + c_ds_ns * off)
        p_l.append(np.full(len(off), P[n] / len(off)))
        ph_l.append(phi[n] + c_asa_deg * off)
        th_l.append(theta[n] + c_zsa_deg * off)
    t = np.concatenate(t_l)
    p = np.concatenate(p_l)
    ph = np.concatenate(ph_l)
    th = np.concatenate(th_l)

    if diffuse_floor:
        n_d = int(diffuse_floor)
        t = np.concatenate([t, rng.uniform(0.0, max(tau.max(), ds_ns) * 3.0, n_d)])
        p = np.concatenate([p, np.full(n_d, P.max() / len(off) * 10 ** (floor_db / 10))])
        ph = np.concatenate([ph, rng.uniform(-180.0, 180.0, n_d)])
        th = np.concatenate([th, np.degrees(np.arccos(rng.uniform(-1.0, 1.0, n_d)))])

    return {
        "rx_index": 0, "tau0_s": 0.0,
        "dtau_s": (t * 1e-9).astype(np.float32),
        "amp": np.sqrt(p).astype(np.complex64),
        "phi_r": np.radians(ph).astype(np.float32),
        "theta_r": np.radians(th).astype(np.float32),
        "theta_t": np.zeros(len(t), np.float32),
        "phi_t": np.zeros(len(t), np.float32),
        "prim_type": np.full(len(t), 2, np.int8),
        "has_los": False, "n_paths_total": int(len(t)),
        # n_eff is the recoverable truth: Step 6 prunes drawn clusters below
        # -25 dB, so the number of clusters actually present is not always N.
        "_truth": {"n_clusters": N,
                   "n_eff": int((P >= P.max() * 10 ** -2.5).sum()),
                   "r_tau": r_tau, "ds_ns": ds_ns, "c_ds_ns": c_ds_ns,
                   "c_asa_deg": c_asa_deg, "c_zsa_deg": c_zsa_deg,
                   "tau": tau, "P": P},
    }


def synth_recovery(n_mc=200, compare_baseline=True, n_clusters=(2, 3, 4),
                   r_taus=(2.0, 2.3, 3.5), ds_list=(10.0, 30.0, 100.0),
                   diffuse_floor=0, seed=0, **kw):
    """Recovery of known cluster parameters, optionally vs the peak-picker."""
    import pandas as pd
    from Ray_Tracing.src.utils.cluster_stats import cluster_rx as _peak_rx

    rows = []
    for N in n_clusters:
        for rt in r_taus:
            for ds in ds_list:
                got, eff, got_b = [], [], []
                rec_rt, rec_cds, rec_asa, rec_zsa = [], [], [], []
                for i in range(n_mc):
                    rec = synth_38901_channel(n_clusters=N, r_tau=rt, ds_ns=ds,
                                              diffuse_floor=diffuse_floor,
                                              seed=seed + i)
                    cl, meta = cluster_id_rx(rec, seed=seed + i, **kw)
                    got.append(meta["n_clusters"])
                    eff.append(rec["_truth"]["n_eff"])
                    if cl:
                        p = np.array([c["power"] for c in cl])
                        rec_rt.append(_r_tau_corrected(cl, meta["ds_ns"]))
                        rec_cds.append(np.average([c["cDS_ns"] for c in cl], weights=p))
                        rec_asa.append(np.average([c["cASA_deg"] for c in cl], weights=p))
                        rec_zsa.append(np.average([c["cZSA_deg"] for c in cl], weights=p))
                    if compare_baseline:
                        got_b.append(_peak_rx(rec)[1]["n_clusters"])

                def _m(v):
                    v = np.asarray(v, float)
                    v = v[np.isfinite(v)]
                    return float(np.mean(v)) if len(v) else np.nan

                g, e = np.array(got, float), np.array(eff, float)
                rows.append({
                    "N_drawn": N, "rTau_true": rt, "ds_ns_true": ds, "n_mc": n_mc,
                    "N_eff": _m(eff),
                    "N_rec": _m(got), "N_bias": _m(g - e),
                    "N_rmse": float(np.sqrt(np.mean((g - e) ** 2))),
                    "N_baseline": _m(got_b) if compare_baseline else np.nan,
                    "rTau_rec": _m(rec_rt), "cDS_rec_ns": _m(rec_cds),
                    "cASA_rec_deg": _m(rec_asa), "cZSA_rec_deg": _m(rec_zsa),
                })
    return pd.DataFrame(rows)


# ─── Diagnostics ──────────────────────────────────────────────────────────────

def cluster_id_sensitivity(res, state="LOS", store="cluster_samples",
                           axes=None, n_rx=None, **base):
    """N and rTau against each method knob, one axis at a time."""
    import pandas as pd
    axes = axes or {
        "zeta": (0.25, 0.5, 1.0, 2.0, 4.0),
        "delay_norm": ("spread", "czink"),
        "prune_db": (20.0, 25.0, 30.0),
        "path_dyn_range_db": (30.0, 40.0, np.inf),
        "validity": ("xb", "ch", "kimparks"),
        "method": ("kpowermeans", "kpd"),
        "k_max": (8, 12, 20),
    }
    want = state == "LOS"
    recs = [r for r in res.get(store, []) if bool(r["has_los"]) == want]
    if n_rx:
        recs = recs[:int(n_rx)]

    rows = []
    for axis, values in axes.items():
        for v in values:
            kw = dict(base)
            kw[axis] = v
            ns, rt = [], []
            for i, rec in enumerate(recs):
                cl, meta = cluster_id_rx(rec, seed=i, **kw)
                if not cl:
                    continue
                ns.append(meta["n_clusters"])
                rt.append(_r_tau_corrected(cl, meta["ds_ns"]))
            rt = np.array(rt, float)
            rt = rt[np.isfinite(rt)]
            rows.append({"axis": axis, "value": v, "n_rx": len(ns),
                         "mu_n_clusters": float(np.mean(ns)) if ns else np.nan,
                         "sigma_n_clusters": float(np.std(ns)) if ns else np.nan,
                         "mu_rTau": float(np.mean(rt)) if len(rt) else np.nan})
    return pd.DataFrame(rows)


def truncation_bias(res, max_paths=2000, store="raw_results", **kw):
    """Full path set vs its own top-N, delay-only, per raw_results record.

    cluster_samples and raw_results share no receivers, so this within-record
    form is the only valid way to bound the top-N capture bias.
    """
    import pandas as pd

    rows = []
    for j, rec in enumerate(res.get(store, [])):
        t = np.asarray(rec["taus"], float)
        a = np.asarray(rec["amps"])
        if len(t) <= max_paths:
            continue
        p = (np.abs(a) ** 2).astype(np.float64)
        sel = np.argsort(p)[::-1][:max_paths]

        out = {}
        for tag, idx in (("full", slice(None)), ("trunc", sel)):
            sub = {"taus": t[idx], "amps": a[idx],
                   "inter_d0": np.asarray(rec["inter_d0"])[idx],
                   "has_los": rec["has_los"]}
            cl, meta = cluster_id_rx(sub, angles_required=False, seed=j, **kw)
            pw = np.array([c["power"] for c in cl]) if cl else np.array([])
            out[tag] = {
                "N": meta["n_clusters"],
                "cDS_ns": float(np.average([c["cDS_ns"] for c in cl], weights=pw))
                          if cl else np.nan,
                "rTau": _r_tau_corrected(cl, meta["ds_ns"]) if cl else np.nan,
                "DS_ns": meta.get("ds_self_ns", np.nan),
            }

        disc = np.setdiff1d(np.arange(len(t)), sel, assume_unique=False)
        rows.append({
            "rx": j, "n_paths": len(t),
            **{f"{k}_full": v for k, v in out["full"].items()},
            **{f"{k}_trunc": v for k, v in out["trunc"].items()},
            "disc_power_frac": float(p[disc].sum() / p.sum()) if len(disc) else 0.0,
            "disc_max_rel_db": float(10 * np.log10(p[disc].max() / p.max()))
                               if len(disc) else -np.inf,
            "disc_mean_delay_ns": float((t[disc] - t.min()).mean() * 1e9)
                                  if len(disc) else np.nan,
            "kept_mean_delay_ns": float((t[sel] - t.min()).mean() * 1e9),
        })
    df = pd.DataFrame(rows)
    for k in ("N", "cDS_ns", "rTau", "DS_ns"):
        df[f"{k}_ratio"] = df[f"{k}_trunc"] / df[f"{k}_full"]
    return df
