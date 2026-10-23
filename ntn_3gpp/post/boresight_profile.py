"""Can a coarse boresight-frame angular power profile of the non-direct power reproduce K at a directive UE antenna?
Frame: boresight = satellite direction (the direct path); psi = angle from boresight, chi = position angle around it
(0 = toward the ground side below the boresight, 90 = right, 180 = above, 270 = left; panel rows horizontal).
Profiles: V1 = power share in psi bins (gain averaged over chi); V2 = psi bins x chi quadrants (below/right/above/left);
V3 = the below-horizon power (ground reflection + ground diffuse) on one ray at the ground image direction, as the
generator's '+ ground power' line does, and V1 bins for the rest.
K_antenna = K_iso / sum_b share_b * gain_b, gain_b = the panel gain averaged uniformly over the bin; compared with the RT
panel K of the same UEs (raw 1-in-50 sample, LoS), pooled and as the TR 38.811 mean of K in dB (per-UE profiles).
Usage: python post/boresight_profile.py RUNS SCENARIO BAND"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_band import panel_gain_db, unit

PSI = np.array(os.environ.get("PSI_EDGES", "0,2,5,10,20,45,90,180").split(","), float)   # psi bin edges [deg]
CHI = np.array([-45, 45, 135, 225, 315.0])                                 # chi quadrants [deg]: below, right, above, left


def frame(b):
    """Unit vectors (y right, z up) of the panel plane; chi = 0 points to -z (ground side)."""
    y = np.cross([0.0, 0.0, 1.0], b); y = y / np.linalg.norm(y) if np.linalg.norm(y) > 1e-9 else np.array([0.0, 1.0, 0.0])
    return y, np.cross(b, y)


def bin_gain(b, n, quadrants):
    """Panel gain [linear, re. boresight] averaged uniformly over each psi bin (x chi quadrant)."""
    y, z = frame(b); g0 = panel_gain_db(b[None], b, n)[0]; out = []
    for i in range(len(PSI) - 1):
        ps = np.radians(np.linspace(PSI[i], PSI[i + 1], 25))[:, None, None]
        for j in (range(len(CHI) - 1) if quadrants else [None]):
            cs = np.radians(np.linspace(CHI[j], CHI[j + 1], 37) if j is not None else np.linspace(0, 360, 145))[None, :, None]
            u = (np.cos(ps) * b + np.sin(ps) * (-np.cos(cs) * z + np.sin(cs) * y)).reshape(-1, 3)
            out.append((10 ** ((panel_gain_db(u, b, n) - g0) / 10)).mean())
    return np.array(out)


def bins(u, b, quadrants):
    y, z = frame(b); psi = np.degrees(np.arccos(np.clip(u @ b, -1, 1)))
    chi = np.degrees(np.arctan2(u @ y, -(u @ z))) % 360; chi = np.where(chi >= 315, chi - 360, chi)
    i = np.clip(np.searchsorted(PSI, psi, side="right") - 1, 0, len(PSI) - 2)
    return i * (len(CHI) - 1) + np.clip(np.searchsorted(CHI, chi, side="right") - 1, 0, 3) if quadrants else i


if __name__ == "__main__":
    runs, scen, band = sys.argv[1:4]
    for e in range(10, 100, 10):
        f = f"{runs}/{scen}_{band}/results_elev{e:03d}deg.pkl"
        if not os.path.exists(f): continue
        r = pickle.load(open(f, "rb")); n = int(r["cfg_snapshot"]["UE_PANEL_N"]); raw = [x for x in r["raw_results"] if x["has_los"]]; del r
        G = {}; acc = np.zeros(5); per = {k: [] for k in ("rt", "v1", "v2", "v3")}
        for x in raw:
            ia = np.asarray(x["inter_all"]); p = np.abs(np.asarray(x["amps"])) ** 2; d = (ia != 0).sum(0)
            u = unit(np.asarray(x["theta_r"], float), np.asarray(x["phi_r"], float)); k0 = d == 0; b = u[k0][0]; nd = ~k0
            key = (round(float(b[2]), 3),)                                  # same elevation -> same gains (azimuth only rotates)
            if key not in G: G[key] = (bin_gain(b, n, False), bin_gain(b, n, True))
            g1, g2 = G[key]; pd = p[k0].sum(); pn = p[nd]
            g_rt = 10 ** ((panel_gain_db(u[nd], b, n) - panel_gain_db(b[None], b, n)[0]) / 10)
            s1 = np.bincount(bins(u[nd], b, False), pn, len(g1)); s2 = np.bincount(bins(u[nd], b, True), pn, len(g2))
            lo = u[nd][:, 2] < 0; gimg = 10 ** ((panel_gain_db(b[None] * [1, 1, -1], b, n)[0] - panel_gain_db(b[None], b, n)[0]) / 10)
            s3 = np.bincount(bins(u[nd][~lo], b, False), pn[~lo], len(g1))
            q = [(pn * g_rt).sum(), (s1 * g1).sum(), (s2 * g2).sum(), pn[lo].sum() * gimg + (s3 * g1).sum()]
            acc += [pd, *q]
            for k, v in zip(("rt", "v1", "v2", "v3"), q): per[k].append(10 * np.log10(pd / v))
        kk = lambda i: 10 * np.log10(acc[0] / acc[i])
        print(f"{scen} {band} {e:2d} deg ({len(raw)} LoS UEs) | pooled K at panel: RT {kk(1):5.1f}, V1 psi bins {kk(2):5.1f}, V2 psi x chi {kk(3):5.1f}, V3 ground ray + bins {kk(4):5.1f} dB"
              f" | mean of K dB: RT {np.mean(per['rt']):5.1f}, V1 {np.mean(per['v1']):5.1f}, V2 {np.mean(per['v2']):5.1f}, V3 {np.mean(per['v3']):5.1f} dB", flush=True)
